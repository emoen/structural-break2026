"""Step 2 sweep: does more / denser training data improve TS-AUC?

Loads once, holds a fixed disjoint 400-series val set, and compares training configs
(number of series, sampled steps/series, and near-tau densification). Val inference uses
PREDICT_EVERY=1 to match the current submission (E6).
"""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import _exp_fast as ef

ef.PREDICT_EVERY = 1
N_VAL = 400
MAX_TRAIN = 9000  # 9000 + 400 val <= 10000 pool


def build(X, yi, ids, steps, rng, near_tau_frac=0.0, near_tau_halfwidth=20):
    """Build training rows. near_tau_frac of picks are drawn near tau (break series)."""
    rows, labels = [], []
    for sid in ids:
        g = X.loc[sid]
        x_h, onl = ef.split_series(g)
        x_on = onl.to_numpy(dtype=np.float64)
        L = len(x_on)
        if L <= ef.MIN_N:
            continue
        ctx = ef.hist_context(x_h, ef.HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        cand = np.arange(ef.MIN_N, L)
        k = min(steps, len(cand))
        if near_tau_frac > 0.0 and tau >= ef.MIN_N:
            k_near = int(round(k * near_tau_frac))
            near_lo = max(ef.MIN_N, tau - near_tau_halfwidth)
            near_hi = min(L, tau + near_tau_halfwidth)
            near_pool = np.arange(near_lo, near_hi)
            picks_near = rng.choice(near_pool, size=min(k_near, len(near_pool)), replace=False) \
                if len(near_pool) else np.array([], dtype=int)
            k_rest = k - len(picks_near)
            picks_rest = rng.choice(cand, size=min(k_rest, len(cand)), replace=False)
            picks = np.concatenate([picks_near, picks_rest])
        else:
            picks = rng.choice(cand, size=k, replace=False)
        for t in picks:
            lo = max(0, t - ef.WINDOW + 1)
            w = x_on[lo:t + 1]
            rows.append(ef.compute_features(w, ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def evaluate(model, val, y):
    ef._T["compute"] = ef._T["predict"] = 0.0
    cols = []
    for sid, x_h, x_on, midx in val:
        preds, _ = ef.stream_predict(model, x_h, x_on)
        cols.append(pd.Series(preds, index=midx))
    preds = pd.concat(cols).to_frame("prediction")
    return ef.ts_auc_from(preds, y)


def main():
    t0 = time.time()
    rng0 = np.random.default_rng(ef.SEED)
    yi = pd.read_parquet(os.path.join(ef.DATA, "y_train_index.parquet"))
    all_ids = yi.index.to_numpy().copy()
    rng0.shuffle(all_ids)
    train_pool = np.sort(all_ids[:MAX_TRAIN])
    val_ids = np.sort(all_ids[MAX_TRAIN:MAX_TRAIN + N_VAL])
    keep = set(train_pool.tolist()) | set(val_ids.tolist())

    print("Loading...")
    X = pd.read_parquet(os.path.join(ef.DATA, "X_train.parquet"))
    y = pd.read_parquet(os.path.join(ef.DATA, "y_train.parquet"))
    X = X[X.index.get_level_values("id").isin(keep)]
    y = y[y.index.get_level_values("id").isin(set(val_ids.tolist()))]
    print(f"  loaded in {time.time()-t0:.1f}s")

    val = []
    for sid in val_ids:
        g = X.loc[sid]
        x_h, onl = ef.split_series(g)
        x_on = onl.to_numpy(dtype=np.float64)
        midx = pd.MultiIndex.from_product([[sid], onl.index], names=["id", "time"])
        val.append((sid, x_h, x_on, midx))

    # (n_series, steps, near_tau_frac)
    configs = [
        (5000, 10, 0.0),   # E4/E6 baseline recipe
        (9000, 10, 0.0),
        (9000, 20, 0.0),
        (9000, 40, 0.0),
        (9000, 20, 0.5),   # densify near tau
    ]
    print(f"{'n_ser':>6} {'steps':>6} {'near':>5} {'rows':>8} {'pos':>6} {'trainS':>7} {'TS-AUC':>8}")
    for n_ser, steps, near in configs:
        rng = np.random.default_rng(ef.SEED)  # same seed for fair comparison
        ids = np.sort(train_pool[:n_ser])
        tb = time.time()
        Xtr, ytr = build(X, yi, ids, steps, rng, near_tau_frac=near)
        model = HistGradientBoostingClassifier(
            max_iter=ef.MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, random_state=ef.SEED,
        )
        model.fit(Xtr, ytr)
        train_s = time.time() - tb
        auc = evaluate(model, val, y)
        print(f"{n_ser:>6} {steps:>6} {near:>5.2f} {Xtr.shape[0]:>8} "
              f"{ytr.mean():>6.3f} {train_s:>7.1f} {auc:>8.4f}")

    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
