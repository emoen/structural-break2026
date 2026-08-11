"""Step 3 sweep: do extra window-scale / within-window features help TS-AUC?

Holds train ids, val set, PREDICT_EVERY and model fixed; varies ONLY the feature set,
so the delta is the pure feature effect. Uses PE=5 for a fast relative comparison
(feature ranking is ~independent of PE); the winner is later confirmed at PE=1.

Feature groups (added on top of the base 14):
  short : mean_z / log_var_ratio / ks on the last SHORT points        (+3)
  split : recent-half vs older-half mean_z / log_var_ratio / ks       (+3)
"""
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import _exp_fast as ef

PE = 5
N_TRAIN = 6000
N_VAL = 400
SHORT = 25
MIN_HALF = 6  # min points per half for the split features


def _pair_stats(recent, older, sample_sorted_for_ks=None):
    """mean_z, log_var_ratio, ks between two samples (recent vs older/reference)."""
    nr, no = len(recent), len(older)
    mr, mo = float(recent.mean()), float(older.mean())
    vr = max(float(recent.var()), 1e-12)
    vo = max(float(older.var()), 1e-12)
    pooled_sd = math.sqrt((vr / max(nr, 1)) + (vo / max(no, 1)))
    mean_z = (mr - mo) / max(pooled_sd, 1e-8)
    log_var_ratio = math.log(vr / vo)
    ks = ef._fast_ks(np.sort(recent), np.sort(older))
    return mean_z, log_var_ratio, ks


def make_feature_fn(use_short, use_split):
    base = ef.compute_features

    def fn(w, ctx):
        feats = list(base(w, ctx))
        n = len(w)
        if use_short:
            k = min(SHORT, n)
            ws = w[-k:]
            mean_ws = float(ws.mean())
            var_ws = max(float(ws.var()), 1e-12)
            se = ctx["sd"] / math.sqrt(k)
            feats.append((mean_ws - ctx["mu"]) / max(se, 1e-8))
            feats.append(math.log(var_ws / ctx["var"]))
            feats.append(ef._fast_ks(np.sort(ws), ctx["sample_sorted"]))
        if use_split:
            h = n // 2
            if h >= MIN_HALF and (n - h) >= MIN_HALF:
                older = w[:h]
                recent = w[h:]
                mz, lvr, ks = _pair_stats(recent, older)
                feats.extend([mz, lvr, ks])
            else:
                feats.extend([0.0, 0.0, 0.0])
        return feats

    return fn


def build(X, yi, ids, feat_fn, rng):
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
        k = min(ef.STEPS_PER_SERIES, len(cand))
        picks = rng.choice(cand, size=k, replace=False)
        for t in picks:
            lo = max(0, t - ef.WINDOW + 1)
            rows.append(feat_fn(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def stream(model, x_h, x_on, feat_fn):
    from collections import deque
    ctx = ef.hist_context(x_h, ef.HIST_CAP)
    buf = deque()
    out = np.zeros(len(x_on))
    score = 0.0
    for idx, point in enumerate(x_on):
        if len(buf) == ef.WINDOW:
            buf.popleft()
        buf.append(float(point))
        if len(buf) < ef.MIN_N:
            out[idx] = 0.0
            continue
        if idx % PE == 0:
            feats = np.asarray(feat_fn(np.asarray(buf), ctx), dtype=np.float64).reshape(1, -1)
            score = float(model.predict_proba(feats)[0, 1])
        out[idx] = score
    return out


def evaluate(model, val, y, feat_fn):
    cols = []
    for sid, x_h, x_on, midx in val:
        cols.append(pd.Series(stream(model, x_h, x_on, feat_fn), index=midx))
    return ef.ts_auc_from(pd.concat(cols).to_frame("prediction"), y)


def main():
    t0 = time.time()
    rng0 = np.random.default_rng(ef.SEED)
    yi = pd.read_parquet(os.path.join(ef.DATA, "y_train_index.parquet"))
    all_ids = yi.index.to_numpy().copy()
    rng0.shuffle(all_ids)
    train_ids = np.sort(all_ids[:N_TRAIN])
    val_ids = np.sort(all_ids[N_TRAIN:N_TRAIN + N_VAL])
    keep = set(train_ids.tolist()) | set(val_ids.tolist())

    print("Loading...")
    X = pd.read_parquet(os.path.join(ef.DATA, "X_train.parquet"))
    y = pd.read_parquet(os.path.join(ef.DATA, "y_train.parquet"))
    X = X[X.index.get_level_values("id").isin(keep)]
    y = y[y.index.get_level_values("id").isin(set(val_ids.tolist()))]
    print(f"  loaded in {time.time()-t0:.1f}s (PE={PE}, train={N_TRAIN}, val={N_VAL})")

    val = []
    for sid in val_ids:
        g = X.loc[sid]
        x_h, onl = ef.split_series(g)
        x_on = onl.to_numpy(dtype=np.float64)
        midx = pd.MultiIndex.from_product([[sid], onl.index], names=["id", "time"])
        val.append((sid, x_h, x_on, midx))

    groups = [
        ("base", False, False),
        ("base+short", True, False),
        ("base+split", False, True),
        ("base+short+split", True, True),
    ]
    print(f"{'group':>18} {'nfeat':>6} {'TS-AUC':>8} {'infer_s':>8}")
    for name, us, up in groups:
        fn = make_feature_fn(us, up)
        rng = np.random.default_rng(ef.SEED)
        Xtr, ytr = build(X, yi, train_ids, fn, rng)
        model = HistGradientBoostingClassifier(
            max_iter=ef.MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, random_state=ef.SEED,
        )
        model.fit(Xtr, ytr)
        ti = time.time()
        auc = evaluate(model, val, y, fn)
        print(f"{name:>18} {Xtr.shape[1]:>6} {auc:>8.4f} {time.time()-ti:>8.1f}")

    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
