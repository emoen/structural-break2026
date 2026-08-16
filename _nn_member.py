"""NN as a DECORRELATED MEMBER in the E36 fixed-CDF blend (not a feature).

Feature-fusion is dead (ov3 + laptop E35). The winning mechanism (E36 ranker, E38 RF) is
rank-averaging DECORRELATED members with the tuned champion HGB. A neural net's errors are
maximally decorrelated from trees -> the untested, on-theme use of the NN.

Method (mirrors E36 deployable form): each member maps its raw per-step score through its
own FIXED empirical CDF (from training scores), then blend  b = w*Fhat_hgb + (1-w)*Fhat_nn.
TS-AUC is per-step rank-based, so Fhat_hgb-alone (w=1) == raw tuned HGB (baseline). We sweep w,
pair each fold's blend delta vs w=1, across POOL SEEDS with a robustness gate (positive all
seeds & t>=2.5), exactly like E36/E38.

Members: tuned champion HGB (45 feats) + seqhist NN (K-seed ensemble, scalar prob).
Both train on the fit pool; val folds are disjoint -> clean, NO cross-fitting needed for a member.

Usage:
  SMOKE=1 .venv/bin/python -u _nn_member.py
  SEEDS=42,7,123,2024 ENSEMBLE=10 STEPS=80 .venv/bin/python -u _nn_member.py
"""
import math
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from _overnight import DATA, build_training
from _exp_fast import ts_auc_from
from _ensemble_tune import (
    FOLD_SIZE, N_FOLDS, N_TRAIN, SMOKE,
    champion_feat, make_hgb, precompute_fold, score_fold, select_picks,
)
from _nn_twosample import WINDOW, STEPS, build_training_pairs, pair_vec, select_picks_n
from _nn_fusion import ENSEMBLE, NN_EPOCHS, TUNED_HGB, EnsembleProba, train_ensemble

WS = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7,123,2024").split(",")]
if SMOKE:
    SEEDS = [42, 7]


def load_pool_strat_seed(seed):
    """_ensemble_tune.load_pool_strat but with an explicit pool seed (varies the split)."""
    rng = np.random.default_rng(seed)
    yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
    ids = yi.index.to_numpy()
    has_break = (yi["tau_index"].to_numpy() >= 0).astype(int)
    pos, neg = ids[has_break == 1].copy(), ids[has_break == 0].copy()
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val = N_FOLDS * FOLD_SIZE
    p = len(pos) / len(ids)
    n_val_pos = int(round(n_val * p))
    n_val_neg = n_val - n_val_pos
    val_pos, val_neg = pos[:n_val_pos], neg[:n_val_neg]
    folds = [[] for _ in range(N_FOLDS)]
    for i, sid in enumerate(val_pos):
        folds[i % N_FOLDS].append(sid)
    for i, sid in enumerate(val_neg):
        folds[i % N_FOLDS].append(sid)
    folds = [np.sort(np.asarray(f)) for f in folds]
    val_ids = np.concatenate(folds)
    train_pool = np.concatenate([pos[n_val_pos:], neg[n_val_neg:]])
    rng.shuffle(train_pool)
    train_ids = np.sort(train_pool[:N_TRAIN])
    keep = set(train_ids.tolist()) | set(val_ids.tolist())
    X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
    X = X[X.index.get_level_values("id").isin(keep)]
    y = pd.read_parquet(os.path.join(DATA, "y_train.parquet"))
    y_val = y[y.index.get_level_values("id").isin(set(val_ids.tolist()))]
    return X, yi, y_val, train_ids, folds


def cdf_fn(train_scores):
    """Fixed empirical CDF from training scores (deployable rank transform)."""
    s = np.sort(np.asarray(train_scores, dtype=np.float64))
    n = max(len(s), 1)
    return lambda x: np.searchsorted(s, x, side="right") / n


def run_seed(seed):
    """Return per-fold TS-AUC arrays for each w (w=1 == tuned HGB alone)."""
    X, yi, y_val, train_ids, folds = load_pool_strat_seed(seed)
    picks_c = select_picks(X, train_ids)
    picks_n = select_picks_n(X, train_ids, STEPS)

    # --- tuned HGB member ---
    Xtr_c, ytr = build_training(X, yi, picks_c, champion_feat, WINDOW)
    hgb = make_hgb(**TUNED_HGB)
    with threadpool_limits(limits=1):
        hgb.fit(Xtr_c, ytr)
    F_hgb = cdf_fn(hgb.predict_proba(Xtr_c)[:, 1])

    # --- NN member (K-seed ensemble, scalar prob) ---
    Xtr_p, ytr_p = build_training_pairs(X, yi, picks_n, WINDOW)
    nets = train_ensemble(Xtr_p, ytr_p, ENSEMBLE, NN_EPOCHS, seed)
    nn = EnsembleProba(nets)
    Xrows_p, _ = build_training_pairs(X, yi, picks_c, WINDOW)  # NN scores on the HGB fit rows
    F_nn = cdf_fn(nn.predict_proba(Xrows_p)[:, 1])

    # --- per-fold val scores -> CDF -> blend sweep ---
    champ_caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
    pair_caches = [precompute_fold(pair_vec, WINDOW, X, f) for f in folds]
    per_w = {w: [] for w in WS}
    for cc, pc in zip(champ_caches, pair_caches):
        hs = score_fold(hgb, cc)
        ns = score_fold(nn, pc)
        fh = F_hgb(hs.values)
        fn = F_nn(ns.values)
        for w in WS:
            blended = pd.Series(w * fh + (1.0 - w) * fn, index=hs.index)
            per_w[w].append(ts_auc_from(blended.to_frame("prediction"), y_val))
    return {w: np.asarray(v) for w, v in per_w.items()}


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== nn member ({mode}) {datetime.now():%F %T} | seeds={SEEDS} "
          f"ENSEMBLE={ENSEMBLE} STEPS={STEPS} ===", flush=True)

    seed_w = {}
    for seed in SEEDS:
        ts = time.time()
        aucs = run_seed(seed)
        base = aucs[1.0]
        seed_w[seed] = {}
        line = [f"seed {seed} base={base.mean():.4f}"]
        for w in WS:
            if w == 1.0:
                continue
            d = aucs[w] - base
            md, sd = float(d.mean()), float(d.std(ddof=1))
            t = md / (sd / math.sqrt(len(d))) if sd > 0 else float("inf")
            seed_w[seed][w] = (md, t)
            line.append(f"w{w}:{md:+.4f}(t{t:+.1f})")
        print("  ".join(line) + f"  [{(time.time() - ts) / 60:.1f}m]", flush=True)

    print("\n=== SUMMARY (blend vs tuned HGB alone) ===", flush=True)
    for w in WS:
        if w == 1.0:
            continue
        ds = [seed_w[s][w][0] for s in SEEDS]
        ts_ = [seed_w[s][w][1] for s in SEEDS]
        npos = sum(1 for x in ds if x > 0)
        gate = "CONFIRM" if npos == len(SEEDS) and min(ts_) >= 2.5 else (
            "hold" if npos == len(SEEDS) else "noise")
        print(f"w={w}: mean_d {np.mean(ds):+.4f} min_d {min(ds):+.4f} "
              f"pos {npos}/{len(SEEDS)} mean_t {np.mean(ts_):+.2f} min_t {min(ts_):+.2f} -> {gate}",
              flush=True)
    print(f"=== DONE in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
