"""Stage 1 of model tuning (roadmap step 5): fast HGB hyperparameter search.

Streaming TS-AUC is expensive (~350-700s per config), so we first RANK a grid of
HistGradientBoostingClassifier hyperparameters with a cheap proxy: series-grouped
K-fold ROC-AUC on the training feature rows (no streaming, seconds per config).

  * GroupKFold on the series id -> train/val rows never share a series (no leakage).
  * Same fold splits for every config -> fair, low-variance comparison.
  * Rank configs by mean CV AUC; the top 2-3 then get CONFIRMED on the real metric
    with `_val_harness.py` (add them as candidates) before anything ships.

This proxy only orders hyperparameters; the streaming harness + |t|>=2 rule remains the
gate for actually changing the notebook.

Run:  .venv/bin/python -u _tune_hgb.py
"""
import itertools
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from _exp_fast import (
    WINDOW, HIST_CAP, compute_features, hist_context, split_series,
)
from _val_harness import load_split, select_picks

SEED = 42
N_FOLDS = 5
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

# Current shipped model (E7) -- the baseline every config is compared against.
# min_samples_leaf=20 is sklearn's default (stated explicitly so the baseline grid
# point is well-defined).
BASE = dict(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, min_samples_leaf=20,
    early_stopping=True, random_state=SEED,
)

# Hyperparameters to sweep (each combined with BASE; overrides win).
GRID = {
    "learning_rate": [0.03, 0.05, 0.08],
    "max_leaf_nodes": [15, 31, 63],
    "l2_regularization": [0.0, 1.0, 5.0],
    "max_iter": [200, 400],
    "min_samples_leaf": [20, 50],
}


def build_matrix(X, yi, picks):
    rows, labels, groups = [], [], []
    for sid, ts in picks.items():
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        for t in ts:
            lo = max(0, t - WINDOW + 1)
            rows.append(compute_features(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
            groups.append(sid)
    return (np.asarray(rows, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
            np.asarray(groups))


def all_configs():
    keys = list(GRID)
    cfgs = [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]
    # Ensure the exact baseline config is present and first.
    base_over = {k: BASE[k] for k in keys}
    cfgs = [base_over] + [c for c in cfgs if c != base_over]
    return cfgs


def eval_cfg(cfg, Xf, y, splits):
    aucs = []
    for tr, va in splits:
        m = HistGradientBoostingClassifier(**{**BASE, **cfg})
        with threadpool_limits(limits=1):
            m.fit(Xf[tr], y[tr])
        p = m.predict_proba(Xf[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
    a = np.asarray(aucs)
    return float(a.mean()), float(a.std(ddof=1))


def main():
    t0 = time.time()
    X, yi, _y_val, train_ids, _folds = load_split()
    picks = select_picks(X, train_ids)
    Xf, y, groups = build_matrix(X, yi, picks)
    print(f"Feature matrix {Xf.shape}, positives {y.mean():.3f}, "
          f"{len(np.unique(groups))} series | N_JOBS={N_JOBS}")

    splits = list(GroupKFold(n_splits=N_FOLDS).split(Xf, y, groups))
    cfgs = all_configs()
    print(f"Evaluating {len(cfgs)} configs x {N_FOLDS} folds "
          f"(row-level grouped ROC-AUC proxy)...\n")

    tb = time.time()
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(eval_cfg)(c, Xf, y, splits) for c in cfgs
    )
    print(f"Done in {time.time()-tb:.1f}s\n")

    base_mean = results[0][0]
    order = sorted(range(len(cfgs)), key=lambda i: results[i][0], reverse=True)
    base_rank = order.index(0) + 1

    keys = list(GRID)
    print(f"Baseline (E7) proxy AUC {base_mean:.4f}  ->  ranks #{base_rank}/{len(cfgs)}\n")
    print("Top 15 configs by proxy CV AUC:")
    print(f"  {'rank':>4} {'AUC':>7} {'std':>6} {'dAUC':>8}  config")
    for r, i in enumerate(order[:15], 1):
        mean, sd = results[i]
        tag = "  <-- BASELINE" if i == 0 else ""
        cfg = " ".join(f"{k}={cfgs[i][k]}" for k in keys)
        print(f"  {r:>4} {mean:.4f} {sd:.4f} {mean-base_mean:+.4f}  {cfg}{tag}")

    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
