"""Window-geometry tuning (roadmap step 5b): sweep WINDOW with the fast proxy.

Same two-stage discipline as _tune_hgb.py: rank WINDOW lengths cheaply with a
series-grouped row-level ROC-AUC proxy (no streaming, seconds each), then confirm the
best 1-2 on _val_harness streaming TS-AUC + paired |t|>=2 before touching the notebook.

The sampled training positions are fixed (MIN_N=15, shared across all windows) so the
only thing that changes per candidate is the trailing-window length feeding
compute_features -> an apples-to-apples comparison. MIN_N (warm-up) mostly affects early
low-information steps the row proxy can't see, so it is a streaming-stage question, not
tuned here.

Run:  .venv/bin/python -u _sweep_window.py
"""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from _exp_fast import HIST_CAP, compute_features, hist_context, split_series
from _val_harness import load_split, select_picks

SEED = 42
N_FOLDS = 5
BASELINE_WINDOW = 75
WINDOW_GRID = [75, 130, 160, 200, 250, 300, 400]

BASE = dict(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, min_samples_leaf=20,
    early_stopping=True, random_state=SEED,
)


def build_matrix(X, yi, picks, window):
    rows, labels, groups = [], [], []
    for sid, ts in picks.items():
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        for t in ts:
            lo = max(0, t - window + 1)
            rows.append(compute_features(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
            groups.append(sid)
    return (np.asarray(rows, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
            np.asarray(groups))


def cv_auc(Xf, y, splits):
    aucs = []
    for tr, va in splits:
        m = HistGradientBoostingClassifier(**BASE)
        with threadpool_limits(limits=1):
            m.fit(Xf[tr], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(Xf[va])[:, 1]))
    a = np.asarray(aucs)
    return float(a.mean()), float(a.std(ddof=1))


def main():
    t0 = time.time()
    X, yi, _y_val, train_ids, _folds = load_split()
    picks = select_picks(X, train_ids)
    print(f"{sum(len(v) for v in picks.values())} rows from {len(picks)} series | "
          f"windows {WINDOW_GRID}\n")

    splits = None
    results = {}
    for w in WINDOW_GRID:
        tb = time.time()
        Xf, y, groups = build_matrix(X, yi, picks, w)
        if splits is None:  # groups identical across windows -> split once
            splits = list(GroupKFold(n_splits=N_FOLDS).split(Xf, y, groups))
        mean, sd = cv_auc(Xf, y, splits)
        results[w] = (mean, sd)
        print(f"  WINDOW={w:>3}: proxy AUC {mean:.4f} std {sd:.4f}  ({time.time()-tb:.0f}s)")

    base = results[BASELINE_WINDOW][0]
    print(f"\nBaseline WINDOW={BASELINE_WINDOW}: {base:.4f}")
    print("Ranked by proxy CV AUC:")
    for w in sorted(results, key=lambda k: results[k][0], reverse=True):
        mean, sd = results[w]
        tag = "  <-- BASELINE" if w == BASELINE_WINDOW else ""
        print(f"  WINDOW={w:>3}: {mean:.4f} (std {sd:.4f})  dAUC {mean-base:+.4f}{tag}")
    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
