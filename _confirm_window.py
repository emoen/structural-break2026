"""Stage 2 of window tuning: confirm candidate WINDOW lengths on streaming TS-AUC.

The proxy (_sweep_window.py) says bigger WINDOW is much better, but it scores uniformly
sampled positions and can't see the reaction-lag near a break that TS-AUC weights. This
harness settles it on the real metric: per-fold streaming TS-AUC across the 5x400 disjoint
folds, paired vs the baseline WINDOW=75 on the same folds, with t = mean_d/(std_d/sqrt(K)).
Ship a new WINDOW only if the paired gain clears the noise floor (|t| >= 2).

MIN_N (warm-up) is held at the current 15 so WINDOW is the only variable.

Run:  .venv/bin/python -u _confirm_window.py
"""
import math
import os
import time
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from _exp_fast import (
    HIST_CAP, PREDICT_EVERY, compute_features, hist_context, split_series, ts_auc_from,
)
from _val_harness import load_split, select_picks

SEED = 42
MIN_N = 15
BASELINE_WINDOW = 75
WINDOWS = [75, 130, 200]     # first is the baseline
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

BASE = dict(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, min_samples_leaf=20,
    early_stopping=True, random_state=SEED,
)


def build_training(X, yi, picks, window):
    rows, labels = [], []
    for sid, ts in picks.items():
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        for t in ts:
            lo = max(0, t - window + 1)
            rows.append(compute_features(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _stream_one(model, window, sid, x_h, x_on, onl_index):
    ctx = hist_context(x_h, HIST_CAP)
    buf = deque()
    out = np.zeros(len(x_on))
    score = 0.0
    for idx, point in enumerate(x_on):
        if len(buf) == window:
            buf.popleft()
        buf.append(float(point))
        if len(buf) < MIN_N:
            out[idx] = 0.0
            continue
        if idx % PREDICT_EVERY == 0:
            feats = np.asarray(compute_features(np.asarray(buf), ctx),
                               dtype=np.float64).reshape(1, -1)
            score = float(model.predict_proba(feats)[0, 1])
        out[idx] = score
    midx = pd.MultiIndex.from_product([[sid], onl_index], names=["id", "time"])
    return pd.Series(out, index=midx)


def fold_auc(model, window, X, y_val, fold_ids):
    jobs = []
    for sid in fold_ids:
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, x_h, onl.to_numpy(dtype=np.float64), onl.index))
    with threadpool_limits(limits=1):
        series = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_stream_one)(model, window, sid, x_h, x_on, oi)
            for (sid, x_h, x_on, oi) in jobs
        )
    return ts_auc_from(pd.concat(series).to_frame("prediction"), y_val)


def evaluate(window, X, yi, y_val, picks, folds):
    tb = time.time()
    Xtr, ytr = build_training(X, yi, picks, window)
    model = HistGradientBoostingClassifier(**BASE)
    with threadpool_limits(limits=1):
        model.fit(Xtr, ytr)
    ti = time.time()
    aucs = np.asarray([fold_auc(model, window, X, y_val, f) for f in folds])
    print(f"[WINDOW={window:>3}] folds {' '.join(f'{v:.4f}' for v in aucs)} | "
          f"mean {aucs.mean():.4f} std {aucs.std(ddof=1):.4f} | "
          f"train {ti-tb:.0f}s infer {time.time()-ti:.0f}s")
    return aucs


def main():
    t0 = time.time()
    X, yi, y_val, train_ids, folds = load_split()
    picks = select_picks(X, train_ids)
    print(f"{sum(len(v) for v in picks.values())} train rows | windows {WINDOWS} | "
          f"N_JOBS={N_JOBS}\n")

    results = {w: evaluate(w, X, yi, y_val, picks, folds) for w in WINDOWS}

    base = results[BASELINE_WINDOW]
    print(f"\nNoise floor (WINDOW={BASELINE_WINDOW} fold std): {base.std(ddof=1):.4f}")
    print("Paired deltas vs baseline WINDOW=75 (same folds):")
    for w in WINDOWS:
        if w == BASELINE_WINDOW:
            continue
        d = results[w] - base
        md, sd = d.mean(), d.std(ddof=1)
        t = md / (sd / math.sqrt(len(d))) if sd > 0 else float("inf")
        verdict = "SHIP" if abs(t) >= 2 and md > 0 else "noise/regression"
        print(f"  WINDOW={w:>3}: delta {' '.join(f'{v:+.4f}' for v in d)} | "
              f"mean {md:+.4f} std {sd:.4f} t={t:+.2f} -> {verdict}")
    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
