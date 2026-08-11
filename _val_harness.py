"""Robust multi-fold validation harness for candidate feature sets.

Motivation (E8): a single 400-series val split is too noisy to trust ~0.006 TS-AUC
deltas -- the recent-vs-older "split" features looked like a +0.006 local win but
LOST 0.24 pt on the full 10000-series leaderboard. This harness fixes the methodology:

  * Train ONE model per candidate on a fixed training pool (disjoint from all folds).
  * Evaluate that model on K DISJOINT validation folds.
  * Report per-fold TS-AUC, mean, std (the noise floor) and min/max.
  * For each candidate, report the PAIRED delta vs the base model on the SAME folds,
    plus a t-like statistic  t = mean_delta / (std_delta / sqrt(K)).

Decision rule: only port a feature change to the submission notebook when its paired
mean gain is clearly larger than the fold-to-fold std (noise floor) -- ideally |t| >= 2.

Candidates are plain module-level functions (w, ctx) -> list[float] so they pickle
cleanly for joblib workers. Add a new one by registering it in CANDIDATES.

Run:  .venv/bin/python _val_harness.py
"""
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

# Reuse the exact feature helpers + base feature set from the validation module so
# the harness measures the same pipeline the notebook ships.
from _exp_fast import (
    WINDOW, MIN_N, HIST_CAP, PREDICT_EVERY, MAX_ITER,
    hist_context, split_series, ts_auc_from, _fast_ks,
    compute_features as base_features,
    FEATURE_NAMES as BASE_NAMES,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_TRAIN_SERIES = 6000     # training pool (disjoint from every fold)
N_FOLDS = 5               # disjoint validation folds
FOLD_SIZE = 400           # series per fold  -> N_FOLDS * FOLD_SIZE distinct val series
STEPS_PER_SERIES = 10     # training rows sampled per series
N_JOBS = max(1, (os.cpu_count() or 4) - 1)


# ---------------------------------------------------------------------------
# Candidate feature functions (module-level so joblib can pickle them)
# ---------------------------------------------------------------------------
def split_features(w, ctx):
    """Base 14 features + 3 recent-vs-older split features (the E8 candidate).

    Kept here (NOT in the submission) purely so the harness can re-test whether the
    single-split "win" survives multi-fold evaluation.
    """
    feats = base_features(w, ctx)
    n = len(w)
    h = n // 2
    if h >= 6 and (n - h) >= 6:
        older = w[:h]
        recent = w[h:]
        vr = max(float(recent.var()), 1e-12)
        vo = max(float(older.var()), 1e-12)
        pooled = math.sqrt(vr / len(recent) + vo / len(older))
        split_mean_z = (float(recent.mean()) - float(older.mean())) / max(pooled, 1e-8)
        split_log_var_ratio = math.log(vr / vo)
        split_ks = _fast_ks(np.sort(recent), np.sort(older))
    else:
        split_mean_z = split_log_var_ratio = split_ks = 0.0
    feats.extend([split_mean_z, split_log_var_ratio, split_ks])
    return feats


SPLIT_NAMES = BASE_NAMES + ["split_mean_z", "split_log_var_ratio", "split_ks"]


# --- Step 4: robust / rank-based window-vs-history contrasts -----------------
# All compare the trailing window to the historical sample (ctx["sample_sorted"],
# already sorted, <= HIST_CAP points). The window analogues of mean_z / log_var_ratio
# / ks, but built from medians, MAD and ranks -> insensitive to heavy tails/outliers.
def _hist_robust(ctx):
    """Median, MAD (scaled), sorted hist sample, and mean/var of |hist - med_h|.
    Cached on the (per-series, mutable) ctx dict so it is computed once per series."""
    r = ctx.get("_rob")
    if r is None:
        s = ctx["sample_sorted"]
        med = float(np.median(s))
        absdev = np.abs(s - med)
        mad = max(float(np.median(absdev)) * 1.4826, 1e-12)
        r = (med, mad, s, float(absdev.mean()), float(absdev.var()))
        ctx["_rob"] = r
    return r


def _med_mad(w):
    med = float(np.median(w))
    mad = max(float(np.median(np.abs(w - med))) * 1.4826, 1e-12)
    return med, mad


def _mwu_z(w, hist_sorted):
    """Standardized Mann-Whitney U (rank-based location shift), z ~ N(0,1)."""
    n, m = len(w), len(hist_sorted)
    left = np.searchsorted(hist_sorted, w, side="left")
    right = np.searchsorted(hist_sorted, w, side="right")
    u = float(left.sum()) + 0.5 * float((right - left).sum())
    sig = math.sqrt(n * m * (n + m + 1) / 12.0)
    return (u - n * m / 2.0) / sig if sig > 1e-12 else 0.0


def robust_loc_features(w, ctx):
    """base14 + robust location/scale: median-z and log-MAD-ratio vs history."""
    feats = base_features(w, ctx)
    med_h, mad_h, _s, _zm, _zv = _hist_robust(ctx)
    med_w, mad_w = _med_mad(w)
    n = len(w)
    med_z = (med_w - med_h) / max(mad_h / math.sqrt(n), 1e-8)
    log_mad_ratio = math.log(mad_w / mad_h)
    feats.extend([med_z, log_mad_ratio])
    return feats


def mwu_features(w, ctx):
    """base14 + Mann-Whitney rank-location shift vs history."""
    feats = base_features(w, ctx)
    _med, _mad, s, _zm, _zv = _hist_robust(ctx)
    feats.append(_mwu_z(w, s))
    return feats


def bf_features(w, ctx):
    """base14 + Brown-Forsythe robust scale change (|dev-from-median| contrast)."""
    feats = base_features(w, ctx)
    med_h, _mad, s, zh_mean, zh_var = _hist_robust(ctx)
    med_w, _ = _med_mad(w)
    n, m = len(w), len(s)
    zw = np.abs(w - med_w)
    denom = math.sqrt(zw.var() / n + zh_var / m)
    bf = (float(zw.mean()) - zh_mean) / max(denom, 1e-8)
    feats.append(bf)
    return feats


def robust_all_features(w, ctx):
    """base14 + all four robust features (median-z, log-MAD, MWU, Brown-Forsythe)."""
    feats = base_features(w, ctx)
    med_h, mad_h, s, zh_mean, zh_var = _hist_robust(ctx)
    med_w, mad_w = _med_mad(w)
    n, m = len(w), len(s)
    med_z = (med_w - med_h) / max(mad_h / math.sqrt(n), 1e-8)
    log_mad_ratio = math.log(mad_w / mad_h)
    mwu = _mwu_z(w, s)
    zw = np.abs(w - med_w)
    denom = math.sqrt(zw.var() / n + zh_var / m)
    bf = (float(zw.mean()) - zh_mean) / max(denom, 1e-8)
    feats.extend([med_z, log_mad_ratio, mwu, bf])
    return feats


# name -> (feature_fn, feature_names).  First entry is treated as the baseline.
CANDIDATES = {
    "base14": (base_features, BASE_NAMES),
    "rob_loc": (robust_loc_features, BASE_NAMES + ["med_z", "log_mad_ratio"]),
    "mwu": (mwu_features, BASE_NAMES + ["mwu_z"]),
    "bf": (bf_features, BASE_NAMES + ["bf_scale"]),
    "rob_all": (robust_all_features,
                BASE_NAMES + ["med_z", "log_mad_ratio", "mwu_z", "bf_scale"]),
}


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
def load_split():
    """Return (X, yi, y_val, train_ids, folds) with disjoint train pool + K folds."""
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "data")
    rng = np.random.default_rng(SEED)
    yi = pd.read_parquet(os.path.join(data, "y_train_index.parquet"))
    all_ids = yi.index.to_numpy().copy()
    rng.shuffle(all_ids)

    train_ids = np.sort(all_ids[:N_TRAIN_SERIES])
    rest = all_ids[N_TRAIN_SERIES:N_TRAIN_SERIES + N_FOLDS * FOLD_SIZE]
    folds = [np.sort(rest[i * FOLD_SIZE:(i + 1) * FOLD_SIZE]) for i in range(N_FOLDS)]
    val_ids = np.concatenate(folds)
    keep = set(train_ids.tolist()) | set(val_ids.tolist())

    print(f"Loading data (train pool {len(train_ids)}, "
          f"{N_FOLDS} folds x {FOLD_SIZE} = {len(val_ids)} val series)...")
    t0 = time.time()
    X = pd.read_parquet(os.path.join(data, "X_train.parquet"))
    y = pd.read_parquet(os.path.join(data, "y_train.parquet"))
    X = X[X.index.get_level_values("id").isin(keep)]
    y_val = y[y.index.get_level_values("id").isin(set(val_ids.tolist()))]
    print(f"  loaded in {time.time()-t0:.1f}s")
    return X, yi, y_val, train_ids, folds


def select_picks(X, train_ids):
    """Pre-select the sampled online indices per series ONCE, so every candidate
    trains on the same rows (only the feature values differ) -- a fair comparison."""
    rng = np.random.default_rng(SEED + 1)
    picks = {}
    for sid in train_ids:
        _, onl = split_series(X.loc[sid])
        L = len(onl)
        if L <= MIN_N:
            continue
        cand = np.arange(MIN_N, L)
        k = min(STEPS_PER_SERIES, len(cand))
        picks[sid] = rng.choice(cand, size=k, replace=False)
    return picks


# ---------------------------------------------------------------------------
# Train / stream
# ---------------------------------------------------------------------------
def build_training(X, yi, picks, feature_fn):
    rows, labels = [], []
    for sid, ts in picks.items():
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        for t in ts:
            lo = max(0, t - WINDOW + 1)
            rows.append(feature_fn(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _stream_one(model, feature_fn, sid, x_h, x_on, onl_index):
    """Stream a single series (runs inside a joblib worker; BLAS capped to 1)."""
    from collections import deque
    ctx = hist_context(x_h, HIST_CAP)
    buf = deque()
    out = np.zeros(len(x_on))
    score = 0.0
    for idx, point in enumerate(x_on):
        if len(buf) == WINDOW:
            buf.popleft()
        buf.append(float(point))
        if len(buf) < MIN_N:
            out[idx] = 0.0
            continue
        if idx % PREDICT_EVERY == 0:
            feats = np.asarray(feature_fn(np.asarray(buf), ctx),
                               dtype=np.float64).reshape(1, -1)
            score = float(model.predict_proba(feats)[0, 1])
        out[idx] = score
    midx = pd.MultiIndex.from_product([[sid], onl_index], names=["id", "time"])
    return pd.Series(out, index=midx)


def fold_auc(model, feature_fn, X, y_val, fold_ids):
    jobs = []
    for sid in fold_ids:
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, x_h, onl.to_numpy(dtype=np.float64), onl.index))
    with threadpool_limits(limits=1):
        series = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_stream_one)(model, feature_fn, sid, x_h, x_on, oi)
            for (sid, x_h, x_on, oi) in jobs
        )
    preds = pd.concat(series).to_frame("prediction")
    return ts_auc_from(preds, y_val)


def evaluate(name, feature_fn, X, yi, y_val, picks, folds):
    tb = time.time()
    Xtr, ytr = build_training(X, yi, picks, feature_fn)
    model = HistGradientBoostingClassifier(
        max_iter=MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, random_state=SEED,
    )
    with threadpool_limits(limits=1):
        model.fit(Xtr, ytr)
    build_t = time.time() - tb

    ti = time.time()
    aucs = [fold_auc(model, feature_fn, X, y_val, f) for f in folds]
    infer_t = time.time() - ti
    a = np.asarray(aucs)
    print(f"[{name:>8}] {Xtr.shape[1]} feats | folds "
          f"{' '.join(f'{v:.4f}' for v in aucs)} | "
          f"mean {a.mean():.4f} std {a.std(ddof=1):.4f} "
          f"[{a.min():.4f},{a.max():.4f}] | "
          f"train {build_t:.0f}s infer {infer_t:.0f}s")
    return a


def main():
    t0 = time.time()
    X, yi, y_val, train_ids, folds = load_split()
    picks = select_picks(X, train_ids)
    print(f"Training pool: {sum(len(v) for v in picks.values())} rows "
          f"from {len(picks)} series | N_JOBS={N_JOBS}\n")

    names = list(CANDIDATES)
    results = {}
    for name in names:
        fn, _ = CANDIDATES[name]
        results[name] = evaluate(name, fn, X, yi, y_val, picks, folds)

    base = names[0]
    print(f"\nNoise floor (base '{base}' fold std): {results[base].std(ddof=1):.4f}")
    print("Paired deltas vs base (same folds):")
    for name in names[1:]:
        d = results[name] - results[base]
        md, sd = d.mean(), d.std(ddof=1)
        t = md / (sd / math.sqrt(len(d))) if sd > 0 else float("inf")
        verdict = "SHIP" if abs(t) >= 2 and md > 0 else "noise/regression"
        print(f"  {name:>8}: delta per fold {' '.join(f'{v:+.4f}' for v in d)} | "
              f"mean {md:+.4f} std {sd:.4f} t={t:+.2f} -> {verdict}")
    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
