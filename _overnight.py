"""Overnight battery runner (roadmap step 5c + feature search).

Runs UNATTENDED for a few hours and writes results incrementally so partial progress
survives anything. Two sections:

  A. WINDOW confirmation at higher power: base14 @ WINDOW=75 vs @ WINDOW=130 across 10
     disjoint folds (settles the borderline t=1.73 from the 5-fold run).
  B. Feature battery: base14 vs base14+<group> for ~40 candidate feature groups/combos,
     at WINDOW=75, on the same 5 folds used by _val_harness (base14 reference == 0.5511).

The 10-fold pool is built so its FIRST 5 folds are identical to the _val_harness 5-fold
split (same seed + carving), so section B reproduces the established baseline exactly and
section A reuses the same streaming.

Every candidate appends one row to _overnight_results.tsv (flushed), wrapped in
try/except so one failure can't kill the night. A ranked summary is written at the end.

Usage:
  SMOKE=1 .venv/bin/python -u _overnight.py     # ~1-2 min end-to-end self-test
  nohup .venv/bin/python -u _overnight.py > _overnight.log 2>&1 &   # real run
"""
import math
import os
import time
from collections import deque
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from _exp_fast import (
    HIST_CAP, PREDICT_EVERY, hist_context, split_series, ts_auc_from,
)
from _feat_lib import build_candidates, base_features

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "_overnight_results.tsv")
SUMMARY = os.path.join(HERE, "_overnight_summary.txt")

SMOKE = os.environ.get("SMOKE") == "1"
SEED = 42
MIN_N = 15
STEPS_PER_SERIES = 10
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

if SMOKE:
    N_TRAIN, N_FOLDS, FOLD_SIZE, BATTERY_FOLDS, MAX_CAND = 300, 2, 40, 2, 3
else:
    N_TRAIN, N_FOLDS, FOLD_SIZE, BATTERY_FOLDS, MAX_CAND = 6000, 10, 400, 5, None

WINDOW_BASELINE = 75
WINDOW_CANDIDATE = 130

BASE = dict(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, min_samples_leaf=20,
    early_stopping=True, random_state=SEED,
)


# --- data -----------------------------------------------------------------
def load_pool():
    rng = np.random.default_rng(SEED)
    yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
    all_ids = yi.index.to_numpy().copy()
    rng.shuffle(all_ids)
    train_ids = np.sort(all_ids[:N_TRAIN])
    rest = all_ids[N_TRAIN:N_TRAIN + N_FOLDS * FOLD_SIZE]
    folds = [np.sort(rest[i * FOLD_SIZE:(i + 1) * FOLD_SIZE]) for i in range(N_FOLDS)]
    val_ids = np.concatenate(folds)
    keep = set(train_ids.tolist()) | set(val_ids.tolist())
    X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
    X = X[X.index.get_level_values("id").isin(keep)]
    y = pd.read_parquet(os.path.join(DATA, "y_train.parquet"))
    y_val = y[y.index.get_level_values("id").isin(set(val_ids.tolist()))]
    return X, yi, y_val, train_ids, folds


def select_picks(X, train_ids):
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


# --- train / stream -------------------------------------------------------
def build_training(X, yi, picks, feat_fn, window):
    rows, labels = [], []
    for sid, ts in picks.items():
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        for t in ts:
            lo = max(0, t - window + 1)
            rows.append(feat_fn(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _stream_one(model, feat_fn, window, sid, x_h, x_on, onl_index):
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
            feats = np.asarray(feat_fn(np.asarray(buf), ctx),
                               dtype=np.float64).reshape(1, -1)
            score = float(model.predict_proba(feats)[0, 1])
        out[idx] = score
    midx = pd.MultiIndex.from_product([[sid], onl_index], names=["id", "time"])
    return pd.Series(out, index=midx)


def fold_auc(model, feat_fn, window, X, y_val, fold_ids):
    jobs = []
    for sid in fold_ids:
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, x_h, onl.to_numpy(dtype=np.float64), onl.index))
    with threadpool_limits(limits=1):
        series = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_stream_one)(model, feat_fn, window, sid, x_h, x_on, oi)
            for (sid, x_h, x_on, oi) in jobs
        )
    return ts_auc_from(pd.concat(series).to_frame("prediction"), y_val)


def evaluate(feat_fn, window, X, yi, y_val, picks, folds):
    Xtr, ytr = build_training(X, yi, picks, feat_fn, window)
    model = HistGradientBoostingClassifier(**BASE)
    with threadpool_limits(limits=1):
        model.fit(Xtr, ytr)
    aucs = np.asarray([fold_auc(model, feat_fn, window, X, y_val, f) for f in folds])
    return aucs, Xtr.shape[1]


def paired(cand, base):
    d = cand - base
    md, sd = float(d.mean()), float(d.std(ddof=1))
    t = md / (sd / math.sqrt(len(d))) if sd > 0 else float("inf")
    return md, sd, t, d


# --- output ---------------------------------------------------------------
def write_row(fh, section, name, nfeat, aucs, base_aucs, secs):
    mean = float(aucs.mean())
    sd = float(aucs.std(ddof=1))
    if base_aucs is not None and len(base_aucs) == len(aucs):
        md, dsd, t, _ = paired(aucs, base_aucs)
        verdict = "SHIP" if abs(t) >= 2 and md > 0 else ("promising" if t >= 1.5 and md > 0 else "noise")
    else:
        md = dsd = t = float("nan")
        verdict = "baseline"
    fh.write("\t".join([
        section, name, str(nfeat),
        ",".join(f"{v:.4f}" for v in aucs),
        f"{mean:.4f}", f"{sd:.4f}", f"{md:+.4f}", f"{dsd:.4f}", f"{t:+.2f}",
        verdict, f"{secs:.0f}", datetime.now().strftime("%H:%M:%S"),
    ]) + "\n")
    fh.flush()
    os.fsync(fh.fileno())
    print(f"[{section}] {name:<18} nf={nfeat:<2} mean={mean:.4f} d={md:+.4f} "
          f"t={t:+.2f} {verdict} ({secs:.0f}s)", flush=True)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== overnight battery ({mode}) start {datetime.now():%Y-%m-%d %H:%M:%S} ===",
          flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train / {sum(len(v) for v in picks.values())} rows, "
          f"{len(folds)} folds x {FOLD_SIZE}, N_JOBS={N_JOBS}", flush=True)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("section\tname\tn_feat\tfold_aucs\tmean\tstd\tdelta\tdelta_std\tt\t"
             "verdict\tsecs\ttime\n")
    fh.flush()

    battery_folds = folds[:BATTERY_FOLDS]

    # --- Section A: WINDOW confirmation (all folds) ---
    print("\n-- Section A: WINDOW 75 vs 130 (higher power) --", flush=True)
    ts = time.time()
    base_win, nf = evaluate(base_features, WINDOW_BASELINE, X, yi, y_val, picks, folds)
    write_row(fh, "A_window", f"WINDOW={WINDOW_BASELINE}", nf, base_win, None, time.time() - ts)
    ts = time.time()
    cand_win, nf = evaluate(base_features, WINDOW_CANDIDATE, X, yi, y_val, picks, folds)
    write_row(fh, "A_window", f"WINDOW={WINDOW_CANDIDATE}", nf, cand_win, base_win, time.time() - ts)

    # base14 reference for the battery = first BATTERY_FOLDS folds of the WINDOW=75 run.
    base_ref = base_win[:BATTERY_FOLDS]

    # --- Section B: feature battery (WINDOW=75, battery folds) ---
    print("\n-- Section B: feature battery (WINDOW=75) --", flush=True)
    ts = time.time()
    write_row(fh, "B_feat", "base14", len(base_features(np.zeros(20), _dummy_ctx())),
              base_ref, None, time.time() - ts)

    cands = build_candidates()
    cands.pop("base14", None)
    names = list(cands)
    if MAX_CAND is not None:
        names = names[:MAX_CAND]
    for i, name in enumerate(names, 1):
        ts = time.time()
        try:
            aucs, nf = evaluate(cands[name], WINDOW_BASELINE, X, yi, y_val, picks, battery_folds)
            write_row(fh, "B_feat", name, nf, aucs, base_ref, time.time() - ts)
        except Exception as e:  # noqa: BLE001 - keep the night alive
            fh.write(f"B_feat\t{name}\tERR\t\t\t\t\t\t\tERROR:{type(e).__name__}\t"
                     f"{time.time()-ts:.0f}\t{datetime.now():%H:%M:%S}\n")
            fh.flush()
            print(f"[B_feat] {name} FAILED: {type(e).__name__}: {e}", flush=True)
        print(f"  ...{i}/{len(names)} done, elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    fh.close()
    _write_summary()
    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)


def _dummy_ctx():
    return hist_context(np.random.default_rng(0).normal(size=200), HIST_CAP)


def _write_summary():
    df = pd.read_csv(RESULTS, sep="\t")
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(f"Overnight battery summary ({datetime.now():%Y-%m-%d %H:%M})\n\n")
        a = df[df.section == "A_window"]
        f.write("Section A - WINDOW confirmation:\n")
        f.write(a[["name", "mean", "std", "delta", "t", "verdict"]].to_string(index=False))
        f.write("\n\nSection B - feature battery (ranked by t):\n")
        b = df[df.section == "B_feat"].copy()
        b["t"] = pd.to_numeric(b["t"], errors="coerce")
        b = b.sort_values("t", ascending=False)
        f.write(b[["name", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))
    print(f"summary -> {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
