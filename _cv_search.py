"""E30 - true k-fold CV re-validation (train ~9k / val ~1k, rotate over ALL series).

WHY: the previous harness (_overnight.py) fixes 6000 train + 4000 held-out, so every
candidate is trained on only 60% of the data while the DEPLOYED model trains on ~100%.
Features that only earn signal near full data (AR/Hurst/sampen/variance families that
ranked high in E25 importance but scored as noise) get a systematic FALSE-NEGATIVE.
This harness removes that: real 10-fold CV, each fold trains on ~9000 series (≈deployed)
and validates on ~1000, rotating so every series is validated exactly once.

Efficiency: per-step features are identical on the train and val paths, and each series
is a val series in exactly ONE fold -> we (a) build the cheap sampled training rows for
all series once per candidate, refit the model 10x on row subsets (seconds each), and
(b) stream each series for TS-AUC only in its own fold (each series streamed once total).

Compares champion (45 feats) vs champion+family and the full bank, paired-t across the
10 fold TS-AUCs. Writes _cv_results.tsv incrementally (crash-safe).

Usage:
  SMOKE=1 .venv/bin/python -u _cv_search.py           # ~2-4 min plumbing self-test
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _cv_search.py > _cv_search.log 2>&1 &
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
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from _exp_fast import HIST_CAP, hist_context, split_series, ts_auc_from
from _overnight import BASE, DATA, MIN_N, STEPS_PER_SERIES, N_JOBS, SEED, _stream_one
from _bank_feats import (
    champion_feat, feat_ar, feat_hurst, feat_ar_hurst, feat_levene,
    feat_fligner, feat_mood, feat_sampen, feat_bank_all,
)

SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 10
SMOKE_SERIES = 800
WINDOW = 250

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cv_results.tsv")

# candidate list: cheap/high-prior first, expensive last (crash-safe partial results)
CANDIDATES = [
    ("+ar4", feat_ar),
    ("+hurst", feat_hurst),
    ("+levene", feat_levene),
    ("+fligner", feat_fligner),
    ("+mood", feat_mood),
    ("+ar4_hurst", feat_ar_hurst),
    ("+bank_all", feat_bank_all),
    ("+sampen", feat_sampen),
]


# --- data ------------------------------------------------------------------
def load_all():
    rng = np.random.default_rng(SEED)
    yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
    all_ids = yi.index.to_numpy().copy()
    rng.shuffle(all_ids)
    if SMOKE:
        all_ids = all_ids[:SMOKE_SERIES]
    fold_arrays = [np.sort(a) for a in np.array_split(all_ids, N_FOLDS)]
    keep = set(all_ids.tolist())
    X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
    X = X[X.index.get_level_values("id").isin(keep)]
    y = pd.read_parquet(os.path.join(DATA, "y_train.parquet"))
    y = y[y.index.get_level_values("id").isin(keep)]
    return X, yi, y, np.sort(all_ids), fold_arrays


# --- CV engine -------------------------------------------------------------
def build_rows(X, yi, ids, feat_fn, window):
    """Sampled STEPS_PER_SERIES training rows for every series (with series id)."""
    rng = np.random.default_rng(SEED + 1)
    rows, labels, sids = [], [], []
    for sid in ids:
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        L = len(x_on)
        if L <= MIN_N:
            continue
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        cand = np.arange(MIN_N, L)
        k = min(STEPS_PER_SERIES, len(cand))
        ts = rng.choice(cand, size=k, replace=False)
        for t in ts:
            lo = max(0, t - window + 1)
            rows.append(feat_fn(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
            sids.append(sid)
    return (np.asarray(rows, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
            np.asarray(sids))


def cv_eval(feat_fn, window, X, yi, y, ids, folds):
    Xall, yall, sid_all = build_rows(X, yi, ids, feat_fn, window)
    fold_aucs = []
    for f in folds:
        train_mask = ~np.isin(sid_all, f)
        model = HistGradientBoostingClassifier(**BASE)
        with threadpool_limits(limits=1):
            model.fit(Xall[train_mask], yall[train_mask])
        jobs = []
        for sid in f:
            x_h, onl = split_series(X.loc[sid])
            jobs.append((sid, x_h, onl.to_numpy(dtype=np.float64), onl.index))
        with threadpool_limits(limits=1):
            series = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(_stream_one)(model, feat_fn, window, sid, x_h, x_on, oi)
                for (sid, x_h, x_on, oi) in jobs
            )
        y_val = y[y.index.get_level_values("id").isin(set(f.tolist()))]
        auc = ts_auc_from(pd.concat(series).to_frame("prediction"), y_val)
        fold_aucs.append(auc)
    return np.asarray(fold_aucs), Xall.shape[1]


def paired(cand, base):
    d = cand - base
    md, sd = float(d.mean()), float(d.std(ddof=1))
    t = md / (sd / math.sqrt(len(d))) if sd > 0 else float("inf")
    return md, sd, t


def verdict(md, t):
    if md > 0 and t >= 2.5:
        return "SHIP(hard)"
    if md > 0 and t >= 2.0:
        return "marginal"
    if md > 0 and t >= 1.5:
        return "weak"
    return "noise"


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== E30 k-fold CV re-validation ({mode}) {datetime.now():%Y-%m-%d %H:%M:%S} ===",
          flush=True)
    X, yi, y, ids, folds = load_all()
    print(f"series={len(ids)}  folds={len(folds)} x ~{len(folds[0])}  "
          f"train/fold≈{len(ids)-len(folds[0])}  N_JOBS={N_JOBS}", flush=True)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("name\tn_feat\tmean\tstd\tdelta\tt\tverdict\tsecs\ttime\n")
    fh.flush()

    ts = time.time()
    base_aucs, nf = cv_eval(champion_feat, WINDOW, X, yi, y, ids, folds)
    fh.write(f"champion\t{nf}\t{base_aucs.mean():.4f}\t{base_aucs.std(ddof=1):.4f}\t"
             f"\t\tbaseline\t{time.time()-ts:.0f}\t{datetime.now():%H:%M:%S}\n")
    fh.flush(); os.fsync(fh.fileno())
    print(f"champion   nf={nf} mean={base_aucs.mean():.4f} "
          f"(train/fold≈{len(ids)-len(folds[0])})  ({(time.time()-ts)/60:.1f} min)", flush=True)

    for name, fn in CANDIDATES:
        ts = time.time()
        try:
            aucs, nf = cv_eval(fn, WINDOW, X, yi, y, ids, folds)
            md, sd, t = paired(aucs, base_aucs)
            v = verdict(md, t)
            fh.write(f"{name}\t{nf}\t{aucs.mean():.4f}\t{aucs.std(ddof=1):.4f}\t"
                     f"{md:+.4f}\t{t:+.2f}\t{v}\t{time.time()-ts:.0f}\t"
                     f"{datetime.now():%H:%M:%S}\n")
            fh.flush(); os.fsync(fh.fileno())
            print(f"{name:<12} nf={nf} mean={aucs.mean():.4f} "
                  f"delta={md:+.4f} t={t:+.2f} -> {v}  ({(time.time()-ts)/60:.1f} min)",
                  flush=True)
        except Exception as e:  # noqa: BLE001 - keep the night alive
            fh.write(f"{name}\tERR\t\t\t\t\tERROR:{type(e).__name__}\t"
                     f"{time.time()-ts:.0f}\t{datetime.now():%H:%M:%S}\n")
            fh.flush()
            print(f"{name} FAILED: {type(e).__name__}: {e}", flush=True)

    fh.close()
    print(f"\n=== E30 DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
