"""Lever A: per-series score calibration — does it lift TS-AUC?

TS-AUC computes a rank-based AUC cross-sectionally at each online step t. Any monotonic
transform applied *uniformly within a step* leaves that AUC unchanged, so the only calibration
that can help is one that differs *by series* — removing each series' idiosyncratic score
offset/scale so the cross-series comparison at step t reflects break-evidence, not a series'
baseline level. Each series' own early (pre-break) online scores estimate that baseline; this is
causal (uses only past points) and needs no cross-series state for the simple variants.

Method: fit the shipped base16 model once, stream each val fold to get RAW per-(series,step)
scores, then re-score TS-AUC under several calibration variants (cheap post-processing on the
same streamed scores). Paired t vs raw over the folds. Ship rule stays |t|>=2 & delta>0.

Usage:
  SMOKE=1 .venv/bin/python -u _lever_a.py
  nohup .venv/bin/python -u _lever_a.py > _lever_a.log 2>&1 &
"""
import math
import os
import time
from datetime import datetime
from functools import partial

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from _overnight import (
    BASE, MIN_N, N_JOBS, SMOKE, WINDOW_CANDIDATE,
    load_pool, select_picks, build_training, _stream_one, paired,
)
from _exp_fast import split_series, ts_auc_from
from _feat_lib import _feat

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "_lever_a_results.tsv")
SUMMARY = os.path.join(HERE, "_lever_a_summary.txt")
WINDOW = WINDOW_CANDIDATE  # 130, shipped
FEAT = partial(_feat, keys=("spec_entropy", "perm_entropy"))  # base16, shipped

# calibration variants applied per-series to the raw score trajectory
VARIANTS = ["raw", "sub_b10", "sub_b20", "sub_b40", "z_b20", "z_b40", "exp_med"]


def _early_base(v, B):
    seg = v[MIN_N:MIN_N + B]
    if len(seg) == 0:
        return 0.0, 1.0
    b = float(np.median(seg))
    sd = 1.4826 * float(np.median(np.abs(seg - b))) + 1e-6
    return b, sd


def apply_variant(s, name):
    v = s.to_numpy().astype(np.float64)
    if name == "raw":
        out = v
    elif name.startswith("sub_b"):
        b, _ = _early_base(v, int(name[5:]))
        out = v - b
    elif name.startswith("z_b"):
        b, sd = _early_base(v, int(name[3:]))
        out = (v - b) / sd
    elif name == "exp_med":
        em = pd.Series(v).expanding(min_periods=MIN_N).median().shift(1).fillna(0.0)
        out = v - em.to_numpy()
    else:
        out = v
    return pd.Series(out, index=s.index)


def fold_raw_series(model, X, fold_ids):
    jobs = []
    for sid in fold_ids:
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, x_h, onl.to_numpy(dtype=np.float64), onl.index))
    with threadpool_limits(limits=1):
        return Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_stream_one)(model, FEAT, WINDOW, sid, x_h, x_on, oi)
            for (sid, x_h, x_on, oi) in jobs
        )


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== Lever A: per-series calibration @W={WINDOW} ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {len(folds[0])}, "
          f"variants={VARIANTS}", flush=True)

    ts = time.time()
    Xtr, ytr = build_training(X, yi, picks, FEAT, WINDOW)
    model = HistGradientBoostingClassifier(**BASE)
    with threadpool_limits(limits=1):
        model.fit(Xtr, ytr)
    print(f"trained base16 {Xtr.shape} in {time.time()-ts:.0f}s", flush=True)

    aucs = {v: [] for v in VARIANTS}
    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("fold\t" + "\t".join(VARIANTS) + "\tsecs\ttime\n")
    fh.flush()

    for fi, fold_ids in enumerate(folds):
        fts = time.time()
        raw = fold_raw_series(model, X, fold_ids)
        row = []
        for v in VARIANTS:
            pred = pd.concat([apply_variant(s, v) for s in raw]).to_frame("prediction")
            a = ts_auc_from(pred, y_val)
            aucs[v].append(a)
            row.append(f"{a:.4f}")
        fh.write(f"{fi}\t" + "\t".join(row) + f"\t{time.time()-fts:.0f}\t"
                 f"{datetime.now():%H:%M:%S}\n")
        fh.flush()
        os.fsync(fh.fileno())
        print(f"fold {fi}: " + "  ".join(f"{v}={aucs[v][-1]:.4f}" for v in VARIANTS)
              + f"  ({time.time()-fts:.0f}s)", flush=True)

    fh.close()
    _summary(aucs)
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)


def _summary(aucs):
    base = np.asarray(aucs["raw"])
    lines = [f"Lever A: per-series calibration @W={WINDOW} ({datetime.now():%Y-%m-%d %H:%M})",
             f"folds={len(base)}  base16 raw mean={base.mean():.4f}\n",
             f"{'variant':<10} {'mean':>8} {'delta':>9} {'t':>7}  verdict"]
    rows = []
    for v in VARIANTS:
        a = np.asarray(aucs[v])
        if v == "raw":
            rows.append((float("inf"), f"{v:<10} {a.mean():>8.4f} {'--':>9} {'--':>7}  baseline"))
            continue
        md, _, t, _ = paired(a, base)
        verdict = "SHIP" if abs(t) >= 2 and md > 0 else ("promising" if t >= 1.5 and md > 0 else "noise")
        rows.append((t, f"{v:<10} {a.mean():>8.4f} {md:>+9.4f} {t:>+7.2f}  {verdict}"))
    for _, line in sorted(rows, key=lambda r: -r[0]):
        lines.append(line)
    text = "\n".join(lines)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
