"""Experiment #2: re-tune BLEND_W and STEPS_PER_SERIES *with the ar8_ms residual feature present*.

The champion's w=0.6 (E41) and STEPS=30 were chosen before the residual feature existed; a stronger,
differently-correlated feature can shift those optima. Feature = champion 45 + AR8 multi-scale residual
var-ratio (scales 60/130/250, 3 feats). Per seed: fit the ranker once (DENSE_K); for each STEPS refit
the HGB; then re-blend the two fixed-CDF percentiles at each weight (free). Report every (STEPS, weight)
cell vs the shipped baseline (STEPS=30, w=0.60), paired-t across ALL 7 seeds (config change -> want
positive on all seeds, like the E41 weight sweep).

Usage:
  SMOKE=1 .venv/bin/python -u _residual_wsteps.py
  .venv/bin/python -u _residual_wsteps.py
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
from threadpoolctl import threadpool_limits

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, champion_feat, load_pool_strat, make_hgb, select_picks
from _exp_fast import HIST_CAP, hist_context, split_series, ts_auc_from
from _lambdarank import fit_ranker, sort_into_groups
from _overnight import MIN_N, N_JOBS, paired

SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = [42, 7] if SMOKE else [42, 7, 123, 31337, 2025, 555, 99]
DENSE_K = 30 if SMOKE else 60
SCALES = (60, 130, 250)
NCHAMP = 45
STEPS_LIST = [10, 30, 60]
WEIGHTS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
BASE = (30, 0.60)
RESULTS = "_wsteps_results.log"

TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)


def _fit_ar(x, p):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < p + 8:
        return None
    cols = [x[p - lag:n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
    return coef, float(np.var(x[p:] - Xd @ coef)) + 1e-12


def _ar_resid(w, coef, p):
    n = len(w)
    if n < p + 3:
        return None
    cols = [w[p - lag:n - lag] for lag in range(1, p + 1)]
    return w[p:] - np.column_stack(cols + [np.ones(n - p)]) @ coef


def _ar8ms(w, ref):
    w = np.asarray(w, dtype=np.float64)
    out = []
    for s in SCALES:
        ws = w if s >= len(w) else w[-s:]
        r = _ar_resid(ws, ref[0], 8) if ref is not None else None
        out.append(0.0 if (r is None or len(r) < 5) else math.log((float(np.var(r)) + 1e-12) / ref[1]))
    return out


def _active(x_on):
    L = len(x_on)
    return np.arange(MIN_N - 1, L) if L >= MIN_N else np.zeros(0, dtype=int)


def _val_series(sid, x_h, x_on, oi):
    ctx = hist_context(x_h, HIST_CAP)
    ref = _fit_ar(x_h, 8)
    active = _active(x_on)
    rows = [np.concatenate([champion_feat(x_on[max(0, t - WINDOW + 1):t + 1], ctx),
                            _ar8ms(x_on[max(0, t - WINDOW + 1):t + 1], ref)]) for t in active]
    feat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, NCHAMP + 3))
    midx = pd.MultiIndex.from_product([[sid], oi], names=["id", "time"])
    return midx, active, feat


def precompute_fold(X, fold_ids):
    jobs = [(sid, np.asarray(split_series(X.loc[sid])[0], dtype=np.float64),
             split_series(X.loc[sid])[1].to_numpy(dtype=np.float64), split_series(X.loc[sid])[1].index)
            for sid in fold_ids]
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series)(sid, x_h, x_on, oi) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, active_all, offset = [], [], [], 0
    for midx, active, feat in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, NCHAMP + 3)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCHAMP + 3))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


def _train_series(sid, x_h, x_on, tau, steps):
    ctx = hist_context(x_h, HIST_CAP)
    ref = _fit_ar(x_h, 8)
    active = _active(x_on)
    base = active[0] if len(active) else 0
    rows, labels, tvals = [], [], []
    for t in steps:
        if t - base < 0 or t - base >= len(active):
            continue
        win = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(np.concatenate([champion_feat(win, ctx), _ar8ms(win, ref)]))
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return (np.asarray(rows) if rows else np.zeros((0, NCHAMP + 3)), labels, tvals)


def build_rows(X, yi, sid_to_steps):
    jobs = []
    for sid, steps in sid_to_steps.items():
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64),
                     int(yi.loc[sid, "tau_index"]), np.asarray(steps)))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_train_series)(sid, x_h, x_on, tau, steps) for (sid, x_h, x_on, tau, steps) in jobs)
    rows = [r for r, _l, _t in parts if len(r)]
    labels = np.concatenate([l for _r, l, _t in parts if len(l)]) if rows else np.zeros(0, int)
    tvals = np.concatenate([t for _r, _l, t in parts if len(t)]) if rows else np.zeros(0, int)
    feat = np.vstack(rows) if rows else np.zeros((0, NCHAMP + 3))
    return feat, labels.astype(np.int64), tvals.astype(np.int64)


def _score(model, cache):
    full = np.zeros(len(cache["index"]))
    if len(cache["feat"]):
        full[cache["active"]] = model.predict_proba(cache["feat"])[:, 1]
    return pd.Series(full, index=cache["index"])


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def _ranker_picks(X, train_ids, seed):
    rng = np.random.default_rng(seed + 202)
    out = {}
    for sid in train_ids:
        L = len(split_series(X.loc[sid])[1].to_numpy())
        if L <= MIN_N:
            continue
        cand = np.arange(MIN_N, L)
        out[sid] = rng.choice(cand, size=min(DENSE_K, len(cand)), replace=False)
    return out


def log(msg):
    with open(RESULTS, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def main():
    log(f"\n===== WSTEPS (#2 weight x steps retune, ar8_ms present) {datetime.now():%F %T} "
        f"(smoke={SMOKE}) =====")
    # cell (steps, w) -> list of per-seed per-fold auc arrays
    cell = {(s, w): [] for s in STEPS_LIST for w in WEIGHTS}
    for seed in SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(X, f) for f in folds]

        # ranker (fixed across STEPS): fit on champion+ar8ms with DENSE_K rows
        Rr, Rl, Rt = build_rows(X, yi, _ranker_picks(X, train_ids, seed))
        rrows, rlab, rgrp = sort_into_groups(Rr, Rl, Rt)
        ranker = fit_ranker("rank_xendcg", rrows, rlab, rgrp)
        ref_rank = np.sort(ranker.r.predict(Rr))
        fr = [pct(_score(ranker, c).to_numpy(), ref_rank) for c in caches]

        for steps in STEPS_LIST:
            ET.STEPS_PER_SERIES = steps
            picks = select_picks(X, train_ids)
            Xtr, ytr, _ = build_rows(X, yi, picks)
            hgb = make_hgb(**TUNED, random_state=seed)
            with threadpool_limits(limits=1):
                hgb.fit(Xtr, ytr)
            ref_hgb = np.sort(hgb.predict_proba(Xtr)[:, 1])
            fh = [pct(_score(hgb, c).to_numpy(), ref_hgb) for c in caches]
            idx = [c["index"] for c in caches]
            for w in WEIGHTS:
                aucs = np.asarray([
                    ts_auc_from(pd.Series(w * a + (1 - w) * b, index=ix).to_frame("prediction"), y_val)
                    for a, b, ix in zip(fh, fr, idx)])
                cell[(steps, w)].append(aucs)
        log(f"[seed {seed}] done ({time.time()-t0:.0f}s) base(30,0.60)="
            f"{np.mean(cell[BASE][-1]):.4f}")
        del X, caches

    # paired-t of every cell vs the shipped baseline (30, 0.60), across seeds x folds
    base_stack = cell[BASE]
    log("\n----- (STEPS, weight) vs shipped (30, 0.60) : per-seed mean_d, 7-seed gate -----")
    rows = []
    for s in STEPS_LIST:
        for w in WEIGHTS:
            ds, ts = [], []
            for aug_seed, base_seed in zip(cell[(s, w)], base_stack):
                md, _sd, t, _ = paired(aug_seed, base_seed)
                ds.append(md)
                ts.append(t)
            npos = sum(1 for d in ds if d > 0)
            mean_auc = float(np.mean([a.mean() for a in cell[(s, w)]]))
            rows.append((s, w, float(np.mean(ds)), float(min(ds)), npos, float(min(ts)), mean_auc))
    for s, w, md, mind, npos, mint, mauc in sorted(rows, key=lambda r: -r[2]):
        tag = "SHIP" if (s, w) != BASE and npos == len(SEEDS) and md >= 0.0010 and mint >= 2.0 else (
            "baseline" if (s, w) == BASE else ("pos" if npos == len(SEEDS) and md > 0 else "noise"))
        log(f"STEPS={s:<3} w={w:.2f}  mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{len(SEEDS)} "
            f"min_t {mint:+.2f} auc {mauc:.4f} -> {tag}")
    log(f"===== WSTEPS DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
