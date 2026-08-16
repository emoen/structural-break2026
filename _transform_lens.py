"""Laptop lane: TRANSFORM-LENS bank on top of the E42 residual champion.

Inspired by the 2025 2nd-place solution (aParsecFromFuture), which harvested breadth from cheap
deterministic transforms (cumsum, dense-rank, abs, rolling-std) recomputed for every stat. That
validates our E42 thesis (the win is a new REPRESENTATION of window-vs-history, not a new raw
statistic) — but via cheap transforms rather than one expensive fitted lens. Everything here is
CAUSAL (computable from x_online[:t+1] and the pre-boundary history) and cheap, so unlike their
TabPFN stack it is deployable under our real-time contract.

Re-baselined vs the CURRENT champion (45 champion feats + AR{1,2,3,4} residual = 49 cols, FROZEN).
Each lens block appends a few contrast features computed from the same raw window `w` and history
context `ctx` that the champion already uses. Real fixed-CDF TS-AUC, paired-t vs champion,
2-stage seed gate (screen {42,7,123} STRONG -> fresh-confirm {31337,2025,555,99}).

Lens blocks (all indices into the 11-col lens vector):
  cusum   [0,1,2]  : standardized cumulative sum (classic CUSUM mean-shift detector) —
                     max|cs|, endpoint, range, all /sqrt(n).  <-- highest prior, we lack this.
  pit     [3,4,5]  : probability-integral / dense-rank transform of the window vs history ecdf —
                     mean(u)-0.5, std(u), KS(u, Uniform).  distribution-free level/shape shift.
  abslens [6,7]    : |value| contrast — log mean-ratio, log var-ratio (volatility/scale break).
  rollstd [8,9]    : rolling-std(16) within window — log level vs history sd, coeff-of-variation
                     (vol-of-vol).
  levene  [10]     : Brown-Forsythe robust variance-equality F (log1p).
  all     [0..10]  : union.

Usage:
  SMOKE=1 .venv/bin/python -u _transform_lens.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _transform_lens.py > _transform_lens.log 2>&1 & disown
"""
import os
import time
import traceback
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, champion_feat, load_pool_strat, select_picks
from _exp_fast import HIST_CAP, hist_context, split_series
from _overnight import MIN_N, N_JOBS, paired
from _residual_overnight import (NCHAMP, _active, _extra_feats, _eval, _fit_refs, _nextra)

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]
FRESH_SEEDS = [31337, 2025] if SMOKE else [31337, 2025, 555, 99]
DENSE_K = 30 if SMOKE else 60
RESULTS = "_transform_lens_results.log"

# Frozen shipped champion residual block (NOT searched here).
SPEC = dict(name="ctrl_ar1234", model="ar", orders=[1, 2, 3, 4], scales=[WINDOW])
NX = _nextra(SPEC)                 # 4
BASE_NCOL = NCHAMP + NX            # 49
NLENS = 11
NCOMB = BASE_NCOL + NLENS          # 60

BLOCKS = {
    "cusum":   [0, 1, 2],
    "pit":     [3, 4, 5],
    "abslens": [6, 7],
    "rollstd": [8, 9],
    "levene":  [10],
    "all":     list(range(NLENS)),
}

_EPS = 1e-12


def log(msg):
    with open(RESULTS, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def _rolling_std(a, k):
    if len(a) < k:
        return np.array([float(np.std(a))]) if len(a) else np.array([0.0])
    c1 = np.concatenate([[0.0], np.cumsum(a)])
    c2 = np.concatenate([[0.0], np.cumsum(a * a)])
    m = (c1[k:] - c1[:-k]) / k
    v = np.maximum((c2[k:] - c2[:-k]) / k - m * m, 0.0)
    return np.sqrt(v)


def lens_feats(w, ctx):
    """11 causal transform-lens contrast features for window w against history ctx."""
    n = len(w)
    mu, sd = ctx["mu"], ctx["sd"]
    hs = ctx["sample_sorted"]
    ho = ctx["sample_ordered"]
    if n < 3 or sd < _EPS or len(hs) < 3:
        return [0.0] * NLENS

    # --- cusum lens (standardized cumulative sum) ---
    z = (w - mu) / sd
    cs = np.cumsum(z)
    rn = 1.0 / np.sqrt(n)
    cusum_maxabs = float(np.max(np.abs(cs))) * rn
    cusum_end = float(cs[-1]) * rn
    cusum_range = float(cs.max() - cs.min()) * rn

    # --- pit / dense-rank lens (empirical CDF of window vs history) ---
    u = np.searchsorted(hs, w, side="right") / len(hs)
    pit_mean_dev = float(u.mean()) - 0.5
    pit_std = float(u.std())
    us = np.sort(u)
    i1 = np.arange(1, n + 1) / n
    i0 = np.arange(n) / n
    pit_ks = float(max(np.max(i1 - us), np.max(us - i0)))

    # --- abs lens (|deviation| contrast) ---
    aw = np.abs(w - mu)
    ah = np.abs(ho - mu)
    abs_mean_ratio = float(np.log((aw.mean() + _EPS) / (ah.mean() + _EPS)))
    abs_var_ratio = float(np.log((aw.var() + _EPS) / (ah.var() + _EPS)))

    # --- rolling-std lens (vol-of-vol within window) ---
    rs = _rolling_std(w, 16)
    rs_mean = float(rs.mean())
    rollstd_level = float(np.log((rs_mean + _EPS) / (sd + _EPS)))
    rollstd_cv = float(rs.std() / (rs_mean + _EPS)) if rs_mean > _EPS else 0.0

    # --- levene (Brown-Forsythe robust variance-equality F) ---
    zw = np.abs(w - np.median(w))
    zh = np.abs(ho - np.median(ho))
    n1, n2 = len(zw), len(zh)
    mw, mh = zw.mean(), zh.mean()
    mg = (n1 * mw + n2 * mh) / (n1 + n2)
    between = n1 * (mw - mg) ** 2 + n2 * (mh - mg) ** 2
    within = float(((zw - mw) ** 2).sum() + ((zh - mh) ** 2).sum())
    F = (n1 + n2 - 2) * between / (within + _EPS)
    levene_f = float(np.log1p(max(F, 0.0)))

    return [cusum_maxabs, cusum_end, cusum_range,
            pit_mean_dev, pit_std, pit_ks,
            abs_mean_ratio, abs_var_ratio,
            rollstd_level, rollstd_cv,
            levene_f]


def _row(win, ctx, refs):
    return np.concatenate([champion_feat(win, ctx), _extra_feats(win, refs, SPEC),
                           np.asarray(lens_feats(win, ctx), dtype=np.float64)])


def _series_feat(x_on, ctx, refs, active):
    rows = [_row(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs) for t in active]
    if not rows:
        return np.zeros((0, NCOMB))
    return np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _val_series(sid, x_h, x_on, oi):
    import pandas as pd
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    feat = _series_feat(x_on, ctx, _fit_refs(x_h, SPEC), active)
    midx = pd.MultiIndex.from_product([[sid], oi], names=["id", "time"])
    return midx, active, feat


def _train_series(sid, x_h, x_on, tau, steps):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return np.zeros((0, NCOMB)), [], []
    refs = _fit_refs(x_h, SPEC)
    base = active[0]
    rows, labels, tvals = [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        win = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(_row(win, ctx, refs))
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    feat = np.nan_to_num(np.asarray(rows), nan=0.0, posinf=0.0, neginf=0.0) if rows \
        else np.zeros((0, NCOMB))
    return feat, labels, tvals


def precompute_fold(X, fold_ids):
    jobs = [(sid, np.asarray(split_series(X.loc[sid])[0], dtype=np.float64),
             split_series(X.loc[sid])[1].to_numpy(dtype=np.float64),
             split_series(X.loc[sid])[1].index) for sid in fold_ids]
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series)(sid, x_h, x_on, oi) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, active_all, offset = [], [], [], 0
    for midx, active, feat in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, NCOMB)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCOMB))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


def build_rows(X, yi, sid_to_steps):
    jobs = []
    for sid, steps in sid_to_steps.items():
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64),
                     int(yi.loc[sid, "tau_index"]), np.asarray(steps)))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_train_series)(sid, x_h, x_on, tau, steps)
            for (sid, x_h, x_on, tau, steps) in jobs)
    rows = [r for r, _l, _t in parts if len(r)]
    labels = np.concatenate([l for _r, l, _t in parts if len(l)]) if rows else np.zeros(0, int)
    tvals = np.concatenate([t for _r, _l, t in parts if len(t)]) if rows else np.zeros(0, int)
    feat = np.vstack(rows) if rows else np.zeros((0, NCOMB))
    return feat, labels.astype(np.int64), tvals.astype(np.int64)


def run_phase(blocks, seeds, phase):
    store = {name: [] for name in blocks}
    base_cols = np.arange(BASE_NCOL)
    for seed in seeds:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        picks = select_picks(X, train_ids)
        rng = np.random.default_rng(seed + 202)
        rank_steps = {}
        for sid in train_ids:
            L = len(split_series(X.loc[sid])[1].to_numpy())
            if L <= MIN_N:
                continue
            cand = np.arange(MIN_N, L)
            rank_steps[sid] = rng.choice(cand, size=min(DENSE_K, len(cand)), replace=False)

        caches = [precompute_fold(X, f) for f in folds]
        Xtr, ytr, _ = build_rows(X, yi, picks)
        Rr, Rl, Rt = build_rows(X, yi, rank_steps)
        base = _eval(base_cols, Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
        log(f"[{phase} seed {seed}] champion base={base.mean():.4f} ({time.time()-t0:.0f}s, "
            f"{BASE_NCOL} cols)")

        for name, idx in blocks.items():
            try:
                cols = np.concatenate([base_cols, BASE_NCOL + np.asarray(idx)])
                aug = _eval(cols, Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
                md, _s, t, _ = paired(aug, base)
                store[name].append((md, t))
                log(f"  [{phase} s{seed}] {name:<9} +{len(idx):>2}f aug={aug.mean():.4f} "
                    f"d={md:+.4f} t={t:+.2f}")
            except Exception:
                log(f"  [{phase} s{seed}] {name:<9} ERROR\n{traceback.format_exc()}")
        del X, caches
    out = {}
    for name, dt in store.items():
        if not dt:
            continue
        ds = [d for d, _t in dt]
        ts = [t for _d, t in dt]
        out[name] = (float(np.mean(ds)), float(min(ds)), sum(1 for d in ds if d > 0),
                     float(min(ts)), len(ds))
    return out


def verdict(mean_d, min_d, npos, min_t, n):
    if npos == n and mean_d >= 0.0015 and min_t >= 2.0:
        return "STRONG"
    return "pos" if npos == n and mean_d > 0 else "noise"


def main():
    log(f"\n===== TRANSFORM-LENS campaign {datetime.now():%F %T} (smoke={SMOKE}) =====")
    log(f"re-baselined vs residual champion ({BASE_NCOL} cols); screening {len(BLOCKS)} lens blocks "
        f"on {SEARCH_SEEDS}")
    screen = run_phase(BLOCKS, SEARCH_SEEDS, "SCREEN")

    log("\n----- SCREEN SUMMARY (search seeds) -----")
    promising = []
    for name in sorted(screen, key=lambda n: -screen[n][0]):
        md, mind, npos, mint, n = screen[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<9} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} min_t {mint:+.2f} -> {v}")
        if npos == n and md >= 0.0015:
            promising.append(name)

    if promising:
        log(f"\nfresh-confirming {len(promising)} blocks on {FRESH_SEEDS}: {promising}")
        fresh = run_phase({k: BLOCKS[k] for k in promising}, FRESH_SEEDS, "FRESH")
        log("\n----- FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<9} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} min_t {mint:+.2f} "
                f"-> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\nno lens block screened promising; nothing to fresh-confirm.")
    log(f"===== TRANSFORM-LENS DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
