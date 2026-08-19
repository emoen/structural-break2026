"""Laptop lane: two NEW residual lenses on top of the multi-scale AR8 residual champion (E54).

Two untried orthogonal REPRESENTATIONS of window-vs-history, both CAUSAL (computable from
x_online[:t+1] and the pre-boundary history) and cheap enough to deploy under the real-time
contract:

  spectral-residual lens [0..4]  : the classic Spectral Residual saliency (Hou & Zhang 2007;
      Microsoft SR-CNN for time-series anomaly detection). Take the window's FFT log-amplitude,
      subtract its smoothed version (the *spectral residual*), invert to a time-domain saliency
      map, and summarise how salient the recent window / its endpoint is. Window-internal, so it
      captures a frequency-domain anomaly structure the champion's AR residual variance ratios and
      spectral-entropy/centroid diffs do not.
        sr_max      max standardized saliency in the window
        sr_end      standardized saliency at the last (most recent) point
        sr_cv       coeff-of-variation of the saliency map (spikiness)
        sr_spike    fraction of points with saliency z > 3 (burst density)
        sr_endrank  fraction of points whose saliency is <= the last point's

  quantile/distribution-residual lens [5..10] : compare the window's empirical quantiles to the
      history's, standardized by history IQR -> a quantile-shift *profile*, plus a distribution
      residual (mean |PIT residual|). Captures where in the distribution the shift lives (median
      vs tails, spread, tail-asymmetry) rather than one scalar mean/var break.
        qr_med      standardized median shift (Q50)
        qr_lower    standardized Q25 shift
        qr_upper    standardized Q75 shift
        qr_tailasym (Q95-shift) - (Q05-shift)  = skew / tail-asymmetry shift
        qr_spread   (Q75-shift) - (Q25-shift)  = standardized IQR change
        qr_l1       mean |sorted PIT - uniform| = Cramer-von-Mises-style distribution residual

Re-baselined vs the multi-scale AR8 residual champion (E54: 45 champion feats + AR8 residual
variance ratio on scales 60/130/250 = 48 cols, FROZEN). Real fixed-CDF TS-AUC, paired-t vs
champion, 2-stage seed gate (screen {42,7,123} -> fresh-confirm {31337,2025,555,99}).

NOTE: the true current LB champion is E56 trail120 (61.58%, 45+6 feats); this re-baselines on the
E54 ar8_ms champion (61.32%, one step behind) as a strong, self-contained laptop-reproducible base.
The lens-additivity verdict transfers to the stronger base.

Usage:
  SMOKE=1 .venv/bin/python -u _resid_lens2.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _resid_lens2.py > _resid_lens2.log 2>&1 & disown
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
RESULTS = "_resid_lens2_results.log"

# Frozen shipped residual block: multi-scale AR8 champion (E54), NOT searched here.
SPEC = dict(name="ar8_ms", model="ar", orders=[8], scales=[60, 130, 250])
NX = _nextra(SPEC)                 # 3
BASE_NCOL = NCHAMP + NX            # 48
NLENS = 11
NCOMB = BASE_NCOL + NLENS          # 59

BLOCKS = {
    "spectral": [0, 1, 2, 3, 4],
    "quantile": [5, 6, 7, 8, 9, 10],
    "all":      list(range(NLENS)),
}

_EPS = 1e-12


def log(msg):
    with open(RESULTS, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def _spectral_resid_feats(w):
    """5 causal Spectral-Residual saliency contrast features from the window."""
    n = len(w)
    if n < 8:
        return [0.0] * 5
    wd = w - w.mean()
    if float(np.std(wd)) < _EPS:
        return [0.0] * 5
    f = np.fft.fft(wd)
    amp = np.abs(f)
    logamp = np.log(amp + _EPS)
    q = 3
    ker = np.ones(q) / q
    smooth = np.convolve(logamp, ker, mode="same")
    sr = logamp - smooth
    recon = np.fft.ifft(np.exp(sr) * np.exp(1j * np.angle(f)))
    sal = np.abs(recon)
    m = float(sal.mean())
    s = float(sal.std())
    if s < _EPS:
        return [0.0] * 5
    salz = (sal - m) / s
    sr_max = float(salz.max())
    sr_end = float(salz[-1])
    sr_cv = float(s / (m + _EPS))
    sr_spike = float(np.mean(salz > 3.0))
    sr_endrank = float(np.mean(sal <= sal[-1]))
    return [sr_max, sr_end, sr_cv, sr_spike, sr_endrank]


def _quantile_resid_feats(w, ctx):
    """6 causal quantile/distribution-residual contrast features (window vs history)."""
    hs = ctx["sample_sorted"]
    n = len(w)
    nh = len(hs)
    if n < 5 or nh < 10:
        return [0.0] * 6
    qlev = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    wq = np.quantile(w, qlev)
    hq = np.quantile(hs, qlev)
    hiqr = hq[3] - hq[1]
    scale = hiqr if hiqr > _EPS else (float(np.std(hs)) + _EPS)
    dq = (wq - hq) / scale
    qr_med = float(dq[2])
    qr_lower = float(dq[1])
    qr_upper = float(dq[3])
    qr_tailasym = float(dq[4] - dq[0])
    qr_spread = float(dq[3] - dq[1])
    u = np.searchsorted(hs, w, side="right") / nh
    us = np.sort(u)
    qr_l1 = float(np.mean(np.abs(us - (np.arange(1, n + 1) - 0.5) / n)))
    return [qr_med, qr_lower, qr_upper, qr_tailasym, qr_spread, qr_l1]


def lens2_feats(w, ctx):
    """11 causal residual-lens contrast features (5 spectral + 6 quantile)."""
    n = len(w)
    if n < 8 or ctx["sd"] < _EPS or len(ctx["sample_sorted"]) < 10:
        return [0.0] * NLENS
    return _spectral_resid_feats(w) + _quantile_resid_feats(w, ctx)


def _row(win, ctx, refs):
    return np.concatenate([champion_feat(win, ctx), _extra_feats(win, refs, SPEC),
                           np.asarray(lens2_feats(win, ctx), dtype=np.float64)])


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
    log(f"\n===== RESID-LENS2 campaign {datetime.now():%F %T} (smoke={SMOKE}) =====")
    log(f"re-baselined vs multi-scale AR8 residual champion ({BASE_NCOL} cols); screening "
        f"{len(BLOCKS)} lens blocks on {SEARCH_SEEDS}")
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
    log(f"===== RESID-LENS2 DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
