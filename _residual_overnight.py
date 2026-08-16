"""Overnight residual-lens orchestrator: screen many reference-model / feature variants, auto
fresh-seed-confirm the strong ones, log incrementally. Designed to run unattended for hours.

Lead going in (fresh-confirmed): a SINGLE ar8 residual-variance-ratio feature scores +0.0208 (min_t
2.72, 4/4) vs the champion -> better than the shipped AR1-4 block (+0.0151). This run optimizes around
that: AR order sets, the champion's multi-scale trick applied to the residual feature, a robust
(MAD-based) variance ratio for short-window stability, a GARCH-lite volatility-clustering reference,
and combinations. Each variant: screen on {42,7,123}; if STRONG -> confirm on {31337,2025,555,99}.

Verdict tags use the shipped hardened intent: STRONG = pos on all seeds AND mean_d>=0.0015 AND min_t>=2.0.
Results append live to _overnight_results.log (survives a mid-run crash).

Usage:
  SMOKE=1 .venv/bin/python -u _residual_overnight.py
  .venv/bin/python -u _residual_overnight.py
"""
import math
import os
import time
import traceback
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, champion_feat, load_pool_strat, make_hgb, select_picks
from _exp_fast import HIST_CAP, hist_context, split_series, ts_auc_from
from _lambdarank import fit_ranker, sort_into_groups
from _overnight import MIN_N, N_JOBS, paired

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]
FRESH_SEEDS = [31337, 2025] if SMOKE else [31337, 2025, 555, 99]
DENSE_K = 30 if SMOKE else 60
RESULTS = "_overnight_results.log"

TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)
NCHAMP = 45


# --- reference models -------------------------------------------------------
def _robust_scale2(a):
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    return mad * mad + 1e-12


def _fit_ar(x, p):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < p + 8:
        return None
    cols = [x[p - lag:n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
    resid = x[p:] - Xd @ coef
    return coef, float(np.var(resid)) + 1e-12, _robust_scale2(resid)


def _ar_resid(w, coef, p):
    n = len(w)
    if n < p + 3:
        return None
    cols = [w[p - lag:n - lag] for lag in range(1, p + 1)]
    return w[p:] - np.column_stack(cols + [np.ones(n - p)]) @ coef


def _fit_garch(x, lam):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 30:
        return None
    mu = float(np.mean(x))
    e = x - mu
    s2 = lfilter([1.0 - lam], [1.0, -lam], e * e)
    s2p = np.concatenate([[float(np.var(e)) + 1e-12], s2[:-1]])
    z = e / np.sqrt(np.maximum(s2p, 1e-12))
    return mu, lam, float(np.var(z)) + 1e-12


def _garch_z(w, state):
    mu, lam, _ref = state
    e = np.asarray(w, dtype=np.float64) - mu
    s2 = lfilter([1.0 - lam], [1.0, -lam], e * e)
    s2p = np.concatenate([[float(np.var(e)) + 1e-12], s2[:-1]])
    return e / np.sqrt(np.maximum(s2p, 1e-12))


def _fit_refs(x_h, spec):
    refs = {}
    if spec.get("orders"):
        refs["ar"] = {p: _fit_ar(x_h, p) for p in spec["orders"]}
    if "garch" in spec.get("model", ""):
        refs["garch"] = _fit_garch(x_h, spec.get("lam", 0.94))
    return refs


def _extra_feats(w, refs, spec):
    scales = spec.get("scales", [WINDOW])
    robust = spec.get("robust", False)
    out = []
    if spec.get("orders"):
        arf = refs["ar"]
        for p in spec["orders"]:
            m = arf.get(p)
            for s in scales:
                ws = w if s >= len(w) else w[-s:]
                r = _ar_resid(ws, m[0], p) if m is not None else None
                if r is None or len(r) < 5:
                    out.append(0.0)
                elif robust:
                    out.append(math.log(_robust_scale2(r) / m[2]))
                else:
                    out.append(math.log((float(np.var(r)) + 1e-12) / m[1]))
    if "garch" in spec.get("model", ""):
        g = refs["garch"]
        for s in scales:
            ws = w if s >= len(w) else w[-s:]
            if g is None or len(ws) < 10:
                out.append(0.0)
            else:
                z = _garch_z(ws, g)
                out.append(math.log((float(np.var(z)) + 1e-12) / g[2]))
    return out


def _nextra(spec):
    ns = len(spec.get("scales", [WINDOW]))
    return len(spec.get("orders", [])) * ns + (ns if "garch" in spec.get("model", "") else 0)


# --- feature build (champion + extra, one pass) -----------------------------
def _active(x_on):
    L = len(x_on)
    return np.arange(MIN_N - 1, L) if L >= MIN_N else np.zeros(0, dtype=int)


def _series_feat(x_on, ctx, refs, spec, active, nx):
    champ, extra = [], []
    for t in active:
        win = x_on[max(0, t - WINDOW + 1):t + 1]
        champ.append(champion_feat(win, ctx))
        extra.append(_extra_feats(win, refs, spec))
    if not champ:
        return np.zeros((0, NCHAMP + nx))
    return np.hstack([np.asarray(champ, dtype=np.float64), np.asarray(extra, dtype=np.float64)])


def _val_series(sid, x_h, x_on, oi, spec, nx):
    ctx = hist_context(x_h, HIST_CAP)
    feat = _series_feat(x_on, ctx, _fit_refs(x_h, spec), spec, _active(x_on), nx)
    midx = pd.MultiIndex.from_product([[sid], oi], names=["id", "time"])
    return midx, _active(x_on), feat


def precompute_fold(X, fold_ids, spec, nx):
    jobs = [(sid, np.asarray(split_series(X.loc[sid])[0], dtype=np.float64),
             split_series(X.loc[sid])[1].to_numpy(dtype=np.float64), split_series(X.loc[sid])[1].index)
            for sid in fold_ids]
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series)(sid, x_h, x_on, oi, spec, nx) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, active_all, offset = [], [], [], 0
    for midx, active, feat in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, NCHAMP + nx)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCHAMP + nx))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


def _train_series(sid, x_h, x_on, tau, steps, spec, nx):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return np.zeros((0, NCHAMP + nx)), [], []
    refs = _fit_refs(x_h, spec)
    base = active[0]
    rows, labels, tvals = [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        win = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(np.concatenate([champion_feat(win, ctx), _extra_feats(win, refs, spec)]))
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return (np.asarray(rows) if rows else np.zeros((0, NCHAMP + nx)), labels, tvals)


def build_rows(X, yi, sid_to_steps, spec, nx):
    jobs = []
    for sid, steps in sid_to_steps.items():
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64),
                     int(yi.loc[sid, "tau_index"]), np.asarray(steps)))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_train_series)(sid, x_h, x_on, tau, steps, spec, nx)
            for (sid, x_h, x_on, tau, steps) in jobs)
    rows = [r for r, _l, _t in parts if len(r)]
    labels = np.concatenate([l for _r, l, _t in parts if len(l)]) if rows else np.zeros(0, int)
    tvals = np.concatenate([t for _r, _l, t in parts if len(t)]) if rows else np.zeros(0, int)
    feat = np.vstack(rows) if rows else np.zeros((0, NCHAMP + nx))
    return feat, labels.astype(np.int64), tvals.astype(np.int64)


def _score(model, cache, cols):
    full = np.zeros(len(cache["index"]))
    if len(cache["feat"]):
        full[cache["active"]] = model.predict_proba(cache["feat"][:, cols])[:, 1]
    return pd.Series(full, index=cache["index"])


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def _eval(cols, Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed):
    hgb = make_hgb(**TUNED, random_state=seed)
    with threadpool_limits(limits=1):
        hgb.fit(Xtr[:, cols], ytr)
    ref_h = np.sort(hgb.predict_proba(Xtr[:, cols])[:, 1])
    hs = [_score(hgb, c, cols) for c in caches]
    rrows, rlab, rgrp = sort_into_groups(Rr[:, cols], Rl, Rt)
    ranker = fit_ranker("rank_xendcg", rrows, rlab, rgrp)
    ref_r = np.sort(ranker.r.predict(Rr[:, cols]))
    rs = [_score(ranker, c, cols) for c in caches]
    fh = [pct(s.to_numpy(), ref_h) for s in hs]
    fr = [pct(s.to_numpy(), ref_r) for s in rs]
    return np.asarray([
        ts_auc_from(pd.Series(0.6 * a + 0.4 * b, index=s.index).to_frame("prediction"), y_val)
        for a, b, s in zip(fh, fr, hs)])


# --- experiment registry ----------------------------------------------------
def experiments():
    S = [WINDOW]
    MS = [60, 130, 250]
    return [
        dict(name="ctrl_ar1234", model="ar", orders=[1, 2, 3, 4], scales=S),
        dict(name="ar8_only", model="ar", orders=[8], scales=S),
        dict(name="ar_4_8", model="ar", orders=[4, 8], scales=S),
        dict(name="ar_2_4_8", model="ar", orders=[2, 4, 8], scales=S),
        dict(name="ar_2_4_8_12", model="ar", orders=[2, 4, 8, 12], scales=S),
        dict(name="ar8_ms", model="ar", orders=[8], scales=MS),
        dict(name="ar_248_ms", model="ar", orders=[2, 4, 8], scales=MS),
        dict(name="ar8_robust", model="ar", orders=[8], scales=S, robust=True),
        dict(name="ar_248_robust", model="ar", orders=[2, 4, 8], scales=S, robust=True),
        dict(name="ar_248_ms_robust", model="ar", orders=[2, 4, 8], scales=MS, robust=True),
        dict(name="garch94", model="garch", scales=S, lam=0.94),
        dict(name="garch97", model="garch", scales=S, lam=0.97),
        dict(name="ar8_garch", model="ar+garch", orders=[8], scales=S, lam=0.94),
        dict(name="ar1234_garch", model="ar+garch", orders=[1, 2, 3, 4], scales=S, lam=0.94),
    ]


def log(msg):
    with open(RESULTS, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def run_phase(specs, seeds, phase):
    """Screen/confirm every spec across seeds; return {name: (mean_d, min_d, npos, min_t, n)}."""
    store = {s["name"]: [] for s in specs}
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
        base_aucs = None
        for spec in specs:
            try:
                nx = _nextra(spec)
                caches = [precompute_fold(X, f, spec, nx) for f in folds]
                Xtr, ytr, _ = build_rows(X, yi, picks, spec, nx)
                Rr, Rl, Rt = build_rows(X, yi, rank_steps, spec, nx)
                if base_aucs is None:
                    base_aucs = _eval(np.arange(NCHAMP), Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
                    log(f"[{phase} seed {seed}] champion base={base_aucs.mean():.4f} "
                        f"({time.time()-t0:.0f}s)")
                cols = np.arange(NCHAMP + nx)
                aug = _eval(cols, Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
                md, _s, t, _ = paired(aug, base_aucs)
                store[spec["name"]].append((md, t))
                log(f"  [{phase} s{seed}] {spec['name']:<18} +{nx:>2}f aug={aug.mean():.4f} "
                    f"d={md:+.4f} t={t:+.2f}")
                del caches
            except Exception:
                log(f"  [{phase} s{seed}] {spec['name']:<18} ERROR\n{traceback.format_exc()}")
        del X
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
    log(f"\n===== OVERNIGHT residual campaign {datetime.now():%F %T} "
        f"(smoke={SMOKE}) =====")
    specs = experiments()
    log(f"screening {len(specs)} variants on search {SEARCH_SEEDS}")
    screen = run_phase(specs, SEARCH_SEEDS, "SCREEN")

    log("\n----- SCREEN SUMMARY (search seeds) -----")
    strong = []
    for name in sorted(screen, key=lambda n: -screen[n][0]):
        md, mind, npos, mint, n = screen[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<18} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} min_t {mint:+.2f} -> {v}")
        if v in ("STRONG", "pos") and md >= 0.0015:
            strong.append(name)

    if strong:
        conf_specs = [s for s in specs if s["name"] in strong]
        log(f"\nfresh-confirming {len(conf_specs)} promising variants on {FRESH_SEEDS}: {strong}")
        fresh = run_phase(conf_specs, FRESH_SEEDS, "FRESH")
        log("\n----- FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<18} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} min_t {mint:+.2f} "
                f"-> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\nno variant screened promising; nothing to fresh-confirm.")
    log(f"===== OVERNIGHT DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
