"""Experiment #1: reference-fit variations of the residual lens.

The shipped/winning residual feature fits AR on the FULL historical segment, on LEVELS. This tests
whether a different REFERENCE fit sharpens (or adds orthogonally to) the AR8 multi-scale var-ratio:
  - trailing history: fit AR on the last k historical points (recent dynamics more relevant),
  - returns: fit AR on Δhistorical, residual on Δwindow (removes trend/unit-root nuisance),
  - AIC order: per-series AIC-selected AR order (1..12) instead of a fixed 8.

Each variant = one or more "refconfigs"; each refconfig -> AR var-ratio at scales 60/130/250. Combos
(ctrl + returns / ctrl + trailing) test orthogonality on top of the ar8_ms base. Added to champion_feat,
both members consume it. 2-stage seed gate: search {42,7,123} -> fresh {31337,2025,555,99}.

Usage:
  SMOKE=1 .venv/bin/python -u _residual_reffit.py
  .venv/bin/python -u _residual_reffit.py
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
RESULTS = "_spanbank_results.log"
SCALES = (60, 130, 250)
NCHAMP = 45

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


def _aic_order(x, pmax=12):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    best_p, best_aic = 1, np.inf
    for p in range(1, pmax + 1):
        m = _fit_ar(x, p)
        if m is None:
            break
        aic = n * math.log(m[1]) + 2.0 * (p + 1)
        if aic < best_aic:
            best_aic, best_p = aic, p
    return best_p


def _fit_refconfig(x_h, rc):
    """rc: dict(p=int | aic=True, trail=int|None, diff=bool). -> (rc, order, (coef, refvar)|None)."""
    x = np.asarray(x_h, dtype=np.float64)
    if rc.get("diff"):
        x = np.diff(x)
    if rc.get("trail"):
        x = x[-rc["trail"]:]
    p = _aic_order(x) if rc.get("aic") else rc["p"]
    return rc, p, _fit_ar(x, p)


def _fit_refs(x_h, spec):
    return [_fit_refconfig(x_h, rc) for rc in spec["refs"]]


def _extra_feats(w, refs, spec):
    w = np.asarray(w, dtype=np.float64)
    out = []
    for rc, p, m in refs:
        for s in SCALES:
            ws = w if s >= len(w) else w[-s:]
            wser = np.diff(ws) if rc.get("diff") else ws
            r = _ar_resid(wser, m[0], p) if m is not None else None
            out.append(0.0 if (r is None or len(r) < 5) else math.log((float(np.var(r)) + 1e-12) / m[1]))
    return out


def _nextra(spec):
    return len(spec["refs"]) * len(SCALES)


# --- feature build (champion + extra) ---------------------------------------
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
    jobs = []
    for sid in fold_ids:
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64), onl.index))
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


def experiments():
    full = dict(p=8)
    def t(k):
        return dict(p=8, trail=k)
    return [
        dict(name="ctrl_full", refs=[full]),                         # ar8_ms base (+0.0232)
        dict(name="full_120", refs=[full, t(120)]),                  # current champion (+0.0303)
        dict(name="full_60", refs=[full, t(60)]),
        dict(name="full_90", refs=[full, t(90)]),
        dict(name="full_200", refs=[full, t(200)]),
        dict(name="full_120_60", refs=[full, t(120), t(60)]),
        dict(name="full_120_200", refs=[full, t(120), t(200)]),
        dict(name="full_120_250", refs=[full, t(120), t(250)]),
        dict(name="full_60_120_200", refs=[full, t(60), t(120), t(200)]),
    ]


def log(msg):
    with open(RESULTS, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def run_phase(specs, seeds, phase):
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
                    log(f"[{phase} seed {seed}] champion base={base_aucs.mean():.4f} ({time.time()-t0:.0f}s)")
                aug = _eval(np.arange(NCHAMP + nx), Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
                md, _s, t, _ = paired(aug, base_aucs)
                store[spec["name"]].append((md, t))
                log(f"  [{phase} s{seed}] {spec['name']:<16} +{nx:>2}f aug={aug.mean():.4f} d={md:+.4f} t={t:+.2f}")
                del caches
            except Exception:
                log(f"  [{phase} s{seed}] {spec['name']:<16} ERROR\n{traceback.format_exc()}")
        del X
    out = {}
    for name, dt in store.items():
        if not dt:
            continue
        ds = [d for d, _t in dt]
        ts = [t for _d, t in dt]
        out[name] = (float(np.mean(ds)), float(min(ds)), sum(1 for d in ds if d > 0), float(min(ts)), len(ds))
    return out


def verdict(md, mind, npos, mint, n):
    if npos == n and md >= 0.0015 and mint >= 2.0:
        return "STRONG"
    return "pos" if npos == n and md > 0 else "noise"


def main():
    log(f"\n===== SPANBANK (reference-span sweep) {datetime.now():%F %T} (smoke={SMOKE}) =====")
    specs = experiments()
    scr = run_phase(specs, SEARCH_SEEDS, "SCREEN")
    log("\n----- SCREEN SUMMARY (search) -----")
    promising = []
    for name in sorted(scr, key=lambda n: -scr[n][0]):
        md, mind, npos, mint, n = scr[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} min_t {mint:+.2f} -> {v}")
        if npos == n and md >= 0.0015:
            promising.append(name)
    if promising:
        cs = [s for s in specs if s["name"] in promising]
        log(f"\nfresh-confirming {promising} on {FRESH_SEEDS}")
        fr = run_phase(cs, FRESH_SEEDS, "FRESH")
        log("\n----- FRESH SUMMARY -----")
        for name in sorted(fr, key=lambda n: -fr[n][0]):
            md, mind, npos, mint, n = fr[name]
            log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    log(f"===== REFFIT DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
