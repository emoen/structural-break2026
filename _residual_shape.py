"""Shape/tail features as add-ons to the residual champion (harness test of the synth-diag candidates).

Synthetic diag found the residual champion is weakest on skew / fat-tail (tdist) breaks, and that
residual higher-moments (skew/kurt of the AR8 residuals), robust bowley skewness, and a tail-heaviness
ratio catch them -- but the champion already carries skew_diff/kurt_diff at 3 scales, so these may be
redundant. This is the definitive test: base = champion(45) + the shipped residual var-ratio block
(AR8 full+120 history, multi-scale, +6), and we add each SHAPE block on top, paired-t vs that base,
2-stage seed gate (search {42,7,123} -> fresh {31337,2025,555,99}).

Usage:
  SMOKE=1 .venv/bin/python -u _residual_shape.py
  .venv/bin/python -u _residual_shape.py
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
RESULTS = "_shape_results.log"
SCALES = (60, 130, 250)
REF_TRAILS = (None, 120)           # shipped champion residual reference spans
NCHAMP = 45
NRESID = len(REF_TRAILS) * len(SCALES)   # 6 -> shipped champion residual block
NBASE = NCHAMP + NRESID                   # 51 = current champion feature width

TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)


def _fit_ar(x, p=8):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < p + 8:
        return None
    cols = [x[p - lag:n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
    return coef, x[p:] - Xd @ coef


def _ar_resid(w, coef, p=8):
    n = len(w)
    if n < p + 3:
        return None
    cols = [w[p - lag:n - lag] for lag in range(1, p + 1)]
    return w[p:] - np.column_stack(cols + [np.ones(n - p)]) @ coef


def _skew(a):
    s = a.std()
    return 0.0 if s < 1e-9 else float(np.mean(((a - a.mean()) / s) ** 3))


def _exkurt(a):
    s = a.std()
    return 0.0 if s < 1e-9 else float(np.mean(((a - a.mean()) / s) ** 4)) - 3.0


def _bowley(a):
    q = np.quantile(a, [0.25, 0.5, 0.75])
    return float((q[2] + q[0] - 2 * q[1]) / (q[2] - q[0] + 1e-9))


def _tailratio(a):
    q = np.quantile(a, [0.05, 0.25, 0.75, 0.95])
    return float((q[3] - q[0]) / (q[2] - q[1] + 1e-9))


def _fit_refs(x_h):
    x = np.asarray(x_h, dtype=np.float64)
    refs = {}
    for trail in REF_TRAILS:
        xr = x if trail is None else x[-trail:]
        m = _fit_ar(xr)
        refs[trail] = None if m is None else (m[0], float(np.var(m[1])) + 1e-12)
    mfull = _fit_ar(x)
    rres = mfull[1] if mfull is not None else np.zeros(1)
    refs["_shape"] = dict(rskew=_skew(rres), rkurt=_exkurt(rres),
                          bowley=_bowley(x), tailratio=_tailratio(x))
    return refs


def _resid_block(w, refs):
    out = []
    for trail in REF_TRAILS:
        m = refs[trail]
        for s in SCALES:
            ws = w if s >= len(w) else w[-s:]
            r = _ar_resid(ws, m[0]) if m is not None else None
            out.append(0.0 if (r is None or len(r) < 5) else math.log((float(np.var(r)) + 1e-12) / m[1]))
    return out


def _shape_block(w, refs, spec):
    sh = refs["_shape"]
    full = refs[None]
    r = _ar_resid(w, full[0]) if full is not None else None
    out = []
    for name in spec["shape"]:
        if name == "resid_skew":
            out.append((_skew(r) - sh["rskew"]) if (r is not None and len(r) >= 5) else 0.0)
        elif name == "resid_kurt":
            out.append((_exkurt(r) - sh["rkurt"]) if (r is not None and len(r) >= 5) else 0.0)
        elif name == "bowley":
            out.append(_bowley(w) - sh["bowley"])
        elif name == "tailratio":
            out.append(_tailratio(w) - sh["tailratio"])
    return out


def _nshape(spec):
    return len(spec["shape"])


def _active(x_on):
    L = len(x_on)
    return np.arange(MIN_N - 1, L) if L >= MIN_N else np.zeros(0, dtype=int)


def _row(win, ctx, refs, spec):
    return list(champion_feat(win, ctx)) + _resid_block(win, refs) + _shape_block(win, refs, spec)


def _val_series(sid, x_h, x_on, oi, spec, nx):
    ctx = hist_context(x_h, HIST_CAP)
    refs = _fit_refs(x_h)
    active = _active(x_on)
    rows = [_row(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs, spec) for t in active]
    feat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, NBASE + nx))
    return pd.MultiIndex.from_product([[sid], oi], names=["id", "time"]), active, feat


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
        feats.append(feat if feat.shape[0] else np.zeros((0, NBASE + nx)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NBASE + nx))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


def _train_series(sid, x_h, x_on, tau, steps, spec, nx):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return np.zeros((0, NBASE + nx)), [], []
    refs = _fit_refs(x_h)
    base = active[0]
    rows, labels, tvals = [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        rows.append(_row(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs, spec))
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return (np.asarray(rows) if rows else np.zeros((0, NBASE + nx)), labels, tvals)


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
    feat = np.vstack(rows) if rows else np.zeros((0, NBASE + nx))
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
    return [
        dict(name="shape_all", shape=["resid_skew", "resid_kurt", "bowley", "tailratio"]),
        dict(name="resid_mom", shape=["resid_skew", "resid_kurt"]),
        dict(name="tailratio", shape=["tailratio"]),
        dict(name="resid_kurt", shape=["resid_kurt"]),
        dict(name="resid_skew", shape=["resid_skew"]),
        dict(name="bowley", shape=["bowley"]),
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
                nx = _nshape(spec)
                caches = [precompute_fold(X, f, spec, nx) for f in folds]
                Xtr, ytr, _ = build_rows(X, yi, picks, spec, nx)
                Rr, Rl, Rt = build_rows(X, yi, rank_steps, spec, nx)
                if base_aucs is None:
                    base_aucs = _eval(np.arange(NBASE), Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
                    log(f"[{phase} seed {seed}] champion+residual base={base_aucs.mean():.4f} "
                        f"({time.time()-t0:.0f}s)")
                aug = _eval(np.arange(NBASE + nx), Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
                md, _s, t, _ = paired(aug, base_aucs)
                store[spec["name"]].append((md, t))
                log(f"  [{phase} s{seed}] {spec['name']:<12} +{nx}f aug={aug.mean():.4f} d={md:+.4f} t={t:+.2f}")
                del caches
            except Exception:
                log(f"  [{phase} s{seed}] {spec['name']:<12} ERROR\n{traceback.format_exc()}")
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
    log(f"\n===== SHAPE (tail/asymmetry add-ons vs champion+residual) {datetime.now():%F %T} "
        f"(smoke={SMOKE}) =====")
    specs = experiments()
    scr = run_phase(specs, SEARCH_SEEDS, "SCREEN")
    log("\n----- SCREEN SUMMARY (search) -----")
    promising = []
    for name in sorted(scr, key=lambda n: -scr[n][0]):
        md, mind, npos, mint, n = scr[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<12} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} min_t {mint:+.2f} -> {v}")
        if npos == n and md >= 0.0015:
            promising.append(name)
    if promising:
        cs = [s for s in specs if s["name"] in promising]
        log(f"\nfresh-confirming {promising} on {FRESH_SEEDS}")
        fr = run_phase(cs, FRESH_SEEDS, "FRESH")
        log("\n----- FRESH SUMMARY -----")
        for name in sorted(fr, key=lambda n: -fr[n][0]):
            md, mind, npos, mint, n = fr[name]
            log(f"{name:<12} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} min_t {mint:+.2f} "
                f"-> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\nno shape feature screened promising; tail/asymmetry lever closed.")
    log(f"===== SHAPE DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
