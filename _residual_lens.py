"""Residual-lens ablation + expansion (Experiment C' follow-up).

C' proved AR(1)+AR(2) forward-residual features add a confirmed +0.011 to the champion blend (7/7 seeds).
This harness (a) ABLATES which of the six residual statistics x AR order drive it, and (b) EXPANDS the
lens: higher AR orders (3,4) and a volatility-dynamics block (squared/abs-residual ACF, ARCH-LM, EWMA
vol ratio) computed on the AR residuals -- the cheap, no-MLE core of Experiment D.

Efficiency: build the FULL residual feature matrix once per seed (champion 45 + all residual cols), then
evaluate every config by slicing columns + refitting champion HGB + rank_xendcg ranker, paired-t vs the
champion (champion cols only). No per-config feature recompute.

Statistics per AR order p: varratio (log resid-var ratio), meanshift (standardized), acf1 (leftover
lag-1), lb (Ljung-Box 3), cusum (CUSUM-of-residuals / Brown-Durbin-Evans), outlier (rate vs ref sigma).
Vol block (from AR-2 residuals): sqacf1, sqacf2 (squared-resid ACF), absacf1, archlm (ARCH-LM q=2),
ewmavol (RiskMetrics EWMA variance ratio vs reference).

Usage:
  SMOKE=1 .venv/bin/python -u _residual_lens.py
  SEEDS=42,7,123 ORDERS=1,2,3,4 .venv/bin/python -u _residual_lens.py
  SEEDS=31337,2025,555,99 .venv/bin/python -u _residual_lens.py    # fresh confirm best config
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
POOL_SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7" if SMOKE else "42,7,123").split(",")]
DENSE_K = 30 if SMOKE else 60
ORDERS = tuple(int(o) for o in os.environ.get("ORDERS", "1,2,3,4").split(","))

TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)
NCHAMP = 45
STATS = ("varratio", "meanshift", "acf1", "lb", "cusum", "outlier")
VOL = ("sqacf1", "sqacf2", "absacf1", "archlm", "ewmavol")
NAMES = [f"ar{p}_{s}" for p in ORDERS for s in STATS] + [f"vol_{v}" for v in VOL]
NEXTRA = len(NAMES)
VORDER = 2 if 2 in ORDERS else ORDERS[0]


# --- reference model + residual statistics ----------------------------------
def _fit_ar(x, p):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < p + 8:
        return None
    cols = [x[p - lag: n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
    return coef, float(np.var(x[p:] - Xd @ coef)) + 1e-12


def _ar_resid(w, coef, p):
    n = len(w)
    if n < p + 3:
        return None
    cols = [w[p - lag: n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    return w[p:] - Xd @ coef


def _acf(x, k):
    if len(x) <= k:
        return 0.0
    x0 = x - x.mean()
    d = float(np.dot(x0, x0)) + 1e-12
    return float(np.dot(x0[k:], x0[:-k])) / d


def _ar_stats(r, ref_var):
    sig = math.sqrt(ref_var)
    rv = float(np.var(r)) + 1e-12
    n = len(r)
    q = 0.0
    for k in (1, 2, 3):
        if n - k > 0:
            ak = _acf(r, k)
            q += ak * ak / (n - k)
    cs = np.cumsum(r) / (sig * math.sqrt(n))
    return [math.log(rv / ref_var), float(np.mean(r)) / math.sqrt(rv), _acf(r, 1),
            n * (n + 2) * q, float(np.max(np.abs(cs))), float(np.mean(np.abs(r) > 3.0 * sig))]


def _vol_stats(r, ref_var, lam=0.94):
    r2 = r * r
    n = len(r2)
    if n > 6:
        y = r2[2:]
        Xd = np.column_stack([np.ones(n - 2), r2[1:-1], r2[:-2]])
        coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        ss_res = float(np.sum((y - Xd @ coef) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        archlm = (n - 2) * max(1.0 - ss_res / ss_tot, 0.0)
    else:
        archlm = 0.0
    v = ref_var
    for e in r2:
        v = lam * v + (1 - lam) * e
    return [_acf(r2, 1), _acf(r2, 2), _acf(np.abs(r), 1), archlm, math.log((v + 1e-12) / ref_var)]


def _resid_feats(w, ref):
    out, cache = [], {}
    for p in ORDERS:
        m = ref.get(p)
        r = _ar_resid(w, m[0], p) if m is not None else None
        if r is None or len(r) < 5:
            out += [0.0] * len(STATS)
            cache[p] = None
        else:
            cache[p] = (r, m[1])
            out += _ar_stats(r, m[1])
    rc = cache.get(VORDER)
    out += _vol_stats(rc[0], rc[1]) if rc is not None else [0.0] * len(VOL)
    return out


def _ref_models(x_h):
    x = np.asarray(x_h, dtype=np.float64)
    return {p: _fit_ar(x, p) for p in ORDERS}


def _series_bank(x_on, ctx, ref, active):
    champ, extra = [], []
    for t in active:
        win = x_on[max(0, t - WINDOW + 1):t + 1]
        champ.append(champion_feat(win, ctx))
        extra.append(_resid_feats(win, ref))
    if not champ:
        return np.zeros((0, NCHAMP + NEXTRA))
    return np.hstack([np.asarray(champ, dtype=np.float64), np.asarray(extra, dtype=np.float64)])


def _active(x_on):
    L = len(x_on)
    return np.arange(MIN_N - 1, L) if L >= MIN_N else np.zeros(0, dtype=int)


def _val_series(sid, x_h, x_on, onl_index):
    ctx = hist_context(x_h, HIST_CAP)
    feat = _series_bank(x_on, ctx, _ref_models(x_h), _active(x_on))
    midx = pd.MultiIndex.from_product([[sid], onl_index], names=["id", "time"])
    return midx, _active(x_on), feat


def precompute_fold_aug(X, fold_ids):
    jobs = []
    for sid in fold_ids:
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64), onl.index))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series)(sid, x_h, x_on, oi) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, active_all, offset = [], [], [], 0
    for midx, active, feat in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, NCHAMP + NEXTRA)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCHAMP + NEXTRA))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


def _train_series(sid, x_h, x_on, tau, steps):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return np.zeros((0, NCHAMP + NEXTRA)), [], []
    ref = _ref_models(x_h)
    base = active[0]
    rows, labels, tvals = [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        win = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(np.concatenate([champion_feat(win, ctx), _resid_feats(win, ref)]))
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return (np.asarray(rows) if rows else np.zeros((0, NCHAMP + NEXTRA)), labels, tvals)


def build_aug(X, yi, sid_to_steps):
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
    feat = np.vstack(rows) if rows else np.zeros((0, NCHAMP + NEXTRA))
    return feat, labels.astype(np.int64), tvals.astype(np.int64)


def _score(model, cache, cols):
    full = np.zeros(len(cache["index"]))
    if len(cache["feat"]):
        full[cache["active"]] = model.predict_proba(cache["feat"][:, cols])[:, 1]
    return pd.Series(full, index=cache["index"])


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


# --- ablation configs (champion + selected residual groups) -----------------
def _idx(pred):
    return [NCHAMP + i for i, n in enumerate(NAMES) if pred(n)]


GROUPS = {f"ar{p}": _idx(lambda n, p=p: n.startswith(f"ar{p}_")) for p in ORDERS}
GROUPS.update({f"stat_{s}": _idx(lambda n, s=s: n.startswith("ar") and n.endswith(f"_{s}")) for s in STATS})
GROUPS["vol"] = _idx(lambda n: n.startswith("vol_"))

_all_ar = [f"ar{p}" for p in ORDERS]
CONFIGS = [
    ("orig_ar12", ["ar1", "ar2"] if 2 in ORDERS else ["ar1"]),
    ("ar1_only", ["ar1"]),
    ("all_ar", list(_all_ar)),
    ("all_ar+vol", list(_all_ar) + ["vol"]),
    ("vol_only", ["vol"]),
    ("stat_varratio", ["stat_varratio"]),
    ("stat_meanshift", ["stat_meanshift"]),
    ("stat_acf1", ["stat_acf1"]),
    ("stat_lb", ["stat_lb"]),
    ("stat_cusum", ["stat_cusum"]),
    ("stat_outlier", ["stat_outlier"]),
    ("cusum+varratio", ["stat_cusum", "stat_varratio"]),
]
if len(ORDERS) > 2:
    CONFIGS.insert(3, (f"ar{ORDERS[-1]}_only", [f"ar{ORDERS[-1]}"]))


def _cols_for(group_names):
    cols = list(range(NCHAMP))
    for g in group_names:
        cols += GROUPS[g]
    return np.array(sorted(set(cols)))


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


def main():
    print(f"=== Residual-lens ablation {datetime.now():%F %T} | seeds={POOL_SEEDS} | ORDERS={ORDERS} "
          f"| {NEXTRA} resid cols | {len(CONFIGS)} configs ===", flush=True)
    d_store = {c[0]: [] for c in CONFIGS}
    t_store = {c[0]: [] for c in CONFIGS}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold_aug(X, f) for f in folds]
        picks = select_picks(X, train_ids)
        Xtr, ytr, _ = build_aug(X, yi, picks)
        rng = np.random.default_rng(seed + 202)
        rank_steps = {}
        for sid in train_ids:
            x_h, onl = split_series(X.loc[sid])
            L = len(onl.to_numpy())
            if L <= MIN_N:
                continue
            cand = np.arange(MIN_N, L)
            rank_steps[sid] = rng.choice(cand, size=min(DENSE_K, len(cand)), replace=False)
        Rr, Rl, Rt = build_aug(X, yi, rank_steps)

        base_aucs = _eval(np.arange(NCHAMP), Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
        print(f"\n[seed {seed}] built ({time.time()-t0:.0f}s) champion base={base_aucs.mean():.4f}",
              flush=True)
        for name, grps in CONFIGS:
            cols = _cols_for(grps)
            aug = _eval(cols, Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
            md, _s, t, _ = paired(aug, base_aucs)
            d_store[name].append(md)
            t_store[name].append(t)
            print(f"  {name:<16} +{len(cols)-NCHAMP:>2}f  aug={aug.mean():.4f} d={md:+.4f} t={t:+.2f}",
                  flush=True)
        del X, caches

    print("\n=== SUMMARY (sorted by mean_d) ===", flush=True)
    for name in sorted(d_store, key=lambda n: -np.mean(d_store[n])):
        ds, ts_ = d_store[name], t_store[name]
        npos = sum(1 for x in ds if x > 0)
        tag = "STRONG" if npos == len(ds) and np.mean(ds) >= 0.0015 and min(ts_) >= 2.0 else (
            "pos" if npos == len(ds) and np.mean(ds) > 0 else "noise")
        print(f"{name:<16} mean_d {np.mean(ds):+.4f} min_d {min(ds):+.4f} pos {npos}/{len(ds)} "
              f"min_t {min(ts_):+.2f} -> {tag}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
