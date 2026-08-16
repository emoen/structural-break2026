"""Experiment C' (the residual lens): fit a reference model, feature-ize its RESIDUALS.

Champion features compare the RAW window to the RAW history. This changes the lens: fit AR(p) to the
historical (pre-break) reference, apply it forward to the current online window, and measure how badly
the reference model mispredicts. If the online window is post-break, the reference residuals inflate,
gain leftover autocorrelation, shift mean, and drift (CUSUM). This strips nuisance structure (trend,
autocorrelation) and looks at what's left -- the canonical structural-break framing (Brown-Durbin-Evans).

Per AR order p in {1,2}: log residual-variance ratio (window vs reference fit), standardized residual
mean shift, lag-1 residual ACF, Ljung-Box(3) on residuals, CUSUM-of-residuals (BDE), outlier rate.
=> 6 x 2 = 12 residual features appended to the 45 champion features, fed to BOTH champion members.

Two-stage gate: screen {42,7,123}; pos 3/3 -> confirm fresh {31337,2025,555,99}; ship = pos all,
mean_d>=0.0015, min_t>=2.5.

Usage:
  SMOKE=1 .venv/bin/python -u _residual_c.py
  SEEDS=42,7,123 .venv/bin/python -u _residual_c.py
  SEEDS=31337,2025,555,99 .venv/bin/python -u _residual_c.py     # fresh confirm
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
AR_ORDERS = (1, 2)

TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)
NCHAMP = 45
NEXTRA = 6 * len(AR_ORDERS)


# --- reference-model residual features --------------------------------------
def _fit_ar(x, p):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < p + 8:
        return None
    cols = [x[p - lag: n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
    resid = x[p:] - Xd @ coef
    return coef, float(np.var(resid)) + 1e-12


def _ar_resid(w, coef, p):
    n = len(w)
    if n < p + 3:
        return None
    cols = [w[p - lag: n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    return w[p:] - Xd @ coef


def _resid_feats(w, ref):
    out = []
    for p in AR_ORDERS:
        m = ref.get(p)
        r = _ar_resid(w, m[0], p) if m is not None else None
        if r is None or len(r) < 5:
            out += [0.0] * 6
            continue
        sig_ref = math.sqrt(m[1])
        r0 = r - r.mean()
        denom = float(np.dot(r0, r0)) + 1e-12
        rv = float(np.var(r)) + 1e-12
        n = len(r)
        acf1 = float(np.dot(r0[1:], r0[:-1])) / denom
        q = 0.0
        for k in (1, 2, 3):
            if n - k > 0:
                ak = float(np.dot(r0[k:], r0[:-k])) / denom
                q += ak * ak / (n - k)
        cs = np.cumsum(r) / (sig_ref * math.sqrt(n))
        out += [
            math.log(rv / m[1]),                         # residual variance ratio
            float(np.mean(r)) / math.sqrt(rv),           # standardized mean shift
            acf1,                                        # leftover lag-1 autocorrelation
            n * (n + 2) * q,                             # Ljung-Box(3)
            float(np.max(np.abs(cs))),                   # CUSUM-of-residuals (BDE)
            float(np.mean(np.abs(r) > 3.0 * sig_ref)),   # outlier rate vs reference sigma
        ]
    return out


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


def _ref_models(x_h):
    x = np.asarray(x_h, dtype=np.float64)
    return {p: _fit_ar(x, p) for p in AR_ORDERS}


# --- validation cache -------------------------------------------------------
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


# --- training rows ----------------------------------------------------------
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
        row = np.concatenate([champion_feat(win, ctx), _resid_feats(win, ref)])
        rows.append(row)
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


def main():
    print(f"=== Experiment C' (residual lens) {datetime.now():%F %T} | seeds={POOL_SEEDS} "
          f"| AR{AR_ORDERS} | {NCHAMP}+{NEXTRA} feats ===", flush=True)
    cols_base = np.arange(NCHAMP)
    cols_aug = np.arange(NCHAMP + NEXTRA)
    d_store, t_store, base_store, aug_store, solo_store = [], [], [], [], []

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
        print(f"\n[seed {seed}] features built ({time.time()-t0:.0f}s) "
              f"hgb_rows={len(Xtr)} rank_rows={len(Rr)}", flush=True)

        res = {}
        for tag, cols in (("base", cols_base), ("aug", cols_aug)):
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
            blend = np.asarray([
                ts_auc_from(pd.Series(0.6 * a + 0.4 * b, index=s.index).to_frame("prediction"), y_val)
                for a, b, s in zip(fh, fr, hs)])
            solo = np.asarray([ts_auc_from(s.to_frame("prediction"), y_val) for s in hs])
            res[tag] = (blend, solo)

        md, _sd, t, _ = paired(res["aug"][0], res["base"][0])
        d_store.append(md)
        t_store.append(t)
        base_store.append(res["base"][0].mean())
        aug_store.append(res["aug"][0].mean())
        solo_store.append((res["base"][1].mean(), res["aug"][1].mean()))
        print(f"  base_blend={res['base'][0].mean():.4f} aug_blend={res['aug'][0].mean():.4f} "
              f"d={md:+.4f} t={t:+.2f} | hgb_solo {res['base'][1].mean():.4f}->{res['aug'][1].mean():.4f}",
              flush=True)
        del X, caches

    npos = sum(1 for x in d_store if x > 0)
    gate = "SHIP(hard)" if npos == len(d_store) and np.mean(d_store) >= 0.0015 and min(t_store) >= 2.5 \
        else ("promising" if npos == len(d_store) and np.mean(d_store) > 0 else "noise")
    print("\n=== SUMMARY (champion + residual-lens vs champion) ===", flush=True)
    print(f"mean_d {np.mean(d_store):+.4f} min_d {min(d_store):+.4f} pos {npos}/{len(d_store)} "
          f"min_t {min(t_store):+.2f} base {np.mean(base_store):.4f} aug {np.mean(aug_store):.4f} "
          f"-> {gate}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
