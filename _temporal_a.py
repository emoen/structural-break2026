"""Experiment A: temporal aggregation of the per-step statistic (feature lever, model fixed).

Every champion feature is a single static window. This adds CAUSAL temporal structure across the
online steps: for three "window-vs-history" base scalars (median shift, log-MAD ratio, Wasserstein),
compute per-step EMA (fast/slow), deviation-from-slow-EMA (surprise), K-step trend, rolling mean and
rolling std -> 6 aggregates x 3 scalars = 18 temporal features appended to the 45 champion features.

Both champion members consume them (HGB + rank_xendcg ranker, same 0.6/0.4 train-CDF blend). We score
the augmented blend vs the champion blend (features off), paired-t across pool seeds. All temporal ops
are strictly causal (only past/current steps) -> no leakage.

Two-stage gate (mandatory): screen {42,7,123}; anything pos 3/3 -> confirm fresh {31337,2025,555,99};
ship = pos on all seeds AND mean_d>=0.0015 AND min_t>=2.5.

Usage:
  SMOKE=1 .venv/bin/python -u _temporal_a.py
  SEEDS=42,7,123 K=10 SF=5 SS=20 .venv/bin/python -u _temporal_a.py
  SEEDS=31337,2025,555,99 .venv/bin/python -u _temporal_a.py     # fresh confirm
"""
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
from _feat_lib import f_mad_ratio, f_median_shift, f_wasserstein
from _lambdarank import fit_ranker, sort_into_groups
from _overnight import MIN_N, N_JOBS, paired

SMOKE = os.environ.get("SMOKE") == "1"
POOL_SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7" if SMOKE else "42,7,123").split(",")]
DENSE_K = 30 if SMOKE else 60
K = int(os.environ.get("K", "10"))          # trend / rolling window (steps)
SF = int(os.environ.get("SF", "5"))         # fast EMA span
SS = int(os.environ.get("SS", "20"))        # slow EMA span

TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)
NCHAMP = 45
BASE_STATS = (f_median_shift, f_mad_ratio, f_wasserstein)   # 3 window-vs-history scalars


def _base_vec(win, ctx):
    return [fn(win, ctx)[0] for fn in BASE_STATS]


def _temporal(bmat):
    """(n, B) time-ordered base scalars -> (n, 6B) causal temporal features."""
    if len(bmat) == 0:
        return np.zeros((0, 6 * len(BASE_STATS)))
    df = pd.DataFrame(bmat)
    ef = df.ewm(span=SF, adjust=False).mean()
    es = df.ewm(span=SS, adjust=False).mean()
    dev = df - es
    delta = df - df.shift(K)
    rmean = df.rolling(K, min_periods=1).mean()
    rstd = df.rolling(K, min_periods=1).std()
    out = pd.concat([ef, es, dev, delta, rmean, rstd], axis=1).to_numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _active(x_on):
    # Match _featmat_one: score from idx == MIN_N-1 onward.
    L = len(x_on)
    return np.arange(MIN_N - 1, L) if L >= MIN_N else np.zeros(0, dtype=int)


def _series_bank(x_on, ctx, active):
    """Champion(45) + temporal(18) for every active step of a series, time-ordered."""
    champ, bvec = [], []
    for t in active:
        win = x_on[max(0, t - WINDOW + 1):t + 1]
        champ.append(champion_feat(win, ctx))
        bvec.append(_base_vec(win, ctx))
    champ = np.asarray(champ, dtype=np.float64) if len(champ) else np.zeros((0, NCHAMP))
    temp = _temporal(np.asarray(bvec, dtype=np.float64))
    return np.hstack([champ, temp]) if len(champ) else np.zeros((0, NCHAMP + 6 * len(BASE_STATS)))


# --- validation cache (full champion+temporal feature matrix per fold) ------
def _val_series(sid, x_h, x_on, onl_index):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    feat = _series_bank(x_on, ctx, active)
    midx = pd.MultiIndex.from_product([[sid], onl_index], names=["id", "time"])
    return midx, active, feat


def precompute_fold_aug(X, fold_ids):
    jobs = []
    for sid in fold_ids:
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, x_h, onl.to_numpy(dtype=np.float64), onl.index))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series)(sid, x_h, x_on, oi) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, active_all, offset = [], [], [], 0
    width = NCHAMP + 6 * len(BASE_STATS)
    for midx, active, feat in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, width)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, width))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


# --- training rows (champion+temporal at sampled steps) ---------------------
def _train_series(sid, x_h, x_on, tau, steps):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return np.zeros((0, NCHAMP + 6 * len(BASE_STATS))), [], []
    bank = _series_bank(x_on, ctx, active)
    base = active[0]
    rows, labels, tvals = [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        rows.append(bank[i])
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return (np.asarray(rows) if rows else np.zeros((0, bank.shape[1])), labels, tvals)


def build_aug(X, yi, sid_to_steps):
    jobs = []
    for sid, steps in sid_to_steps.items():
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, x_h, onl.to_numpy(dtype=np.float64),
                     int(yi.loc[sid, "tau_index"]), np.asarray(steps)))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_train_series)(sid, x_h, x_on, tau, steps)
            for (sid, x_h, x_on, tau, steps) in jobs)
    rows = [r for r, _l, _t in parts if len(r)]
    labels = np.concatenate([l for _r, l, _t in parts if len(l)]) if rows else np.zeros(0, int)
    tvals = np.concatenate([t for _r, _l, t in parts if len(t)]) if rows else np.zeros(0, int)
    feat = np.vstack(rows) if rows else np.zeros((0, NCHAMP + 6 * len(BASE_STATS)))
    return feat, labels.astype(np.int64), tvals.astype(np.int64)


def _score(model, cache, cols):
    full = np.zeros(len(cache["index"]))
    if len(cache["feat"]):
        full[cache["active"]] = model.predict_proba(cache["feat"][:, cols])[:, 1]
    return pd.Series(full, index=cache["index"])


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def main():
    print(f"=== Experiment A (temporal aggregation) {datetime.now():%F %T} | seeds={POOL_SEEDS} "
          f"| K={K} SF={SF} SS={SS} | {NCHAMP}+{6*len(BASE_STATS)} feats ===", flush=True)
    cols_base = np.arange(NCHAMP)
    cols_aug = np.arange(NCHAMP + 6 * len(BASE_STATS))
    d_store, t_store, base_store, aug_store, solo_store = [], [], [], [], []

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold_aug(X, f) for f in folds]

        # HGB training rows (pointwise member)
        picks = select_picks(X, train_ids)
        Xtr, ytr, _ = build_aug(X, yi, picks)

        # ranker training rows (grouped by step)
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
    print("\n=== SUMMARY (champion + temporal vs champion) ===", flush=True)
    print(f"mean_d {np.mean(d_store):+.4f} min_d {min(d_store):+.4f} pos {npos}/{len(d_store)} "
          f"min_t {min(t_store):+.2f} base {np.mean(base_store):.4f} aug {np.mean(aug_store):.4f} "
          f"-> {gate}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
