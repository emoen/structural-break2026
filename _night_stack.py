"""Shared build machinery + stacking evaluator for the overnight big-run.

Two pieces:
  1. Wide-matrix feature build (`build_rows_wide`, `precompute_fold_wide`) over a selectable
     `mode` -> {"champ" (51), "champ_dep" (65), "wide" (1102)}. Features are computed ONCE per
     seed; every experiment then selects column blocks of the same matrices (cheap).
  2. `eval_stack` -- a leak-free stacking evaluator: diverse first-level models -> out-of-fold
     predictions via SERIES-level K-fold (no row from a series appears in both train and OOF for
     that series) -> fixed-CDF calibration -> a second-level meta-learner. Mirrors the 2025
     first-place Alphabot pipeline (independent members + meta stack) under our causal TS-AUC.

`_eval` (the shipped 0.6/0.4 HGB+ranker champion blend) is imported unchanged as the control.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, load_pool_strat, make_hgb, select_picks
from _exp_fast import HIST_CAP, hist_context, split_series, ts_auc_from
from _lambdarank import fit_ranker, sort_into_groups
from _overnight import MIN_N, N_JOBS, paired  # noqa: F401 (re-exported)
from _residual_overnight import TUNED, _active, _eval, pct  # noqa: F401 (re-exported)

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]
FRESH_SEEDS = [31337, 2025] if SMOKE else [31337, 2025, 555, 99]
DENSE_K = 30 if SMOKE else 60

from _night_banks import (
    NCHAMP_BASE, NDEP, NDEP2, NSUPF2, NSEQ, NCALIB, NCALIBZ, NALPHA, NBAYESCP,
    fit_champ_refs, champ_base_row, fit_dep_refs, dep_feats,
    fit_dep2_refs, dep2_feats, fit_supf2_refs, supf2_feats, seq_feats,
    fit_calib_refs, calib_feats, fit_calibz_refs, calibz_feats, alpha_full,
    bayescp_feats,
)

MODE_NCOL = {"champ": NCHAMP_BASE, "champ_dep": NCHAMP_BASE + NDEP,
             "champ_dep2": NCHAMP_BASE + NDEP + NDEP2,
             "champ_supf2": NCHAMP_BASE + NDEP + NSUPF2,
             "champ_seq": NCHAMP_BASE + NDEP + NSEQ,
             "champ_calib": NCHAMP_BASE + NDEP + NCALIB,
             "champ_calibz": NCHAMP_BASE + NDEP + NCALIBZ,
             "champ_bayescp": NCHAMP_BASE + NDEP + NBAYESCP,
             "wide": NCHAMP_BASE + NDEP + NALPHA}


# ---------------------------------------------------------------------------
# Feature build (row = champion base [+ dep] [+ alpha], selected by `mode`)
# ---------------------------------------------------------------------------
def _row(w, ctx, refs, mode):
    out = champ_base_row(w, ctx, refs["champ"])
    if mode != "champ":
        out = np.concatenate([out, np.asarray(dep_feats(w, ctx, refs["dep"]), dtype=np.float64)])
    if mode == "champ_dep2":
        out = np.concatenate([out, np.asarray(dep2_feats(w, ctx, refs["dep2"]), dtype=np.float64)])
    if mode == "champ_supf2":
        out = np.concatenate([out, np.asarray(supf2_feats(w, ctx, refs["supf2"]), dtype=np.float64)])
    if mode == "champ_seq":
        out = np.concatenate([out, np.asarray(seq_feats(w, ctx, refs["dep"]), dtype=np.float64)])
    if mode == "champ_bayescp":
        out = np.concatenate([out, np.asarray(bayescp_feats(w), dtype=np.float64)])
    if mode == "champ_calib":
        out = np.concatenate([out, np.asarray(calib_feats(out[:NCHAMP_BASE], refs["calib"]), dtype=np.float64)])
    if mode == "champ_calibz":
        out = np.concatenate([out, np.asarray(calibz_feats(out[:NCHAMP_BASE], refs["calibz"]), dtype=np.float64)])
    if mode == "wide":
        out = np.concatenate([out, np.asarray(alpha_full(w, ctx), dtype=np.float64)])
    return out


def _fit_refs(x_h, mode):
    refs = {"champ": fit_champ_refs(x_h)}
    if mode != "champ":
        refs["dep"] = fit_dep_refs(x_h)
    if mode == "champ_dep2":
        refs["dep2"] = fit_dep2_refs(x_h)
    if mode == "champ_supf2":
        refs["supf2"] = fit_supf2_refs(x_h)
    if mode == "champ_calib":
        refs["calib"] = fit_calib_refs(x_h, refs["champ"], hist_context(x_h, HIST_CAP))
    if mode == "champ_calibz":
        refs["calibz"] = fit_calibz_refs(x_h, refs["champ"], hist_context(x_h, HIST_CAP))
    return refs


def _sanitize(a, ncol):
    if not len(a):
        return np.zeros((0, ncol))
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _val_series(sid, x_h, x_on, oi, mode, ncol):
    ctx = hist_context(x_h, HIST_CAP)
    refs = _fit_refs(x_h, mode)
    active = _active(x_on)
    rows = [_row(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs, mode) for t in active]
    feat = _sanitize(rows, ncol)
    midx = pd.MultiIndex.from_product([[sid], oi], names=["id", "time"])
    return midx, active, feat


def _train_series(sid, x_h, x_on, tau, steps, mode, ncol):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return np.zeros((0, ncol)), [], [], []
    refs = _fit_refs(x_h, mode)
    base = active[0]
    rows, labels, tvals = [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        rows.append(_row(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs, mode))
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return _sanitize(rows, ncol), labels, tvals, [int(sid)] * len(labels)


def precompute_fold_wide(X, fold_ids, mode):
    ncol = MODE_NCOL[mode]
    jobs = [(sid, np.asarray(split_series(X.loc[sid])[0], dtype=np.float64),
             split_series(X.loc[sid])[1].to_numpy(dtype=np.float64),
             split_series(X.loc[sid])[1].index) for sid in fold_ids]
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series)(sid, x_h, x_on, oi, mode, ncol) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, active_all, offset = [], [], [], 0
    for midx, active, feat in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, ncol)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, ncol))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


def build_rows_wide(X, yi, sid_to_steps, mode):
    ncol = MODE_NCOL[mode]
    jobs = []
    for sid, steps in sid_to_steps.items():
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64),
                     int(yi.loc[sid, "tau_index"]), np.asarray(steps)))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_train_series)(sid, x_h, x_on, tau, steps, mode, ncol)
            for (sid, x_h, x_on, tau, steps) in jobs)
    rows = [r for r, _l, _t, _s in parts if len(r)]
    labels = np.concatenate([l for _r, l, _t, _s in parts if len(l)]) if rows else np.zeros(0, int)
    tvals = np.concatenate([t for _r, _l, t, _s in parts if len(t)]) if rows else np.zeros(0, int)
    sids = np.concatenate([s for _r, _l, _t, s in parts if len(s)]) if rows else np.zeros(0, int)
    feat = np.vstack(rows) if rows else np.zeros((0, ncol))
    return feat, labels.astype(np.int64), tvals.astype(np.int64), sids.astype(np.int64)


# ---------------------------------------------------------------------------
# Stacking evaluator (leak-free series-level OOF + meta-learner)
# ---------------------------------------------------------------------------
def _fit_member(spec, Xc, y, t, seed):
    kind = spec["kind"]
    if kind == "hgb":
        m = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            m.fit(Xc, y)
        return m
    if kind == "lgbm":
        m = lgb.LGBMClassifier(objective="binary", n_estimators=spec.get("n_estimators", 400),
                               learning_rate=spec.get("learning_rate", 0.03),
                               num_leaves=spec.get("num_leaves", 31),
                               colsample_bytree=spec.get("colsample", 0.7), subsample=0.8,
                               subsample_freq=1, reg_lambda=1.0, random_state=seed,
                               n_jobs=N_JOBS, verbose=-1)
        with threadpool_limits(limits=1):
            m.fit(Xc, y)
        return m
    if kind == "extratrees":
        m = ExtraTreesClassifier(n_estimators=spec.get("n_estimators", 300),
                                 max_features=spec.get("max_features", "sqrt"),
                                 min_samples_leaf=spec.get("min_samples_leaf", 20),
                                 n_jobs=N_JOBS, random_state=seed)
        m.fit(Xc, y)
        return m
    if kind == "ranker":
        rows, lab, grp = sort_into_groups(Xc, y, t)
        return fit_ranker(spec.get("obj", "rank_xendcg"), rows, lab, grp)
    raise ValueError(kind)


def _member_pred(model, Xc):
    return model.predict_proba(Xc)[:, 1]


def _fit_meta(meta, F, y, seed):
    if meta == "mean":
        return ("mean", None)
    if meta == "logit":
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=3000, random_state=seed))
        m.fit(F, y)
        return ("logit", m)
    if meta == "gbm":
        m = lgb.LGBMClassifier(objective="binary", n_estimators=250, learning_rate=0.05,
                               num_leaves=15, colsample_bytree=0.8, subsample=0.8,
                               subsample_freq=1, reg_lambda=1.0, random_state=seed,
                               n_jobs=N_JOBS, verbose=-1)
        m.fit(F, y)
        return ("gbm", m)
    raise ValueError(meta)


def _meta_pred(mm, F):
    tag, m = mm
    if tag == "mean":
        return F.mean(axis=1)
    return m.predict_proba(F)[:, 1]


def eval_stack(members, meta, Xtr, ytr, Ttr, Str, caches, y_val, seed, mfold=3):
    """Leak-free stack: series-CV OOF meta-features -> fixed-CDF -> meta-learner -> per-fold TS-AUC.

    members: list of dicts {name, kind, cols, ...}. meta: 'mean' | 'logit' | 'gbm'.
    Returns per-fold TS-AUC array (comparable to `_eval`).
    """
    n, M = len(ytr), len(members)
    Foof = np.zeros((n, M))

    sids = np.unique(Str)
    rng = np.random.default_rng(seed + 7)
    rng.shuffle(sids)
    folds = np.array_split(sids, mfold)
    for f in range(mfold):
        held = np.isin(Str, folds[f])
        tr = ~held
        for mi, spec in enumerate(members):
            cols = spec["cols"]
            mdl = _fit_member(spec, Xtr[tr][:, cols], ytr[tr], Ttr[tr], seed)
            Foof[held, mi] = _member_pred(mdl, Xtr[held][:, cols])

    full, refs = [], []
    for mi, spec in enumerate(members):
        cols = spec["cols"]
        mdl = _fit_member(spec, Xtr[:, cols], ytr, Ttr, seed)
        ref = np.sort(_member_pred(mdl, Xtr[:, cols]))
        full.append((mdl, cols))
        refs.append(ref)
        Foof[:, mi] = pct(Foof[:, mi], ref)

    mm = _fit_meta(meta, Foof, ytr, seed)

    aucs = []
    for c in caches:
        L = len(c["index"])
        Fval = np.zeros((L, M))
        for mi, (mdl, cols) in enumerate(full):
            s = np.zeros(L)
            if len(c["feat"]):
                s[c["active"]] = _member_pred(mdl, c["feat"][:, cols])
            Fval[:, mi] = pct(s, refs[mi])
        score = _meta_pred(mm, Fval)
        aucs.append(ts_auc_from(pd.Series(score, index=c["index"]).to_frame("prediction"), y_val))
    return np.asarray(aucs)


# ---------------------------------------------------------------------------
# Per-seed harness (shared by the three runners) + verdict/logging
# ---------------------------------------------------------------------------
def prepare_seed(seed, mode, rr_mode=None):
    """Build every matrix a runner needs for one seed.

    caches/Xtr are in `mode` (>=51 cols). The control ranker rows `Rr` are in `rr_mode`
    (default = mode); pass rr_mode='champ' to keep the champion control cheap when `mode`
    is the 1102-col wide bank. Returns a dict of matrices.
    """
    rr_mode = rr_mode or mode
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
    caches = [precompute_fold_wide(X, f, mode) for f in folds]
    Xtr, ytr, Ttr, Str = build_rows_wide(X, yi, picks, mode)
    Rr, Rl, Rt, _ = build_rows_wide(X, yi, rank_steps, rr_mode)
    return dict(caches=caches, Xtr=Xtr, ytr=ytr, Ttr=Ttr, Str=Str,
                Rr=Rr, Rl=Rl, Rt=Rt, y_val=y_val)


def champion_base(S, seed, ncol_champ=NCHAMP_BASE):
    """Shipped 0.6/0.4 HGB+ranker champion blend on the first `ncol_champ` (=51) columns."""
    cols = np.arange(ncol_champ)
    return _eval(cols, S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"], S["caches"],
                 S["y_val"], seed)


def verdict(mean_d, min_d, npos, min_t, n):
    if npos == n and mean_d >= 0.0015 and min_t >= 2.0:
        return "STRONG"
    return "pos" if npos == n and mean_d > 0 else "noise"


def make_logger(path):
    def log(msg):
        with open(path, "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)
    return log


def summarize(store):
    out = {}
    for name, dt in store.items():
        if not dt:
            continue
        ds = [d for d, _t in dt]
        ts = [t for _d, t in dt]
        out[name] = (float(np.mean(ds)), float(min(ds)), sum(1 for d in ds if d > 0),
                     float(min(ts)), len(ds))
    return out
