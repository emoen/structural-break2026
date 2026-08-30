"""ALTERNATE RESIDUAL-LENS probe on the CURRENT (E75) ranker+rf 79-col SHIP base.

AUDIT FACT (verified vs _night_banks/_residual_reffit/_rshape_rankerrf): every residual column in
base79 reads the residual through ONE lens only -- a fixed AR(8) fit -- and only ever as its
log-VARIANCE-RATIO log(var(window_resid)/var_ref) per MS scale. The residual LENS itself (the AR
order, the estimator family) has never been varied as a dedicated var-ratio column on the current
base. The prewhitening order is a free hyper the whole feature axis has silently pinned at 8.

FEATURE: add the SAME affine-invariant statistic the base uses -- log((var(window_resid_s)+1e-12) /
(ref_resid_var+1e-12)) at each SHIP scale -- but computed through a DIFFERENT residual lens, and
screen whether it exposes break structure the AR(8) lens absorbs or misses. This is a pure
lens-diversity test: identical statistic, identical scales, different prewhitening.

BASE (always in, = the E75 champion 79, byte-identical build to _rshape_rankerrf, SHIP_SCALES here):
  champ51 (AR8 full + AR8 trail120) + dep10 + ship12 + [AR8 trail60, AR8 trail200] x 3 scales.
  The base residual lens is AR(8); NONE of these base columns change in this probe.
CANDIDATE arms (each added in ISOLATION on base79; each lens is fit once per series in _refs, ~free):
  ar2    AR(2)  lens var-ratio  @ SHIP_SCALES   (3 cols)
  ar4    AR(4)  lens var-ratio  @ SHIP_SCALES   (3 cols)
  ar16   AR(16) lens var-ratio  @ SHIP_SCALES   (3 cols)
  rob8   robust/Huber AR(8) lens var-ratio @ SHIP_SCALES (3 cols) -- the one different FAMILY
  armix  ar2 + ar4 + ar16 together (9 cols)     -- tests ORDER-diversity as a set
rob8 is the heavy-tail hedge: the OLS var lens caps out on the t-dist / outlier-innovation families
(late-offset AUC plateaus ~0.57) because a few fat residuals dominate the plain variance; a Huber
lens down-weights them so the ref/window variance contrast reflects the bulk regime, not the tail.

rob8 IMPLEMENTATION (Huber-IRLS AR(8), affine-EQUIVARIANT): start from the OLS AR(8) fit, then run
2-3 IRLS reweights with Huber weights w_i = min(1, c/|r_i/scale|), c=1.345, scale = 1.4826*MAD(r)
(guarded against scale==0). The lens's history reference variance is the WEIGHTED residual variance
sum(w r^2)/sum(w); the window is residualized through the robust coef and read as its PLAIN var, same
as every other lens. Affine argument (see _affine_check): AR-with-intercept is affine-equivariant, so
r -> a*r under x -> a*x+b; MAD is scale-equivariant (MAD(a*r)=a*MAD(r)) so scale -> a*scale and the
Huber argument r/scale is INVARIANT -> identical weights -> the weighted fit is affine-equivariant too
-> both numerator and denominator variances scale by a^2 and cancel in the ratio. Only the +1e-12
epsilons perturb it (~1e-12 in practice), well under the 1e-6 gate.

AFFINE-INVARIANT to machine precision: each added column is a log-variance-RATIO. The AR fit (OLS or
Huber) is affine-equivariant, residuals scale by a, and var(window)/var(ref) -> (a^2..)/(a^2..) is
unchanged; the +1e-12 epsilons are the only perturbation. Validated by the smoke affine-check
(all arms maxdiff < 1e-6; the OLS-only arms separately < 1e-9).

PRIOR (honest, LOW): the AR order was a dead axis before -- AR(4) refconfigs were noise in the
feature banks (E25/E29) and the feature axis has been broadly saturated E52-E79. This is the FIRST
time alternate lenses (orders 2/4/16 + a robust AR8 family) are tested as DEDICATED var-ratio columns
on the current ranker+rf base79, and rob8 is the first genuinely different-FAMILY residual lens tried.
FAILURE WATCH: an alt-order lens is near-collinear with AR(8) on well-behaved series, so a win must
come from the heavy-tail families (where rob8 or the low-order lenses diverge from AR(8)); a broad
mean_d that is not concentrated there is likely noise. Do NOT tune to force positive deltas.

Reuses _fit_ar/_ar_resid from _residual_reffit and _fit_span/_span_feats from the base build. Same
seeds/gate/recipe/harness as _rshape_rankerrf.py. Own cache dir (_altlens_cache).

Run:
  SMOKE=1 .venv/bin/python -u _altlens_rankerrf.py         # tiny self-test + affine check
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _altlens_rankerrf.py \
      > _altlens_rankerrf.out 2>&1 & disown
"""
import math
import os
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestClassifier

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, _subwin, load_pool_strat, select_picks
from _exp_fast import HIST_CAP, hist_context, split_series
from _overnight import MIN_N, N_JOBS
from _residual_overnight import _active, pct
from _residual_reffit import _fit_ar, _ar_resid
from _night_banks import (NCHAMP_BASE, fit_champ_refs, champ_base_row,
                          fit_dep_refs, dep_feats)
from _lambdarank import sort_into_groups, fit_ranker

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]

RF_TREES = 120 if SMOKE else 500
DENSE_K = 20 if SMOKE else 40
W_RANK, W_RF = 0.45, 0.55
RF_KW = dict(max_features=0.5, min_samples_leaf=20)

# env-configurable so the screen can align to the shipped E78 champion window (40,100,250)
SHIP_SCALES = tuple(int(s) for s in os.environ.get("SHIP_SCALES", "40,100,250").split(","))

# shipped dep block keeps 10 of dep_feats' 14 cols (drops ACF-lag dacf1/2/5/10) -> b_no_accf.
_DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_altlens_cache")
RAW = os.path.join(HERE, "_altlens_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_altlens_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_altlens_rankerrf.log")

AR_P = 8  # order of the SHIPPED residual lens (base columns only; untouched here)

# --- base AR8 spans (SHIPPED, byte-identical to _spearac_rankerrf) ----------
BASE_SPAN_SPECS = [
    ("ar8_t60",  dict(p=8, trail=60)),
    ("ar8_t200", dict(p=8, trail=200)),
]

# --- candidate alternate-lens arms: (name, kind, order); each emits |SHIP_SCALES| cols ------
#   kind "ols"   -> plain OLS AR(order) fit on full history (via _fit_ar)
#   kind "huber" -> robust Huber-IRLS AR(order) fit on full history (via _fit_ar_huber)
ALT_LENSES = [
    ("ar2",  "ols",   2),
    ("ar4",  "ols",   4),
    ("ar16", "ols",   16),
    ("rob8", "huber", 8),
]

SPAN_W = len(SHIP_SCALES)                          # 3 cols per AR8 base span
NBASE = (NCHAMP_BASE + len(_DEP_KEEP) + 4 * len(SHIP_SCALES)
         + len(BASE_SPAN_SPECS) * SPAN_W)          # champ51 + dep10 + ship12 + base-spans6 = 79
_CAND_W = len(ALT_LENSES) * len(SHIP_SCALES)       # 4 lenses x 3 scales = 12
NCOL = NBASE + _CAND_W
BASE_COLS = np.arange(NBASE)

CAND_SLICE = {}
_off = NBASE
for _name, _kind, _p in ALT_LENSES:
    CAND_SLICE[_name] = np.arange(_off, _off + len(SHIP_SCALES))
    _off += len(SHIP_SCALES)


def cand_cols(cand_names):
    return np.concatenate([BASE_COLS] + [CAND_SLICE[n] for n in cand_names])


CAND_SINGLES = [n for n, _k, _p in ALT_LENSES]                    # ar2 ar4 ar16 rob8
CANDIDATES = [(n, [n]) for n in CAND_SINGLES] + [("armix", ["ar2", "ar4", "ar16"])]

# OLS-only cand cols (exclude the robust lens) -- these must be tighter than the whole set.
_OLS_CAND_COLS = np.concatenate([CAND_SLICE[n] for n in ("ar2", "ar4", "ar16")])


# --- AR8 base-span fit / residual-var-ratio feats (verbatim from _spearac) ---
def _fit_span(x_h, spec):
    x = np.asarray(x_h, dtype=np.float64)
    if spec.get("trail"):
        x = x[-spec["trail"]:]
    p = spec["p"]
    return p, _fit_ar(x, p)


def _span_feats(w, p, m):
    w = np.asarray(w, dtype=np.float64)
    out = []
    for s in SHIP_SCALES:
        ws = w if s >= len(w) else w[-s:]
        r = _ar_resid(ws, m[0], p) if m is not None else None
        out.append(0.0 if (r is None or len(r) < 5) else math.log((float(np.var(r)) + 1e-12) / m[1]))
    return out


# --- robust (Huber-IRLS) AR(p) lens, affine-equivariant ---------------------
def _fit_ar_huber(x, p, c=1.345, n_iter=3):
    """OLS-initialized Huber-IRLS AR(p) with intercept. Returns (coef, robust_var + 1e-12) or None.

    Weights w_i = min(1, c/|r_i/scale|), scale = 1.4826*MAD(r). Affine-equivariant: r -> a*r and
    scale -> a*scale under x -> a*x+b, so r/scale (hence w) is invariant -> coef equivariant and the
    weighted residual variance scales by a^2 (cancels in the log var-ratio)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < p + 8:
        return None
    cols = [x[p - lag:n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    y = x[p:]
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)   # OLS start
    w = None
    for _ in range(n_iter):
        r = y - Xd @ coef
        mad = np.median(np.abs(r - np.median(r)))
        scale = 1.4826 * mad
        if scale <= 1e-12:                          # near-degenerate residual -> stop reweighting
            break
        u = np.abs(r) / scale
        w = np.minimum(1.0, c / np.maximum(u, 1e-12))
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(Xd * sw[:, None], y * sw, rcond=None)
    r = y - Xd @ coef
    if w is None:
        rv = float(np.var(r))
    else:
        rv = float(np.sum(w * r * r) / np.sum(w))   # robust (weighted) residual variance
    return coef, rv + 1e-12


def _fit_lens(x_h, kind, p):
    x = np.asarray(x_h, dtype=np.float64)
    return _fit_ar_huber(x, p) if kind == "huber" else _fit_ar(x, p)


# --- ship feature funcs (verbatim from the shipped notebook) ---------------
_SHIP_EPS = 1e-12
_SHIP_MSPLIT_MIN_SEG = 5


def _ship_perm_entropy4(w):
    n = len(w)
    if n < 8:
        return [0.0]
    counts = {}
    for i in range(n - 3):
        pat = tuple(np.argsort(w[i:i + 4]))
        counts[pat] = counts.get(pat, 0) + 1
    p = np.array(list(counts.values()), dtype=np.float64)
    p /= p.sum()
    return [float(-(p * np.log(p)).sum()) / math.log(24)]


def _ship_msplit_prefix(w):
    n = len(w)
    lo, hi = _SHIP_MSPLIT_MIN_SEG, n - _SHIP_MSPLIT_MIN_SEG
    if hi <= lo:
        return None
    c1 = np.cumsum(w)
    c2 = np.cumsum(w * w)
    tot1 = c1[-1]
    tot2 = c2[-1]
    ks = np.arange(lo, hi + 1, dtype=np.float64)
    kidx = ks.astype(np.int64) - 1
    s1l = c1[kidx]
    s2l = c2[kidx]
    nl = ks
    nr = n - ks
    s1r = tot1 - s1l
    s2r = tot2 - s2l
    ssl = np.maximum(s2l - s1l * s1l / nl, 0.0)
    ssr = np.maximum(s2r - s1r * s1r / nr, 0.0)
    return n, nl, nr, s1l, s1r, ssl, ssr


def _ship_max_split_lr(w):
    pf = _ship_msplit_prefix(w)
    if pf is None:
        return [0.0]
    n, nl, nr, s1l, s1r, ssl, ssr = pf
    var0 = max(float(np.sum(w * w) / n - (float(np.sum(w)) / n) ** 2), 1e-12)
    varl = np.maximum(ssl / nl, 1e-12)
    varr = np.maximum(ssr / nr, 1e-12)
    llr = 0.5 * (n * math.log(var0) - nl * np.log(varl) - nr * np.log(varr))
    return [float(np.max(np.maximum(llr, 0.0)))]


def _ship_mosum(w, ctx, k=10):
    z = (np.asarray(w, dtype=np.float64) - ctx["mu"]) / max(ctx["sd"], 1e-8)
    sq = z * z
    n = len(sq)
    if n < 2 * k:
        return [0.0, 1.0]
    c = np.concatenate([[0.0], np.cumsum(sq)])
    blocks = (c[k:] - c[:-k]) / k
    mx = float(blocks.max())
    mn = float(blocks.mean())
    return [math.log(mx + _SHIP_EPS), mx / max(mn, _SHIP_EPS)]


def _ship_block(ws, ctx):
    return _ship_perm_entropy4(ws) + _ship_max_split_lr(ws) + _ship_mosum(ws, ctx)


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# --- feature row build -----------------------------------------------------
def _refs(x_h):
    return {"champ": fit_champ_refs(x_h), "dep": fit_dep_refs(x_h),
            "base_span": {name: _fit_span(x_h, spec) for name, spec in BASE_SPAN_SPECS},
            "alt_lens": {name: (p, _fit_lens(x_h, kind, p)) for name, kind, p in ALT_LENSES}}


def _cand_feats(w, refs):
    """base79 (+) the alt-lens var-ratio columns, row-aligned, in ALT_LENSES order."""
    out = []
    for name, _kind, _p in ALT_LENSES:
        p, m = refs["alt_lens"][name]
        out.extend(_span_feats(w, p, m))
    return out


def _row(w, ctx, refs):
    w = np.asarray(w, dtype=np.float64)
    dep_full = dep_feats(w, ctx, refs["dep"])
    out = [np.asarray(champ_base_row(w, ctx, refs["champ"]), dtype=np.float64),
           np.asarray([dep_full[i] for i in _DEP_KEEP], dtype=np.float64)]
    ship = []
    for s in SHIP_SCALES:
        ship.extend(_ship_block(_subwin(w, s), ctx))
    out.append(np.asarray(ship, dtype=np.float64))
    base_span = []
    for name, _spec in BASE_SPAN_SPECS:
        p, m = refs["base_span"][name]
        base_span.extend(_span_feats(w, p, m))
    out.append(np.asarray(base_span, dtype=np.float64))
    out.append(np.asarray(_cand_feats(w, refs), dtype=np.float64))
    return np.concatenate(out)


def _san(a):
    if not len(a):
        return np.zeros((0, NCOL))
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _train_series(sid, x_h, x_on, tau, steps):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return np.zeros((0, NCOL)), [], [], []
    refs = _refs(x_h)
    base = active[0]
    rows, labels, tvals = [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        rows.append(_row(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs))
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return _san(rows), labels, tvals, [int(sid)] * len(labels)


def _val_series(sid, x_h, x_on, oi):
    ctx = hist_context(x_h, HIST_CAP)
    refs = _refs(x_h)
    active = _active(x_on)
    rows = [_row(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs) for t in active]
    feat = _san(rows)
    midx = pd.MultiIndex.from_product([[sid], oi], names=["id", "time"])
    return midx, active, feat


def build_train(X, yi, sid_to_steps):
    jobs = []
    for sid, steps in sid_to_steps.items():
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64),
                     int(yi.loc[sid, "tau_index"]), np.asarray(steps)))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_train_series)(sid, x_h, x_on, tau, steps) for (sid, x_h, x_on, tau, steps) in jobs)
    rows = [r for r, _l, _t, _s in parts if len(r)]
    lab = np.concatenate([l for _r, l, _t, _s in parts if len(l)]) if rows else np.zeros(0, int)
    tv = np.concatenate([t for _r, _l, t, _s in parts if len(t)]) if rows else np.zeros(0, int)
    sids = np.concatenate([s for _r, _l, _t, s in parts if len(s)]) if rows else np.zeros(0, int)
    feat = np.vstack(rows) if rows else np.zeros((0, NCOL))
    return feat, lab.astype(np.int64), tv.astype(np.int64), sids.astype(np.int64)


def build_val(X, fold_ids):
    jobs = [(sid, np.asarray(split_series(X.loc[sid])[0], dtype=np.float64),
             split_series(X.loc[sid])[1].to_numpy(dtype=np.float64),
             split_series(X.loc[sid])[1].index) for sid in fold_ids]
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series)(sid, x_h, x_on, oi) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, active_all, offset = [], [], [], 0
    for midx, active, feat in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, NCOL)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCOL))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


def prepare_seed(seed):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"seed_{seed}{'_smoke' if SMOKE else ''}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
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
    Xtr, ytr, Ttr, Str = build_train(X, yi, picks)
    Rr, Rl, Rt, Rs = build_train(X, yi, rank_steps)
    caches = [build_val(X, f) for f in folds]
    sd = dict(Xtr=Xtr, ytr=ytr, Rr=Rr, Rl=Rl, Rt=Rt, caches=caches, y_val=y_val)
    with open(path, "wb") as f:
        pickle.dump(sd, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  [cache] seed {seed} built in {time.time()-t0:.0f}s  Xtr{Xtr.shape} Rr{Rr.shape} -> {path}")
    return sd


# --- scoring ---------------------------------------------------------------
def _proba(m, X):
    return m.predict_proba(X)[:, 1]


def _auc(lab, sc):
    r = pd.Series(sc).rank(method="average").to_numpy()
    npos = float(lab.sum())
    nneg = float(len(lab) - npos)
    if npos == 0 or nneg == 0:
        return None
    return (r[lab == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def make_folds_meta(sd):
    metas = []
    for c in sd["caches"]:
        idx = c["index"]
        L = len(idx)
        df = pd.Series(np.arange(L), index=idx).to_frame("row")
        merged = df.merge(sd["y_val"], how="left", left_index=True, right_index=True)
        merged["time_online"] = merged.groupby("id").cumcount()
        groups = []
        for _, g in merged.groupby("time_online"):
            lab = g["target"].to_numpy(dtype=np.float64)
            npos = int(lab.sum())
            nneg = int(len(lab) - npos)
            if npos == 0 or nneg == 0:
                continue
            groups.append((g["row"].to_numpy(), lab, float(npos * nneg)))
        metas.append((L, c["active"], groups))
    return metas


def score_cols(sd, metas, cols, seed):
    Xtr, ytr = sd["Xtr"], sd["ytr"]
    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=N_JOBS, random_state=seed, **RF_KW)
    rf.fit(Xtr[:, cols], ytr)
    ref_rf = np.sort(_proba(rf, Xtr[:, cols]))
    rows, lab, grp = sort_into_groups(sd["Rr"][:, cols], sd["Rl"], sd["Rt"])
    rk = fit_ranker("rank_xendcg", rows, lab, grp)
    ref_rk = np.sort(_proba(rk, sd["Rr"][:, cols]))

    aucs = []
    for (L, active, groups), c in zip(metas, sd["caches"]):
        srf = np.zeros(L)
        srk = np.zeros(L)
        if len(c["feat"]):
            srf[active] = _proba(rf, c["feat"][:, cols])
            srk[active] = _proba(rk, c["feat"][:, cols])
        blend = W_RANK * pct(srk, ref_rk) + W_RF * pct(srf, ref_rf)
        wsum = tot = 0.0
        for r, lb, w in groups:
            a = _auc(lb, blend[r])
            if a is None:
                continue
            wsum += w * a
            tot += w
        aucs.append(wsum / tot if tot > 0 else 0.5)
    return float(np.mean(aucs))


def _affine_check():
    """Every added alt-lens column must be invariant to x -> a*x + b (a>0).

    Builds a synthetic history+window with mild AR structure, computes the FULL added-column vector
    for x and for a*x+b applied to BOTH history and window, and asserts the max delta over the
    candidate cols is < 1e-6 (loose, to allow the robust lens's iterative rounding). The OLS-only
    arms (ar2/ar4/ar16) are asserted separately at < 1e-9."""
    rng = np.random.default_rng(0)
    xh = rng.standard_normal(600) * 3.0 + 5.0
    for i in range(2, len(xh)):                        # give it mild AR structure
        xh[i] += 0.3 * xh[i - 1] - 0.1 * xh[i - 2]
    w = xh[-260:].copy()
    ctx = hist_context(xh, HIST_CAP)
    r1 = _row(w, ctx, _refs(xh))
    a, b = 3.7, 5.2
    xh2, w2 = a * xh + b, a * w + b
    ctx2 = hist_context(xh2, HIST_CAP)
    r2 = _row(w2, ctx2, _refs(xh2))
    cand_idx = np.arange(NBASE, NCOL)
    md = float(np.max(np.abs(r1[cand_idx] - r2[cand_idx]))) if len(cand_idx) else 0.0
    md_ols = float(np.max(np.abs(r1[_OLS_CAND_COLS] - r2[_OLS_CAND_COLS])))
    md_rob = float(np.max(np.abs(r1[CAND_SLICE["rob8"]] - r2[CAND_SLICE["rob8"]])))
    nz = int(np.sum(np.abs(r1[cand_idx]) > 1e-9))
    log(f"[affine-check] alt-lens cand cols maxdiff under x->{a}x+{b}: {md:.2e}  "
        f"(OLS arms {md_ols:.2e}; rob8 {md_rob:.2e}; nonzero cand cols {nz}/{len(cand_idx)})")
    assert md < 1e-6, f"alt-lens affine invariance broken: maxdiff {md:.2e}"
    assert md_ols < 1e-9, f"OLS-lens affine invariance broken: maxdiff {md_ols:.2e}"


def main():
    open(LOG, "w").close()
    log(f"=== ALTERNATE-LENS probe on ranker+rf E75 base79  {datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF}")
    log(f"NBASE={NBASE} NCOL={NCOL} ship_scales={SHIP_SCALES}  base_lens=AR({AR_P})")
    log(f"lenses={[(n, k, p) for n, k, p in ALT_LENSES]}")
    log(f"candidates={[n for n, _ in CANDIDATES]}")
    _affine_check()

    if not os.path.exists(RAW):
        with open(RAW, "w") as f:
            f.write("seed\tcandidate\tbase_auc\tcand_auc\tdelta\n")

    per_seed = {}
    for seed in SEARCH_SEEDS:
        sd = prepare_seed(seed)
        metas = make_folds_meta(sd)
        t0 = time.time()
        base_auc = score_cols(sd, metas, BASE_COLS, seed)
        log(f"[seed {seed}] base79 ranker+rf = {base_auc:.4f}  ({time.time()-t0:.0f}s)")
        res = {}
        for name, cand_names in CANDIDATES:
            tc = time.time()
            cols = cand_cols(cand_names)
            auc = score_cols(sd, metas, cols, seed)
            d = auc - base_auc
            res[name] = d
            with open(RAW, "a") as f:
                f.write(f"{seed}\t{name}\t{base_auc:.6f}\t{auc:.6f}\t{d:+.6f}\n")
            log(f"[seed {seed}] {name:<12} {auc:.4f}  d={d:+.4f}  ({time.time()-tc:.0f}s)")
        per_seed[seed] = res

    names = [n for n, _ in CANDIDATES]
    agg = []
    for n in names:
        ds = [per_seed[s][n] for s in SEARCH_SEEDS if n in per_seed[s]]
        if not ds:
            continue
        md = float(np.mean(ds))
        npos = sum(1 for d in ds if d > 0)
        agg.append((n, md, min(ds), max(ds), npos, len(ds)))
    agg.sort(key=lambda r: -r[1])
    with open(SUMMARY, "w") as f:
        f.write(f"# alternate residual-LENS probe on ranker+rf E75 base79  {datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF}\n")
        f.write(f"# base79 = E75 champion (AR8 var-ratio lens); arms = SAME var-ratio, DIFFERENT lens\n")
        f.write(f"# ar2 ar4 ar16 (OLS orders) + rob8 (robust Huber AR8 family) @ {SHIP_SCALES}; ISOLATION vs base79\n")
        f.write(f"# RELAXED gate: believe only if positive on ALL screen seeds AND mean_d over threshold,\n")
        f.write(f"#   then fresh-confirm {{31337,2025,555,99}}. This is the noise gate, not a law.\n\n")
        f.write(f"{'candidate':<14}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for n, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md > 0 and npos == tot) else ""
            f.write(f"{n:<14}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
