"""PREWHITENED-RESIDUAL SHAPE/TAIL probe on the CURRENT (E75) ranker+rf 79-col SHIP base.

Origin: 2026-08-27 2-agent ideation #2 (proposer A1+A2, critic KEEP #1). AUDIT FACT (verified vs
_night_banks/_residual_reffit): every residual column in base79 is a log-VARIANCE-RATIO -- the AR(8)
residual is only ever read through its variance. Its skew/kurtosis/tail are NEVER computed. The base's
skew_diff/kurt_diff/tail2/tail3 are all on the RAW window (= the E59 dead axis, dependence-contaminated).
So the residual's SHAPE is the most under-derived object in the whole feature set.

FEATURE: fit AR(8) on the full history (the same lens the base var-ratio uses), take the residual (the
prewhitened innovations), and contrast the WINDOW residual's shape against the HISTORY residual's shape.
Prewhitening strips the linear-dependence contamination so the statistic isolates the INNOVATION
distribution -- the exact tdist (heavy tail) / garch / skew signature. It is a PERSISTENT post-break
regime property (fat innovations stay fat), so it fights the transition-detector's late-offset DECAY
instead of chasing the saturated transition.

BASE (always in, = the E75 champion 79, byte-identical to _spearac_rankerrf):
  champ51 (AR8 full + AR8 trail120) + dep10 + ship12 + [AR8 trail60, AR8 trail200] x 3 scales.
CANDIDATE arms (each added in ISOLATION on base79; residual reused across arms, ~free):
  rkurt    excess-kurtosis(resid_window) - kurtosis(resid_history)          @ scales 130, 250
  rqkurt   quantile-kurtosis (q95-q05)/(q75-q25) contrast (ROBUST hedge)    @ scales 60, 130, 250
  rhill    Hill upper-tail index on |resid-median| top-k, contrast          @ scale  250 only
  rshape_all   all three together
rhill is 250-only by design: it needs >=~15 tail exceedances; s=60 residual gives ~6 -> pure noise.
rkurt skips s=60 (4th moment too noisy at n~52); rqkurt carries the small scale (robust).

AFFINE-INVARIANT to machine precision: the AR fit is affine-equivariant (residual of a*x+b scales by a),
and kurtosis (4th-moment ratio), quantile-kurtosis (ratio of scale-equivariant spacings), and Hill
(log-differences of order statistics) are each dimensionless -> the window-minus-history contrast cancels
a and b exactly. Validated by the smoke affine-check (maxdiff ~0).

FAILURE WATCH (critic): rkurt is outlier-dominated -- a single fat point at the window edge spikes it as a
TRANSITION artifact that decays. Only believe a win if the ROBUST arm (rqkurt/rhill) moves too.

Reuses _fit_ar/_ar_resid from _residual_reffit. Same seeds/gate/recipe/harness as _spearac_rankerrf.py.

Run:
  SMOKE=1 .venv/bin/python -u _rshape_rankerrf.py           # tiny self-test + affine check
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _rshape_rankerrf.py \
      > _rshape_rankerrf.out 2>&1 & disown
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

SHIP_SCALES = (60, 130, 250)

# shipped dep block keeps 10 of dep_feats' 14 cols (drops ACF-lag dacf1/2/5/10) -> b_no_accf.
_DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_rshape_cache")
RAW = os.path.join(HERE, "_rshape_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_rshape_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_rshape_rankerrf.log")

AR_P = 8  # same order as the shipped residual lens

# --- base AR8 spans (SHIPPED, byte-identical to _spearac_rankerrf) ----------
BASE_SPAN_SPECS = [
    ("ar8_t60",  dict(p=8, trail=60)),
    ("ar8_t200", dict(p=8, trail=200)),
]

# --- candidate residual-shape arms: (name, stat_key, scales) ----------------
RSHAPE_ARMS = [
    ("rkurt",  "kurt",  (130, 250)),
    ("rqkurt", "qkurt", (60, 130, 250)),
    ("rhill",  "hill",  (250,)),
]
if SMOKE:
    RSHAPE_ARMS = [
        ("rkurt",  "kurt",  (130, 250)),
        ("rqkurt", "qkurt", (60, 130, 250)),
        ("rhill",  "hill",  (250,)),
    ]
_ALL_RSHAPE_SCALES = tuple(sorted({s for _n, _k, scs in RSHAPE_ARMS for s in scs}))

SPAN_W = len(SHIP_SCALES)                          # 3 cols per AR8 base span
NBASE = (NCHAMP_BASE + len(_DEP_KEEP) + 4 * len(SHIP_SCALES)
         + len(BASE_SPAN_SPECS) * SPAN_W)          # champ51 + dep10 + ship12 + base-spans6 = 79
_CAND_W = sum(len(scs) for _n, _k, scs in RSHAPE_ARMS)
NCOL = NBASE + _CAND_W
BASE_COLS = np.arange(NBASE)

CAND_SLICE = {}
_off = NBASE
for _name, _key, _scs in RSHAPE_ARMS:
    CAND_SLICE[_name] = np.arange(_off, _off + len(_scs))
    _off += len(_scs)


def cand_cols(cand_names):
    return np.concatenate([BASE_COLS] + [CAND_SLICE[n] for n in cand_names])


CAND_SINGLES = [n for n, _k, _s in RSHAPE_ARMS]
CANDIDATES = [(n, [n]) for n in CAND_SINGLES] + [("rshape_all", CAND_SINGLES)]


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


# --- residual-shape statistics (all affine-invariant, dimensionless) --------
def _kurt(r):
    if r is None or len(r) < 8:
        return 0.0
    r = np.asarray(r, dtype=np.float64)
    m = r.mean()
    v = float(r.var())
    if v < 1e-18:
        return 0.0
    return float((((r - m) ** 4).mean()) / (v * v) - 3.0)


def _qkurt(r):
    if r is None or len(r) < 8:
        return 0.0
    q05, q25, q75, q95 = np.quantile(np.asarray(r, dtype=np.float64), [0.05, 0.25, 0.75, 0.95])
    iqr = q75 - q25
    if abs(iqr) < 1e-18:
        return 0.0
    return float((q95 - q05) / iqr)


def _hill(r, frac=0.15, kmin=10):
    if r is None or len(r) < 30:
        return 0.0
    a = np.abs(np.asarray(r, dtype=np.float64) - np.median(r))
    a.sort()                                   # ascending
    n = len(a)
    k = min(max(kmin, int(frac * n)), n - 2)
    if k < kmin:
        return 0.0
    thr = a[n - k - 1]
    if thr <= 1e-18:
        return 0.0
    top = a[n - k:]
    top = top[top > 1e-18]
    if len(top) < kmin:
        return 0.0
    return float(np.mean(np.log(top) - math.log(thr)))


_STAT = {"kurt": _kurt, "qkurt": _qkurt, "hill": _hill}


def _fit_rshape_ref(x_h):
    """AR8 fit on full history + the reference (history-null) residual shape scalars."""
    m = _fit_ar(np.asarray(x_h, dtype=np.float64), AR_P)
    rh = _ar_resid(np.asarray(x_h, dtype=np.float64), m[0], AR_P) if m is not None else None
    ref = {key: fn(rh) for key, fn in _STAT.items()}
    return m, ref


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
    m_rshape, ref_rshape = _fit_rshape_ref(x_h)
    return {"champ": fit_champ_refs(x_h), "dep": fit_dep_refs(x_h),
            "base_span": {name: _fit_span(x_h, spec) for name, spec in BASE_SPAN_SPECS},
            "rshape_ar": m_rshape, "rshape_ref": ref_rshape}


def _cand_feats(w, refs):
    m = refs["rshape_ar"]
    ref = refs["rshape_ref"]
    resid_by_scale = {}
    for s in _ALL_RSHAPE_SCALES:
        ws = w if s >= len(w) else w[-s:]
        resid_by_scale[s] = _ar_resid(ws, m[0], AR_P) if m is not None else None
    out = []
    for _name, key, scales in RSHAPE_ARMS:
        fn = _STAT[key]
        rval = ref[key]
        for s in scales:
            out.append(fn(resid_by_scale[s]) - rval)
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
    """Candidate residual-shape cols must be invariant to x -> a*x + b (a>0)."""
    rng = np.random.default_rng(0)
    xh = rng.standard_normal(600) * 3.0 + 5.0
    for i in range(2, len(xh)):                        # give it mild AR structure
        xh[i] += 0.3 * xh[i - 1] - 0.1 * xh[i - 2]
    w = xh[-260:].copy()
    ctx = hist_context(xh, HIST_CAP)
    r1 = _row(w, ctx, _refs(xh))
    a, b = 7.3, -2.1
    xh2, w2 = a * xh + b, a * w + b
    ctx2 = hist_context(xh2, HIST_CAP)
    r2 = _row(w2, ctx2, _refs(xh2))
    cand_idx = np.arange(NBASE, NCOL)
    md = float(np.max(np.abs(r1[cand_idx] - r2[cand_idx]))) if len(cand_idx) else 0.0
    nz = int(np.sum(np.abs(r1[cand_idx]) > 1e-9))
    log(f"[affine-check] cand cols maxdiff under x->{a}x+{b}: {md:.2e}  (nonzero cand cols {nz}/{len(cand_idx)})")


def main():
    open(LOG, "w").close()
    log(f"=== RESIDUAL-SHAPE probe on ranker+rf E75 base79  {datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF}")
    log(f"NBASE={NBASE} NCOL={NCOL} ship_scales={SHIP_SCALES}  cand_scales={_ALL_RSHAPE_SCALES}")
    log(f"arms={[(n, k, s) for n, k, s in RSHAPE_ARMS]}")
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
        f.write(f"# residual-shape probe on ranker+rf E75 base79  {datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF}\n")
        f.write(f"# base79 = E75 champion; arms = prewhitened AR8-residual shape contrast vs history\n")
        f.write(f"# rkurt@130,250  rqkurt@60,130,250  rhill@250 ; ISOLATION vs base79\n")
        f.write(f"# RELAXED gate: any mean_d>0 is a candidate; rank by mean_d. BELIEVE only if robust arm moves.\n\n")
        f.write(f"{'candidate':<14}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for n, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md > 0 and npos == tot) else ""
            f.write(f"{n:<14}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
