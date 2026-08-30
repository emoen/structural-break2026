"""METRIC-WEIGHTED LOSS probe on the CURRENT (E78) ranker+rf base79 champion.

Origin: 2026-08-28 2-agent ideation #3 (proposer idea #2, critic KEEP #1 / build-first). The
training loss is FLAT over rows while the leaderboard metric (TS-AUC) is a per-`time_online`
weighted cross-sectional rank-AUC: at absolute step t it weights the within-step ordering by the
pair count n_pos(t)*n_neg(t). That is a train<->metric mismatch nobody has closed from the loss
side with the ACTUAL metric weight (E26/E72 tried hand-crafted EARLY/hazard ramps on offset and
they hurt; this is different).

MECHANISM (no new features -- base79 is byte-identical to the E78 champion): importance-weight
each training row by its exact metric pairing count at its own step t:
    positive row at step t  ->  weight = n_neg(t)   (# negatives it is ranked against)
    negative row at step t  ->  weight = n_pos(t)   (# positives it is ranked against)
This is the correct AUC-surrogate importance weight; it is the empirical, HUMPED w(t) the critic
insisted on (n_pos*n_neg peaks mid-stream and falls at both ends) -- NOT the monotone late-ramp the
proposer assumed. Applied to BOTH members: RandomForest (pointwise sample_weight) and the
LGBMRanker rank_xendcg (within-group sample_weight; each group == one step t, so positives carry
n_neg(t) and negatives n_pos(t)).

ARMS = emphasis exponent gamma on the (mean-normalized) metric weight w(t):
    sw = (w_metric(t)/mean)**gamma , renormalized to mean 1.  gamma=0 == uniform == the champion.
Sweep gamma in {0.5, 1.0, 1.5}. Baseline (gamma=0) reproduces the champion exactly (all weights 1).

BASE (always in, = E78 champion 79, byte-identical build to _rshape_rankerrf, SHIP_SCALES=40,100,250):
  champ51 (AR8 full + AR8 trail120) + dep10 + ship12 + [AR8 trail60, AR8 trail200] x 3 scales.

No new columns => no affine-invariance concern (inherited from the shipped base79). This probe
ONLY changes fit-time sample weights; scoring/blend/CDF are identical to the champion.

FAILURE WATCH (critic): at late offset the signal is near the info floor, so up-weighting can teach
the model to fit noise where there's little to learn and degrade the well-separated rows -> net
wash / small loss (the generic reason E26/E72 target-shaping died). BELIEVE-RULE before an LB slot:
a gamma must clear mean_d >= 0.0015 AND be positive on all screen seeds, then repeat on the fresh
confirm seeds. Given the 0-for-N marginal reversal history, prior ~20%.

Shares the base79 feature cache dir `_struct79_cache` with _twohead_rankerrf.py (identical rows).
RUN ONE AT A TIME (memory). Run:
  SMOKE=1 .venv/bin/python -u _wloss_rankerrf.py           # tiny self-test
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _wloss_rankerrf.py \
      > _wloss_rankerrf.out 2>&1 & disown
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

import lightgbm as lgb
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
from _lambdarank import sort_into_groups, RankerProba, LGB_COMMON

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]

RF_TREES = 120 if SMOKE else 500
DENSE_K = 20 if SMOKE else 40
W_RANK, W_RF = 0.45, 0.55
RF_KW = dict(max_features=0.5, min_samples_leaf=20)

# E78 champion window triple (MS_SCALES / RESIDUAL_SCALES). Override => delete _struct79_cache.
SHIP_SCALES = tuple(int(s) for s in os.environ.get("SHIP_SCALES", "40,100,250").split(","))

# shipped dep block keeps 10 of dep_feats' 14 cols (drops ACF-lag dacf1/2/5/10) -> b_no_accf.
_DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]

# emphasis exponents on the metric weight w(t); gamma=0 (uniform) is the baseline.
GAMMAS = [0.5, 1.0] if SMOKE else [0.5, 1.0, 1.5]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE_DIR = os.path.join(HERE, "_struct79_cache")   # SHARED with _twohead_rankerrf.py
RAW = os.path.join(HERE, "_wloss_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_wloss_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_wloss_rankerrf.log")

AR_P = 8  # same order as the shipped residual lens

BASE_SPAN_SPECS = [
    ("ar8_t60",  dict(p=8, trail=60)),
    ("ar8_t200", dict(p=8, trail=200)),
]

SPAN_W = len(SHIP_SCALES)
NBASE = (NCHAMP_BASE + len(_DEP_KEEP) + 4 * len(SHIP_SCALES)
         + len(BASE_SPAN_SPECS) * SPAN_W)          # champ51 + dep10 + ship12 + base-spans6 = 79
NCOL = NBASE


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# --- metric weight profile w(t): n_pos(t), n_neg(t) over absolute online step t ---
def _metric_weight_profile():
    Xp = pd.read_parquet(os.path.join(DATA, "X_train.parquet"), columns=["period"])
    yi_all = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
    period = Xp["period"].to_numpy(np.int64)
    ids = Xp.index.get_level_values("id").to_numpy()
    uniq, start = np.unique(ids, return_index=True)
    ends = np.append(start[1:], len(ids))
    L = {int(u): int((period[s:e] == 2).sum()) for u, s, e in zip(uniq, start, ends)}
    TAU = {int(k): int(v) for k, v in yi_all["tau_index"].items()}
    series = [s for s in L if L[s] > 0]
    maxL = max(L[s] for s in series)
    n_pos = np.zeros(maxL, dtype=np.float64)
    n_alive = np.zeros(maxL, dtype=np.float64)
    for s in series:
        Ls, tau = L[s], TAU.get(s, -1)
        n_alive[:Ls] += 1.0
        if 0 <= tau < Ls:
            n_pos[tau:Ls] += 1.0
    n_neg = n_alive - n_pos
    return n_pos, n_neg, maxL


NPOS_T, NNEG_T, MAXL = _metric_weight_profile()


def scheme_weight(tv, lab, gamma):
    """Per-row sample weight: metric pairing count at step t, ^gamma, mean-normalized to 1.

    gamma=0 -> all ones (== champion). Positives weighted by n_neg(t), negatives by n_pos(t)."""
    if gamma == 0.0:
        return np.ones(len(tv), dtype=np.float64)
    t = np.clip(np.asarray(tv, dtype=np.int64), 0, MAXL - 1)
    wb = np.where(np.asarray(lab) == 1, NNEG_T[t], NPOS_T[t]).astype(np.float64)
    wb = wb / max(wb.mean(), 1e-12)                 # shape only, mean 1
    w = np.power(np.maximum(wb, 1e-9), gamma)
    w *= len(w) / w.sum()                            # renormalize to mean 1
    return w


def sort_into_groups_w(rows, labels, tvals, sw):
    """sort_into_groups that also carries the sample weights through the same permutation."""
    order = np.argsort(tvals, kind="stable")
    rows, labels, tvals = rows[order], labels[order], tvals[order]
    swo = None if sw is None else np.asarray(sw)[order]
    _, counts = np.unique(tvals, return_counts=True)
    return rows, labels, counts, swo


def fit_ranker_w(objective, rows, labels, groups, sw=None):
    r = lgb.LGBMRanker(objective=objective, **LGB_COMMON)
    with threadpool_limits(limits=1):
        r.fit(rows, labels, group=groups, sample_weight=sw)
    return RankerProba(r)


# --- AR8 base-span fit / residual-var-ratio feats (verbatim from _rshape) ----
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


# --- feature row build (base79, byte-identical to the E78 champion) --------
def _refs(x_h):
    return {"champ": fit_champ_refs(x_h), "dep": fit_dep_refs(x_h),
            "base_span": {name: _fit_span(x_h, spec) for name, spec in BASE_SPAN_SPECS}}


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


def _offsets(tv, sids, tau_map):
    tau = np.array([tau_map.get(int(s), -1) for s in sids], dtype=np.int64)
    off = np.asarray(tv, dtype=np.int64) - tau
    return np.where((tau < 0) | (off < 0), -1, off).astype(np.int64)


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
    tau_map = {int(s): int(yi.loc[s, "tau_index"]) for s in train_ids}
    off_tr = _offsets(Ttr, Str, tau_map)
    off_r = _offsets(Rt, Rs, tau_map)
    caches = [build_val(X, f) for f in folds]
    sd = dict(Xtr=Xtr, ytr=ytr, Ttr=Ttr, off_tr=off_tr,
              Rr=Rr, Rl=Rl, Rt=Rt, off_r=off_r, caches=caches, y_val=y_val)
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


def score_gamma(sd, metas, gamma, seed):
    Xtr, ytr = sd["Xtr"], sd["ytr"]
    sw_rf = scheme_weight(sd["Ttr"], ytr, gamma)
    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=N_JOBS, random_state=seed, **RF_KW)
    rf.fit(Xtr, ytr, sample_weight=sw_rf)
    ref_rf = np.sort(_proba(rf, Xtr))

    sw_rk = scheme_weight(sd["Rt"], sd["Rl"], gamma)
    rows, lab, grp, swo = sort_into_groups_w(sd["Rr"], sd["Rl"], sd["Rt"], sw_rk)
    rk = fit_ranker_w("rank_xendcg", rows, lab, grp, swo)
    ref_rk = np.sort(_proba(rk, sd["Rr"]))

    aucs = []
    for (L, active, groups), c in zip(metas, sd["caches"]):
        srf = np.zeros(L)
        srk = np.zeros(L)
        if len(c["feat"]):
            srf[active] = _proba(rf, c["feat"])
            srk[active] = _proba(rk, c["feat"])
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


def main():
    open(LOG, "w").close()
    log(f"=== METRIC-WEIGHTED-LOSS probe on ranker+rf E78 base79  {datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} ship_scales={SHIP_SCALES}")
    log(f"NBASE={NBASE} NCOL={NCOL}  metric-weight w(t): maxL={MAXL} "
        f"n_pos[peak]={NPOS_T.max():.0f} n_neg[0]={NNEG_T[0]:.0f}  gammas={GAMMAS}")

    if not os.path.exists(RAW):
        with open(RAW, "w") as f:
            f.write("seed\tgamma\tbase_auc\tcand_auc\tdelta\n")

    per_seed = {}
    for seed in SEARCH_SEEDS:
        sd = prepare_seed(seed)
        metas = make_folds_meta(sd)
        t0 = time.time()
        base_auc = score_gamma(sd, metas, 0.0, seed)
        log(f"[seed {seed}] base79 (gamma=0) ranker+rf = {base_auc:.4f}  ({time.time()-t0:.0f}s)")
        res = {}
        for g in GAMMAS:
            tc = time.time()
            auc = score_gamma(sd, metas, g, seed)
            d = auc - base_auc
            res[g] = d
            with open(RAW, "a") as f:
                f.write(f"{seed}\t{g}\t{base_auc:.6f}\t{auc:.6f}\t{d:+.6f}\n")
            log(f"[seed {seed}] gamma={g:<4} {auc:.4f}  d={d:+.4f}  ({time.time()-tc:.0f}s)")
        per_seed[seed] = res

    agg = []
    for g in GAMMAS:
        ds = [per_seed[s][g] for s in SEARCH_SEEDS if g in per_seed[s]]
        if not ds:
            continue
        md = float(np.mean(ds))
        npos = sum(1 for d in ds if d > 0)
        agg.append((g, md, min(ds), max(ds), npos, len(ds)))
    agg.sort(key=lambda r: -r[1])
    with open(SUMMARY, "w") as f:
        f.write(f"# metric-weighted-loss probe on ranker+rf E78 base79  {datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} ship={SHIP_SCALES}\n")
        f.write(f"# sample_weight = (metric pairing count at step t)^gamma, mean-normalized; gamma=0 == champion.\n")
        f.write(f"# pos row weighted by n_neg(t), neg row by n_pos(t) -> the humped empirical w(t).\n")
        f.write(f"# BELIEVE only if mean_d>=+0.0015 AND all-seeds-positive, then confirm on fresh seeds.\n\n")
        f.write(f"{'gamma':<10}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for g, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md > 0 and npos == tot) else ""
            f.write(f"{g:<10}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
