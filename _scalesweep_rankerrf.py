"""SCALE / WINDOW re-sweep on the CURRENT ranker+rf 73-col SHIP base (2026-08-25).

Motivation: E16 (window) and E19 (scale-set) found WINDOW=130 / scales (60,130,250)
optimal, but those sweeps were run on the OLD base15 *HGB single model*. The shipped
base is completely different (73 feats = champ_base 51 + dep 10 + ship 12, ranker+rf
blend). So re-test whether an EXTRA trailing scale adds signal on the new base.

The core (60,130,250) triple is hardcoded across champion_feat / residual / dep, so a
full global re-spacing is a refactor. Instead this probes ADDITIVELY: keep the shipped
73-col base UNCHANGED and append the multiscale families (champion base15 + the 3 ship
families perm_entropy4/max_split_lr/mosum) recomputed at one EXTRA trailing scale s, then
paired-delta the ranker+rf blend (base73 vs base73 + extra-scale block). All probe scales
< WINDOW=250 (longer-than-250 needs a larger buffer -> out of scope here).

Two variants per probe scale:
  add{s}_ship  = base73 + [perm4, maxlr, mosum] at s        (+4 cols)
  add{s}_full  = base73 + [champ15 + perm4, maxlr, mosum]   (+19 cols)

Reuses the proven data/ranker/rf/TS-AUC machinery from _night_rankerrf.py verbatim; only
the row builder and candidate registry differ. Screen recipe (rf 500, dense_k 40),
seeds [42,7,123], relaxed gate (rank by mean delta). Separate feature cache
(_scalesweep_cache) since the column layout differs from the night screen.

Run:
  SMOKE=1 .venv/bin/python -u _scalesweep_rankerrf.py
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _scalesweep_rankerrf.py \
      > _scalesweep_rankerrf.out 2>&1 & disown
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
from _ensemble_tune import WINDOW, _subwin, load_pool_strat, select_picks, BASE_KEYS
from _feat_lib import _feat
from _exp_fast import HIST_CAP, hist_context, split_series
from _overnight import MIN_N, N_JOBS
from _residual_overnight import _active, pct
from _night_banks import (NCHAMP_BASE, NDEP, fit_champ_refs, champ_base_row,
                          fit_dep_refs, dep_feats)
from _lambdarank import sort_into_groups, fit_ranker

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]

RF_TREES = 120 if SMOKE else 500
DENSE_K = 20 if SMOKE else 40
W_RANK, W_RF = 0.45, 0.55
RF_KW = dict(max_features=0.5, min_samples_leaf=20)

SHIP_SCALES = (60, 130, 250)                    # scales already in the shipped ship block
PROBE_SCALES = [90, 160] if SMOKE else [40, 90, 160, 190]   # extra trailing scales to test

# shipped dep block keeps 10 of _night_banks.dep_feats' 14 cols (drops the ACF-lag
# block dacf1/2/5/10 at indices 6..9) -> b_no_accf, matching the ship notebook exactly.
_DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_scalesweep_cache")
RAW = os.path.join(HERE, "_scalesweep_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_scalesweep_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_scalesweep_rankerrf.log")

# --- column layout ---------------------------------------------------------
NBASE = NCHAMP_BASE + len(_DEP_KEEP) + 3 * 4    # champ 51 + dep 10 + ship 12 = 73
CHAMP15 = 15                                    # champion_feat width per scale
SHIP4 = 4                                       # perm4(1)+maxlr(1)+mosum(2)
PROBE_W = CHAMP15 + SHIP4                        # 19 cols per probe scale
NCOL = NBASE + PROBE_W * len(PROBE_SCALES)
BASE_COLS = np.arange(NBASE)

# per-probe-scale column blocks (champ15 first, then ship4) within the probe region
PROBE_SLICE = {}
for i, s in enumerate(PROBE_SCALES):
    start = NBASE + i * PROBE_W
    PROBE_SLICE[s] = dict(champ=(start, start + CHAMP15),
                          ship=(start + CHAMP15, start + PROBE_W))


def cand_cols(scale, variant):
    """base73 + one probe scale's block (ship-only or champ15+ship)."""
    sl = PROBE_SLICE[scale]
    if variant == "ship":
        lo, hi = sl["ship"]
    else:  # full
        lo, hi = sl["champ"][0], sl["ship"][1]
    return np.concatenate([BASE_COLS, np.arange(lo, hi)])


CANDIDATES = [(f"add{s}_ship", (s, "ship")) for s in PROBE_SCALES] + \
             [(f"add{s}_full", (s, "full")) for s in PROBE_SCALES]


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
def _row(w, ctx, refs):
    w = np.asarray(w, dtype=np.float64)
    dep_full = dep_feats(w, ctx, refs["dep"])
    out = [np.asarray(champ_base_row(w, ctx, refs["champ"]), dtype=np.float64),
           np.asarray([dep_full[i] for i in _DEP_KEEP], dtype=np.float64)]
    # shipped ship block at (60,130,250)
    ship = []
    for s in SHIP_SCALES:
        ship.extend(_ship_block(_subwin(w, s), ctx))
    out.append(np.asarray(ship, dtype=np.float64))
    # probe blocks: champ15 + ship4 at each extra scale
    probe = []
    for s in PROBE_SCALES:
        ws = _subwin(w, s)
        probe.extend(_feat(ws, ctx, keys=BASE_KEYS))
        probe.extend(_ship_block(ws, ctx))
    out.append(np.asarray(probe, dtype=np.float64))
    return np.concatenate(out)


def _refs(x_h):
    return {"champ": fit_champ_refs(x_h), "dep": fit_dep_refs(x_h)}


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


def main():
    open(LOG, "w").close()
    log(f"=== SCALE re-sweep on ranker+rf ship base  {datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF}")
    log(f"NBASE={NBASE} NCOL={NCOL} ship_scales={SHIP_SCALES} probe_scales={PROBE_SCALES}")
    log(f"candidates={[n for n, _ in CANDIDATES]}")

    if not os.path.exists(RAW):
        with open(RAW, "w") as f:
            f.write("seed\tcandidate\tbase_auc\tcand_auc\tdelta\n")

    per_seed = {}
    for seed in SEARCH_SEEDS:
        sd = prepare_seed(seed)
        metas = make_folds_meta(sd)
        t0 = time.time()
        base_auc = score_cols(sd, metas, BASE_COLS, seed)
        log(f"[seed {seed}] base73 ranker+rf = {base_auc:.4f}  ({time.time()-t0:.0f}s)")
        res = {}
        for name, (scale, variant) in CANDIDATES:
            tc = time.time()
            cols = cand_cols(scale, variant)
            auc = score_cols(sd, metas, cols, seed)
            d = auc - base_auc
            res[name] = d
            with open(RAW, "a") as f:
                f.write(f"{seed}\t{name}\t{base_auc:.6f}\t{auc:.6f}\t{d:+.6f}\n")
            log(f"[seed {seed}] {name:<16} {auc:.4f}  d={d:+.4f}  ({time.time()-tc:.0f}s)")
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
        f.write(f"# scale re-sweep on ranker+rf ship base  {datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF}\n")
        f.write(f"# base73 = champ51 + dep10 + ship12 @ {SHIP_SCALES}; probe adds one extra scale\n")
        f.write(f"# RELAXED gate: any mean_d>0 is a candidate; rank by mean_d\n\n")
        f.write(f"{'candidate':<18}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for n, md, mn, mx, npos, k in agg:
            flag = "  <== WIN" if md > 0 and npos == k else ("  ~" if md > 0 else "")
            f.write(f"{n:<18}{md:>+10.5f}{mn:>+10.5f}{mx:>+10.5f}{npos:>4}/{k}{flag}\n")
    log(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
