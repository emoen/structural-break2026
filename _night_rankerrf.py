"""OVERNIGHT (2026-08-23): re-screen the WHOLE feature battery on the NEW ranker+rf blend.

Why: the deployed 4-member stack dropped hgb to weight 0; the gain is carried by ranker+rf
(cached-OOF sweep: ranker~.45/rf~.55 = +0.0016 vs champ, 7/7 seeds, == the full stack). Every
feature was previously screened on the pointwise HGB champion and rejected. A different base
model (listwise ranker + bagged forest) may accept features HGB could not. So re-screen the
entire _feat_lib.EXTRA/COMBOS battery PLUS the ADIA-2025 4th-place NEW families (_screen_feats:
coefficient-of-variation "magic feature", Lempel-Ziv complexity, MOSUM local variance, SSA
top-mode share) on the ranker+rf blend.

RELAXED GATE (user directive "all marginal wins we take, no t>2"): rank candidates by mean
paired delta across the search seeds; flag ANY with mean_d > 0 (and note per-seed positivity).
Winners get a full-recipe fresh-seed confirm in a follow-up (not here).

Design (leak-free, incremental):
  * Per seed: build the 61-col champion base (champ_base_row 51 + dep_feats 10) PLUS the union of
    every candidate's extra columns, over the SAME rows (pointwise train rows + dense ranker rows
    + val fold caches). Cache to _screen_cache/seed_{seed}.pkl (train/val series disjoint from
    load_pool_strat -> no OOF needed for a fixed-weight blend).
  * Per candidate: select base cols + that candidate's extra-column block, refit ranker + rf,
    fixed train-CDF percentile blend (W_RANK*pct_rank + W_RF*pct_rf), weighted TS-AUC on the val
    folds, paired delta vs the base ranker+rf (no extra cols) on the same folds.
  * Screen on SEARCH_SEEDS only (3). Reduced screen recipe (rf 500 trees, dense_k 40) for speed;
    winners re-confirmed at full recipe. Incremental TSV survives a crash.

  SMOKE=1 .venv/bin/python -u _night_rankerrf.py     # tiny plumbing/correctness pass
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _night_rankerrf.py > _night_rankerrf.log 2>&1 &
"""
import os
import time
import pickle
import warnings
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestClassifier

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, SCALES, _subwin, load_pool_strat, select_picks
from _exp_fast import HIST_CAP, hist_context, split_series
from _overnight import MIN_N, N_JOBS
from _residual_overnight import _active, pct
from _night_banks import (NCHAMP_BASE, NDEP, fit_champ_refs, champ_base_row,
                          fit_dep_refs, dep_feats)
from _lambdarank import sort_into_groups, fit_ranker
from _feat_lib import EXTRA, COMBOS
import _screen_feats as SF

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]

NBASE = NCHAMP_BASE + NDEP                     # 61 champion-stack columns
RF_TREES = 120 if SMOKE else 500
DENSE_K = 20 if SMOKE else 40
W_RANK, W_RF = 0.45, 0.55                      # fixed blend (matches the cached-OOF optimum plateau)
RF_KW = dict(max_features=0.5, min_samples_leaf=20)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_screen_cache")
RAW = os.path.join(HERE, "_night_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_night_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_night_rankerrf.log")

# ---------------------------------------------------------------------------
# candidate registry: NEW families first, then the full EXTRA battery + COMBOS
# ---------------------------------------------------------------------------
BASE_IN = ("spec_entropy",)                     # already in the champion base -> not a candidate
NEW_KEYS = ["cov", "magic", "lz", "mosum", "ssa"]
EXTRA_KEYS = sorted(k for k in EXTRA if k not in BASE_IN)
ALLREG = {**{k: SF.REG[k] for k in NEW_KEYS}, **{k: EXTRA[k] for k in EXTRA_KEYS}}
UNION_ORDER = NEW_KEYS + EXTRA_KEYS             # column layout order (grouped by key, 3 scales each)

if SMOKE:
    EXTRA_KEYS = EXTRA_KEYS[:3]
    UNION_ORDER = NEW_KEYS + EXTRA_KEYS
    ALLREG = {k: ALLREG[k] for k in UNION_ORDER}


def _widths():
    rng = np.random.default_rng(0)
    x_h = np.cumsum(rng.standard_normal(600)) * 0.1 + rng.standard_normal(600)
    ctx = hist_context(x_h, HIST_CAP)
    win = x_h[-WINDOW:]
    w = {}
    for k in UNION_ORDER:
        w[k] = len(ALLREG[k](win, ctx))
    return w


WIDTHS = _widths()
# contiguous per-key blocks in the extra matrix: block_k = 3 scales * width_k
KEY_SLICE = {}
off = 0
for k in UNION_ORDER:
    bw = 3 * WIDTHS[k]
    KEY_SLICE[k] = (off, off + bw)
    off += bw
NEXTRA = off
NCOL = NBASE + NEXTRA
BASE_COLS = np.arange(NBASE)


def cand_cols(keys):
    """Base 61 cols + the extra-column block(s) for the given candidate key(s).
    Keys already resident in the champion base (e.g. spec_entropy) carry no extra
    block and are simply skipped."""
    extra = []
    for k in keys:
        if k not in KEY_SLICE:
            continue
        lo, hi = KEY_SLICE[k]
        extra.extend(range(NBASE + lo, NBASE + hi))
    return np.concatenate([BASE_COLS, np.asarray(extra, dtype=int)]) if extra else BASE_COLS


# candidate list: (name, keys-tuple)
CANDIDATES = [(k, (k,)) for k in NEW_KEYS]
CANDIDATES.append(("new_all", tuple(NEW_KEYS)))
CANDIDATES += [(k, (k,)) for k in EXTRA_KEYS]
CANDIDATES += [(name, COMBOS[name]) for name in sorted(COMBOS)]
if SMOKE:
    CANDIDATES = CANDIDATES[:6]


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# feature row build (champ base + dep + union-extra), mirrors _night_stack
# ---------------------------------------------------------------------------
def _row_screen(w, ctx, refs):
    out = [np.asarray(champ_base_row(w, ctx, refs["champ"]), dtype=np.float64),
           np.asarray(dep_feats(w, ctx, refs["dep"]), dtype=np.float64)]
    extra = []
    for k in UNION_ORDER:
        fn = ALLREG[k]
        for s in SCALES:
            extra.extend(fn(_subwin(w, s), ctx))
    out.append(np.asarray(extra, dtype=np.float64))
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
        rows.append(_row_screen(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs))
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return _san(rows), labels, tvals, [int(sid)] * len(labels)


def _val_series(sid, x_h, x_on, oi):
    ctx = hist_context(x_h, HIST_CAP)
    refs = _refs(x_h)
    active = _active(x_on)
    rows = [_row_screen(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs) for t in active]
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


# ---------------------------------------------------------------------------
# scoring: fixed-weight ranker+rf blend, weighted TS-AUC over val folds
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def main():
    open(LOG, "w").close()
    log(f"=== NIGHT ranker+rf feature re-screen  {datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF}")
    log(f"NBASE={NBASE} NEXTRA={NEXTRA} NCOL={NCOL} candidates={len(CANDIDATES)}")
    log(f"widths={WIDTHS}")

    if not os.path.exists(RAW):
        with open(RAW, "w") as f:
            f.write("seed\tcandidate\tbase_auc\tcand_auc\tdelta\n")

    per_seed = {}
    for seed in SEARCH_SEEDS:
        sd = prepare_seed(seed)
        metas = make_folds_meta(sd)
        t0 = time.time()
        base_auc = score_cols(sd, metas, BASE_COLS, seed)
        log(f"[seed {seed}] base ranker+rf = {base_auc:.4f}  ({time.time()-t0:.0f}s)")
        res = {}
        for name, keys in CANDIDATES:
            tc = time.time()
            cols = cand_cols(keys)
            auc = score_cols(sd, metas, cols, seed)
            d = auc - base_auc
            res[name] = d
            with open(RAW, "a") as f:
                f.write(f"{seed}\t{name}\t{base_auc:.6f}\t{auc:.6f}\t{d:+.6f}\n")
            log(f"[seed {seed}] {name:<22} {auc:.4f}  d={d:+.4f}  ({time.time()-tc:.0f}s)")
        per_seed[seed] = res

    # ---- aggregate ----
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
        f.write(f"# ranker+rf feature re-screen  {datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF}\n")
        f.write(f"# RELAXED gate: any mean_d>0 is a candidate; rank by mean_d\n\n")
        f.write(f"{'candidate':<24}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for n, md, mn, mx, npos, k in agg:
            flag = "  <== WIN" if md > 0 and npos == k else ("  ~" if md > 0 else "")
            f.write(f"{n:<24}{md:>+10.5f}{mn:>+10.5f}{mx:>+10.5f}{npos:>4}/{k}{flag}\n")
    log(f"\n[done] wrote {SUMMARY}")
    log("TOP 15:")
    for n, md, mn, mx, npos, k in agg[:15]:
        log(f"  {n:<22} mean_d={md:+.5f} min={mn:+.5f} npos={npos}/{k}")


if __name__ == "__main__":
    main()
