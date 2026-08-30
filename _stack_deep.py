"""Deep, principled stacking search over the 5-member pool (hgb + ranker + rf + xgb + catboost).

Why this exists: the first pass (_stack_zoo.py) only scored 7 hand-named full-pool combos and let
Caruana's *integer* greedy pick odd weights (e.g. hgb=1/14~=0.07). This replaces that with an
exhaustive, sensible weight search that costs almost nothing once the member fits are cached.

Two stages, one launch (nohup-friendly, no approvals):

  STAGE 1 -- CACHE (expensive, ~20 min/seed, done once).
    For every seed reuse the *exact* leak-free member pipeline from _stack_zoo.member_preds:
    OOF meta-features Foof (series K-fold, ranker OOF-fit on dense rows of non-held series) +
    per-val-fold percentile matrices Vmat. Pickle to _stack_cache/seed_{seed}.pkl so re-runs skip
    all model fitting.

  STAGE 2 -- SEARCH (cheap, seconds; a blend eval is a matmul + a rank-AUC).
    * Exhaustive 5-simplex weight grid at step 0.05 (10626 blends). This grid *contains* equal
      weights (0.2 each), the deployed champ (hgb=0.40, ranker=0.60), every subset mean
      (mean5/4/3/2), trees-only (ranker=0), and the whole "ranker 0.3-0.4 / hgb 0.3-0.4 / others
      0.1-0.4" region -- so greedy/mean-K/trees-K are all just points in it.
    * Per-subset learned metas: logistic + LightGBM stackers on every subset of size >= 2.
    Selection is HONEST: weights/metas are chosen on the SEARCH seeds only, then scored on the
    held-out FRESH seeds (and ALL) to report an out-of-sample delta vs champ.

The fast val scorer reproduces _exp_fast.ts_auc_from bit-for-bit (groupby id cumcount -> time_online,
weighted n_pos*n_neg rank-AUC with tie-averaged ranks == roc_auc_score).

Env STACK_SMOKE=1 -> tiny pool / 2 seeds end-to-end smoke.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

STACK_SMOKE = os.environ.get("STACK_SMOKE") == "1"
if STACK_SMOKE:
    os.environ["SMOKE"] = "1"

import time
import json
import pickle
import warnings
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# reuse the *exact* leak-free member pipeline + constants from the first pass
from _stack_zoo import (member_preds, NAMES, IDX, W_CHAMP, SEEDS, FRESH, COLS, MEMBERS)
from _overnight import N_JOBS, paired

M = len(MEMBERS)
CACHE_DIR = "_stack_cache"
os.makedirs("experiments/json", exist_ok=True)
LOG = "experiments/_stack_deep.log"
STEP = 0.05                                   # weight-grid resolution (full 5-simplex)
TOPK = 40                                     # weight blends carried from search -> fresh/all
ALL_SEEDS = SEEDS + FRESH


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# STAGE 1 -- cache member predictions (fit once, reuse forever)
# ---------------------------------------------------------------------------
def get_seed_cache(seed):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"seed_{seed}{'_smoke' if STACK_SMOKE else ''}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    sd = member_preds(seed)
    with open(path, "wb") as f:
        pickle.dump(sd, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  [cache] seed {seed} fit+dumped in {time.time()-t0:.0f}s -> {path}")
    return sd


# ---------------------------------------------------------------------------
# STAGE 2 -- fast exact val TS-AUC scorer
# ---------------------------------------------------------------------------
def _auc(lab, sc):
    """roc_auc_score via tie-averaged ranks (Mann-Whitney U)."""
    r = rankdata(sc)                          # method='average' -> matches sklearn on ties
    npos = float(lab.sum())
    nneg = float(len(lab) - npos)
    return (r[lab == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def build_folds(sd):
    """Per fold: (mat[L,M], groups=[(rows, labels, weight)]) mirroring ts_auc_from grouping."""
    folds = []
    for idx, mat in zip(sd["Vidx"], sd["Vmat"]):
        df = pd.Series(np.arange(len(idx)), index=idx).to_frame("row")
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
        folds.append((np.asarray(mat, dtype=np.float64), groups))
    return folds


def score_folds(folds, get_s):
    """Weighted TS-AUC over a seed's folds; get_s(mat)->per-row score. Returns mean-over-folds."""
    aucs = []
    for mat, groups in folds:
        s = get_s(mat)
        wsum = tot = 0.0
        for rows, lab, w in groups:
            wsum += w * _auc(lab, s[rows])
            tot += w
        aucs.append(wsum / tot if tot > 0 else 0.5)
    return float(np.mean(aucs))


def lin(w):
    return lambda mat: mat @ w


def score_grid(folds, Wmat, chunk=1500):
    """Vectorized TS-AUC for MANY linear blends at once.

    Wmat: (M, B) weight columns. Returns (B,) mean-over-folds TS-AUC. Uses stable-argsort ranks
    (no tie averaging) for speed; the handful of selected winners are re-scored exactly afterwards.
    """
    B = Wmat.shape[1]
    out = np.zeros(B)
    for lo in range(0, B, chunk):
        W = Wmat[:, lo:lo + chunk]
        b = W.shape[1]
        acc = np.zeros((len(folds), b))
        for fi, (mat, groups) in enumerate(folds):
            S = mat @ W                                    # L x b
            wsum = np.zeros(b)
            tot = 0.0
            for rows, lab, w in groups:
                sub = S[rows, :]                           # g x b
                g = sub.shape[0]
                order = np.argsort(sub, axis=0, kind="stable")
                ranks = np.empty((g, b))
                np.put_along_axis(ranks, order,
                                  np.broadcast_to(np.arange(1, g + 1)[:, None], (g, b)), axis=0)
                npos = float(lab.sum())
                nneg = float(g - npos)
                auc = (ranks[lab == 1, :].sum(axis=0) - npos * (npos + 1) / 2.0) / (npos * nneg)
                wsum += w * auc
                tot += w
            acc[fi] = wsum / tot if tot > 0 else 0.5
        out[lo:lo + chunk] = acc.mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# weight-grid helpers
# ---------------------------------------------------------------------------
def simplex(m, q):
    """All integer m-tuples summing to q (stars & bars)."""
    if m == 1:
        yield (q,)
        return
    for i in range(q + 1):
        for tail in simplex(m - 1, q - i):
            yield (i,) + tail


def wname(w):
    return " ".join(f"{n}={v:.2f}" for n, v in zip(NAMES, w) if v > 1e-9)


# ---------------------------------------------------------------------------
# learned metas (per subset)
# ---------------------------------------------------------------------------
def fit_logit(Foof, y, cols, seed):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(C=1.0, max_iter=3000, random_state=seed))
    m.fit(Foof[:, cols], y)
    return lambda mat: m.predict_proba(mat[:, cols])[:, 1]


def fit_gbm(Foof, y, cols, seed):
    m = lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=0.05,
                           num_leaves=15, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                           n_jobs=N_JOBS, verbose=-1)
    m.fit(Foof[:, cols], y)
    return lambda mat: m.predict_proba(mat[:, cols])[:, 1]


# ---------------------------------------------------------------------------
def main():
    open(LOG, "w").close()
    log(f"=== STACK DEEP  {datetime.now():%F %T}  smoke={STACK_SMOKE} ===")
    log(f"members={NAMES}  search={SEEDS}  fresh={FRESH}  step={STEP}")

    # ---- stage 1: caches + fold structures ----
    seed_folds, seed_meta = {}, {}
    for seed in ALL_SEEDS:
        sd = get_seed_cache(seed)
        seed_folds[seed] = build_folds(sd)
        seed_meta[seed] = (sd["Foof"], sd["ytr"])
    log(f"[cache ready] {len(ALL_SEEDS)} seeds")

    def mean_over(seeds, get_s_by_seed):
        return {s: score_folds(seed_folds[s], get_s_by_seed(s)) for s in seeds}

    # champ reference (in-grid point: hgb=0.40, ranker=0.60)
    wc = np.zeros(M); wc[IDX["hgb"]] = W_CHAMP; wc[IDX["ranker"]] = 1 - W_CHAMP
    champ = {s: score_folds(seed_folds[s], lin(wc)) for s in ALL_SEEDS}
    log(f"[sanity] champ ALL mean={np.mean([champ[s] for s in ALL_SEEDS]):.4f} "
        f"(expect ~0.6236 from _stack_zoo)")

    # ---- stage 2a: exhaustive weight-simplex sweep, SELECT on search seeds ----
    q = int(round(1 / STEP))
    grid = [np.asarray(t, dtype=np.float64) / q for t in simplex(M, q)]
    Wmat = np.asarray(grid).T                             # M x B
    log(f"\n-- weight sweep: {len(grid)} blends, select on SEARCH seeds --")
    t0 = time.time()
    search_scores = np.mean([score_grid(seed_folds[s], Wmat) for s in SEEDS], axis=0)
    order = np.argsort(-search_scores)
    log(f"   swept {len(grid)} blends in {time.time()-t0:.0f}s")

    # evaluate the top-K search blends everywhere
    def rowstats(get_s_by_seed):
        sm = np.mean([score_folds(seed_folds[s], get_s_by_seed(s)) for s in SEEDS])
        fm = np.mean([score_folds(seed_folds[s], get_s_by_seed(s)) for s in FRESH]) if FRESH else float("nan")
        am = np.mean([score_folds(seed_folds[s], get_s_by_seed(s)) for s in ALL_SEEDS])
        npos = sum(1 for s in ALL_SEEDS
                   if score_folds(seed_folds[s], get_s_by_seed(s)) > champ[s])
        return sm, fm, am, npos

    log("\n-- TOP weight blends (selected on SEARCH, honest FRESH/ALL) --")
    log(f"   {'search':>7} {'fresh':>7} {'all':>7} {'d_all':>7} {'np/7':>5}  weights")
    champ_all = np.mean([champ[s] for s in ALL_SEEDS])
    log(f"   {np.mean([champ[s] for s in SEEDS]):7.4f} "
        f"{np.mean([champ[s] for s in FRESH]) if FRESH else float('nan'):7.4f} "
        f"{champ_all:7.4f} {0.0:+7.4f} {'-':>5}  CHAMP [{wname(wc)}]")
    best_fresh = None
    for j in order[:TOPK]:
        w = grid[j]
        sm, fm, am, npos = rowstats(lambda s, w=w: lin(w))
        log(f"   {sm:7.4f} {fm:7.4f} {am:7.4f} {am-champ_all:+7.4f} {npos:>3}/7  {wname(w)}")
        if FRESH and (best_fresh is None or fm > best_fresh[1]):
            best_fresh = (w, fm, am, npos)

    # ---- sensible region the user asked for: ranker in [.30,.40], hgb in [.30,.40] ----
    reg = [(j, grid[j]) for j in range(len(grid))
           if 0.30 - 1e-9 <= grid[j][IDX["ranker"]] <= 0.40 + 1e-9
           and 0.30 - 1e-9 <= grid[j][IDX["hgb"]] <= 0.40 + 1e-9]
    reg.sort(key=lambda t: -search_scores[t[0]])
    log("\n-- sensible region (ranker 0.30-0.40, hgb 0.30-0.40): top 10 by SEARCH --")
    log(f"   {'search':>7} {'fresh':>7} {'all':>7} {'d_all':>7} {'np/7':>5}  weights")
    for _, w in reg[:10]:
        sm, fm, am, npos = rowstats(lambda s, w=w: lin(w))
        log(f"   {sm:7.4f} {fm:7.4f} {am:7.4f} {am-champ_all:+7.4f} {npos:>3}/7  {wname(w)}")

    # ---- stage 2b: per-subset learned metas, SELECT on search seeds ----
    log("\n-- learned metas per subset (select on SEARCH, honest FRESH/ALL) --")
    log(f"   {'method':<7} {'search':>7} {'fresh':>7} {'all':>7} {'d_all':>7} {'np/7':>5}  subset")
    subsets = [c for r in range(2, M + 1) for c in itertools.combinations(range(M), r)]
    for kind, fitter in (("logit", fit_logit), ("gbm", fit_gbm)):
        best = None
        for cols in subsets:
            cols = list(cols)
            fitted = {s: fitter(seed_meta[s][0], seed_meta[s][1], cols, s) for s in ALL_SEEDS}
            sm = np.mean([score_folds(seed_folds[s], fitted[s]) for s in SEEDS])
            if best is None or sm > best[0]:
                best = (sm, cols, fitted)
        sm, cols, fitted = best
        fm = np.mean([score_folds(seed_folds[s], fitted[s]) for s in FRESH]) if FRESH else float("nan")
        am = np.mean([score_folds(seed_folds[s], fitted[s]) for s in ALL_SEEDS])
        npos = sum(1 for s in ALL_SEEDS if score_folds(seed_folds[s], fitted[s]) > champ[s])
        nm = "+".join(NAMES[c] for c in cols)
        log(f"   {kind:<7} {sm:7.4f} {fm:7.4f} {am:7.4f} {am-champ_all:+7.4f} {npos:>3}/7  {nm}")

    # ---- verdict ----
    log("\n-- VERDICT (out-of-sample: best-on-SEARCH scored on FRESH vs champ) --")
    if best_fresh is not None:
        wb = grid[order[0]]                    # best SEARCH blend
        wb_fresh = np.mean([score_folds(seed_folds[s], lin(wb)) for s in FRESH])
        champ_fresh = np.mean([champ[s] for s in FRESH])
        log(f"   best-SEARCH blend [{wname(wb)}]")
        log(f"     search={search_scores[order[0]]:.4f}  fresh={wb_fresh:.4f}  "
            f"champ_fresh={champ_fresh:.4f}  d_fresh={wb_fresh-champ_fresh:+.4f}")

    with open("experiments/json/_stack_deep_best.json", "w") as f:
        json.dump({"members": NAMES, "step": STEP,
                   "champ_weights": {n: float(v) for n, v in zip(NAMES, wc)},
                   "best_search_weights": {n: float(v) for n, v in zip(NAMES, grid[order[0]])},
                   "champ_all": float(champ_all)}, f, indent=2)
    log(f"\n=== DONE {datetime.now():%F %T} ===")


if __name__ == "__main__":
    main()
