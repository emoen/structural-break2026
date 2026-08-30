"""Fine local refine around the deep-search winner + meta regularization probe.

Reuses the cached member predictions from _stack_deep (no model re-fitting), so this is seconds.

Part A -- FINE WEIGHTS: a 0.01-step bounded 5-simplex box around the plateau center
    (ranker~0.45 rf~0.40 xgb~0.10 cat~0.05, hgb 0-0.10). Honest: select on SEARCH seeds,
    score on held-out FRESH + ALL, count seeds beating champ.

Part B -- META OVERFIT PROBE: the logit/gbm metas lost out-of-sample. Sweep the logit L2
    strength (C from very strong shrinkage -> weak) on the winning subset. As C->0 the meta
    collapses toward its prior (a shrunken average) -- this quantifies whether *regularization
    alone* recovers a blend-like generalizer, which is exactly what 'pretrain on synthetic'
    would buy as data-augmentation. Reported honestly on FRESH.
"""
import os
import time
import json
import itertools
from datetime import datetime

import numpy as np

from _stack_zoo import NAMES, IDX, W_CHAMP, SEEDS, FRESH
from _stack_deep import (get_seed_cache, build_folds, score_folds, score_grid,
                         lin, M, ALL_SEEDS)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("experiments/json", exist_ok=True)
LOG = "experiments/_stack_refine.log"

# plateau center (deep-search winner), in 0.01 integer units summing to 100
BEST = {"hgb": 0, "ranker": 45, "rf": 40, "xgb": 10, "catboost": 5}
Q = 100                                   # 0.01 resolution
# per-member integer bounds (a box around the plateau); clipped >= 0
BOUNDS = {"hgb": (0, 12), "ranker": (36, 54), "rf": (28, 50),
          "xgb": (0, 22), "catboost": (0, 16)}


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def wname(w):
    return " ".join(f"{n}={v:.2f}" for n, v in zip(NAMES, w) if v > 1e-9)


def bounded_simplex(bounds, q):
    """Yield integer m-tuples in per-member [lo,hi] bounds summing to q (pruned recursion)."""
    m = len(bounds)
    suf_lo = [0] * (m + 1)
    suf_hi = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suf_lo[i] = suf_lo[i + 1] + bounds[i][0]
        suf_hi[i] = suf_hi[i + 1] + bounds[i][1]

    def rec(i, rem):
        lo, hi = bounds[i]
        if i == m - 1:
            if lo <= rem <= hi:
                yield (rem,)
            return
        v_lo = max(lo, rem - suf_hi[i + 1])
        v_hi = min(hi, rem - suf_lo[i + 1])
        for v in range(v_lo, v_hi + 1):
            for tail in rec(i + 1, rem - v):
                yield (v,) + tail

    yield from rec(0, q)


def main():
    open(LOG, "w").close()
    log(f"=== STACK REFINE  {datetime.now():%F %T} ===")
    log(f"members={NAMES}  search={SEEDS}  fresh={FRESH}")

    seed_folds, seed_meta = {}, {}
    for seed in ALL_SEEDS:
        sd = get_seed_cache(seed)
        seed_folds[seed] = build_folds(sd)
        seed_meta[seed] = (sd["Foof"], sd["ytr"])
    log(f"[cache ready] {len(ALL_SEEDS)} seeds")

    # champ reference (in-grid: hgb=0.40, ranker=0.60)
    wc = np.zeros(M); wc[IDX["hgb"]] = W_CHAMP; wc[IDX["ranker"]] = 1 - W_CHAMP
    champ = {s: score_folds(seed_folds[s], lin(wc)) for s in ALL_SEEDS}
    champ_all = np.mean([champ[s] for s in ALL_SEEDS])
    champ_fresh = np.mean([champ[s] for s in FRESH])
    champ_search = np.mean([champ[s] for s in SEEDS])
    log(f"[champ] search={champ_search:.4f} fresh={champ_fresh:.4f} all={champ_all:.4f}")

    # ---- Part A: fine 0.01 bounded refine ----
    bounds = [BOUNDS[n] for n in NAMES]
    grid_i = list(bounded_simplex(bounds, Q))
    grid = [np.asarray(t, dtype=np.float64) / Q for t in grid_i]
    Wmat = np.asarray(grid).T
    log(f"\n-- Part A: fine 0.01 refine, {len(grid)} blends in box --")
    t0 = time.time()
    search_scores = np.mean([score_grid(seed_folds[s], Wmat) for s in SEEDS], axis=0)
    log(f"   swept in {time.time()-t0:.0f}s; select on SEARCH")
    order = np.argsort(-search_scores)

    def rowstats(w):
        sm = np.mean([score_folds(seed_folds[s], lin(w)) for s in SEEDS])
        fm = np.mean([score_folds(seed_folds[s], lin(w)) for s in FRESH])
        am = np.mean([score_folds(seed_folds[s], lin(w)) for s in ALL_SEEDS])
        npos = sum(1 for s in ALL_SEEDS if score_folds(seed_folds[s], lin(w)) > champ[s])
        return sm, fm, am, npos

    log(f"   {'search':>7} {'fresh':>7} {'all':>7} {'d_all':>7} {'d_frsh':>7} {'np/7':>5}  weights")
    log(f"   {champ_search:7.4f} {champ_fresh:7.4f} {champ_all:7.4f} {0.0:+7.4f} {0.0:+7.4f} {'-':>5}  CHAMP")
    wb05 = np.zeros(M)
    for n, v in BEST.items():
        wb05[IDX[n]] = v / Q
    sm, fm, am, npos = rowstats(wb05)
    log(f"   {sm:7.4f} {fm:7.4f} {am:7.4f} {am-champ_all:+7.4f} {fm-champ_fresh:+7.4f} {npos:>3}/7  0.05-WINNER [{wname(wb05)}]")

    # top-20 by SEARCH, and separately the best-on-FRESH among top-200 (honest robustness view)
    log("   -- top 20 by SEARCH --")
    for j in order[:20]:
        w = grid[j]
        sm, fm, am, npos = rowstats(w)
        log(f"   {sm:7.4f} {fm:7.4f} {am:7.4f} {am-champ_all:+7.4f} {fm-champ_fresh:+7.4f} {npos:>3}/7  {wname(w)}")

    # average the top-K search blends -> a robust centroid (less noise-fit than the single argmax)
    for K in (25, 100, 400):
        cent = np.mean([grid[j] for j in order[:K]], axis=0)
        cent = cent / cent.sum()
        sm, fm, am, npos = rowstats(cent)
        log(f"   centroid(top{K:>3}): search={sm:.4f} fresh={fm:.4f} all={am:.4f} "
            f"d_all={am-champ_all:+.4f} d_fresh={fm-champ_fresh:+.4f} {npos}/7  [{wname(cent)}]")

    best = grid[order[0]]
    with open("experiments/json/_stack_refine_best.json", "w") as f:
        json.dump({"members": NAMES,
                   "best_fine_weights": {n: float(v) for n, v in zip(NAMES, best)},
                   "champ_all": float(champ_all)}, f, indent=2)

    # ---- Part B: meta regularization probe (informs 'pretrain on synthetic') ----
    log("\n-- Part B: logit meta L2 sweep on winning subset (honest FRESH) --")
    log("   (strong shrinkage C->0 == meta collapses toward its prior; a proxy for what")
    log("    synthetic-data augmentation would regularize toward)")
    subset = [IDX[n] for n in ("ranker", "rf", "xgb", "catboost")]
    log(f"   subset = ranker+rf+xgb+catboost")
    log(f"   {'C':>8} {'search':>7} {'fresh':>7} {'all':>7} {'d_frsh':>7} {'np/7':>5}")
    for C in (0.001, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0):
        fitted = {}
        for s in ALL_SEEDS:
            Foof, y = seed_meta[s]
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=C, max_iter=5000, random_state=s))
            m.fit(Foof[:, subset], y)
            fitted[s] = (lambda mat, m=m: m.predict_proba(mat[:, subset])[:, 1])
        sm = np.mean([score_folds(seed_folds[s], fitted[s]) for s in SEEDS])
        fm = np.mean([score_folds(seed_folds[s], fitted[s]) for s in FRESH])
        am = np.mean([score_folds(seed_folds[s], fitted[s]) for s in ALL_SEEDS])
        npos = sum(1 for s in ALL_SEEDS if score_folds(seed_folds[s], fitted[s]) > champ[s])
        log(f"   {C:8.3f} {sm:7.4f} {fm:7.4f} {am:7.4f} {fm-champ_fresh:+7.4f} {npos:>3}/7")

    log(f"\n=== DONE {datetime.now():%F %T} ===")


if __name__ == "__main__":
    main()
