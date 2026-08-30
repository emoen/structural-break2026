"""Two rank-matched meta-combiners vs the fixed percentile blend (honest SEARCH/FRESH split).

Motivated by the L2 probe: the logit meta's loss was flat in regularisation -> a BIAS (wrong
functional form), not variance. So try combiners whose geometry matches TS-AUC's rank metric:

  A) LambdaMART meta  -- LGBMRanker (rank_xendcg / lambdarank) on the 5 member percentile
     columns, query group == one online step (Ttr), exactly the champion ranker recipe but
     stacked on member outputs instead of raw features.
  B) Isotonic-then-blend -- monotone-calibrate each member percentile to P(break) with isotonic
     regression, then weighted-sum at the deep-search winner weights (vs the raw percentile blend
     at the same weights). Isotonic depends only on order, so fitting on the stored percentiles is
     identical to fitting on raw scores.

All off the cached member preds (_stack_cache); no member refits. Fit each meta on a seed's OOF
(Foof/ytr/Ttr), score on that seed's val folds; SELECT nothing on FRESH -> honest OOS deltas.
"""
import os
import json
from datetime import datetime

import numpy as np

from _stack_zoo import NAMES, IDX, W_CHAMP, SEEDS, FRESH
from _stack_deep import get_seed_cache, build_folds, score_folds, lin, M, ALL_SEEDS
from _lambdarank import sort_into_groups, fit_ranker

from sklearn.isotonic import IsotonicRegression

os.makedirs("experiments/json", exist_ok=True)
LOG = "experiments/_stack_meta2.log"
WIN = {"ranker": 0.45, "rf": 0.40, "xgb": 0.10, "catboost": 0.05, "hgb": 0.0}   # deep-search winner
W = np.array([WIN.get(n, 0.0) for n in NAMES])
POS = [i for i in range(M) if W[i] > 0]


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def proba(m, X):
    return m.predict_proba(X)[:, 1]


def fit_isos(Foof, y):
    isos = [None] * M
    for i in POS:
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(Foof[:, i], y)
        isos[i] = ir
    return isos


def iso_get_s(isos):
    def gs(mat):
        out = np.zeros(mat.shape[0])
        for i in POS:
            out += W[i] * isos[i].predict(mat[:, i])
        return out
    return gs


def rank_get_s(rk):
    return lambda mat: proba(rk, mat)


def main():
    open(LOG, "w").close()
    log(f"=== STACK META2  {datetime.now():%F %T} ===")
    log(f"members={NAMES}  search={SEEDS}  fresh={FRESH}  winner_w={WIN}")

    seed_folds, seed_meta = {}, {}
    for s in ALL_SEEDS:
        sd = get_seed_cache(s)
        seed_folds[s] = build_folds(sd)
        seed_meta[s] = (sd["Foof"], sd["ytr"], sd["Ttr"])
    log(f"[cache ready] {len(ALL_SEEDS)} seeds")

    # references
    wc = np.zeros(M); wc[IDX["hgb"]] = W_CHAMP; wc[IDX["ranker"]] = 1 - W_CHAMP
    champ = {s: score_folds(seed_folds[s], lin(wc)) for s in ALL_SEEDS}
    win = {s: score_folds(seed_folds[s], lin(W)) for s in ALL_SEEDS}
    champ_fresh = np.mean([champ[s] for s in FRESH])
    champ_all = np.mean([champ[s] for s in ALL_SEEDS])

    # build per-seed metas
    getters = {"champ": lambda s: lin(wc), "winner_pct": lambda s: lin(W)}
    isos = {s: fit_isos(*seed_meta[s][:2]) for s in ALL_SEEDS}
    getters["iso_blend"] = lambda s: iso_get_s(isos[s])
    for obj in ("rank_xendcg", "lambdarank"):
        rks = {}
        for s in ALL_SEEDS:
            Foof, y, T = seed_meta[s]
            rows, lab, grp = sort_into_groups(Foof, y, T)
            rks[s] = fit_ranker(obj, rows, lab, grp)
        getters[f"lgbmrank_{obj}"] = (lambda rks: lambda s: rank_get_s(rks[s]))(rks)

    def stats(get_by_seed):
        sm = np.mean([score_folds(seed_folds[s], get_by_seed(s)) for s in SEEDS])
        fm = np.mean([score_folds(seed_folds[s], get_by_seed(s)) for s in FRESH])
        am = np.mean([score_folds(seed_folds[s], get_by_seed(s)) for s in ALL_SEEDS])
        npos = sum(1 for s in ALL_SEEDS
                   if score_folds(seed_folds[s], get_by_seed(s)) > champ[s])
        return sm, fm, am, npos

    log("\n-- combiners (honest: fit on OOF, FRESH held out) --")
    log(f"   {'method':<18} {'search':>7} {'fresh':>7} {'all':>7} {'d_all':>8} {'d_frsh':>8} {'np/7':>5}")
    log(f"   {'champ':<18} {np.mean([champ[s] for s in SEEDS]):7.4f} {champ_fresh:7.4f} "
        f"{champ_all:7.4f} {0.0:+8.4f} {0.0:+8.4f} {'-':>5}")
    out = {}
    for name, get in getters.items():
        if name == "champ":
            continue
        sm, fm, am, npos = stats(get)
        log(f"   {name:<18} {sm:7.4f} {fm:7.4f} {am:7.4f} {am-champ_all:+8.4f} "
            f"{fm-champ_fresh:+8.4f} {npos:>3}/7")
        out[name] = dict(search=sm, fresh=fm, all=am,
                         d_all=am - champ_all, d_fresh=fm - champ_fresh, npos=npos)

    with open("experiments/json/_stack_meta2_best.json", "w") as f:
        json.dump({"winner_w": WIN, "champ_all": float(champ_all),
                   "champ_fresh": float(champ_fresh), "results": out}, f, indent=2)
    log(f"\n=== DONE {datetime.now():%F %T} ===")


if __name__ == "__main__":
    main()
