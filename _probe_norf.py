"""Cost of removing the slow 800-tree RF from the blend, scored on the cached OOF across
all 7 seeds (no retrain). RF costs 45ms/point (10x the whole rest) -> streaming timeout.
Question: does dropping RF (or moving its weight to ranker) hurt TS-AUC?
"""
import numpy as np
from _stack_deep import get_seed_cache, build_folds, score_folds, lin, SEEDS, FRESH, ALL_SEEDS
from _stack_zoo import NAMES, IDX

print("NAMES:", NAMES, "IDX:", IDX)

folds_by_seed = {s: build_folds(get_seed_cache(s)) for s in ALL_SEEDS}


def wvec(d):
    w = np.zeros(len(NAMES))
    for k, v in d.items():
        w[IDX[k]] = v
    s = w.sum()
    return w / s if s else w


def score(d):
    w = wvec(d)
    per = {s: score_folds(folds_by_seed[s], lin(w)) for s in ALL_SEEDS}
    srch = np.mean([per[s] for s in SEEDS])
    frsh = np.mean([per[s] for s in FRESH])
    allm = np.mean([per[s] for s in ALL_SEEDS])
    return srch, frsh, allm, per


CAND = {
    "stack4 (has RF)":        {"ranker": .45, "rf": .40, "xgb": .10, "catboost": .05},
    "stack5 (has RF)":        {"ranker": .45, "rf": .37, "xgb": .11, "catboost": .06, "hgb": .01},
    "noRF renorm":            {"ranker": .45, "xgb": .10, "catboost": .05},
    "noRF RF->ranker":        {"ranker": .85, "xgb": .10, "catboost": .05},
    "noRF RF->rank+hgb":      {"ranker": .60, "xgb": .10, "catboost": .05, "hgb": .25},
    "noRF ranker+hgb+xgb+cat":{"ranker": .50, "hgb": .30, "xgb": .12, "catboost": .08},
}

# quick simplex search over the 4 fast members (no RF): hgb, ranker, xgb, catboost
best = None
grid = [i / 20 for i in range(21)]
fast = ["ranker", "hgb", "xgb", "catboost"]
import itertools
for a in grid:
    for b in grid:
        if a + b > 1: continue
        for c in grid:
            if a + b + c > 1: continue
            d = 1 - a - b - c
            if d < 0: continue
            w = {"ranker": a, "hgb": b, "xgb": c, "catboost": d}
            sr, fr, al, _ = score(w)
            if best is None or al > best[0]:
                best = (al, sr, fr, dict(w))

print(f"\n{'blend':<26} {'search':>7} {'fresh':>7} {'all':>7}   np/7")
for name, d in CAND.items():
    sr, fr, al, per = score(d)
    npos = sum(1 for s in ALL_SEEDS if per[s] >= score(CAND['stack4 (has RF)'])[3][s] - 1e-9)
    print(f"{name:<26} {sr:7.4f} {fr:7.4f} {al:7.4f}")

al, sr, fr, w = best
print(f"\nbest no-RF simplex (fast members only): all={al:.4f} search={sr:.4f} fresh={fr:.4f}")
print("  weights:", {k: round(v, 3) for k, v in w.items() if v > 0})
