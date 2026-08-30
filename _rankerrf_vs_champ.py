"""Score the ranker+rf 2-member blend vs the deployed hgb+ranker champion on the cached
7-seed OOF (no retrain). Answers: does ranker+rf match/beat the old champ, and at what weight?
"""
import numpy as np
from _stack_deep import get_seed_cache, build_folds, score_folds, lin, SEEDS, FRESH, ALL_SEEDS
from _stack_zoo import NAMES, IDX, W_CHAMP

folds = {s: build_folds(get_seed_cache(s)) for s in ALL_SEEDS}


def wvec(d):
    w = np.zeros(len(NAMES))
    for k, v in d.items():
        w[IDX[k]] = v
    s = w.sum()
    return w / s if s else w


def score(d):
    per = {s: score_folds(folds[s], lin(wvec(d))) for s in ALL_SEEDS}
    return (float(np.mean([per[s] for s in SEEDS])),
            float(np.mean([per[s] for s in FRESH])),
            float(np.mean([per[s] for s in ALL_SEEDS])), per)


CHAMP = {"hgb": W_CHAMP, "ranker": 1 - W_CHAMP}
_, _, _, champ_per = score(CHAMP)

REF = {
    "champ hgb.40/rank.60": CHAMP,
    "stack4 (r.45/rf.40/x.10/c.05)": {"ranker": .45, "rf": .40, "xgb": .10, "catboost": .05},
}

print(f"{'blend':<34} {'search':>7} {'fresh':>7} {'all':>7}  {'d_vs_champ':>10} npos/7")
for name, d in REF.items():
    sr, fr, al, per = score(d)
    dv = np.mean([per[s] - champ_per[s] for s in ALL_SEEDS])
    npos = sum(1 for s in ALL_SEEDS if per[s] > champ_per[s] + 1e-12)
    print(f"{name:<34} {sr:7.4f} {fr:7.4f} {al:7.4f}  {dv:+10.4f}  {npos}/7")

print("\n-- ranker+rf 2-member weight sweep --")
best = None
for a in np.round(np.arange(0.0, 1.001, 0.05), 3):
    d = {"ranker": float(a), "rf": float(1 - a)}
    sr, fr, al, per = score(d)
    dv = np.mean([per[s] - champ_per[s] for s in ALL_SEEDS])
    npos = sum(1 for s in ALL_SEEDS if per[s] > champ_per[s] + 1e-12)
    tag = ""
    if best is None or al > best[0]:
        best = (al, a, dv, npos, per)
        tag = "  <== best all"
    print(f"  ranker={a:.2f} rf={1-a:.2f}   all={al:.4f}  d={dv:+.4f}  npos={npos}/7{tag}")

al, a, dv, npos, per = best
print(f"\nBEST ranker+rf: ranker={a:.2f} rf={1-a:.2f}  all={al:.4f}  d_vs_champ={dv:+.4f}  npos={npos}/7")
print("per-seed (ranker+rf - champ):")
for s in ALL_SEEDS:
    tag = "search" if s in SEEDS else "fresh"
    print(f"  seed {s:<6} ({tag}): rankerrf={per[s]:.4f}  champ={champ_per[s]:.4f}  d={per[s]-champ_per[s]:+.4f}")
