"""Settle the borderline: multi-seed LENGTH-MATCHED T1b for the two-sample block.
T1 gave ws_split +0.0195 but that was a window-length artifact (block-alone 0.88->0.55 under
length-matching). Residual was ~+0.003 single-seed with ALL=+0.000. Screen across seeds to
decide noise vs real. Ship bar: mean_d>=0.0015 AND all seeds positive AND min_t>=2.0.
"""
import os, math, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import stats
from _twosample_t1 import (usable, TAU, ON, OFFSET, N_EACH, build, IDX, NAMES,
                           StratifiedKFold, HistGradientBoostingClassifier, roc_auc_score,
                           cross_val_predict, np as _np)

SEEDS = [42, 7, 123]
res = {nm: [] for nm in NAMES}


def auc_seed(F, yy, sd):
    cv = StratifiedKFold(5, shuffle=True, random_state=sd)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
                                       random_state=sd)
    oof = cross_val_predict(m, np.nan_to_num(F), yy, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(yy, oof)


brm_all = [s for s in usable if TAU[s] >= 270 and len(ON[s]) > TAU[s] + OFFSET]
print(f"pool len-matched series = {len(brm_all)}")
for sd in SEEDS:
    rng = np.random.default_rng(sd)
    brm = brm_all.copy(); rng.shuffle(brm); brm = brm[:N_EACH]
    pairs = []
    for s in brm:
        pairs.append((s, TAU[s] + OFFSET, 1))
        pairs.append((s, int(rng.integers(249, TAU[s] - 1)), 0))
    Xc, Bl, y = build(pairs)
    b0 = auc_seed(Xc, y, sd)
    line = f"seed {sd:>4}  n={int(y.sum())}  champ={b0:.4f}"
    for nm in NAMES:
        a = auc_seed(np.hstack([Xc, Bl[:, IDX[nm]]]), y, sd)
        res[nm].append(a - b0); line += f"  {nm}={a-b0:+.4f}"
    print(line)

print("\n=== verdict ===")
for nm in NAMES:
    d = np.array(res[nm]); t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-12)
    ok = (d.mean() >= 0.0015) and (d > 0).all() and abs(t) >= 2.0
    print(f"{nm:<9} mean_d={d.mean():+.4f}  min={d.min():+.4f}  t={t:+.2f}  "
          f"{'SHIP-CANDIDATE' if ok else 'noise/below-bar'}")
