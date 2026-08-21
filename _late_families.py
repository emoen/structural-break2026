"""Which feature FAMILY carries late-offset discrimination vs REAL (no-break) negatives?

Per-series contrasts (window vs that series' own history), AUC vs no-break negatives
at several offsets. Families: mean-shift, var-ratio, KS(CDF), skew/kurt shift, ac1 shift,
plus a small HGB over all of them, and pooled (localized) variants of the strongest.
Random = 0.5.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
vals = X["value"].to_numpy(np.float64)
per = X["period"].to_numpy(np.int64)
ids = X.index.get_level_values("id").to_numpy()
uniq, start = np.unique(ids, return_index=True)
ends = np.append(start[1:], len(ids))
ON, HI = {}, {}
for u, s, e in zip(uniq, start, ends):
    p = per[s:e]; v = vals[s:e]
    HI[int(u)] = v[p == 1]; ON[int(u)] = v[p == 2]
TAU = {int(k): int(v) for k, v in yi["tau_index"].items()}
rng = np.random.default_rng(0)


def moments(a):
    m = a.mean(); s = a.std() + 1e-9
    z = (a - m) / s
    sk = float((z ** 3).mean()); ku = float((z ** 4).mean() - 3.0)
    return m, s, sk, ku


def ac1(a):
    if len(a) < 5:
        return 0.0
    a = a - a.mean()
    d = (a * a).sum() + 1e-12
    return float((a[:-1] * a[1:]).sum() / d)


def ks(win, hsorted):
    ws = np.sort(win); n = len(ws)
    # KS between win ecdf and history ecdf via merge
    allv = np.concatenate([ws, hsorted])
    cdf_w = np.searchsorted(ws, allv, side="right") / n
    cdf_h = np.searchsorted(hsorted, allv, side="right") / len(hsorted)
    return float(np.max(np.abs(cdf_w - cdf_h)))


def feats(x, h, hsorted, win_slice):
    w = x[win_slice]
    if len(w) < 5:
        return [0.0] * 6
    hm, hs, hsk, hku = moments(h)
    wm, ws, wsk, wku = moments(w)
    mz = abs((wm - hm) / (hs / np.sqrt(len(w))))
    lvr = abs(np.log((w.var() + 1e-9) / (h.var() + 1e-9)))
    kss = ks(w, hsorted)
    dsk = abs(wsk - hsk); dku = abs(wku - hku)
    dac = abs(ac1(w) - ac1(h))
    return [mz, lvr, kss, dsk, dku, dac]


FAM = ["mean_z", "lvr", "ks", "dskew", "dkurt", "dac1"]
OFFSETS = [60, 200, 400]
N_EACH = 1400
br = [s for s in TAU if TAU[s] >= 15 and len(ON[s]) > TAU[s] + 5 and len(HI[s]) > 30]
nb = [s for s in TAU if TAU[s] < 0 and len(ON[s]) > 60 and len(HI[s]) > 30]
print(f"break={len(br)} nobreak={len(nb)}\n")
print(f"{'offset':>7} {'npos':>5} " + " ".join(f"{f:>7}" for f in FAM) + f" {'HGBall':>7}")

for k in OFFSETS:
    Pos, Neg, pos_t = [], [], []
    cand = [s for s in br if len(ON[s]) > TAU[s] + k]
    rng.shuffle(cand); cand = cand[:N_EACH]
    for s in cand:
        tau = TAU[s]; t = tau + k
        x = ON[s][:t + 1]; h = HI[s]; hs = np.sort(h)
        Pos.append(feats(x, h, hs, slice(max(0, len(x) - 250), len(x))))
        pos_t.append(t)
    rng.shuffle(nb); ti = 0
    for s in nb:
        if ti >= len(pos_t):
            break
        t = pos_t[ti]
        if len(ON[s]) <= t:
            continue
        x = ON[s][:t + 1]; h = HI[s]; hs = np.sort(h)
        Neg.append(feats(x, h, hs, slice(max(0, len(x) - 250), len(x))))
        ti += 1
    Pos = np.array(Pos[:len(Neg)]); Neg = np.array(Neg)
    Xf = np.vstack([Pos, Neg]); y = np.r_[np.ones(len(Pos)), np.zeros(len(Neg))]
    aucs = [roc_auc_score(y, Xf[:, j]) for j in range(len(FAM))]
    # small HGB over all families, OOF
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                         max_leaf_nodes=31, random_state=0)
    oof = cross_val_predict(hgb, Xf, y, cv=StratifiedKFold(4, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    auc_all = roc_auc_score(y, oof)
    print(f"{k:>7} {len(Pos):>5} " + " ".join(f"{a:7.3f}" for a in aucs) + f" {auc_all:7.3f}")
