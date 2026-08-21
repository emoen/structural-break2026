"""Does POOLING more post-break samples beat trailing-250 for the LIVE late families
(variance, KS, dependence)? If yes -> the champion's trailing-250 cap leaves late signal
on the table. AUC vs REAL no-break negatives at high offset.

windows:  trail250 (champion cap) | expand (all online) | localize (CUSUM-argmax pool post)
features: lvr (|log var ratio|) | ks | dac1 (|ac1 shift|)
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
vals = X["value"].to_numpy(np.float64); per = X["period"].to_numpy(np.int64)
ids = X.index.get_level_values("id").to_numpy()
uniq, start = np.unique(ids, return_index=True); ends = np.append(start[1:], len(ids))
ON, HI = {}, {}
for u, s, e in zip(uniq, start, ends):
    p = per[s:e]; v = vals[s:e]; HI[int(u)] = v[p == 1]; ON[int(u)] = v[p == 2]
TAU = {int(k): int(v) for k, v in yi["tau_index"].items()}
rng = np.random.default_rng(0)


def ac1(a):
    if len(a) < 5:
        return 0.0
    a = a - a.mean(); d = (a * a).sum() + 1e-12
    return float((a[:-1] * a[1:]).sum() / d)


def ks(win, hsorted):
    ws = np.sort(win); n = len(ws)
    allv = np.concatenate([ws, hsorted])
    cw = np.searchsorted(ws, allv, side="right") / n
    ch = np.searchsorted(hsorted, allv, side="right") / len(hsorted)
    return float(np.max(np.abs(cw - ch)))


def cusum_argmax(x):
    n = len(x)
    if n < 20:
        return n // 2
    c = np.cumsum(x - x.mean()); m = int(np.argmax(np.abs(c[:-1]))) + 1
    return max(5, min(n - 5, m))


def seg_feats(w, h, hsorted, hac):
    if len(w) < 5:
        return 0.0, 0.0, 0.0
    lvr = abs(np.log((w.var() + 1e-9) / (h.var() + 1e-9)))
    kss = ks(w, hsorted)
    dac = abs(ac1(w) - hac)
    return lvr, kss, dac


def windows(x):
    n = len(x)
    return {"trail250": x[max(0, n - 250):], "expand": x, "localize": x[cusum_argmax(x):]}


WINS = ["trail250", "expand", "localize"]
FEATS = ["lvr", "ks", "dac1"]
OFFSETS = [100, 200, 400]
N_EACH = 1400
br = [s for s in TAU if TAU[s] >= 15 and len(ON[s]) > TAU[s] + 5 and len(HI[s]) > 30]
nb = [s for s in TAU if TAU[s] < 0 and len(ON[s]) > 60 and len(HI[s]) > 30]

for k in OFFSETS:
    posw = {w: {f: [] for f in FEATS} for w in WINS}
    negw = {w: {f: [] for f in FEATS} for w in WINS}
    cand = [s for s in br if len(ON[s]) > TAU[s] + k]
    rng.shuffle(cand); cand = cand[:N_EACH]
    pos_t = []
    for s in cand:
        tau = TAU[s]; t = tau + k; x = ON[s][:t + 1]
        h = HI[s]; hs = np.sort(h); hac = ac1(h)
        for wn, w in windows(x).items():
            lvr, kss, dac = seg_feats(w, h, hs, hac)
            posw[wn]["lvr"].append(lvr); posw[wn]["ks"].append(kss); posw[wn]["dac1"].append(dac)
        pos_t.append(t)
    rng.shuffle(nb); ti = 0
    for s in nb:
        if ti >= len(pos_t):
            break
        t = pos_t[ti]
        if len(ON[s]) <= t:
            continue
        x = ON[s][:t + 1]; h = HI[s]; hs = np.sort(h); hac = ac1(h)
        for wn, w in windows(x).items():
            lvr, kss, dac = seg_feats(w, h, hs, hac)
            negw[wn]["lvr"].append(lvr); negw[wn]["ks"].append(kss); negw[wn]["dac1"].append(dac)
        ti += 1
    npos = len(negw[WINS[0]]["lvr"])
    print(f"offset={k}  npos={npos}")
    print(f"    {'feat':>6} " + " ".join(f"{w:>9}" for w in WINS))
    for f in FEATS:
        row = []
        for w in WINS:
            pp = np.array(posw[w][f][:npos]); nn = np.array(negw[w][f])
            y = np.r_[np.ones(len(pp)), np.zeros(len(nn))]
            row.append(roc_auc_score(y, np.r_[pp, nn]))
        print(f"    {f:>6} " + " ".join(f"{a:9.3f}" for a in row))
    print()
