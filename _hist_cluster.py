"""Does clustering the HISTORIC series help? Two questions, at late offset vs REAL negatives:
(A) does break detectability differ across history-clusters? (diagnostic)
(B) is a DEPLOYABLE per-series cluster assignment (from history alone) additive to the live
    families? i.e. does HGB{live + cluster-onehot + centroid-distances} beat HGB{live}?
Cluster assignment uses ONLY the series' own history -> per-series, deterministic at inference.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

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


def ac(a, L):
    if len(a) < L + 5:
        return 0.0
    a = a - a.mean(); d = (a * a).sum() + 1e-12
    return float((a[:-L] * a[L:]).sum() / d)


def spec_centroid(a):
    a = a - a.mean()
    if len(a) < 8 or a.std() < 1e-9:
        return 0.0
    p = np.abs(np.fft.rfft(a))[1:] ** 2
    f = np.arange(1, len(p) + 1)
    return float((f * p).sum() / (p.sum() + 1e-12) / len(p))


def hist_vec(h):
    z = (h - h.mean()) / (h.std() + 1e-9)
    return [float(h.mean()), float(h.std()), float(((z**3).mean())), float((z**4).mean() - 3),
            ac(h, 1), ac(h, 2), ac(h, 5), spec_centroid(h),
            float(np.abs(h - h.mean()).mean()), float(np.percentile(h, 75) - np.percentile(h, 25))]


def ks(win, hsorted):
    ws = np.sort(win); n = len(ws)
    allv = np.concatenate([ws, hsorted])
    cw = np.searchsorted(ws, allv, side="right") / n
    ch = np.searchsorted(hsorted, allv, side="right") / len(hsorted)
    return float(np.max(np.abs(cw - ch)))


def live_feats(w, h, hs, hac):
    return [abs(np.log((w.var() + 1e-9) / (h.var() + 1e-9))), ks(w, hs), abs(ac(w, 1) - hac)]


K = 8
OFFSET = 200
N_EACH = 2500

# fit clustering on a pool of TRAIN histories (all usable series)
usable = [s for s in TAU if len(HI[s]) > 40 and len(ON[s]) > 60]
Hmat = np.array([hist_vec(HI[s]) for s in usable])
scaler = StandardScaler().fit(Hmat)
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(scaler.transform(Hmat))
print(f"clustered {len(usable)} histories into K={K}; sizes={np.bincount(km.labels_)}")

# build late-offset samples
br = [s for s in usable if TAU[s] >= 15 and len(ON[s]) > TAU[s] + OFFSET]
nb = [s for s in usable if TAU[s] < 0]
rng.shuffle(br); rng.shuffle(nb)
br = br[:N_EACH]

def feats_for(s, t):
    x = ON[s][:t + 1]; w = x[max(0, len(x) - 250):]
    h = HI[s]; hs = np.sort(h); hac = ac(h, 1)
    hv = scaler.transform([hist_vec(h)])[0]
    dist = km.transform([hv])[0]           # distance to each centroid
    cl = int(np.argmin(dist))
    onehot = np.zeros(K); onehot[cl] = 1.0
    return live_feats(w, h, hs, hac), list(onehot) + list(dist), cl

Pl, Pc, Pcl, pos_t = [], [], [], []
for s in br:
    t = TAU[s] + OFFSET
    lf, cf, cl = feats_for(s, t)
    Pl.append(lf); Pc.append(cf); Pcl.append(cl); pos_t.append(t)
Nl, Nc, Ncl = [], [], []
ti = 0
for s in nb:
    if ti >= len(pos_t):
        break
    t = pos_t[ti]
    if len(ON[s]) <= t:
        continue
    lf, cf, cl = feats_for(s, t)
    Nl.append(lf); Nc.append(cf); Ncl.append(cl); ti += 1
n = len(Nl)
Pl, Pc, Pcl = np.array(Pl[:n]), np.array(Pc[:n]), np.array(Pcl[:n])
Nl, Nc, Ncl = np.array(Nl), np.array(Nc), np.array(Ncl)
y = np.r_[np.ones(n), np.zeros(n)]


def hgb_oof(Xf):
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                         max_leaf_nodes=31, random_state=0)
    oof = cross_val_predict(hgb, Xf, y, cv=StratifiedKFold(4, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    return roc_auc_score(y, oof)


live_only = hgb_oof(np.vstack([Pl, Nl]))
live_clu = hgb_oof(np.vstack([np.hstack([Pl, Pc]), np.hstack([Nl, Nc])]))
print(f"\n(B) late offset={OFFSET}, n={n}/class")
print(f"    HGB live           = {live_only:.4f}")
print(f"    HGB live + cluster = {live_clu:.4f}   (delta {live_clu-live_only:+.4f})")

# (A) per-cluster detectability spread (using live score = HGB live OOF)
oof = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
      max_leaf_nodes=31, random_state=0), np.vstack([Pl, Nl]), y,
      cv=StratifiedKFold(4, shuffle=True, random_state=0), method="predict_proba")[:, 1]
allcl = np.r_[Pcl, Ncl]
print(f"\n(A) per-cluster detectability (live-feature AUC within each history-cluster):")
for c in range(K):
    m = allcl == c
    yc = y[m]
    if yc.sum() < 20 or (1 - yc).sum() < 20:
        print(f"    cluster {c}: n={m.sum():5d}  (too few of one class)")
        continue
    print(f"    cluster {c}: n={m.sum():5d}  npos={int(yc.sum()):4d}  AUC={roc_auc_score(yc, oof[m]):.3f}")
