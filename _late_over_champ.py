"""Cheap orthogonality test BEFORE the full gate: do history-CLUSTER features and the ACF-VECTOR
contrast add over the TRUE 61-col champion base (champ_base 51 + dep KEEP 10), at late offset vs
REAL negatives? If either still lifts OOF AUC over the real champion, it earns the full multi-seed
gate; if it vanishes, the champion already captures it.
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

from _exp_fast import hist_context, HIST_CAP
from _night_banks import fit_champ_refs, champ_base_row, fit_dep_refs, dep_feats

DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]   # = b_no_accf
WINDOW = 250

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
    return [float(h.mean()), float(h.std()), float((z**3).mean()), float((z**4).mean() - 3),
            ac(h, 1), ac(h, 2), ac(h, 5), spec_centroid(h),
            float(np.abs(h - h.mean()).mean()), float(np.percentile(h, 75) - np.percentile(h, 25))]


def acf_vec(a, lags=10):
    if len(a) < lags + 5:
        return np.zeros(lags)
    a = a - a.mean(); d = (a * a).sum() + 1e-12
    return np.array([(a[:-L] * a[L:]).sum() / d for L in range(1, lags + 1)])


K = 8
OFFSET = int(os.environ.get("OFFSET", "200"))
N_EACH = 2400

usable = [s for s in TAU if len(HI[s]) > 40 and len(ON[s]) > 60]
Hmat = np.array([hist_vec(HI[s]) for s in usable])
scaler = StandardScaler().fit(Hmat)
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(scaler.transform(Hmat))
print(f"K={K} clusters, offset={OFFSET}, sizes={np.bincount(km.labels_)}")

br = [s for s in usable if TAU[s] >= 15 and len(ON[s]) > TAU[s] + OFFSET]
nb = [s for s in usable if TAU[s] < 0]
rng.shuffle(br); rng.shuffle(nb); br = br[:N_EACH]


def champ61(w, ctx, cref, dref):
    base = champ_base_row(w, ctx, cref)
    dep = np.asarray(dep_feats(w, ctx, dref), dtype=np.float64)
    return np.concatenate([base, dep[DEP_KEEP]])


def rowset(s, t):
    x = ON[s][:t + 1]; w = x[max(0, len(x) - WINDOW + 1):]
    h = HI[s]; ctx = hist_context(h, HIST_CAP)
    cref = fit_champ_refs(h); dref = fit_dep_refs(h)
    ch = champ61(w, ctx, cref, dref)
    hv = scaler.transform([hist_vec(h)])[0]
    dist = km.transform([hv])[0]; cl = int(np.argmin(dist))
    onehot = np.zeros(K); onehot[cl] = 1.0
    clu = np.concatenate([onehot, dist])
    hac_v = acf_vec(h)
    acfd = acf_vec(w) - hac_v          # ACF-vector contrast (10 dims)
    return ch, clu, acfd


Pch, Pcl, Pac, pos_t = [], [], [], []
for s in br:
    t = TAU[s] + OFFSET
    a, b, c = rowset(s, t); Pch.append(a); Pcl.append(b); Pac.append(c); pos_t.append(t)
Nch, Ncl, Nac = [], [], []
ti = 0
for s in nb:
    if ti >= len(pos_t):
        break
    t = pos_t[ti]
    if len(ON[s]) <= t:
        continue
    a, b, c = rowset(s, t); Nch.append(a); Ncl.append(b); Nac.append(c); ti += 1
n = len(Nch)
Pch, Pcl, Pac = np.array(Pch[:n]), np.array(Pcl[:n]), np.array(Pac[:n])
Nch, Ncl, Nac = np.array(Nch), np.array(Ncl), np.array(Nac)
y = np.r_[np.ones(n), np.zeros(n)]
print(f"n={n}/class  champ61 dim={Pch.shape[1]}")


def hgb_oof(P, N):
    Xf = np.nan_to_num(np.vstack([P, N]))
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03,
                                         max_leaf_nodes=31, random_state=0)
    oof = cross_val_predict(hgb, Xf, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    return roc_auc_score(y, oof)


base = hgb_oof(Pch, Nch)
c_clu = hgb_oof(np.hstack([Pch, Pcl]), np.hstack([Nch, Ncl]))
c_acf = hgb_oof(np.hstack([Pch, Pac]), np.hstack([Nch, Nac]))
c_both = hgb_oof(np.hstack([Pch, Pcl, Pac]), np.hstack([Nch, Ncl, Nac]))
print(f"\nHGB champ61              = {base:.4f}")
print(f"HGB champ61 + cluster    = {c_clu:.4f}   (delta {c_clu-base:+.4f})")
print(f"HGB champ61 + acf_vec    = {c_acf:.4f}   (delta {c_acf-base:+.4f})")
print(f"HGB champ61 + both       = {c_both:.4f}   (delta {c_both-base:+.4f})")
