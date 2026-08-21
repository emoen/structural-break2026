"""Broad clustering sweep: can ANY clustering of the HISTORIES (any algorithm, any distance metric)
add deployable orthogonal signal OVER the true 61-col champion at late offset vs REAL step-matched
negatives? Clustering is unsupervised (label-free) so fitting on the full history pool does not leak
the break target; the supervised leakage is handled by 5-fold OOF on the champion+cluster features.

For every method we build a DEPLOYABLE per-series block = onehot(nearest cluster) + distances to all
cluster centroids (GMM uses EM responsibilities), computed from the series' OWN history -> per-series,
causal. Delta = OOF AUC(champ61 + block) - OOF AUC(champ61).
"""
import os, warnings, time
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import (KMeans, AgglomerativeClustering, SpectralClustering, Birch,
                             DBSCAN, HDBSCAN, MeanShift)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier

from _exp_fast import hist_context, HIST_CAP
from _night_banks import fit_champ_refs, champ_base_row, fit_dep_refs, dep_feats

DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
WINDOW = 250
OFFSET = int(os.environ.get("OFFSET", "200"))
N_EACH = 2400

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


def champ61(w, ctx, cref, dref):
    base = champ_base_row(w, ctx, cref)
    dep = np.asarray(dep_feats(w, ctx, dref), dtype=np.float64)
    return np.concatenate([base, dep[DEP_KEEP]])


usable = [s for s in TAU if len(HI[s]) > 40 and len(ON[s]) > 60]
sid2row = {s: i for i, s in enumerate(usable)}
Hraw = np.array([hist_vec(HI[s]) for s in usable])
scaler = StandardScaler().fit(Hraw)
Hstd = scaler.transform(Hraw)

br = [s for s in usable if TAU[s] >= 15 and len(ON[s]) > TAU[s] + OFFSET]
nb = [s for s in usable if TAU[s] < 0]
rng.shuffle(br); rng.shuffle(nb); br = br[:N_EACH]
nb_len = np.array([len(ON[s]) for s in nb]); nb_arr = np.array(nb)


def sample_row(s, t):
    x = ON[s][:t + 1]; w = x[max(0, len(x) - WINDOW):]
    h = HI[s]; ctx = hist_context(h, HIST_CAP)
    return champ61(w, ctx, fit_champ_refs(h), fit_dep_refs(h))


t0 = time.time()
Pc, Nc, Psid, Nsid = [], [], [], []
for s in br:
    t = TAU[s] + OFFSET
    if len(ON[s]) <= t:
        continue
    elig = nb_arr[nb_len > t]
    if len(elig) == 0:
        continue
    sn = int(rng.choice(elig))
    Pc.append(sample_row(s, t)); Psid.append(s)
    Nc.append(sample_row(sn, t)); Nsid.append(sn)
n = min(len(Pc), len(Nc))
Pc, Nc = np.array(Pc[:n]), np.array(Nc[:n])
Psid, Nsid = Psid[:n], Nsid[:n]
Xc = np.nan_to_num(np.vstack([Pc, Nc]))
rows_sid = np.array([sid2row[s] for s in Psid] + [sid2row[s] for s in Nsid])
y = np.r_[np.ones(n), np.zeros(n)]
print(f"offset={OFFSET}  n={n}/class  champ61 dim={Xc.shape[1]}  built in {time.time()-t0:.0f}s")

# distance-metric spaces (clustering + centroid distances computed in these)
SPACES = {
    "euclid": Hstd,
    "mahalanobis": PCA(whiten=True, random_state=0).fit_transform(Hstd),
    "cosine": normalize(Hstd),
}

cv = StratifiedKFold(5, shuffle=True, random_state=0)


def hgb_auc(F):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
                                       random_state=0)
    oof = cross_val_predict(m, np.nan_to_num(F), y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, oof)


base_auc = hgb_auc(Xc)
print(f"champ61 base AUC = {base_auc:.4f}\n")


def centroid_block(labels, Xspace):
    """onehot(nearest centroid) + distances to every cluster centroid, per usable series."""
    labs = np.unique(labels[labels >= 0])
    if len(labs) < 2:
        return None
    cents = np.array([Xspace[labels == c].mean(0) for c in labs])
    d = np.linalg.norm(Xspace[:, None, :] - cents[None, :, :], axis=2)  # n_series x K
    onehot = np.zeros_like(d); onehot[np.arange(len(d)), d.argmin(1)] = 1.0
    return np.hstack([onehot, d])


def eval_block(name, per_series_block):
    if per_series_block is None:
        print(f"  {name:<34} -> <2 clusters, skip"); return
    blk = per_series_block[rows_sid]
    a = hgb_auc(np.hstack([Xc, blk]))
    print(f"  {name:<34} AUC={a:.4f}  delta {a-base_auc:+.4f}  (Kfeat={per_series_block.shape[1]})")


def labels_for(method, Xspace, K):
    if method == "kmeans":
        return KMeans(K, n_init=10, random_state=0).fit_predict(Xspace)
    if method == "agglo_ward":
        return AgglomerativeClustering(K, linkage="ward").fit_predict(Xspace)
    if method == "agglo_avg":
        return AgglomerativeClustering(K, linkage="average").fit_predict(Xspace)
    if method == "spectral":
        return SpectralClustering(K, affinity="nearest_neighbors", random_state=0,
                                  assign_labels="kmeans").fit_predict(Xspace)
    if method == "birch":
        return Birch(n_clusters=K).fit_predict(Xspace)
    raise ValueError(method)


print("== centroid-feature blocks over champ61 ==")
for metric, Xsp in SPACES.items():
    for K in (8, 16):
        eval_block(f"kmeans K={K} [{metric}]", centroid_block(labels_for("kmeans", Xsp, K), Xsp))

for method in ("agglo_ward", "agglo_avg", "spectral", "birch"):
    lab = labels_for(method, SPACES["euclid"], 8)
    eval_block(f"{method} K=8 [euclid]", centroid_block(lab, SPACES["euclid"]))

# GMM (EM) — soft responsibilities as the block
for K in (8, 16):
    gm = GaussianMixture(K, covariance_type="full", random_state=0, reg_covar=1e-4).fit(Hstd)
    resp = gm.predict_proba(Hstd)                       # n_series x K responsibilities
    eval_block(f"gmm(EM) K={K} [full-cov]", resp)

# density methods (variable K, noise=-1); assign by nearest non-noise centroid
db = DBSCAN(eps=1.2, min_samples=10).fit_predict(SPACES["euclid"])
print(f"  (dbscan clusters={len(set(db))-(1 if -1 in db else 0)}, noise={int((db==-1).sum())})")
eval_block("dbscan [euclid]", centroid_block(db, SPACES["euclid"]))
hd = HDBSCAN(min_cluster_size=50).fit_predict(SPACES["euclid"])
print(f"  (hdbscan clusters={len(set(hd))-(1 if -1 in hd else 0)}, noise={int((hd==-1).sum())})")
eval_block("hdbscan [euclid]", centroid_block(hd, SPACES["euclid"]))

# cluster-CONDITIONAL detection: separate champion model per KMeans-8 cluster vs global
print("\n== cluster-conditional detection (KMeans K=8 euclid) ==")
km8 = KMeans(8, n_init=10, random_state=0).fit_predict(SPACES["euclid"])
samp_cl = km8[rows_sid]
oof = np.zeros(len(y))
for c in np.unique(samp_cl):
    m = samp_cl == c
    if y[m].sum() < 25 or (1 - y[m]).sum() < 25:
        oof[m] = 0.5; continue
    oof[m] = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
                                       random_state=0),
        Xc[m], y[m], cv=StratifiedKFold(4, shuffle=True, random_state=0),
        method="predict_proba")[:, 1]
print(f"  per-cluster champ models AUC={roc_auc_score(y, oof):.4f}  delta {roc_auc_score(y, oof)-base_auc:+.4f}")
