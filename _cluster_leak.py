"""Is the cluster-feature gain REAL break-detection signal or a spurious HISTORY<->break-membership
selection proxy? Two diagnostics:

(T2) Predict br(has-break) vs nb(no-break) MEMBERSHIP from history alone (hist_vec, 5-fold OOF AUC).
     If >0.5, break assignment is NOT independent of history in the training set -> a static per-series
     cluster feature can inflate offline AUC as a selection proxy that need not hold on the LB.

(T1) Fair break-DETECTION test: positives = break series at tau+OFFSET, negatives = the SAME break
     series population at an EARLY not-yet-broken step (t < tau). Now pos & neg pools share the same
     history-cluster distribution, so cluster-id cannot act as a selection proxy. Does KMeans-8 cluster
     STILL add over champ61 here? If it collapses to ~0, the offset-200 gain was selection leakage.
"""
import os, warnings, time
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier

from _exp_fast import hist_context, HIST_CAP
from _night_banks import fit_champ_refs, champ_base_row, fit_dep_refs, dep_feats

DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
WINDOW = 250
OFFSET = 200
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


def crow(s, t):
    x = ON[s][:t + 1]; w = x[max(0, len(x) - WINDOW):]
    h = HI[s]; ctx = hist_context(h, HIST_CAP)
    return champ61(w, ctx, fit_champ_refs(h), fit_dep_refs(h))


usable = [s for s in TAU if len(HI[s]) > 40 and len(ON[s]) > 60]
sid2row = {s: i for i, s in enumerate(usable)}
Hraw = np.array([hist_vec(HI[s]) for s in usable])
scaler = StandardScaler().fit(Hraw)
Hstd = scaler.transform(Hraw)
km = KMeans(8, n_init=10, random_state=0).fit(Hstd)
cents = km.cluster_centers_


def clublock(sid):
    hv = Hstd[sid2row[sid]]
    d = np.linalg.norm(cents - hv, axis=1)
    oh = np.zeros(8); oh[d.argmin()] = 1.0
    return np.concatenate([oh, d])


cv = StratifiedKFold(5, shuffle=True, random_state=0)


def auc(F, yy):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
                                       random_state=0)
    oof = cross_val_predict(m, np.nan_to_num(F), yy, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(yy, oof)


# ---- T2: history -> break-membership ----
br_all = [s for s in usable if TAU[s] >= 15]
nb_all = [s for s in usable if TAU[s] < 0]
rng.shuffle(br_all); rng.shuffle(nb_all)
m = min(len(br_all), len(nb_all), 4000)
Hm = np.vstack([Hstd[[sid2row[s] for s in br_all[:m]]], Hstd[[sid2row[s] for s in nb_all[:m]]]])
ym = np.r_[np.ones(m), np.zeros(m)]
print(f"(T2) predict has-break vs no-break from HISTORY alone: AUC={auc(Hm, ym):.4f}  (0.5=independent)  m={m}/class")

# ---- T1: same-population break-detection (neg = early not-yet-broken window of break series) ----
br = [s for s in usable if TAU[s] >= 120 and len(ON[s]) > TAU[s] + OFFSET]
rng.shuffle(br); br = br[:N_EACH]
t0 = time.time()
Pc, Nc, Ps, Ns = [], [], [], []
for s in br:
    tp = TAU[s] + OFFSET
    tn = int(rng.integers(60, TAU[s] - 20))     # early, safely before the break
    Pc.append(crow(s, tp)); Ps.append(s)
    Nc.append(crow(s, tn)); Ns.append(s)
Xc = np.nan_to_num(np.vstack([Pc, Nc]))
y = np.r_[np.ones(len(Pc)), np.zeros(len(Nc))]
blk = np.vstack([[clublock(s) for s in Ps], [clublock(s) for s in Ns]])
b0 = auc(Xc, y)
b1 = auc(np.hstack([Xc, blk]), y)
print(f"(T1) same-series (pos=tau+{OFFSET}, neg=early t<tau): n={len(Pc)}/class built {time.time()-t0:.0f}s")
print(f"     champ61            AUC={b0:.4f}")
print(f"     champ61 + cluster  AUC={b1:.4f}   delta {b1-b0:+.4f}   (cluster-id identical for pos&neg)")

# ---- T3: vs-no-break design (reproduces the sweep) -> attribute the gain to SAMPLE selection ----
br3 = [s for s in usable if TAU[s] >= 15 and len(ON[s]) > TAU[s] + OFFSET]
nb3 = [s for s in usable if TAU[s] < 0]
rng.shuffle(br3); rng.shuffle(nb3); br3 = br3[:N_EACH]
nl = np.array([len(ON[s]) for s in nb3]); na = np.array(nb3)
Pc3, Nc3, Ps3, Ns3 = [], [], [], []
for s in br3:
    tp = TAU[s] + OFFSET
    elig = na[nl > tp]
    if len(elig) == 0:
        continue
    sn = int(rng.choice(elig))
    Pc3.append(crow(s, tp)); Ps3.append(s)
    Nc3.append(crow(sn, tp)); Ns3.append(sn)
Xc3 = np.nan_to_num(np.vstack([Pc3, Nc3]))
y3 = np.r_[np.ones(len(Pc3)), np.zeros(len(Nc3))]
blk3 = np.vstack([[clublock(s) for s in Ps3], [clublock(s) for s in Ns3]])
a0 = auc(Xc3, y3)
a1 = auc(np.hstack([Xc3, blk3]), y3)
aClu = auc(blk3, y3)                    # cluster block ALONE -> the selection proxy
print(f"(T3) vs-no-break (matched step, distinct series): n={len(Pc3)}/class")
print(f"     champ61            AUC={a0:.4f}")
print(f"     champ61 + cluster  AUC={a1:.4f}   delta {a1-a0:+.4f}   (reproduces sweep gain)")
print(f"     cluster block ALONE AUC={aClu:.4f}   (>0.5 = sampled break vs no-break histories differ = SELECTION artifact)")
