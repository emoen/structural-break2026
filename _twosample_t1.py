"""Grind: two-sample DISTRIBUTIONAL distances as causal window features over champ61, re-run
under the leak-safe T1 protocol (E21 evaluated these in the old confounded framework).
Mechanistically orthogonal to the champion's moment/F-statistic contrasts: these look at the
whole distribution shift, not just mean/variance. All are DYNAMIC (trailing window), so T1
(same-series, neg=early not-yet-broken window) is leak-safe AND rewards genuine signal.

  ws_hist  : online-window vs HISTORY full-distribution distances -> Wasserstein-1, Energy,
             MMD(RBF, median-heuristic), Cramer-von-Mises (integrated squared CDF diff).
  ws_split : within-window sup-k distributional change-point -> max_k Wasserstein-1(w[:k],w[k:])
             + split recency (a distribution-based analogue of sup-F, no Gaussian assumption).
Only a positive T1 delta counts; if meaningful, escalate to the full multi-seed ship gate.
"""
import os, math, warnings, time
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier

from _exp_fast import hist_context, HIST_CAP
from _night_banks import fit_champ_refs, champ_base_row, fit_dep_refs, dep_feats

DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
WINDOW = 250
OFFSET = 200
N_EACH = 2400
CAP = 200          # subsample cap for O(n^2) energy/MMD
EPS = 1e-12

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
_sub = np.random.default_rng(12345)


def champ61(w, ctx, cref, dref):
    base = champ_base_row(w, ctx, cref)
    dep = np.asarray(dep_feats(w, ctx, dref), dtype=np.float64)
    return np.concatenate([base, dep[DEP_KEEP]])


def _cap(a):
    return a if len(a) <= CAP else a[_sub.choice(len(a), CAP, replace=False)]


def wasser1(a, b):
    qs = np.linspace(0.02, 0.98, 49)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def energy(a, b):
    a = _cap(a); b = _cap(b)
    dab = np.abs(a[:, None] - b[None, :]).mean()
    daa = np.abs(a[:, None] - a[None, :]).mean()
    dbb = np.abs(b[:, None] - b[None, :]).mean()
    return float(max(2 * dab - daa - dbb, 0.0))


def mmd_rbf(a, b):
    a = _cap(a); b = _cap(b)
    z = np.concatenate([a, b])
    d = np.abs(z[:, None] - z[None, :])
    med = np.median(d[d > 0]) + EPS
    g = 1.0 / (2 * med * med)
    na, nb = len(a), len(b)
    Kaa = np.exp(-g * (a[:, None] - a[None, :]) ** 2)
    Kbb = np.exp(-g * (b[:, None] - b[None, :]) ** 2)
    Kab = np.exp(-g * (a[:, None] - b[None, :]) ** 2)
    t1 = (Kaa.sum() - np.trace(Kaa)) / (na * (na - 1))
    t2 = (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1))
    return float(max(t1 + t2 - 2 * Kab.mean(), 0.0))


def cvm2(a, b):
    grid = np.sort(np.concatenate([a, b]))
    sa = np.sort(a); sb = np.sort(b)
    Fa = np.searchsorted(sa, grid, side="right") / len(a)
    Fb = np.searchsorted(sb, grid, side="right") / len(b)
    return float(np.mean((Fa - Fb) ** 2))


def sup_split_wass(z):
    n = len(z)
    if n < 40:
        return 0.0, 0.0
    lo, hi = max(15, n // 6), n - max(15, n // 6)
    ks = np.arange(lo, hi, max(1, (hi - lo) // 20))
    best, bpos = 0.0, 0.0
    for k in ks:
        d = wasser1(z[:int(k)], z[int(k):])
        if d > best:
            best, bpos = d, k / n
    return float(best), float(bpos)


def twosample_block(w, h):
    m, s = h.mean(), h.std() + 1e-9
    hz = (h - m) / s; wz = (w - m) / s
    z = (w - w.mean()) / (w.std() + 1e-9)
    ss = sup_split_wass(z)
    return np.array([wasser1(hz, wz), energy(hz, wz), mmd_rbf(hz, wz), cvm2(hz, wz),
                     ss[0], ss[1]])


def crow(s, t, with_block=False):
    x = ON[s][:t + 1]; w = x[max(0, len(x) - WINDOW):]
    h = HI[s]; ctx = hist_context(h, HIST_CAP)
    cref = fit_champ_refs(h); dref = fit_dep_refs(h)
    c61 = champ61(w, ctx, cref, dref)
    if with_block:
        return c61, twosample_block(w, h)
    return c61


usable = [s for s in TAU if len(HI[s]) > 40 and len(ON[s]) > 60]
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def auc(F, yy):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
                                       random_state=0)
    oof = cross_val_predict(m, np.nan_to_num(F), yy, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(yy, oof)


def build(pairs):
    C, B, Y = [], [], []
    for s, t, lab in pairs:
        c, b = crow(s, t, with_block=True)
        C.append(c); B.append(b); Y.append(lab)
    return np.nan_to_num(np.vstack(C)), np.nan_to_num(np.vstack(B)), np.array(Y, float)


NAMES = ["ws_hist", "ws_split", "ALL"]
IDX = {"ws_hist": [0, 1, 2, 3], "ws_split": [4, 5], "ALL": list(range(6))}

# ---- T1 honest: pos=broken tau+OFFSET, neg=early not-yet-broken window of SAME series ----
br = [s for s in usable if TAU[s] >= 120 and len(ON[s]) > TAU[s] + OFFSET]
rng.shuffle(br); br = br[:N_EACH]
pairs = []
for s in br:
    pairs.append((s, TAU[s] + OFFSET, 1))
    pairs.append((s, int(rng.integers(60, TAU[s] - 20)), 0))
t0 = time.time()
Xc, Bl, y = build(pairs)
b0 = auc(Xc, y)
print(f"(T1 honest) n={int(y.sum())}/class built {time.time()-t0:.0f}s   champ61 AUC={b0:.4f}")
for nm in NAMES:
    a = auc(np.hstack([Xc, Bl[:, IDX[nm]]]), y)
    print(f"   champ61 + {nm:<9} AUC={a:.4f}   delta {a-b0:+.4f}")

# alone: ws block by itself (sanity on signal magnitude)
aw = auc(Bl[:, IDX["ALL"]], y)
print(f"   ws-block ALONE     AUC={aw:.4f}")

# ---- T1b LENGTH-MATCHED: both pos & neg are FULL 250-len windows of the SAME series;
#      neg is entirely PRE-break (t<tau) so any within-window change-point is genuine, not a
#      length/step-position artifact. Requires tau>=270 so a full pre-break window exists. ----
brm = [s for s in usable if TAU[s] >= 270 and len(ON[s]) > TAU[s] + OFFSET]
rng.shuffle(brm); brm = brm[:N_EACH]
pairsm = []
for s in brm:
    pairsm.append((s, TAU[s] + OFFSET, 1))                       # break at pos ~49/250
    pairsm.append((s, int(rng.integers(249, TAU[s] - 1)), 0))    # full 250 window, pre-break
Xcm, Blm, ym = build(pairsm)
bm = auc(Xcm, ym)
print(f"\n(T1b len-matched) n={int(ym.sum())}/class   champ61 AUC={bm:.4f}")
for nm in NAMES:
    a = auc(np.hstack([Xcm, Blm[:, IDX[nm]]]), ym)
    print(f"   champ61 + {nm:<9} AUC={a:.4f}   delta {a-bm:+.4f}")
awm = auc(Blm[:, IDX["ALL"]], ym)
print(f"   ws-block ALONE     AUC={awm:.4f}")

# ---- T3 contrast: vs no-break distinct series (selection-prone) ----
br3 = [s for s in usable if TAU[s] >= 15 and len(ON[s]) > TAU[s] + OFFSET]
nb3 = [s for s in usable if TAU[s] < 0]
rng.shuffle(br3); rng.shuffle(nb3); br3 = br3[:N_EACH]
nl = np.array([len(ON[s]) for s in nb3]); na = np.array(nb3)
pairs3 = []
for s in br3:
    tp = TAU[s] + OFFSET
    elig = na[nl > tp]
    if len(elig) == 0:
        continue
    pairs3.append((s, tp, 1)); pairs3.append((int(rng.choice(elig)), tp, 0))
Xc3, Bl3, y3 = build(pairs3)
a0 = auc(Xc3, y3)
print(f"\n(T3 vs-nobreak) n={int(y3.sum())}/class   champ61 AUC={a0:.4f}")
for nm in NAMES:
    a = auc(np.hstack([Xc3, Bl3[:, IDX[nm]]]), y3)
    print(f"   champ61 + {nm:<9} AUC={a:.4f}   delta {a-a0:+.4f}")
