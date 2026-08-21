"""Second-order / volatility-clustering dependence probe at LATE offset vs REAL negatives.

Distinct from level-dependence (ac1) and spectral: captures change in the PERSISTENCE of
volatility (ARCH-type, autocorrelation of |residuals| / squared residuals) and the full
ACF-vector shape. Synth per-family analysis flagged garch/arma as the separable real families.
Tests trail250 vs localize, and orthogonality to the live families (var/ks/dep) via HGB.
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
vals = X["value"].to_numpy(np.float64); per = X["period"].to_numpy(np.int64)
ids = X.index.get_level_values("id").to_numpy()
uniq, start = np.unique(ids, return_index=True); ends = np.append(start[1:], len(ids))
ON, HI = {}, {}
for u, s, e in zip(uniq, start, ends):
    p = per[s:e]; v = vals[s:e]; HI[int(u)] = v[p == 1]; ON[int(u)] = v[p == 2]
TAU = {int(k): int(v) for k, v in yi["tau_index"].items()}
rng = np.random.default_rng(0)


def acf_vec(a, lags):
    if len(a) < lags + 5:
        return np.zeros(lags)
    a = a - a.mean(); d = (a * a).sum() + 1e-12
    return np.array([(a[:-L] * a[L:]).sum() / d for L in range(1, lags + 1)])


def vol_ac1(a):
    """lag-1 autocorrelation of |residual| = volatility clustering."""
    if len(a) < 6:
        return 0.0
    r = np.abs(a - a.mean()); r = r - r.mean(); d = (r * r).sum() + 1e-12
    return float((r[:-1] * r[1:]).sum() / d)


def arch_lm(a):
    """crude ARCH: lag-1 autocorr of squared residual."""
    if len(a) < 6:
        return 0.0
    r = (a - a.mean()) ** 2; r = r - r.mean(); d = (r * r).sum() + 1e-12
    return float((r[:-1] * r[1:]).sum() / d)


def ks(win, hsorted):
    ws = np.sort(win); n = len(ws)
    allv = np.concatenate([ws, hsorted])
    cw = np.searchsorted(ws, allv, side="right") / n
    ch = np.searchsorted(hsorted, allv, side="right") / len(hsorted)
    return float(np.max(np.abs(cw - ch)))


def ac1(a):
    if len(a) < 5:
        return 0.0
    a = a - a.mean(); d = (a * a).sum() + 1e-12
    return float((a[:-1] * a[1:]).sum() / d)


def cusum_argmax(x):
    n = len(x)
    if n < 20:
        return n // 2
    c = np.cumsum(x - x.mean()); m = int(np.argmax(np.abs(c[:-1]))) + 1
    return max(5, min(n - 5, m))


LAGS = 10


def dep2_feats(w, hac_vec, hvol, harch):
    dacf = float(np.sqrt(((acf_vec(w, LAGS) - hac_vec) ** 2).mean()))
    dvol = abs(vol_ac1(w) - hvol)
    darch = abs(arch_lm(w) - harch)
    return [dacf, dvol, darch]


def live_feats(w, h, hsorted, hac):
    return [abs(np.log((w.var() + 1e-9) / (h.var() + 1e-9))), ks(w, hsorted), abs(ac1(w) - hac)]


SPEC = ["dacf_l2", "dvol_ac1", "darch"]
OFFSETS = [60, 200, 400]
N_EACH = 1400
br = [s for s in TAU if TAU[s] >= 15 and len(ON[s]) > TAU[s] + 5 and len(HI[s]) > 40]
nb = [s for s in TAU if TAU[s] < 0 and len(ON[s]) > 60 and len(HI[s]) > 40]
print(f"break={len(br)} nobreak={len(nb)}\n")


def build(k, win_mode):
    Ps, Ns, Pl, Nl, pos_t = [], [], [], [], []
    cand = [s for s in br if len(ON[s]) > TAU[s] + k]
    rng.shuffle(cand); cand = cand[:N_EACH]
    for s in cand:
        tau = TAU[s]; t = tau + k; x = ON[s][:t + 1]
        w = x[cusum_argmax(x):] if win_mode == "localize" else x[max(0, len(x) - 250):]
        h = HI[s]; hv = acf_vec(h, LAGS); hs = np.sort(h); hac = ac1(h)
        Ps.append(dep2_feats(w, hv, vol_ac1(h), arch_lm(h)))
        Pl.append(live_feats(w, h, hs, hac)); pos_t.append(t)
    rng.shuffle(nb); ti = 0
    for s in nb:
        if ti >= len(pos_t):
            break
        t = pos_t[ti]
        if len(ON[s]) <= t:
            continue
        x = ON[s][:t + 1]
        w = x[cusum_argmax(x):] if win_mode == "localize" else x[max(0, len(x) - 250):]
        h = HI[s]; hv = acf_vec(h, LAGS); hs = np.sort(h); hac = ac1(h)
        Ns.append(dep2_feats(w, hv, vol_ac1(h), arch_lm(h)))
        Nl.append(live_feats(w, h, hs, hac)); ti += 1
    n = len(Ns)
    return np.array(Ps[:n]), np.array(Ns), np.array(Pl[:n]), np.array(Nl)


def hgb_oof(Xf, y):
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                         max_leaf_nodes=31, random_state=0)
    oof = cross_val_predict(hgb, Xf, y, cv=StratifiedKFold(4, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    return roc_auc_score(y, oof)


for win_mode in ("trail250", "localize"):
    print(f"===== window = {win_mode} =====")
    print(f"{'offset':>7} " + " ".join(f"{c:>9}" for c in SPEC) + f" {'live':>7} {'live+dep2':>9}")
    for k in OFFSETS:
        Ps, Ns, Pl, Nl = build(k, win_mode)
        y = np.r_[np.ones(len(Ps)), np.zeros(len(Ns))]
        specA = [roc_auc_score(y, np.r_[Ps[:, j], Ns[:, j]]) for j in range(len(SPEC))]
        live_all = hgb_oof(np.vstack([Pl, Nl]), y)
        both = hgb_oof(np.vstack([np.hstack([Pl, Ps]), np.hstack([Nl, Ns])]), y)
        print(f"{k:>7} " + " ".join(f"{a:9.3f}" for a in specA) + f" {live_all:7.3f} {both:9.3f}")
    print()
