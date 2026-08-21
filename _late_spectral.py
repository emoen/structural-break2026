"""New representation probe: full PSD-shape DIVERGENCE (window vs history), at LATE offset,
with POOLED (change-point-localized) post-break samples that sharpen the spectrum.

Prior spectral work tested single scalar summaries (flatness/rolloff/centroid/entropy) on an
old base, screened globally -> noise. This tests (a) a whole-spectrum divergence contrast, (b)
specifically at late offset vs REAL no-break negatives, (c) whether pooling more post-break
samples helps the spectrum, and (d) whether it is ORTHOGONAL to the champion's live families
(var/ks/dep) -- i.e. does a small HGB over {live + spectral} beat {live} alone at late offset.
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

NB = 48  # common PSD grid bins


def psd_norm(a, nb=NB):
    a = a - a.mean()
    if len(a) < 8 or a.std() < 1e-9:
        return np.full(nb, 1.0 / nb)
    f = np.fft.rfft(a)
    p = (f.real ** 2 + f.imag ** 2)[1:]  # drop DC
    if p.sum() < 1e-12:
        return np.full(nb, 1.0 / nb)
    # resample onto nb bins on normalized freq [0,1)
    x = np.linspace(0, 1, len(p))
    xi = np.linspace(0, 1, nb)
    pi = np.interp(xi, x, p)
    pi = pi / (pi.sum() + 1e-12)
    return pi


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


def spectral_feats(w, h_psd):
    pw = psd_norm(w)
    l2 = float(np.sqrt(((np.log(pw + 1e-9) - np.log(h_psd + 1e-9)) ** 2).mean()))
    m = 0.5 * (pw + h_psd)
    kl = float(0.5 * (np.sum(pw * np.log((pw + 1e-12) / (m + 1e-12)))
                      + np.sum(h_psd * np.log((h_psd + 1e-12) / (m + 1e-12)))))  # JS divergence
    lo = slice(0, NB // 3); mid = slice(NB // 3, 2 * NB // 3); hi = slice(2 * NB // 3, NB)
    dlo = abs(pw[lo].sum() - h_psd[lo].sum())
    dhi = abs(pw[hi].sum() - h_psd[hi].sum())
    return [l2, kl, dlo, dhi]


def live_feats(w, h, hsorted, hac):
    lvr = abs(np.log((w.var() + 1e-9) / (h.var() + 1e-9)))
    return [lvr, ks(w, hsorted), abs(ac1(w) - hac)]


SPEC = ["psd_l2", "psd_js", "dlow", "dhigh"]
OFFSETS = [60, 200, 400]
N_EACH = 1400
br = [s for s in TAU if TAU[s] >= 15 and len(ON[s]) > TAU[s] + 5 and len(HI[s]) > 40]
nb = [s for s in TAU if TAU[s] < 0 and len(ON[s]) > 60 and len(HI[s]) > 40]
print(f"break={len(br)} nobreak={len(nb)}  (window: trailing-250)\n")


def build(k, win_mode):
    Pos_s, Neg_s, Pos_l, Neg_l, pos_t = [], [], [], [], []
    cand = [s for s in br if len(ON[s]) > TAU[s] + k]
    rng.shuffle(cand); cand = cand[:N_EACH]
    for s in cand:
        tau = TAU[s]; t = tau + k; x = ON[s][:t + 1]
        w = x[cusum_argmax(x):] if win_mode == "localize" else x[max(0, len(x) - 250):]
        h = HI[s]; hp = psd_norm(h); hs = np.sort(h); hac = ac1(h)
        Pos_s.append(spectral_feats(w, hp)); Pos_l.append(live_feats(w, h, hs, hac)); pos_t.append(t)
    rng.shuffle(nb); ti = 0
    for s in nb:
        if ti >= len(pos_t):
            break
        t = pos_t[ti]
        if len(ON[s]) <= t:
            continue
        x = ON[s][:t + 1]
        w = x[cusum_argmax(x):] if win_mode == "localize" else x[max(0, len(x) - 250):]
        h = HI[s]; hp = psd_norm(h); hs = np.sort(h); hac = ac1(h)
        Neg_s.append(spectral_feats(w, hp)); Neg_l.append(live_feats(w, h, hs, hac)); ti += 1
    n = len(Neg_s)
    return (np.array(Pos_s[:n]), np.array(Neg_s), np.array(Pos_l[:n]), np.array(Neg_l))


def hgb_oof(Xf, y):
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                         max_leaf_nodes=31, random_state=0)
    oof = cross_val_predict(hgb, Xf, y, cv=StratifiedKFold(4, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    return roc_auc_score(y, oof)


for win_mode in ("trail250", "localize"):
    print(f"===== window = {win_mode} =====")
    print(f"{'offset':>7} " + " ".join(f"{c:>7}" for c in SPEC)
          + f" {'live':>7} {'live+spec':>9}")
    for k in OFFSETS:
        Ps, Ns, Pl, Nl = build(k, win_mode)
        y = np.r_[np.ones(len(Ps)), np.zeros(len(Ns))]
        specA = [roc_auc_score(y, np.r_[Ps[:, j], Ns[:, j]]) for j in range(len(SPEC))]
        live_all = hgb_oof(np.vstack([Pl, Nl]), y)
        both = hgb_oof(np.vstack([np.hstack([Pl, Ps]), np.hstack([Nl, Ns])]), y)
        print(f"{k:>7} " + " ".join(f"{a:7.3f}" for a in specA)
              + f" {live_all:7.3f} {both:9.3f}")
    print()
