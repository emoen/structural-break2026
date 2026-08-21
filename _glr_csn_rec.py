"""Grind: GLR + CSN + REC-CUSUM as causal window features over champ61.
These are DYNAMIC (trailing-window) detectors, so T1 (same-series, neg=early not-yet-broken window)
is both leak-safe AND able to reward genuine orthogonal signal. T3 (vs no-break) shown for contrast.

  GLR      : sup over interior splits k of the Gaussian joint mean+VARIANCE log-likelihood ratio
             (break-at-unknown-k vs one regime) -> max LR + split recency; on AR8 residuals & raw.
  CSN      : Chu-Stinchcombe-Nason fluctuation monitor -> max_j |S_j|/sqrt(j) with the expanding
             sqrt(t) boundary + endpoint + recency, on the history-standardized window.
  REC-CUSUM: Brown-Durbin-Evans recursive residuals (one-step-ahead, 1/sqrt(1+1/t) scaled) ->
             cumulative-sum max / sqrt(n), endpoint, recency.
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
from _residual_reffit import _ar_resid

DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
WINDOW = 250
OFFSET = 200
N_EACH = 2400
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


def champ61(w, ctx, cref, dref):
    base = champ_base_row(w, ctx, cref)
    dep = np.asarray(dep_feats(w, ctx, dref), dtype=np.float64)
    return np.concatenate([base, dep[DEP_KEEP]])


def glr_meanvar(v):
    """sup-k Gaussian joint mean+variance log-LR (break-at-k vs one regime)."""
    n = len(v)
    if n < 24:
        return 0.0, 0.0
    csum = np.cumsum(v); csq = np.cumsum(v * v)
    tot, totsq = csum[-1], csq[-1]
    vfull = max(totsq / n - (tot / n) ** 2, EPS)
    ll_full = -0.5 * n * math.log(vfull)
    lo, hi = max(6, n // 8), n - max(6, n // 8)
    ks = np.arange(lo, hi, max(1, (hi - lo) // 40))
    best, bpos = 0.0, 0.0
    for k in ks:
        m1 = csum[k - 1] / k; v1 = max(csq[k - 1] / k - m1 * m1, EPS)
        m2 = (tot - csum[k - 1]) / (n - k); v2 = max((totsq - csq[k - 1]) / (n - k) - m2 * m2, EPS)
        lr = ll_full - (-0.5 * (k * math.log(v1) + (n - k) * math.log(v2)))  # >=0
        if lr > best:
            best, bpos = lr, k / n
    return float(best), float(bpos)


def csn(z):
    """Chu-Stinchcombe-Nason fluctuation: max_j |S_j|/sqrt(j) + endpoint + recency."""
    n = len(z)
    if n < 8:
        return 0.0, 0.0, 0.0
    S = np.cumsum(z - z.mean())
    j = np.arange(1, n + 1)
    q = np.abs(S) / np.sqrt(j)
    im = int(np.argmax(q))
    return float(q[im]), float(im / n), float(abs(S[-1]) / math.sqrt(n))


def rec_cusum(z):
    """Brown-Durbin-Evans recursive-residual CUSUM: max|W_t|/sqrt(n) + endpoint + recency."""
    n = len(z)
    if n < 12:
        return 0.0, 0.0, 0.0
    csum = np.cumsum(z)
    mprev = csum[:-1] / np.arange(1, n)                 # running mean up to t-1
    rr = (z[1:] - mprev) / np.sqrt(1.0 + 1.0 / np.arange(1, n))
    s = np.std(rr) + EPS
    W = np.cumsum(rr) / (s * math.sqrt(n))
    im = int(np.argmax(np.abs(W)))
    return float(np.max(np.abs(W))), float(im / n), float(W[-1])


def detector_block(w, dref):
    z = (w - w.mean()) / (w.std() + 1e-9)
    coef, refvar = dref["ar8"]
    r = _ar_resid(w, coef, 8) if coef is not None else None
    if r is not None and len(r) >= 24:
        rs = r / (math.sqrt(refvar) + EPS)
        glr_r = glr_meanvar(rs)
    else:
        glr_r = (0.0, 0.0)
    glr_w = glr_meanvar(z)
    c = csn(z)
    rc = rec_cusum(z)
    return np.array([glr_r[0], glr_r[1], glr_w[0], glr_w[1], c[0], c[1], c[2], rc[0], rc[1], rc[2]])


def crow(s, t, with_block=False):
    x = ON[s][:t + 1]; w = x[max(0, len(x) - WINDOW):]
    h = HI[s]; ctx = hist_context(h, HIST_CAP)
    cref = fit_champ_refs(h); dref = fit_dep_refs(h)
    c61 = champ61(w, ctx, cref, dref)
    if with_block:
        return c61, detector_block(w, dref)
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


NAMES = ["glr_resid", "csn", "rec_cusum", "ALL"]
IDX = {"glr_resid": [0, 1, 2, 3], "csn": [4, 5, 6], "rec_cusum": [7, 8, 9],
       "ALL": list(range(10))}

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
    print(f"   champ61 + {nm:<10} AUC={a:.4f}   delta {a-b0:+.4f}")

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
    print(f"   champ61 + {nm:<10} AUC={a:.4f}   delta {a-a0:+.4f}")
