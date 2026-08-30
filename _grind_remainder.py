"""Grind the remainder: Bai-Perron (2-break), batch Bayesian change-point, particle/Kalman filter
as causal window features over champ61. Same protocol as _glr_csn_rec.py:
  T1 honest  = pos: broken series at tau+OFFSET; neg: EARLY not-yet-broken window of the SAME series.
  T3 contrast= pos: broken at tau+OFFSET; neg: length-matched no-break distinct series (selection-prone).
Only a positive T1 delta counts; if meaningful -> escalate to the multi-seed ship gate.

  baiperron: 2-break DP-ish sup over interior grid -> incremental SSR gain of a 2nd break beyond 1
             (targets MULTIPLE breaks; champion's sup-F is single-break), + both break positions.
  bayescp  : offline single-changepoint conjugate-Gaussian log Bayes factor (evidence for a change),
             MAP change location, normalized posterior entropy (localisation certainty).
  kalman   : local-level Kalman filter (= exact particle filter for the linear-Gaussian model);
             standardized one-step innovations spike at a break -> max|inn|, recency, endpoint, cusum.
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


def _prefix(v):
    return np.concatenate([[0.0], np.cumsum(v)]), np.concatenate([[0.0], np.cumsum(v * v)])


def baiperron(v):
    """Incremental sup-F for a 2nd break beyond the best single break, + both positions."""
    n = len(v)
    if n < 40:
        return np.array([0.0, 0.0, 0.0, 0.0])
    cs, cq = _prefix(v)

    def ssr(a, b):
        cnt = b - a
        if cnt <= 1:
            return 0.0
        s = cs[b] - cs[a]; sq = cq[b] - cq[a]
        return max(sq - s * s / cnt, 0.0)

    tot = ssr(0, n)
    lo, hi = max(6, n // 10), n - max(6, n // 10)
    grid = np.arange(lo, hi, max(1, (hi - lo) // 30))
    best1, k1 = tot, 0
    for k in grid:
        r = ssr(0, k) + ssr(k, n)
        if r < best1:
            best1, k1 = r, int(k)
    best2, ka, kb = best1, k1, k1
    for i, a in enumerate(grid):
        s0a = ssr(0, a)
        for b in grid[i + 1:]:
            r = s0a + ssr(a, b) + ssr(b, n)
            if r < best2:
                best2, ka, kb = r, int(a), int(b)
    f1 = (tot - best1) / (best1 / n + EPS)          # 1-break gain (single-break sup-F proxy)
    f2 = (best1 - best2) / (best2 / n + EPS)         # incremental 2nd-break gain
    return np.array([f1, f2, ka / n, kb / n])


def bayescp(v):
    """Offline single-changepoint conjugate-Gaussian log Bayes factor + MAP loc + posterior entropy."""
    n = len(v)
    if n < 24:
        return np.array([0.0, 0.0, 0.0])
    z = (v - v.mean()) / (v.std() + 1e-9)
    cs, cq = _prefix(z)
    tau2 = 10.0

    def lml(a, b):
        cnt = b - a
        if cnt <= 0:
            return 0.0
        s = cs[b] - cs[a]; sq = cq[b] - cq[a]
        prec = cnt + 1.0 / tau2
        return -0.5 * sq + 0.5 * s * s / prec - 0.5 * math.log(prec * tau2)

    ml0 = lml(0, n)
    lo, hi = max(4, n // 12), n - max(4, n // 12)
    ks = np.arange(lo, hi)
    lps = np.array([lml(0, int(k)) + lml(int(k), n) for k in ks])
    m = lps.max()
    ev1 = m + math.log(np.exp(lps - m).sum()) - math.log(len(ks))   # evidence for 1-change
    logbf = ev1 - ml0
    post = np.exp(lps - m); post /= post.sum()
    kmap = ks[int(np.argmax(lps))] / n
    ent = -float(np.sum(post * np.log(post + EPS))) / math.log(len(ks))
    return np.array([logbf, kmap, ent])


def kalman(v):
    """Local-level Kalman (exact PF for linear-Gaussian): standardized innovations spike at a break."""
    n = len(v)
    if n < 12:
        return np.array([0.0, 0.0, 0.0, 0.0])
    z = (v - v.mean()) / (v.std() + 1e-9)
    q, r = 0.01, 1.0
    x, P = z[0], 1.0
    inn = np.zeros(n)
    for t in range(1, n):
        Pp = P + q
        S = Pp + r
        e = z[t] - x
        inn[t] = e / math.sqrt(S)
        K = Pp / S
        x = x + K * e
        P = (1.0 - K) * Pp
    a = np.abs(inn)
    im = int(np.argmax(a))
    cus = np.cumsum(inn) / math.sqrt(n)
    return np.array([float(a.max()), float(im / n), float(inn[-1]), float(np.max(np.abs(cus)))])


def detector_block(w):
    return np.concatenate([baiperron(w), bayescp(w), kalman(w)])


def crow(s, t, with_block=False):
    x = ON[s][:t + 1]; w = x[max(0, len(x) - WINDOW):]
    h = HI[s]; ctx = hist_context(h, HIST_CAP)
    cref = fit_champ_refs(h); dref = fit_dep_refs(h)
    c61 = champ61(w, ctx, cref, dref)
    if with_block:
        return c61, detector_block(w)
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


NAMES = ["baiperron", "bayescp", "kalman", "ALL"]
IDX = {"baiperron": [0, 1, 2, 3], "bayescp": [4, 5, 6], "kalman": [7, 8, 9, 10],
       "ALL": list(range(11))}


def main():
    # ---- T1 honest ----
    br = [s for s in usable if TAU[s] >= 120 and len(ON[s]) > TAU[s] + OFFSET]
    rng.shuffle(br); brc = br[:N_EACH]
    pairs = []
    for s in brc:
        pairs.append((s, TAU[s] + OFFSET, 1))
        pairs.append((s, int(rng.integers(60, TAU[s] - 20)), 0))
    t0 = time.time()
    Xc, Bl, y = build(pairs)
    b0 = auc(Xc, y)
    print(f"(T1 honest) n={int(y.sum())}/class built {time.time()-t0:.0f}s   champ61 AUC={b0:.4f}", flush=True)
    for nm in NAMES:
        a = auc(np.hstack([Xc, Bl[:, IDX[nm]]]), y)
        print(f"   champ61 + {nm:<10} AUC={a:.4f}   delta {a-b0:+.4f}", flush=True)

    # ---- T3 contrast (length-matched vs no-break) ----
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
    print(f"\n(T3 vs-nobreak) n={int(y3.sum())}/class   champ61 AUC={a0:.4f}", flush=True)
    for nm in NAMES:
        a = auc(np.hstack([Xc3, Bl3[:, IDX[nm]]]), y3)
        print(f"   champ61 + {nm:<10} AUC={a:.4f}   delta {a-a0:+.4f}", flush=True)


if __name__ == "__main__":
    main()
