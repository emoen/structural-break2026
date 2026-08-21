"""Late-offset estimator bake-off: does pooling MORE post-break samples beat the
champion's trailing-250 cap at high offset?

For target offsets k, build balanced pos (break series scored at t=tau+k) vs neg
(no-break series scored at a matched absolute step t). Compare per-offset AUC of the
same mean-shift feature under different windows:
  trail250  : |mean_z| on trailing 250 (champion proxy)
  trailBig  : |mean_z| on trailing 500
  expand    : |mean_z| on ALL online 0..t
  localize  : CUSUM-argmax change point mhat, pool x[mhat:], studentize post mean
Also variance-ratio analogues for the winner window. Random=0.5.
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

vals = X["value"].to_numpy(np.float64)
per = X["period"].to_numpy(np.int64)
ids = X.index.get_level_values("id").to_numpy()
uniq, start = np.unique(ids, return_index=True)
ends = np.append(start[1:], len(ids))
ON = {}   # online values per series
HI = {}   # historical values per series
for u, s, e in zip(uniq, start, ends):
    p = per[s:e]; v = vals[s:e]
    HI[int(u)] = v[p == 1]
    ON[int(u)] = v[p == 2]
TAU = {int(k): int(v) for k, v in yi["tau_index"].items()}

rng = np.random.default_rng(0)


def hist_stats(h):
    return float(h.mean()), float(h.std() + 1e-9), float(h.var() + 1e-9)


def mean_z(seg, hm, hs):
    if len(seg) < 5:
        return 0.0
    return abs((seg.mean() - hm) / (hs / np.sqrt(len(seg))))


def cusum_argmax(x):
    # argmax |cumulative deviation from global mean| -> single change-point estimate
    n = len(x)
    if n < 20:
        return n // 2
    c = np.cumsum(x - x.mean())
    m = int(np.argmax(np.abs(c[:-1]))) + 1
    return max(5, min(n - 5, m))


def score_set(x, hm, hs):
    """return dict of estimator scores at current end of online array x (0..t)."""
    n = len(x)
    out = {}
    out["trail250"] = mean_z(x[max(0, n - 250):], hm, hs)
    out["trail500"] = mean_z(x[max(0, n - 500):], hm, hs)
    out["expand"] = mean_z(x, hm, hs)
    mhat = cusum_argmax(x)
    out["localize"] = mean_z(x[mhat:], hm, hs)
    return out


EST = ["trail250", "trail500", "expand", "localize"]
OFFSETS = [30, 60, 100, 200, 300, 400]
N_EACH = 1200

br = [s for s in TAU if TAU[s] >= 15 and len(ON[s]) > TAU[s] + 5]
nb = [s for s in TAU if TAU[s] < 0 and len(ON[s]) > 40]
print(f"break-usable={len(br)}  nobreak={len(nb)}\n")

print(f"{'offset':>7} {'npos':>5} " + " ".join(f"{e:>9}" for e in EST))
for k in OFFSETS:
    pos_scores = {e: [] for e in EST}
    neg_scores = {e: [] for e in EST}
    pos_t = []
    # positives
    cand = [s for s in br if len(ON[s]) > TAU[s] + k]
    rng.shuffle(cand)
    cand = cand[:N_EACH]
    for s in cand:
        tau = TAU[s]; t = tau + k
        x = ON[s][:t + 1]
        hm, hs, hv = hist_stats(HI[s])
        sc = score_set(x, hm, hs)
        for e in EST:
            pos_scores[e].append(sc[e])
        pos_t.append(t)
    # negatives: no-break, scored at a t matched to the positive-t distribution
    rng.shuffle(nb)
    ti = 0
    for s in nb:
        if ti >= len(pos_t):
            break
        t = pos_t[ti]
        if len(ON[s]) <= t:
            continue
        x = ON[s][:t + 1]
        hm, hs, hv = hist_stats(HI[s])
        sc = score_set(x, hm, hs)
        for e in EST:
            neg_scores[e].append(sc[e])
        ti += 1
    npos = min(len(pos_t), len(neg_scores[EST[0]]))
    aucs = []
    for e in EST:
        y = np.r_[np.ones(len(pos_scores[e])), np.zeros(len(neg_scores[e]))]
        sarr = np.r_[pos_scores[e], neg_scores[e]]
        aucs.append(roc_auc_score(y, sarr))
    print(f"{k:>7} {npos:>5} " + " ".join(f"{a:9.3f}" for a in aucs))
