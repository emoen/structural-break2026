"""Decompose the TS-AUC metric's pairwise WEIGHT across post-break offset k=t-tau.

The metric numerator = sum over absolute online step t of concordant (pos,neg) pairs.
Each positive occurrence (series s alive at t, t>=tau_s) contributes n_neg(t) pairs.
Bucketing those contributions by offset k=t-tau_s tells us how much of the metric's
total weight is recoverable at each offset -> where late-vs-early detection actually pays.

Overlays the clean fixed-window per-offset AUC (measured earlier) to turn weight into
'contribution' (W*AUC) and 'headroom' (W*(1-AUC))."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))

period = X["period"].to_numpy(np.int64)
ids = X.index.get_level_values("id").to_numpy()
uniq, start = np.unique(ids, return_index=True)
ends = np.append(start[1:], len(ids))
L = {}  # online length per series
for u, s, e in zip(uniq, start, ends):
    L[int(u)] = int((period[s:e] == 2).sum())
TAU = {int(k): int(v) for k, v in yi["tau_index"].items()}

series = [s for s in L if L[s] > 0]
maxL = max(L[s] for s in series)

# per absolute step t: n_pos(t), n_neg(t)
n_pos = np.zeros(maxL, dtype=np.int64)
n_alive = np.zeros(maxL, dtype=np.int64)
for s in series:
    Ls, tau = L[s], TAU.get(s, -1)
    n_alive[:Ls] += 1
    if tau >= 0 and tau < Ls:
        n_pos[tau:Ls] += 1
n_neg = n_alive - n_pos

# total metric denominator = sum_t n_pos(t)*n_neg(t)
D = float((n_pos.astype(np.float64) * n_neg.astype(np.float64)).sum())

# offset weight W(k) = sum over positive occurrences at offset k of n_neg(t)
W = np.zeros(maxL, dtype=np.float64)
for s in series:
    Ls, tau = L[s], TAU.get(s, -1)
    if tau < 0 or tau >= Ls:
        continue
    t_arr = np.arange(tau, Ls)
    off = t_arr - tau
    np.add.at(W, off, n_neg[t_arr].astype(np.float64))

assert abs(W.sum() - D) < 1e-6, (W.sum(), D)

# measured clean fixed-window per-offset AUC (anchor points), interpolate between
anchor_k = np.array([0, 1, 2, 5, 8, 10, 20, 40, 60, 80, 120, 160, 200])
anchor_a = np.array([0.692, 0.723, 0.742, 0.754, 0.757, 0.751, 0.734,
                     0.703, 0.685, 0.678, 0.659, 0.662, 0.644])
kk = np.arange(maxL)
auc = np.interp(kk, anchor_k, anchor_a)  # flat-extrapolate beyond 200 at 0.644

Wf = W / W.sum()
contrib = W * auc
head = W * (1.0 - auc)

print(f"series={len(series)}  maxL={maxL}  total pair-weight D={D:,.0f}\n")

buckets = [(0, 1), (1, 3), (3, 6), (6, 11), (11, 21), (21, 41), (41, 61),
           (61, 101), (101, 161), (161, 251), (251, maxL)]
print(f"{'offset':>12} {'wt%':>7} {'cumwt%':>7} {'AUC':>6} {'head%':>7} {'cumhead%':>8}")
cw = 0.0
ch = 0.0
Htot = head.sum()
for lo, hi in buckets:
    w = Wf[lo:hi].sum()
    h = head[lo:hi].sum() / Htot
    a = (W[lo:hi] * auc[lo:hi]).sum() / (W[lo:hi].sum() + 1e-12)
    cw += w
    ch += h
    lab = f"{lo}-{hi-1}"
    print(f"{lab:>12} {100*w:7.2f} {100*cw:7.2f} {a:6.3f} {100*h:7.2f} {100*ch:8.2f}")

print()
for thr in (20, 40, 60, 100, 160, 250):
    wpast = Wf[thr:].sum()
    hpast = head[thr:].sum() / Htot
    print(f"offset >= {thr:4d}:  weight={100*wpast:5.2f}%   headroom={100*hpast:5.2f}%")
