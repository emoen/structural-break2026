"""Step 0: is there a drift/trend break signal in REAL data that the champion's
mean/variance contrasts miss? Real series have no family labels, so instead we test
the drift hypothesis directly: break vs no-break separability of a studentized
recent-slope detector, and whether it adds orthogonal signal over |mean_z|+|log_var_ratio|."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.getcwd()
X = pd.read_parquet("data/X_train.parquet")
yi = pd.read_parquet("data/y_train_index.parquet")

vals = X["value"].to_numpy(np.float64)
per = X["period"].to_numpy(np.int64)
ids = X.index.get_level_values("id").to_numpy()
uniq, start = np.unique(ids, return_index=True)
ends = np.append(start[1:], len(ids))
MAP = {int(u): (vals[s:e], per[s:e]) for u, s, e in zip(uniq, start, ends)}
TAU = {int(k): int(v) for k, v in yi["tau_index"].items()}

WINDOW = 250
KDRIFT = 60          # recent-slope span
NULL_STEP = 10
rng = np.random.default_rng(0)


def slope(seg):
    n = len(seg)
    if n < 5:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    return float((t @ (seg - seg.mean())) / (t @ t + 1e-12))


def detectors(h, w):
    hstd = h.std() + 1e-9
    mean_z = (w.mean() - h.mean()) / hstd
    lvr = np.log((w.var() + 1e-9) / (h.var() + 1e-9))
    k = min(KDRIFT, len(w))
    s_win = slope(w[-k:])
    # studentize recent slope vs the series' own historical sliding-window slope null
    sl = [slope(h[i:i + k]) for i in range(0, max(1, len(h) - k), NULL_STEP)]
    sl = np.asarray(sl) if len(sl) else np.array([0.0])
    z_drift = (s_win - sl.mean()) / (sl.std() + 1e-9)
    return abs(mean_z), abs(lvr), abs(z_drift)


def build(offset, n_each=1500):
    br = [s for s, t in TAU.items() if t >= 5]
    nb = [s for s, t in TAU.items() if t < 0]
    rng.shuffle(br); rng.shuffle(nb)
    rows, y = [], []
    used = 0
    for s in br:
        if used >= n_each:
            break
        v, p = MAP[s]
        h, on = v[p == 1], v[p == 2]
        tau = TAU[s]
        t = tau + offset
        if len(h) < 30 or t >= len(on) or t < 20:
            continue
        w = on[max(0, t - WINDOW + 1):t + 1]
        rows.append(detectors(h, w)); y.append(1); used += 1
    used = 0
    for s in nb:
        if used >= n_each:
            break
        v, p = MAP[s]
        h, on = v[p == 1], v[p == 2]
        if len(h) < 30 or len(on) < 40:
            continue
        t = int(rng.integers(20, len(on)))
        w = on[max(0, t - WINDOW + 1):t + 1]
        rows.append(detectors(h, w)); y.append(0); used += 1
    return np.asarray(rows), np.asarray(y)


def oof_auc(X, y):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                           max_leaf_nodes=31, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


print(f"{'offset':>6} {'n_br':>5} | {'|mean_z|':>8} {'|lvr|':>6} {'|z_drift|':>9} | "
      f"{'mv_pair':>7} {'mv+drift':>8}  {'increment':>9}")
for o in (10, 20, 40, 80):
    Xr, y = build(o)
    a_mz = roc_auc_score(y, Xr[:, 0])
    a_lv = roc_auc_score(y, Xr[:, 1])
    a_dr = roc_auc_score(y, Xr[:, 2])
    mv = oof_auc(Xr[:, :2], y)
    mvd = oof_auc(Xr, y)
    print(f"{o:>6} {int(y.sum()):>5} | {a_mz:>8.3f} {a_lv:>6.3f} {a_dr:>9.3f} | "
          f"{mv:>7.3f} {mvd:>8.3f}  {mvd - mv:>+9.3f}")
