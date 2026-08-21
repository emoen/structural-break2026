"""History-STABILITY as SOFT features (not a hard trim): does telling the model how stable each
series' history is add over champ61? A stability scalar is STATIC per series, so we test BOTH:
  T3 (vs no-break, distinct series)      -> KNOWN to inflate via selection bias (clustering lesson)
  T1 (same series, neg = early t<tau)    -> HONEST: static feat can only help via INTERACTION with
                                            the time-varying online contrast; no selection possible.
Only a positive T1 delta counts. STAB block = sup-F mean/var break strength on history, break recency,
last-stable-segment ratio, AR8 logVR null-band width (the champion's existing studentizer, exposed),
and log history length.
"""
import os, warnings, time
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier

from _exp_fast import hist_context, HIST_CAP
from _night_banks import fit_champ_refs, champ_base_row, fit_dep_refs, dep_feats, _chow_supf

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


def champ61(w, ctx, cref, dref):
    base = champ_base_row(w, ctx, cref)
    dep = np.asarray(dep_feats(w, ctx, dref), dtype=np.float64)
    return np.concatenate([base, dep[DEP_KEEP]])


def crow(s, t):
    x = ON[s][:t + 1]; w = x[max(0, len(x) - WINDOW):]
    h = HI[s]; ctx = hist_context(h, HIST_CAP)
    return champ61(w, ctx, fit_champ_refs(h), fit_dep_refs(h))


def stab_block(s):
    h = HI[s]
    t_sf, v_sf, pos_sf = _chow_supf(h)
    dref = fit_dep_refs(h)
    null_sd = dref["null"][1]
    stable_ratio = (1.0 - pos_sf) if t_sf > 4.0 else 1.0
    return np.array([t_sf, v_sf, pos_sf, stable_ratio, null_sd, np.log(len(h) + 1.0)])


usable = [s for s in TAU if len(HI[s]) > 40 and len(ON[s]) > 60]
STAB = {s: stab_block(s) for s in usable}
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def auc(F, yy):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
                                       random_state=0)
    oof = cross_val_predict(m, np.nan_to_num(F), yy, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(yy, oof)


# ---- T1: same-series (pos=broken tau+OFFSET, neg=early t<tau) -> HONEST modulation test ----
br = [s for s in usable if TAU[s] >= 120 and len(ON[s]) > TAU[s] + OFFSET]
rng.shuffle(br); br = br[:N_EACH]
t0 = time.time()
Pc, Nc, Ps, Ns = [], [], [], []
for s in br:
    tp = TAU[s] + OFFSET
    tn = int(rng.integers(60, TAU[s] - 20))
    Pc.append(crow(s, tp)); Ps.append(s)
    Nc.append(crow(s, tn)); Ns.append(s)
Xc = np.nan_to_num(np.vstack([Pc, Nc]))
y = np.r_[np.ones(len(Pc)), np.zeros(len(Nc))]
blk = np.vstack([[STAB[s] for s in Ps], [STAB[s] for s in Ns]])
b0 = auc(Xc, y); b1 = auc(np.hstack([Xc, blk]), y)
print(f"(T1 honest) n={len(Pc)}/class built {time.time()-t0:.0f}s")
print(f"   champ61            AUC={b0:.4f}")
print(f"   champ61 + stability AUC={b1:.4f}   delta {b1-b0:+.4f}   <-- the verdict (interaction only)")

# ---- T3: vs no-break (distinct series) -> expected to inflate via selection (for contrast) ----
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
    Pc3.append(crow(s, tp)); Ps3.append(s); Nc3.append(crow(sn, tp)); Ns3.append(sn)
Xc3 = np.nan_to_num(np.vstack([Pc3, Nc3]))
y3 = np.r_[np.ones(len(Pc3)), np.zeros(len(Nc3))]
blk3 = np.vstack([[STAB[s] for s in Ps3], [STAB[s] for s in Ns3]])
a0 = auc(Xc3, y3); a1 = auc(np.hstack([Xc3, blk3]), y3)
aS = auc(blk3, y3)
print(f"(T3 vs-nobreak) n={len(Pc3)}/class  champ61={a0:.4f}  +stability={a1:.4f}  delta {a1-a0:+.4f}")
print(f"   stability block ALONE AUC={aS:.4f}  (>0.5 = selection artifact, NOT deployable)")
