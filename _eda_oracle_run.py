import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

rng = np.random.default_rng(42)
DATA      = "data"
N_SERIES  = 2500                 # subsample for runtime; raise for a tighter estimate
WINDOW    = 250                  # trailing buffer (matches champion)
MIN_N     = 15                   # warm-up
# online "ages" to evaluate on (1-based count of online points seen). Late ages carry most weight.
AGES      = [15, 20, 30, 45, 65, 90, 130, 180, 250, 350, 500]

yi = pd.read_parquet(f"{DATA}/y_train_index.parquet")          # tau_index, tau per id
X  = pd.read_parquet(f"{DATA}/X_train.parquet").sort_index()   # (id,time) -> period, value
all_ids = yi.index.to_numpy()
pick = rng.choice(all_ids, size=min(N_SERIES, len(all_ids)), replace=False)
print(f"pool {len(all_ids)} series; break_rate {(yi['tau_index'] >= 0).mean():.3f}; sampling {len(pick)}")

# ===== next cell =====

def hist_stats(x_h):
    mu = float(x_h.mean())
    sd = float(x_h.std(ddof=1))
    return mu, (sd if sd > 1e-9 else 1.0)

_FACT = np.array([1.0, 2.0, 6.0, 24.0, 120.0])   # k! null-variance scaling for moments 1..5

def five_moment_mag(w, mu, sd):
    """Length-INDEPENDENT effect size of a window's 5 standardized moments vs (mu, sd).

    Measures HOW different the window's distribution is from (mu, sd) -- NOT the amount of evidence.
    Dropping the earlier n-multiply (a GLR test statistic) is essential for the tau-knowing oracle:
    otherwise a null series with a long trailing window (n up to WINDOW) mechanically outscores a
    break series whose known post-window x_on[tau:a] is short (recent tau), inverting the AUC below
    0.5. As a pure effect size, the oracle that isolates the true post-break window can only help.
    """
    n = len(w)
    if n < 5:
        return 0.0
    z = (np.asarray(w, float) - mu) / sd
    m = np.array([z.mean(), (z**2).mean() - 1.0, (z**3).mean(),
                  (z**4).mean() - 3.0, (z**5).mean()])
    return float(np.sum(m * m / _FACT))

def fit_ar_bic(x, pmax=12):
    x = np.asarray(x, float); n = len(x); best = None
    for p in range(0, pmax + 1):
        if n <= p + 10:
            break
        if p == 0:
            coef = np.array([x.mean()]); resid = x - x.mean()
        else:
            Xd = np.column_stack([x[p - l:n - l] for l in range(1, p + 1)] + [np.ones(n - p)])
            coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
            resid = x[p:] - Xd @ coef
        mres = len(resid); rss = float(np.sum(resid**2)) + 1e-12
        bic = mres * np.log(rss / mres) + (p + 1) * np.log(mres)
        if best is None or bic < best[1]:
            best = (p, bic, coef)
    return best[0], best[2]

def _ar_resid(seq, p, coef):
    n = len(seq)
    if p == 0:
        return seq - coef[0]
    if n <= p:
        return np.array([])
    Xd = np.column_stack([seq[p - l:n - l] for l in range(1, p + 1)] + [np.ones(n - p)])
    return seq[p:] - Xd @ coef

def whiten(x_h, x_on, p, coef):
    """Rosenblatt-style whitened online stream (len == len(x_on)): AR innovations PIT->normal."""
    xh = np.asarray(x_h, float); xo = np.asarray(x_on, float)
    rh = _ar_resid(xh, p, coef)
    if len(rh) < 10:
        mu, sd = hist_stats(xh)
        return (xo - mu) / sd
    primed = np.concatenate([xh[-p:], xo]) if p > 0 else xo
    ro = _ar_resid(primed, p, coef)                 # length == len(xo)
    rh_sorted = np.sort(rh)
    ranks = np.searchsorted(rh_sorted, ro, side="right") / (len(rh_sorted) + 1.0)
    ranks = np.clip(ranks, 1e-4, 1 - 1e-4)
    return norm.ppf(ranks)

# ===== next cell =====

S = {}
for sid in pick:
    g = X.loc[sid]
    x_h = g.loc[g["period"] == 1, "value"].to_numpy(float)
    x_on = g.loc[g["period"] == 2, "value"].to_numpy(float)
    if len(x_on) < MIN_N or len(x_h) < 30:
        continue
    tau = int(yi.loc[sid, "tau_index"])       # -1 == null (no break)
    mu, sd = hist_stats(x_h)
    p, coef = fit_ar_bic(x_h)
    wh = whiten(x_h, x_on, p, coef)
    S[sid] = dict(x_h=x_h, x_on=x_on, tau=tau, mu=mu, sd=sd, wh=wh, p=p)

lens = np.array([len(v["x_on"]) for v in S.values()])
brk  = np.array([v["tau"] >= 0 for v in S.values()])
print(f"usable series {len(S)}  |  break_rate {brk.mean():.3f}  |  "
      f"online len median {int(np.median(lens))} p90 {int(np.percentile(lens,90))}  |  "
      f"mean AR order {np.mean([v['p'] for v in S.values()]):.1f}")

# ===== next cell =====

def eval_scorer(score_fn, uses_label):
    wsum = tot = 0.0
    per_age = {}
    for a in AGES:
        labs, scs = [], []
        for v in S.values():
            if len(v["x_on"]) < a:
                continue
            lab = 1 if (0 <= v["tau"] < a) else 0
            scs.append(score_fn(v, a, lab if uses_label else None))
            labs.append(lab)
        labs = np.asarray(labs); scs = np.asarray(scs, float)
        npos = int(labs.sum()); nneg = len(labs) - npos
        if npos == 0 or nneg == 0:
            continue
        auc = roc_auc_score(labs, scs); w = npos * nneg
        wsum += w * auc; tot += w; per_age[a] = auc
    return wsum / tot if tot else 0.5, per_age

def _trailing(v, a):
    lo = max(0, a - WINDOW)
    return v["x_on"][lo:a]

# model proxy: plain trailing-window 5-moment magnitude on the RAW stream (no tau knowledge)
def s_model(v, a, _):
    return five_moment_mag(_trailing(v, a), v["mu"], v["sd"])

# tau-KNOWING oracle (raw): if broken, read ONLY the true post-break window x_on[tau:a]
def s_oracleA_raw(v, a, lab):
    w = v["x_on"][v["tau"]:a] if lab == 1 else _trailing(v, a)
    return five_moment_mag(w, v["mu"], v["sd"])

# tau-KNOWING oracle (whitened): same, on the whitened stream (mu=0, sd=1)
def s_oracleA_white(v, a, lab):
    w = v["wh"][v["tau"]:a] if lab == 1 else v["wh"][max(0, a - WINDOW):a]
    return five_moment_mag(w, 0.0, 1.0)

print("scorers defined")

# ===== next cell =====

auc_model, pa_model   = eval_scorer(s_model,        uses_label=False)
auc_A_raw, pa_A_raw    = eval_scorer(s_oracleA_raw,   uses_label=True)
gap_A = auc_A_raw - auc_model
print(f"model proxy (raw trailing 5-moment)  TS-AUC = {auc_model:.4f}")
print(f"oracle A  tau-KNOWING (raw)          TS-AUC = {auc_A_raw:.4f}")
print(f"gap_A (ceiling of the statistic + perfect tau) = {gap_A:+.4f}")

# ===== next cell =====

def s_oracleB(v, a, _):
    lo = MIN_N
    if a - lo < 4:
        return five_moment_mag(_trailing(v, a), v["mu"], v["sd"])
    cand = np.unique(np.linspace(lo, a - 2, 12).astype(int))
    return float(np.mean([five_moment_mag(v["x_on"][s:a], v["mu"], v["sd"]) for s in cand]))

auc_B, pa_B = eval_scorer(s_oracleB, uses_label=False)
gap_B = auc_B - auc_model
price = gap_A - gap_B
frac  = price / gap_A if gap_A > 1e-9 else float("nan")
print(f"oracle B  tau-INTEGRATING (raw)      TS-AUC = {auc_B:.4f}")
print(f"gap_A (tau-knowing)  = {gap_A:+.4f}")
print(f"gap_B (tau-blind)    = {gap_B:+.4f}")
print(f"price of not knowing tau = gap_A - gap_B = {price:+.4f}")
print(f"fraction of ceiling that is tau-location = {frac:.0%}   (post reports ~82%)")

# ===== next cell =====

auc_A_white, pa_A_white = eval_scorer(s_oracleA_white, uses_label=True)
sig_raw   = auc_A_raw   - 0.5
sig_white = auc_A_white - 0.5
print(f"oracle A tau-knowing  RAW      TS-AUC = {auc_A_raw:.4f}  (signal {sig_raw:+.4f})")
print(f"oracle A tau-knowing  WHITENED TS-AUC = {auc_A_white:.4f}  (signal {sig_white:+.4f})")
print(f"whitened / raw usable-signal ratio = {sig_white / sig_raw:.2f}x   (post reports ~2.5x)")
print()
print("per-age AUC (raw -> whitened):")
for a in AGES:
    if a in pa_A_raw and a in pa_A_white:
        print(f"  age {a:4d}:  raw {pa_A_raw[a]:.4f}   white {pa_A_white[a]:.4f}   "
              f"d {pa_A_white[a]-pa_A_raw[a]:+.4f}")

# ===== next cell =====

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

def hist_descriptors(x_h):
    x = np.asarray(x_h, float); n = len(x)
    d = x - x.mean()
    sd = x.std() + 1e-9
    ac1 = float(np.dot(d[1:], d[:-1]) / (np.dot(d, d) + 1e-12))
    ac2 = float(np.dot(d[2:], d[:-2]) / (np.dot(d, d) + 1e-12))
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return [n, x.mean(), sd, ac1, ac2, float(q75 - q25),
            float(((x - x.mean())**3).mean() / sd**3),
            float(((x - x.mean())**4).mean() / sd**4 - 3.0),
            float(np.log(n))]

Xd = np.array([hist_descriptors(v["x_h"]) for v in S.values()])
yd = np.array([len(v["x_on"]) for v in S.values()], float)
pred = cross_val_predict(Ridge(alpha=1.0), Xd, yd, cv=5)
print(f"OOF R^2 predicting online length from {Xd.shape[1]} history descriptors = {r2_score(yd, pred):+.4f}")
print("(<= 0 means the history carries no usable prior on how long the online segment runs)")