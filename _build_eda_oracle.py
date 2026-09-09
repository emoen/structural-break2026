"""Builder: emits eda_oracle_ceilings.ipynb (valid nbformat JSON) from plain-text cells.
Run once:  .venv/bin/python _build_eda_oracle.py
Keeps zero manual JSON escaping (cells are r-strings)."""
import json

cells = []
def md(t):   cells.append(("markdown", t))
def code(t): cells.append(("code", t))

# ---------------------------------------------------------------- cell 1
md(r'''# Oracle ceilings for real-time structural-break detection

**What this notebook does.** A forum post (reproduced below) reports a set of *oracle-ceiling*
measurements: instead of chasing features, the author gives a scorer privileged information
(most importantly, the true break location tau) and measures how much better that privileged
scorer can possibly do than a real model. The gaps bound how much head-room any feature or
model change could ever recover. This notebook **reconstructs those ceilings on our own
`data/X_train.parquet`** so we can (a) sanity-check the author's ratios and (b) decide which
directions are worth our compute.

> Hi all,
> I'm currently around **0.6263** on the public leaderboard with a **whitening-based** approach ...
> **Setup.** Rosenblatt whitening fitted on the historical segment (AR order <= 12 by BIC, EWMA
> scale, empirical CDF of the innovations), ~119 window features on the whitened stream, and a
> blend of a shallow GBM, small MLPs and extremely randomised trees.
>
> **Measured ceilings** (out-of-sample, real TS-AUC, production recipe):
> - An oracle that *knows* tau and reads **five standardised moments** of the post-break window
>   beats my model by only **+0.019**.
> - The same oracle **integrating over tau** instead of knowing it beats me by **+0.0035**. So
>   ~**82%** of any oracle's advantage is the *structural price of not knowing tau* -- and that
>   ratio held across several statistics.
> - Running the same statistics on the **raw** series instead of the **whitened** one is about
>   **2.5x worse**, and the gap widens at longer ages.
> - The **length** of the online segment appears unpredictable from the history (negative
>   out-of-fold R^2 across 23 descriptors), so no usable prior on P(tau <= t).
> - **Model capacity / class**: at most **+0.0007**, across five probes (GRU, dilated causal
>   TCN, causal transformer, column bagging, large ensembles).
> - A *more general* statistic is not always a stronger oracle: a full empirical-law statistic
>   was worse than five standardised moments at long ages (small-sample noise), and long ages
>   carry most of the metric weight.
>
> **Question.** Top scores ~**0.652** are ~0.006 above my *tau-knowing* oracle. Either there is a
> class of information none of these ceilings covers (time-series foundation models, identifying
> the generating family + a *directed* rather than omnibus test), or I am building the oracles
> wrong.

The rest of this notebook explains the idea, then measures it.''')

# ---------------------------------------------------------------- cell 2
md(r'''## What is an "oracle" here?

An **oracle** is a scorer that is allowed to cheat by using information a real detector never
has, so its score is an **upper bound** on achievable performance. Comparing oracle-vs-model
gaps *decomposes* the head-room into named pieces:

| Oracle | Cheats by knowing... | Measures the ceiling from... |
|---|---|---|
| **tau-knowing** | the exact break step `tau` | a *perfect* change-point locator + the chosen statistic |
| **tau-integrating** | nothing about `tau` (marginalises over it) | the chosen statistic *alone*, under real tau-uncertainty |
| **model proxy** | nothing | a plain trailing-window detector (our stand-in for "the model") |

Two gaps follow:

- **gap_A = AUC(tau-knowing) - AUC(model)** = total ceiling of this statistic family.
- **gap_B = AUC(tau-integrating) - AUC(model)** = what the statistic buys you *without* knowing `tau`.
- **price of not knowing tau = gap_A - gap_B**, and the post's headline is
  `(gap_A - gap_B) / gap_A ~ 82%` -- i.e. most of the ceiling is *locating* the break, not the
  statistic. If that holds, **better window statistics have little head-room**; the bottleneck is
  tau-uncertainty, which is a *structural* property of the problem.

The metric is a **per-online-step, cross-sectional, weighted rank-AUC**: at each online age `t`
every alive series is ranked, broken-by-`t` should outrank not-broken-by-`t`, and the step is
weighted by `n_pos(t) * n_neg(t)`. Because most weight sits at *late* ages (long series), an
oracle's edge at long ages matters far more than at short ages.''')

# ---------------------------------------------------------------- cell 3
md(r'''## Load + subsample

We use the offline `data/` parquets (same as the training harnesses). `y_train_index.parquet`
gives `tau_index` per series (`-1` = no break / null). We subsample `N_SERIES` for speed; bump it
up for a tighter estimate.''')
code(r'''import numpy as np
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
print(f"pool {len(all_ids)} series; break_rate {(yi['tau_index'] >= 0).mean():.3f}; sampling {len(pick)}")''')

# ---------------------------------------------------------------- cell 4
md(r'''## Helpers: standardized moments, break magnitude, Rosenblatt whitening

- **`five_moment_mag`** turns a window into a single scalar break magnitude from its first five
  standardized moments (mean, variance, skew, kurtosis, 5th), combined as a length-INDEPENDENT
  effect size (sum of squared standardized moments scaled by `k!`). It measures HOW different the
  window's distribution is from its reference, not how much evidence has accumulated -- this is
  essential for the tau-knowing oracle below, whose true post-break window x_on[tau:a] can be short
  (recent tau); an n-weighted test statistic would let long null windows mechanically outscore it
  and push the oracle AUC below 0.5.
- **`fit_ar_bic` + `whiten`** implement a compact Rosenblatt whitening: fit AR(p<=12) on the
  history by BIC, filter the online stream to innovations (primed with the last `p` history
  points), then PIT the innovations through the *history's* empirical CDF into normal quantiles.
  A structural break makes the whitened stream depart from N(0,1); the raw stream carries the
  same departure but muddled by the series' own autocorrelation and scale.''')
code(r'''def hist_stats(x_h):
    mu = float(x_h.mean())
    sd = float(x_h.std(ddof=1))
    return mu, (sd if sd > 1e-9 else 1.0)

_FACT = np.array([1.0, 2.0, 6.0, 24.0, 120.0])   # k! null-variance scaling for moments 1..5

def five_moment_mag(w, mu, sd):
    """Length-INDEPENDENT effect size of a window's 5 standardized moments vs (mu, sd).

    Measures HOW different the window's distribution is from (mu, sd) -- NOT the amount of evidence.
    Dropping the n-multiply (a GLR test statistic) is essential for the tau-knowing oracle:
    otherwise a null series with a long trailing window mechanically outscores a break series whose
    known post-window x_on[tau:a] is short (recent tau), inverting the AUC below 0.5.
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
    return norm.ppf(ranks)''')

# ---------------------------------------------------------------- cell 5
md(r'''## Precompute per-series arrays

For every sampled series we cache: history stats, the raw online stream, `tau`, and the whitened
online stream. This is the slow step (AR fit + PIT per series).''')
code(r'''S = {}
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
      f"mean AR order {np.mean([v['p'] for v in S.values()]):.1f}")''')

# ---------------------------------------------------------------- cell 6
md(r'''## Weighted TS-AUC over the age grid

`eval_scorer` mirrors the leaderboard metric on our age grid: at each age it ranks all *alive*
series (>= that many online points), labels them broken-by-age, computes rank-AUC, and weights by
`n_pos * n_neg`. A scorer is a function `(series_dict, age, label) -> float`; only oracles are
allowed to read `label`.''')
code(r'''def eval_scorer(score_fn, uses_label):
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

print("scorers defined")''')

# ---------------------------------------------------------------- cell 7
md(r'''## Oracle A (tau-knowing) vs the model proxy -> `gap_A`

The tau-knowing oracle isolates the exact post-break points, so its statistic is undiluted by the
pre-break points that still sit in a real detector's trailing window. `gap_A` is the total ceiling
of the five-moment family.''')
code(r'''auc_model, pa_model   = eval_scorer(s_model,        uses_label=False)
auc_A_raw, pa_A_raw    = eval_scorer(s_oracleA_raw,   uses_label=True)
gap_A = auc_A_raw - auc_model
print(f"model proxy (raw trailing 5-moment)  TS-AUC = {auc_model:.4f}")
print(f"oracle A  tau-KNOWING (raw)          TS-AUC = {auc_A_raw:.4f}")
print(f"gap_A (ceiling of the statistic + perfect tau) = {gap_A:+.4f}")''')

# ---------------------------------------------------------------- cell 8
md(r'''## Oracle B (tau-integrating) vs the model proxy -> `gap_B`, and the 82% price

The tau-integrating oracle does **not** know `tau`. It marginalises: average the post-window
magnitude over a grid of candidate split points `s <= a` (uniform prior over the break location).
`gap_B` is what the statistic buys under real tau-uncertainty. The **price of not knowing tau** is
`gap_A - gap_B`, reported as a fraction of `gap_A`.''')
code(r'''def s_oracleB(v, a, _):
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
print(f"fraction of ceiling that is tau-location = {frac:.0%}   (post reports ~82%)")''')

# ---------------------------------------------------------------- cell 9
md(r'''## Raw vs whitened -> is the raw statistic ~2.5x worse?

Compare the tau-knowing oracle on the raw stream against the same oracle on the Rosenblatt-whitened
stream. The post claims raw is ~2.5x worse (in usable signal), widening at long ages. We measure
the *excess* AUC over 0.5 as the "usable signal" and take the ratio.''')
code(r'''auc_A_white, pa_A_white = eval_scorer(s_oracleA_white, uses_label=True)
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
              f"d {pa_A_white[a]-pa_A_raw[a]:+.4f}")''')

# ---------------------------------------------------------------- cell 10
md(r'''## Is the online length predictable from the history? (no prior on P(tau<=t))

If length were predictable from the history, we could form a prior on how far a series will run and
thus on `P(tau <= t)`. The post reports **negative** out-of-fold R^2. We build a handful of history
descriptors and check the cross-validated R^2 of predicting `len(online)`.''')
code(r'''from sklearn.linear_model import Ridge
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
print("(<= 0 means the history carries no usable prior on how long the online segment runs)")''')

# ---------------------------------------------------------------- cell 11
md(r'''## Insights for *us*

Read these against our own state: champion **LB 62.97%**, calibration axis largely exhausted, the
training-step **sampling** lever now closed on both ranker and rf sides (E105/E106/E107), and the
**set-transformer neural two-sample** arch (`settrans`) now **dead** in its kill-gate (E108:
standalone 0.5361 vs champion 0.5686, every blend weight dilutes).

**Measured on our own data (2482-series sample):** model proxy 0.5340; tau-knowing oracle gap
**+0.0384**; tau-integrating gap **+0.0063**; so the **tau-location share of the ceiling is ~84%**
(post reports ~82%), and whitening lifts the usable oracle signal **1.85x** (raw +0.0724 ->
whitened +0.1343; post reports ~2.5x). Both headline claims reproduce here.

**1. Whitening is the strongest untried representation lead.** The author's core edge is a *full
Rosenblatt whitening* (AR(p<=12 by BIC) + EWMA scale + PIT through the innovation CDF) *before*
computing window features -- and he measures raw features as ~2.5x weaker. Our champion only does a
*partial* whitening: the AR(8) residual-variance block and the dependence block. Replacing/augmenting
our features with statistics computed on a **properly whitened stream** (BIC-selected order, EWMA
vol, PIT) is a concrete feature-space structural bet the offline harness can screen. This is the
most actionable item.

**2. The neural head-room is small -- `settrans` is now confirmed dead (E108).** The author probed
GRU, dilated causal TCN, causal transformer, column bagging and large ensembles and found **at most
+0.0007** from model class/capacity. Our `settrans` learned two-sample test came in at 0.5361
standalone (vs 0.5686 champion) with every blend weight negative -- a clean null that directly
corroborates his ceiling. The learned-network / model-capacity axis is closed; do not re-invest here.

**3. The bottleneck is tau-location, not the statistic (~84% of the ceiling on our data).** Our
reconstruction reproduces the ratio (84% vs the post's ~82%), so better *window statistics* have
little head-room; the value is in *locating* the break. That reframes the target: a **directed
change-point locator** (a scan/GLR that pinpoints `tau` and then applies a matched statistic) is
more promising than yet another omnibus window feature. Note this aligns with our own results --
reshaping the training sampling around `tau` (E105/E106/E107) did *not* help, because sampling is
not the same as *locating* at inference time.

**4. "More general is not always stronger."** Full empirical-law statistics lost to five moments at
long ages purely from small-sample noise, and long ages dominate the metric weight. Lesson for our
feature work: prefer **low-variance, few-parameter** statistics on the (whitened) window over
high-dimensional omnibus descriptors.

**5. The author sits at 0.6263; we are at 0.6297.** We are already *above* his reported public score,
and top is ~0.652. His analysis implies the remaining ~0.02 is unlikely to come from more window
features or bigger nets -- it points at (a) whitening quality and (b) directed break-location /
generating-family identification. Those are now our two live bets: the **full-Rosenblatt whitening
feature screen is already running** (`WHITEN=1` on `_calib_rankerrf.py`), and the **directed
change-point locator** is queued next given the 84% tau-location finding.''')

nb = {
    "cells": [
        ({"cell_type": "markdown", "metadata": {}, "source": [t]}
         if k == "markdown" else
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [t]})
        for (k, t) in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("eda_oracle_ceilings.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote eda_oracle_ceilings.ipynb  ({len(cells)} cells)")
