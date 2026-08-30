"""AR-RESIDUAL DENSITY-RATIO probe: a LEARNED 3rd member on the E78 ranker+rf base79 champion.

Origin: 2026-08-28. Target-side (E80 metric-weighted loss) and model-side (E63/E81 offset
specialists) re-weighting are both exhausted; the feature axis (E52-E79) is saturated. The one
remaining head-room the whole arc keeps pointing at is REPRESENTATION -- a learned matched-filter /
density-ratio over the residual DISTRIBUTION, attacking the early-detection cliff (clean-window AUC
~0.97 collapses to ~0.71 at 10 post-break points) and the skew/ma/tdist families the hand-engineered
kurt/skew features cap out on (~0.57 late-offset AUC). This probe is the smallest honest test of that
learned-representation fork.

MECHANISM. The base79 residual columns read the AR(8) innovations only through their log-variance
ratio (a single scalar per span/scale). They never look at the innovation distribution's SHAPE
(skew, tail asymmetry, higher order structure). This member does exactly that, and lets a small MLP
learn the discriminant instead of hand-coding a moment:

  per online step t, causal (history x_h + x_on[:t+1] only):
    1. fit AR(8) on the FULL history segment ONCE per series  (_fit_span(x_h, dict(p=8)) -> (p, m),
       m = (coef, resid_var); if m is None -> emit a zero embedding).
    2. take the trailing window w = x_on[max(0,t-WINDOW+1):t+1]  (the SAME slice base79 uses).
    3. residualize the window through the history AR(8):  r = _ar_resid(w, m[0], p)
       (the identical call _span_feats uses). If <8 residuals -> zero embedding.
    4. STANDARDIZE by the history residual scale:  r_std = r / sqrt(m[1] + 1e-12).
       The AR fit (with intercept) already removes mean + linear dependence, so dividing by the
       history residual sd makes the embedding invariant to x -> a*x + b for a > 0.
    5. EMBED as a fixed-length order-statistic vector: KQ evenly-spaced quantiles of r_std,
       q = np.quantile(r_std, linspace(0,1,KQ)). This "qrow" is an affine-invariant summary of the
       residual distribution SHAPE (captures the skew / tail / asymmetry the scalar kurt/skew miss).

  A small MLPClassifier (sklearn, no torch) is trained on (Qtr, ytr) -- the qrows aligned row-for-row
  with the base79 training rows Xtr -- and its per-step P(break | qrow) becomes a 3rd CDF-normalized
  member of the frozen ranker+rf blend.

LIKELIHOOD-RATIO / DENSITY-RATIO JUSTIFICATION. Under TS-AUC (a per-step cross-sectional rank metric)
the only thing that matters is the WITHIN-STEP ORDERING of series. The Bayes-optimal ranker of
"break vs no-break" given the residual-embedding e is the posterior p(break|e), and for a fixed prior
that posterior is a strictly monotone function of the density ratio p(e|break)/p(e|no-break) -- the
Neyman-Pearson matched filter. A classifier trained to separate break/no-break embeddings estimates
exactly this monotone-in-LR discriminant, which is precisely the object a rank metric needs. So a
LEARNED density-ratio on the residual-shape embedding is the theoretically right complement to the
variance-only residual columns, whereas a single moment (kurtosis/skew) is one fixed, low-power
projection of that ratio.

AFFINE-INVARIANCE (hard constraint). For x -> a*x + b (a>0): the AR fit with a constant column is
affine-equivariant, so the window residual scales r -> a*r and the history residual variance scales
m[1] -> a^2*m[1]; hence r_std = a*r / sqrt(a^2*m[1] + eps) = r / sqrt(m[1] + eps/a^2) -> r/sqrt(m[1])
to machine precision (the 1e-12 epsilon is the only perturbation). Quantiles of an invariant vector
are invariant, so the qrow is invariant to a,b. The SMOKE self-test builds a qrow for a random window
and for 3.7*w + 5.2 and asserts max|delta| < 1e-8.

WHY THIS IS GENUINELY DIFFERENT FROM THE DEAD LEARNED/RE-WEIGHT MEMBERS:
  * E45 / E47 (trajectory GRU): a stateful SEQUENCE model over the per-step 45-dim trajectory. It
    decorrelated on some data draws and anti-aligned on others (seed 42 +, seed 7 -), a real cross-pool
    disagreement no ensembling fixed. This probe is NOT a trajectory model -- it is a per-STEP,
    stateless classifier over a within-window order-statistic embedding; no temporal state, no
    per-series persistence to disagree about.
  * E80 (metric-weighted loss) / E81 (two-head early/late): both are RE-WEIGHTINGS of the SAME base79
    features / model (sample weights, head splits). They monotonically damaged the score because they
    starve the separable early-offset rows. This probe adds NO re-weighting -- it fits the frozen
    champion members unchanged (baseline at w_mdn=0 reproduces the champion EXACTLY) and introduces a
    NEW representation (the residual-distribution embedding) the base never reads.

BASE (always in, = E78 ranker+rf base79, byte-identical to the shipped CHAMPION):
  champ51 (AR8 full + AR8 trail120) + dep10 + ship12 + [AR8 trail60, AR8 trail200] x 3 scales.
  The ranker+rf halves are the imported _wloss build machinery (_row / _refs / fit_ranker_w /
  sort_into_groups_w). The RF is fit with NO sample_weight -> provably identical to the champion
  (matching the shipped notebook + _rshape_rankerrf). NOTE: this is NOT identical to _wloss gamma=0,
  which passes an explicit all-ones weight vector -- sklearn RF routes ones through a different
  (weighted-multiplicity vs bootstrap) path under min_samples_leaf=20, so _wloss gamma=0 is itself a
  slightly-off baseline (0.5431 smoke) while this probe reproduces the true champion (0.5384 smoke).
  Only the 3rd member and the blend arm are new.

BLEND (3-way fixed-CDF percentile, single arm w_mdn in [0,1)):
  wr = W_RANK*(1-w_mdn); wf = W_RF*(1-w_mdn); wm = w_mdn
  blend = wr*pct(srk,ref_rk) + wf*pct(srf,ref_rf) + wm*pct(smdn,ref_mdn)
  w_mdn = 0  ==  the 2-member champion (baseline), byte-identical.
Arms: w_mdn in {0.10, 0.20, 0.30}.

BELIEVE-RULE (low prior ~15%; the learned-representation fork mostly died -- E45/E47 GRU, E31 CNN).
Believe an arm only if mean_d >= +0.0015 AND positive on all screen seeds (npos == total), then
confirm on fresh seeds {31337,2025,555,99}. A learned member that only helps on some seeds is E45 all
over again -> no ship.

Uses its OWN cache dir _armdn_cache (the sd dict carries extra Q matrices, so it is NOT compatible
with the shared _struct79_cache). Do NOT delete _struct79_cache. RUN ONE 15-job harness at a time.
Run:
  SMOKE=1 PYTHONWARNINGS=ignore .venv/bin/python -u _armdn_rankerrf.py      # tiny self-test + affine check
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _armdn_rankerrf.py \
      > _armdn_rankerrf.out 2>&1 & disown
"""
import math
import os
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, load_pool_strat, select_picks
from _exp_fast import HIST_CAP, hist_context, split_series
from _overnight import MIN_N, N_JOBS
from _residual_overnight import _active
from _residual_reffit import _fit_ar, _ar_resid  # _fit_ar imported for import-parity (used via _fit_span)
# base79 build machinery imported so the ranker+rf half is provably identical to the champion.
from _wloss_rankerrf import (W_RANK, W_RF, RF_TREES, RF_KW, pct, _proba, _auc,
                             make_folds_meta, sort_into_groups_w, fit_ranker_w,
                             _san, _row, _fit_span, _offsets,
                             _refs as _refs_base, build_train as _build_train_base,
                             SHIP_SCALES, NCOL)

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]

DENSE_K = 20 if SMOKE else 40           # ranker rows per series (matches _wloss_rankerrf)
KQ = 32                                 # quantile-embedding width of the residual-shape qrow
MLP_MAX_ITER = 60 if SMOKE else 300
MLP_KW = dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3)
W_MDN_ARMS = [0.20] if SMOKE else [0.10, 0.20, 0.30]

AR_P = 8                                # residual-lens AR order (== the base residual lens)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_armdn_cache")   # OWN cache (sd carries Q matrices)
RAW = os.path.join(HERE, "_armdn_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_armdn_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_armdn_rankerrf.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# --- extended per-series refs: base79 refs + a full-history AR8 fit for the residual embedding ---
def _refs_ext(x_h):
    r = _refs_base(x_h)
    r["ar8_hist"] = _fit_span(x_h, dict(p=AR_P))     # (p, m) with m = (coef, resid_var) or None
    return r


# --- the learned member's per-step embedding (affine-invariant residual-shape quantile vector) ---
def _qrow(w, ctx, refs):
    """KQ evenly-spaced quantiles of the history-standardized AR(8) window residuals (a zero vector
    for degenerate cases). `ctx` is unused (kept for signature parity with _row)."""
    p, m = refs["ar8_hist"]
    if m is None:
        return np.zeros(KQ, dtype=np.float64)
    r = _ar_resid(np.asarray(w, dtype=np.float64), m[0], p)
    if r is None or len(r) < 8:
        return np.zeros(KQ, dtype=np.float64)
    r_std = r / math.sqrt(m[1] + 1e-12)              # affine-invariant (AR removes a,b; sd removes scale)
    return np.quantile(r_std, np.linspace(0.0, 1.0, KQ)).astype(np.float64)


def _qsan(a):
    """Sanitize a stack of qrows to a finite (n, KQ) matrix (mirrors _san but at KQ width)."""
    if not len(a):
        return np.zeros((0, KQ))
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


# --- extended train/val builders that ALSO emit qrows aligned row-for-row with the base79 rows ---
def _train_series_q(sid, x_h, x_on, tau, steps):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return np.zeros((0, NCOL)), [], [], [], np.zeros((0, KQ))
    refs = _refs_ext(x_h)
    base = active[0]
    rows, labels, tvals, qrows = [], [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        w = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(_row(w, ctx, refs))              # base79 row (identical to _wloss)
        qrows.append(_qrow(w, ctx, refs))            # aligned residual-shape embedding
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return _san(rows), labels, tvals, [int(sid)] * len(labels), _qsan(qrows)


def _val_series_q(sid, x_h, x_on, oi):
    ctx = hist_context(x_h, HIST_CAP)
    refs = _refs_ext(x_h)
    active = _active(x_on)
    rows, qrows = [], []
    for t in active:
        w = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(_row(w, ctx, refs))
        qrows.append(_qrow(w, ctx, refs))
    feat = _san(rows)
    q = _qsan(qrows)
    midx = pd.MultiIndex.from_product([[sid], oi], names=["id", "time"])
    return midx, active, feat, q


def build_train_q(X, yi, sid_to_steps):
    """base79 Xtr + row-aligned Qtr (the ranker rows Rr are built separately, no qrows needed)."""
    jobs = []
    for sid, steps in sid_to_steps.items():
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64),
                     int(yi.loc[sid, "tau_index"]), np.asarray(steps)))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_train_series_q)(sid, x_h, x_on, tau, steps)
            for (sid, x_h, x_on, tau, steps) in jobs)
    keep = [p for p in parts if len(p[0])]
    if not keep:
        return (np.zeros((0, NCOL)), np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(0, np.int64), np.zeros((0, KQ)))
    feat = np.vstack([r for r, _l, _t, _s, _q in keep])
    lab = np.concatenate([l for _r, l, _t, _s, _q in keep])
    tv = np.concatenate([t for _r, _l, t, _s, _q in keep])
    sids = np.concatenate([s for _r, _l, _t, s, _q in keep])
    Q = np.vstack([q for _r, _l, _t, _s, q in keep])
    return feat, lab.astype(np.int64), tv.astype(np.int64), sids.astype(np.int64), Q


def build_val_q(X, fold_ids):
    """Fold cache dict = the _wloss schema (index/feat/active) PLUS a row-aligned 'q' matrix."""
    jobs = [(sid, np.asarray(split_series(X.loc[sid])[0], dtype=np.float64),
             split_series(X.loc[sid])[1].to_numpy(dtype=np.float64),
             split_series(X.loc[sid])[1].index) for sid in fold_ids]
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series_q)(sid, x_h, x_on, oi) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, qs, active_all, offset = [], [], [], [], 0
    for midx, active, feat, q in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, NCOL)))
        qs.append(q if q.shape[0] else np.zeros((0, KQ)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCOL))
    q = np.vstack(qs) if qs else np.zeros((0, KQ))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active, "q": q}


def prepare_seed(seed):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"seed_{seed}{'_smoke' if SMOKE else ''}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    ET.SEED = seed
    X, yi, y_val, train_ids, folds = load_pool_strat()
    picks = select_picks(X, train_ids)
    rng = np.random.default_rng(seed + 202)
    rank_steps = {}
    for sid in train_ids:
        L = len(split_series(X.loc[sid])[1].to_numpy())
        if L <= MIN_N:
            continue
        cand = np.arange(MIN_N, L)
        rank_steps[sid] = rng.choice(cand, size=min(DENSE_K, len(cand)), replace=False)
    Xtr, ytr, Ttr, Str, Qtr = build_train_q(X, yi, picks)
    Rr, Rl, Rt, Rs = _build_train_base(X, yi, rank_steps)   # ranker rows (base79, no qrows)
    tau_map = {int(s): int(yi.loc[s, "tau_index"]) for s in train_ids}
    off_tr = _offsets(Ttr, Str, tau_map)
    off_r = _offsets(Rt, Rs, tau_map)
    caches = [build_val_q(X, f) for f in folds]
    sd = dict(Xtr=Xtr, ytr=ytr, Ttr=Ttr, off_tr=off_tr, Qtr=Qtr,
              Rr=Rr, Rl=Rl, Rt=Rt, off_r=off_r, caches=caches, y_val=y_val)
    with open(path, "wb") as f:
        pickle.dump(sd, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  [cache] seed {seed} built in {time.time()-t0:.0f}s  "
        f"Xtr{Xtr.shape} Qtr{Qtr.shape} Rr{Rr.shape} -> {path}")
    return sd


# --- fit the 3 members ONCE per seed, then evaluate every w_mdn arm from cached percentiles -------
def prep_scores(sd, metas, seed):
    """Fit ranker+rf (uniform weights == champion) + the learned MLP member, and return per-fold
    (active, groups, prk, prf, pmdn) where prk/prf/pmdn are the fixed-CDF percentiles at full length."""
    Xtr, ytr = sd["Xtr"], sd["ytr"]

    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=N_JOBS, random_state=seed, **RF_KW)
    rf.fit(Xtr, ytr)                                  # NO sample_weight -> identical to shipped champion
    ref_rf = np.sort(_proba(rf, Xtr))

    rows, lab, grp, _swo = sort_into_groups_w(sd["Rr"], sd["Rl"], sd["Rt"], None)
    rk = fit_ranker_w("rank_xendcg", rows, lab, grp, None)
    ref_rk = np.sort(_proba(rk, sd["Rr"]))

    scaler = StandardScaler().fit(sd["Qtr"])
    Qtr_s = scaler.transform(sd["Qtr"])
    clf = MLPClassifier(random_state=seed, max_iter=MLP_MAX_ITER, **MLP_KW)
    with threadpool_limits(limits=1):
        clf.fit(Qtr_s, ytr)
    ref_mdn = np.sort(_proba(clf, Qtr_s))

    folds = []
    for (L, active, groups), c in zip(metas, sd["caches"]):
        prk = np.zeros(L)
        prf = np.zeros(L)
        pmdn = np.zeros(L)
        if len(c["feat"]):
            srf = np.zeros(L)
            srk = np.zeros(L)
            smdn = np.zeros(L)
            srf[active] = _proba(rf, c["feat"])
            srk[active] = _proba(rk, c["feat"])
            smdn[active] = _proba(clf, scaler.transform(c["q"]))
            prf = pct(srf, ref_rf)
            prk = pct(srk, ref_rk)
            pmdn = pct(smdn, ref_mdn)
        folds.append((active, groups, prk, prf, pmdn))
    return folds


def score_wmdn(folds, w_mdn):
    """3-way fixed-CDF blend at arm w_mdn; w_mdn=0 reproduces the 2-member champion exactly."""
    wr = W_RANK * (1.0 - w_mdn)
    wf = W_RF * (1.0 - w_mdn)
    wm = w_mdn
    aucs = []
    for _active, groups, prk, prf, pmdn in folds:
        blend = wr * prk + wf * prf + wm * pmdn
        wsum = tot = 0.0
        for r, lb, w in groups:
            a = _auc(lb, blend[r])
            if a is None:
                continue
            wsum += w * a
            tot += w
        aucs.append(wsum / tot if tot > 0 else 0.5)
    return float(np.mean(aucs))


def _affine_check():
    """qrow(x) must equal qrow(a*x+b) for a>0 to machine precision (asserts max|delta| < 1e-8)."""
    rng = np.random.default_rng(0)
    xh = rng.standard_normal(600) * 3.0 + 5.0
    for i in range(2, len(xh)):                       # mild AR structure so the fit is non-trivial
        xh[i] += 0.3 * xh[i - 1] - 0.1 * xh[i - 2]
    w = xh[-260:].copy()
    ctx = hist_context(xh, HIST_CAP)
    q1 = _qrow(w, ctx, _refs_ext(xh))
    a, b = 3.7, 5.2
    xh2, w2 = a * xh + b, a * w + b
    ctx2 = hist_context(xh2, HIST_CAP)
    q2 = _qrow(w2, ctx2, _refs_ext(xh2))
    md = float(np.max(np.abs(q1 - q2)))
    nz = int(np.sum(np.abs(q1) > 1e-9))
    log(f"[affine-check] qrow maxdiff under x->{a}x+{b}: {md:.2e}  (nonzero {nz}/{KQ})")
    assert md < 1e-8, f"AFFINE INVARIANCE FAILED: max|delta|={md:.2e} >= 1e-8 (standardization is wrong)"
    return md


def main():
    open(LOG, "w").close()
    log(f"=== AR-RESIDUAL DENSITY-RATIO 3rd-member probe on ranker+rf base79  "
        f"{datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
        f"ship_scales={SHIP_SCALES}")
    log(f"NCOL={NCOL} KQ={KQ} mlp={MLP_KW} max_iter={MLP_MAX_ITER}  arms={W_MDN_ARMS}")
    _affine_check()

    if not os.path.exists(RAW):
        with open(RAW, "w") as f:
            f.write("seed\tarm\tbase_auc\tcand_auc\tdelta\n")

    per_seed = {}
    for seed in SEARCH_SEEDS:
        sd = prepare_seed(seed)
        metas = make_folds_meta(sd)
        t0 = time.time()
        folds = prep_scores(sd, metas, seed)
        base_auc = score_wmdn(folds, 0.0)
        log(f"[seed {seed}] base79 (w_mdn=0) ranker+rf = {base_auc:.4f}  ({time.time()-t0:.0f}s)")
        res = {}
        for w_mdn in W_MDN_ARMS:
            tc = time.time()
            auc = score_wmdn(folds, w_mdn)
            d = auc - base_auc
            res[w_mdn] = d
            arm = f"wmdn{w_mdn:.2f}"
            with open(RAW, "a") as f:
                f.write(f"{seed}\t{arm}\t{base_auc:.6f}\t{auc:.6f}\t{d:+.6f}\n")
            log(f"[seed {seed}] {arm} {auc:.4f}  d={d:+.4f}  ({time.time()-tc:.0f}s)")
        per_seed[seed] = res

    agg = []
    for w_mdn in W_MDN_ARMS:
        ds = [per_seed[s][w_mdn] for s in SEARCH_SEEDS if w_mdn in per_seed[s]]
        if not ds:
            continue
        md = float(np.mean(ds))
        npos = sum(1 for d in ds if d > 0)
        agg.append((f"wmdn{w_mdn:.2f}", md, min(ds), max(ds), npos, len(ds)))
    agg.sort(key=lambda r: -r[1])
    with open(SUMMARY, "w") as f:
        f.write(f"# AR-residual density-ratio 3rd-member probe on ranker+rf base79  "
                f"{datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
                f"ship={SHIP_SCALES} KQ={KQ} mlp={MLP_KW} max_iter={MLP_MAX_ITER}\n")
        f.write(f"# 3rd member = MLP on the history-standardized AR8 window-residual quantile embedding\n")
        f.write(f"# (affine-invariant density-ratio matched filter); w_mdn=0 == the 2-member champion.\n")
        f.write(f"# BELIEVE only if mean_d>=+0.0015 AND all-seeds-positive, then confirm on fresh seeds\n")
        f.write(f"# {{31337,2025,555,99}}. Prior LOW: learned-rep members E45/E47/E31 died.\n\n")
        f.write(f"{'arm':<12}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for arm, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md >= 0.0015 and npos == tot) else ""
            f.write(f"{arm:<12}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
