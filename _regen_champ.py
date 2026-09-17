#%pip install crunch-cli lightgbm xgboost catboost onnxruntime skl2onnx onnx --upgrade --quiet --progress-bar off
#!crunch setup-notebook structural-break-real-time EaYXndGIvcQUzXnCXXMW4kgN


import os

# Keep BLAS / OpenMP single-threaded. infer() runs in several parallel
# processes, and single-row model predictions are dramatically faster without
# per-call thread-pool spin-up. Env vars only work if set before numpy loads
# (not guaranteed on the cloud), so infer() ALSO caps threads at call time via
# threadpool_limits below.
#os.environ.setdefault("OMP_NUM_THREADS", "1")
#os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
#os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
from collections import deque
from typing import Iterable, List, Optional, Tuple

import joblib
from joblib.externals.loky import get_reusable_executor
import numpy as np
import pandas as pd
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from scipy.stats import norm


import crunch

# Load the Crunch Toolings (data loader, local tester, submitter).
#crunch_tools = crunch.load_notebook()


# Load the data (only needed for local exploration / testing).
#train_data, test_data = crunch_tools.load_data()
#print(f"train series: {len(train_data)}  |  test series: {len(test_data)}")


#FEATURE_NAMES = [
#    "mean_z", "log_var_ratio", "ks", "skew_diff", "kurt_diff",
#    "ac1_diff", "ac2_diff", "iqr_ratio", "tail2", "tail3",
#    "zero_cross", "slope_norm", "spec_centroid_diff", "rng_ratio",
#    "spec_entropy",
#]


def _autocorr(a, lag):
    if len(a) <= lag:
        return 0.0
    x = a[:-lag]
    y = a[lag:]
    xm = x.mean()
    ym = y.mean()
    denom = math.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def _spectral_centroid(a):
    a = a - a.mean()
    sp = np.abs(np.fft.rfft(a))
    freqs = np.fft.rfftfreq(len(a))
    s = sp.sum()
    if s < 1e-12:
        return 0.0
    return float((freqs * sp).sum() / s)


def _skew_kurt(w, mean_w, pop_var):
    """Population skew and excess kurtosis from central moments (pure numpy)."""
    if pop_var < 1e-12:
        return 0.0, 0.0
    d = w - mean_w
    m3 = float(np.mean(d ** 3))
    m4 = float(np.mean(d ** 4))
    sk = m3 / (pop_var ** 1.5)
    ku = m4 / (pop_var ** 2) - 3.0
    return sk, ku


def _fast_ks(w_sorted, h_sorted):
    """Two-sample KS statistic from pre-sorted arrays (numpy searchsorted)."""
    h = len(h_sorted)
    n = len(w_sorted)
    allv = np.concatenate([h_sorted, w_sorted])
    cdf_h = np.searchsorted(h_sorted, allv, side="right") / h
    cdf_w = np.searchsorted(w_sorted, allv, side="right") / n
    return float(np.max(np.abs(cdf_h - cdf_w)))


def hist_context(x_h, hist_cap):
    """Precompute historical-segment statistics + a sorted sample for the KS test."""
    n = len(x_h)
    mu = float(x_h.mean()) if n else 0.0
    var = float(x_h.var(ddof=1)) if n > 1 else 1.0
    var = max(var, 1e-12)
    sd = math.sqrt(var)
    pop_var = float(x_h.var()) if n else 1.0
    sk, ku = _skew_kurt(x_h, mu, pop_var) if n > 2 else (0.0, 0.0)
    ac1 = _autocorr(x_h, 1)
    ac2 = _autocorr(x_h, 2)
    q25, q75 = np.quantile(x_h, [0.25, 0.75]) if n else (0.0, 0.0)
    iqr = float(q75 - q25)
    iqr = iqr if iqr > 1e-12 else 1e-12
    rng = float(x_h.max() - x_h.min()) if n else 1.0
    rng = rng if rng > 1e-12 else 1e-12
    sc = _spectral_centroid(x_h) if n > 4 else 0.0
    if n > hist_cap:
        stride = n // hist_cap
        sample = x_h[::stride][:hist_cap]
    else:
        sample = x_h
    sample_sorted = np.sort(np.asarray(sample, dtype=np.float64))
    return {
        "mu": mu, "sd": sd, "var": var, "skew": sk, "kurt": ku,
        "ac1": ac1, "ac2": ac2, "iqr": iqr, "rng": rng, "sc": sc,
        "sample_sorted": sample_sorted,
    }


def compute_features(w, ctx):
    """w: 1-D np.array trailing window. ctx: hist_context dict. -> feature list (len 15).

    Pure-numpy (no scipy.stats) so it is fast enough for per-step streaming inference.
    """
    n = len(w)
    mean_w = float(w.mean())
    var_w = max(float(w.var()), 1e-12)
    se = ctx["sd"] / math.sqrt(n)
    mean_z = (mean_w - ctx["mu"]) / max(se, 1e-8)
    log_var_ratio = math.log(var_w / ctx["var"])
    w_sorted = np.sort(w)
    ks = _fast_ks(w_sorted, ctx["sample_sorted"])
    sk, ku = _skew_kurt(w, mean_w, var_w)
    skew_diff = sk - ctx["skew"]
    kurt_diff = ku - ctx["kurt"]
    ac1_diff = _autocorr(w, 1) - ctx["ac1"]
    ac2_diff = _autocorr(w, 2) - ctx["ac2"]
    q25 = w_sorted[int(0.25 * (n - 1))]
    q75 = w_sorted[int(0.75 * (n - 1))]
    iqr_w = max(float(q75 - q25), 1e-12)
    iqr_ratio = iqr_w / ctx["iqr"]
    dev = np.abs(w - ctx["mu"]) / ctx["sd"]
    tail2 = float((dev > 2.0).mean())
    tail3 = float((dev > 3.0).mean())
    centered = w - ctx["mu"]
    zero_cross = float((np.diff(np.sign(centered)) != 0).mean()) if n > 1 else 0.0
    i = np.arange(n, dtype=np.float64)
    sx = i.sum()
    sxx = (i * i).sum()
    sy = w.sum()
    sxy = (i * w).sum()
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if abs(denom) > 1e-12 else 0.0
    slope_norm = slope * n / ctx["sd"]
    spec_centroid_diff = _spectral_centroid(w) - ctx["sc"]
    rng_ratio = float(w_sorted[-1] - w_sorted[0]) / ctx["rng"]
    # spectral entropy of the window (E13: robust +0.0050 TS-AUC @ WINDOW=130, t=+4.67)
    ps = np.abs(np.fft.rfft(w - mean_w)) ** 2
    ps_sum = ps.sum()
    if ps_sum < 1e-12 or len(ps) < 2:
        spec_entropy = 0.0
    else:
        p = ps / ps_sum
        p = p[p > 0]
        spec_entropy = float(-(p * np.log(p)).sum()) / math.log(len(ps))
    return [
        mean_z, log_var_ratio, ks, skew_diff, kurt_diff,
        ac1_diff, ac2_diff, iqr_ratio, tail2, tail3,
        zero_cross, slope_norm, spec_centroid_diff, rng_ratio,
        spec_entropy,
    ]


def compute_features_ms(w, ctx):
    """Multi-scale wrapper (E17 winner ms_60_130_250, 45 feats, 10-fold t=+3.06).

    Compute the base15 feature set on nested trailing sub-windows w[-s:] for each
    scale s in MS_SCALES and concatenate. Short scales are responsive; long scales
    are slow -> the model sees several views of the same series-vs-history contrast.
    Guarded so a scale larger than the current window just reuses the full window.
    """
    n = len(w)
    out = []
    for s in MS_SCALES:
        ws = w if s >= n else w[-s:]
        out.extend(compute_features(ws, ctx))
    return out


# @crunch/keep:on
WINDOW           = 250   # buffer = largest MS scale (E17 multi-scale)
MS_SCALES        = (40, 100, 250)  # winsearch t_40_100_250: inner scales 60/130 -> 40/100 (best mean +0.00076, 2/3 gamble)
MIN_N            = 15    # warm-up: emit 0.0 until this many online points seen
HIST_CAP         = 150   # max historical points sampled for the KS test
PREDICT_EVERY    = 1     # recompute features + prediction every N steps (1 = no staleness)
STEPS_PER_SERIES = 10    # sampled online steps per training series for the pointwise models
DENSE_K          = 60    # sampled online steps per series for the ranker (dense trajectory)
MAX_TRAIN_SERIES = 10**9 # consume ALL series the platform provides (no cap)
MAX_ITER         = 400   # E32 tuned HGB winner
SEED             = 42
RANK_SEED        = 42    # ranker model seed

# E116 promoted blend: white GLOC on ranker+RF only, then bounded 0.025 weight search.
# Weights are percentile-space (each member's live score -> its train-CDF percentile). Sum = 1.0.
BLEND = {
    "ranker":   0.55,
    "rf":       0.30,
    "xgb":      0.05,
    "catboost": 0.10,
    "hgb":      0.00,
}

# LGBM listwise ranker: E116 model-seed screen accepted the slower 500-tree recipe.
LGB_PARAMS = dict(
    objective="rank_xendcg", n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, reg_lambda=1.0, subsample=1.0, colsample_bytree=0.7,
    random_state=RANK_SEED, verbose=-1,
)

# Pointwise-member recipes. E116 retained RF/XGB and accepted CatBoost depth 8.
RF_PARAMS  = dict(n_estimators=800, max_features=0.5, min_samples_leaf=20)
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=1.0, min_child_weight=5)
CAT_PARAMS = dict(iterations=600, learning_rate=0.03, depth=8, l2_leaf_reg=10.0)
HGB_PARAMS = dict(max_iter=MAX_ITER, learning_rate=0.021145795969329348, max_leaf_nodes=63,
                  min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)


# @crunch/keep:on
# =====================================================================================
# Residual-variance-ratio feature block  (E53 multi-scale + E54 trailing-history reference)
# -------------------------------------------------------------------------------------
# A structural break shows up first as the REFERENCE model's residuals blowing up. Fit AR(p) on the
# historical segment, apply forward onto trailing SUB-windows of the online buffer, and use
# log( var(sub-window residuals) / var(reference-fit residuals) ) at each scale.
# E54: fit the reference on TWO history spans -- the FULL history AND only the LAST 120 points. The
# recent-history reference (dynamics closer to the break region) adds ORTHOGONAL signal on top of the
# full-history one. Fresh-confirmed +0.0303 (min_t 2.89, 4/4) vs +0.0232 for the full-history-only block.
# E75: add trail60 + trail200 spans (short+long recency pair). refspan mining on the ranker+rf 73-col
# base -> trail60+trail200 = +0.00282 mean_d, 3/3 seeds; LB-confirmed +0.11 pt (62.50 -> 62.61).
#
#   feats = |REF_TRAILS| * |AR_ORDERS| * |RESIDUAL_SCALES|   (min mode: var-ratio only)
# =====================================================================================
RESIDUAL_MODE    = "min"
AR_ORDERS        = (8,)                # the AR8 residual is the sharpest single reference
RESIDUAL_SCALES  = (40, 100, 250)      # winsearch t_40_100_250: mirrors MS_SCALES (was 60,130,250)
REF_TRAILS       = (None, 120, 60, 200)  # E54 full+120; E75 add trail60+trail200 (+0.00282, 3/3, refspan)
STEPS_PER_SERIES = 30                  # secondary confirmed lever; set back to 10 to disable


def _fit_ar(x, p):
    """AR(p) least-squares fit on an ordered series -> (coef[p+1], residual variance)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < p + 8:
        return None
    cols = [x[p - lag:n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
    return coef, float(np.var(x[p:] - Xd @ coef)) + 1e-12


def _ar_resid(w, coef, p):
    n = len(w)
    if n < p + 3:
        return None
    cols = [w[p - lag:n - lag] for lag in range(1, p + 1)]
    return w[p:] - np.column_stack(cols + [np.ones(n - p)]) @ coef


def _acf(x, k):
    if len(x) <= k:
        return 0.0
    x0 = x - x.mean()
    d = float(np.dot(x0, x0)) + 1e-12
    return float(np.dot(x0[k:], x0[:-k])) / d


def _fit_ar_ref(x_h):
    """AR fits per (trail-span, order). trail=None uses the full history."""
    x = np.asarray(x_h, dtype=np.float64)
    ref = {}
    for trail in REF_TRAILS:
        xr = x if trail is None else x[-trail:]
        ref[trail] = {p: _fit_ar(xr, p) for p in AR_ORDERS}
    return ref


def _resid_stats(r, refvar):
    rv = float(np.var(r)) + 1e-12
    varratio = math.log(rv / refvar)
    if RESIDUAL_MODE == "min":
        return [varratio]
    sig = math.sqrt(refvar)
    n = len(r)
    q = 0.0
    for k in (1, 2, 3):
        if n - k > 0:
            ak = _acf(r, k)
            q += ak * ak / (n - k)
    cs = np.cumsum(r) / (sig * math.sqrt(n))
    return [varratio, float(np.mean(r)) / math.sqrt(rv), _acf(r, 1),
            n * (n + 2) * q, float(np.max(np.abs(cs))), float(np.mean(np.abs(r) > 3.0 * sig))]


def _residual_feats(w, ar_ref):
    w = np.asarray(w, dtype=np.float64)
    per = 1 if RESIDUAL_MODE == "min" else 6
    out = []
    for trail in REF_TRAILS:
        for p in AR_ORDERS:
            m = ar_ref[trail].get(p)
            for s in RESIDUAL_SCALES:
                ws = w if s >= len(w) else w[-s:]
                r = _ar_resid(ws, m[0], p) if m is not None else None
                out += ([0.0] * per) if (r is None or len(r) < 5) else _resid_stats(r, m[1])
    return out


# Wrap the champion's context + feature builders so the residual block rides along everywhere.
# train() and infer() call hist_context / compute_features_ms BY NAME, so no other change is needed.
_base_hist_context = hist_context
_base_compute_features_ms = compute_features_ms


def hist_context(x_h, hist_cap):
    ctx = _base_hist_context(x_h, hist_cap)
    ctx["ar_ref"] = _fit_ar_ref(x_h)
    return ctx


def compute_features_ms(w, ctx):
    return _base_compute_features_ms(w, ctx) + _residual_feats(w, ctx["ar_ref"])


_n_extra = len(REF_TRAILS) * len(AR_ORDERS) * len(RESIDUAL_SCALES) * (1 if RESIDUAL_MODE == "min" else 6)
print(f"residual block: mode={RESIDUAL_MODE}, AR{AR_ORDERS} x trails{REF_TRAILS} x scales{RESIDUAL_SCALES} "
      f"-> +{_n_extra} feats; STEPS_PER_SERIES={STEPS_PER_SERIES}")


# @crunch/keep:on
# =============================================================================
# Dependence / early-onset residual block  (E62: EDA levers 1 + 2 -> b_no_accf)
# -----------------------------------------------------------------------------
# +10 causal features on top of the 51-col trail120 champion (fresh-confirmed +0.0066, 4/4):
#   [0] logvr        log AR8-residual variance ratio (window vs full-history reference)
#   [1] z_student    (logvr - hist_mean) / hist_sd  -- AR-studentized (specificity fix, lever 2)
#   [2..5] cusum_*   early-onset residual CUSUM  max / end / argmax-pos / drift  (lever 1)
#   [6] arch_contrast  lag-1 autocorr of squared residuals, window vs history (ARCH/BDS proxy)
#   [7..9] supf_*    sup-F / Chow max-split scan inside the window  t / |log var-ratio| / pos
# The 4 ACF-lag contrasts (dacf1/2/5/10) from the full 14-feat bank were noise -> DROPPED.
# =============================================================================
_HIST_NULL_WINS = 24          # historical windows used to estimate the per-series logVR null
_DEP_EPS = 1e-12
_DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]   # drop dacf1/2/5/10 (indices 6..9)


def _dep_autocorr(a, lag):
    if len(a) <= lag:
        return 0.0
    x = a[:-lag]
    y = a[lag:]
    xm = x.mean()
    ym = y.mean()
    denom = math.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def _dep_fit_ar(x, p):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < p + 8:
        return None
    cols = [x[p - lag:n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
    return coef, float(np.var(x[p:] - Xd @ coef)) + 1e-12


def _dep_ar_resid(w, coef, p):
    n = len(w)
    if n < p + 3:
        return None
    cols = [w[p - lag:n - lag] for lag in range(1, p + 1)]
    return w[p:] - np.column_stack(cols + [np.ones(n - p)]) @ coef


def _dep_fit_refs(x_h):
    x = np.asarray(x_h, dtype=np.float64)
    m8 = _dep_fit_ar(x, 8)
    coef, refvar = (m8[0], m8[1]) if m8 is not None else (None, 1.0)
    logvr = []
    arch_vals = []
    if coef is not None and len(x) >= 300:
        win = 250
        starts = np.linspace(0, len(x) - win, _HIST_NULL_WINS).astype(int)
        for s in starts:
            seg = x[s:s + win]
            r = _dep_ar_resid(seg, coef, 8)
            if r is None or len(r) < 20:
                continue
            logvr.append(math.log((float(np.var(r)) + _DEP_EPS) / refvar))
            arch_vals.append(_dep_autocorr(r * r, 1))
    null = (float(np.mean(logvr)), float(np.std(logvr) + _DEP_EPS)) if len(logvr) >= 3 else (0.0, 1.0)
    arch_hist = float(np.mean(arch_vals)) if arch_vals else 0.0
    hist_acf = [_dep_autocorr(x, lag) for lag in (1, 2, 5, 10)]
    return {"ar8": (coef, refvar), "null": null, "hist_acf": hist_acf, "arch_hist": arch_hist}


def _dep_chow_supf(v):
    n = len(v)
    if n < 24:
        return 0.0, 0.0, 0.0
    csum = np.cumsum(v)
    csq = np.cumsum(v * v)
    tot = csum[-1]
    lo, hi = max(6, n // 8), n - max(6, n // 8)
    ks = np.arange(lo, hi, max(1, (hi - lo) // 40))
    best_t, best_v, best_pos = 0.0, 0.0, 0.0
    for k in ks:
        m1 = csum[k - 1] / k
        m2 = (tot - csum[k - 1]) / (n - k)
        v1 = max(csq[k - 1] / k - m1 * m1, _DEP_EPS)
        v2 = max((csq[-1] - csq[k - 1]) / (n - k) - m2 * m2, _DEP_EPS)
        t = abs(m1 - m2) / math.sqrt(v1 / k + v2 / (n - k) + _DEP_EPS)
        lvr = abs(math.log(v2 / v1))
        if t > best_t:
            best_t, best_pos = t, k / n
        if lvr > best_v:
            best_v = lvr
    return best_t, best_v, best_pos


def _dep_feats_full(w, dref):
    w = np.asarray(w, dtype=np.float64)
    coef, refvar = dref["ar8"]
    out = []
    r = _dep_ar_resid(w, coef, 8) if coef is not None else None
    if r is not None and len(r) >= 8:
        logvr = math.log((float(np.var(r)) + _DEP_EPS) / refvar)
        m, s = dref["null"]
        z_student = (logvr - m) / s
        rs = r / (math.sqrt(refvar) + _DEP_EPS)
        surprise = rs * rs - 1.0
        cusum = np.cumsum(surprise)
        nn = len(cusum)
        cmax = float(np.max(cusum)) / math.sqrt(nn)
        cend = float(cusum[-1]) / math.sqrt(nn)
        cpos = float(np.argmax(cusum)) / nn
        cdrift = float(cusum[-1] - cusum[0]) / nn
        arch_w = _dep_autocorr(r * r, 1)
    else:
        logvr = z_student = cmax = cend = cpos = cdrift = arch_w = 0.0
    out += [logvr, z_student, cmax, cend, cpos, cdrift]
    for lag, h_ac in zip((1, 2, 5, 10), dref["hist_acf"]):
        out.append(_dep_autocorr(w, lag) - h_ac)
    out.append(arch_w - dref["arch_hist"])
    t_sf, v_sf, pos_sf = _dep_chow_supf(w)
    out += [t_sf, v_sf, pos_sf]
    return out


# Chain onto the residual block's wrappers -- train()/infer() call these BY NAME (late binding),
# so appending here needs no change to the training or inference code.
_dep_base_hist_context = hist_context
_dep_base_compute_features_ms = compute_features_ms


def hist_context(x_h, hist_cap):
    ctx = _dep_base_hist_context(x_h, hist_cap)
    ctx["dep_ref"] = _dep_fit_refs(x_h)
    return ctx


def compute_features_ms(w, ctx):
    base = _dep_base_compute_features_ms(w, ctx)
    df = _dep_feats_full(np.asarray(w, dtype=np.float64), ctx["dep_ref"])
    return base + [df[i] for i in _DEP_KEEP]


print(f"dependence block (b_no_accf): +{len(_DEP_KEEP)} feats "
      f"(studentized resid + early-onset CUSUM + arch contrast + sup-F; ACF-lag block dropped)")


# @crunch/keep:on
# =============================================================================
# Ship-feature block (2026-08-24): perm_entropy4 + max_split_lr + mosum
# -----------------------------------------------------------------------------
# 3-seed isolation on the ranker+rf blend: the three families TOGETHER add
# +0.00222 TS-AUC (t=5.88, 3/3 seeds) on top of the 61-col champion base. Each
# family is computed on the same MS_SCALES sub-windows as the champion base and
# appended (+12 cols = 3 families [perm4:1, maxlr:1, mosum:2] x 3 scales).
# Rides along via the compute_features_ms wrapper (late binding), so train() and
# infer() pick it up unchanged.
# =============================================================================
_SHIP_EPS = 1e-12
_SHIP_MSPLIT_MIN_SEG = 5


def _ship_perm_entropy4(w):
    n = len(w)
    if n < 8:
        return [0.0]
    counts = {}
    for i in range(n - 3):
        pat = tuple(np.argsort(w[i:i + 4]))
        counts[pat] = counts.get(pat, 0) + 1
    p = np.array(list(counts.values()), dtype=np.float64)
    p /= p.sum()
    return [float(-(p * np.log(p)).sum()) / math.log(24)]


def _ship_msplit_prefix(w):
    n = len(w)
    lo, hi = _SHIP_MSPLIT_MIN_SEG, n - _SHIP_MSPLIT_MIN_SEG
    if hi <= lo:
        return None
    c1 = np.cumsum(w)
    c2 = np.cumsum(w * w)
    tot1 = c1[-1]
    tot2 = c2[-1]
    ks = np.arange(lo, hi + 1, dtype=np.float64)
    kidx = ks.astype(np.int64) - 1
    s1l = c1[kidx]
    s2l = c2[kidx]
    nl = ks
    nr = n - ks
    s1r = tot1 - s1l
    s2r = tot2 - s2l
    ssl = np.maximum(s2l - s1l * s1l / nl, 0.0)
    ssr = np.maximum(s2r - s1r * s1r / nr, 0.0)
    return n, nl, nr, s1l, s1r, ssl, ssr


def _ship_max_split_lr(w):
    pf = _ship_msplit_prefix(w)
    if pf is None:
        return [0.0]
    n, nl, nr, s1l, s1r, ssl, ssr = pf
    var0 = max(float(np.sum(w * w) / n - (float(np.sum(w)) / n) ** 2), 1e-12)
    varl = np.maximum(ssl / nl, 1e-12)
    varr = np.maximum(ssr / nr, 1e-12)
    llr = 0.5 * (n * math.log(var0) - nl * np.log(varl) - nr * np.log(varr))
    return [float(np.max(np.maximum(llr, 0.0)))]


def _ship_mosum(w, ctx, k=10):
    z = (np.asarray(w, dtype=np.float64) - ctx["mu"]) / max(ctx["sd"], 1e-8)
    sq = z * z
    n = len(sq)
    if n < 2 * k:
        return [0.0, 1.0]
    c = np.concatenate([[0.0], np.cumsum(sq)])
    blocks = (c[k:] - c[:-k]) / k
    mx = float(blocks.max())
    mn = float(blocks.mean())
    return [math.log(mx + _SHIP_EPS), mx / max(mn, _SHIP_EPS)]


def _ship_feats(w, ctx):
    w = np.asarray(w, dtype=np.float64)
    out = []
    for s in MS_SCALES:
        ws = w if s >= len(w) else w[-s:]
        out.extend(_ship_perm_entropy4(ws))
        out.extend(_ship_max_split_lr(ws))
        out.extend(_ship_mosum(ws, ctx))
    return out


# Chain onto the dependence block's wrappers (late binding); train()/infer() call by name.
_ship_base_compute_features_ms = compute_features_ms


def compute_features_ms(w, ctx):
    return _ship_base_compute_features_ms(w, ctx) + _ship_feats(w, ctx)


_n_ship = len(MS_SCALES) * 4
print(f"ship feature block: perm_entropy4 + max_split_lr + mosum x {len(MS_SCALES)} scales -> +{_n_ship} feats")


# @crunch/keep:on
# =============================================================================
# Per-series CALIBRATION block  (czrb_all79: robust median/MAD studentize base79)
# -----------------------------------------------------------------------------
# Append a per-series calibration readout of the 79 base features: studentize each
# base feature against THAT series' OWN pre-break null (32 sliding windows of length
# 250 over the history; ROBUST median/MAD). The TS-AUC metric ranks series against
# each other every step, so a raw feature is not comparable across series (a noisy
# series has a wider null band); the studentized z makes a value mean "k MAD above
# THIS series' own null". Fresh-confirmed +0.00577 mean_d, 4/4 seeds (7/7 all seeds)
# on the base79 ranker+rf blend (_calib_rankerrf.py, arm czrb_all79).
# CAUSAL: the null is built from x_h (pre-break history) ONLY. AFFINE-INVARIANT: every
# base79 feature is affine-invariant and med/mad are affine-covariant, so z cancels
# a,b under x -> a*x+b to machine precision. base79 rides FIRST, byte-unchanged; the
# +79 calibration cols are appended AFTER (158 total). Rides along via the same
# compute_features_ms late binding; train()/infer() only add one ctx["calib"]= hook.
# =============================================================================
_CALIB_WIN       = 250      # null-window length (matches champion top scale)
_CALIB_NULL_WINS = 32       # sliding windows used to estimate the per-series null
_CALIB_EPS       = 1e-9
_CALIB_NCOL      = 79        # base79 contract width
_WBLOCK_N        = 8         # additive whitened readout width

# Capture the ship-wrapped base79 producer BEFORE we replace compute_features_ms.
_base79_producer = compute_features_ms


def _fit_ar_bic(x, pmax=12):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    best = None
    for p in range(0, pmax + 1):
        if n <= p + 10:
            break
        if p == 0:
            coef = np.array([x.mean()])
            resid = x - x.mean()
        else:
            Xd = np.column_stack([x[p - l:n - l] for l in range(1, p + 1)] + [np.ones(n - p)])
            coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
            resid = x[p:] - Xd @ coef
        mres = len(resid)
        rss = float(np.sum(resid ** 2)) + 1e-12
        bic = mres * np.log(rss / mres) + (p + 1) * np.log(mres)
        if best is None or bic < best[1]:
            best = (p, bic, coef)
    return best[0], best[2]


def _ar_resid_wh(seq, p, coef):
    seq = np.asarray(seq, dtype=np.float64)
    n = len(seq)
    if p == 0:
        return seq - coef[0]
    if n <= p:
        return np.array([])
    Xd = np.column_stack([seq[p - l:n - l] for l in range(1, p + 1)] + [np.ones(n - p)])
    return seq[p:] - Xd @ coef


def whiten_pair(x_h, x_on):
    xh = np.asarray(x_h, dtype=np.float64)
    xo = np.asarray(x_on, dtype=np.float64)
    if len(xh) < 30 or len(xo) < 3:
        return xh, xo
    loc = float(np.median(xh))
    scl = 1.4826 * float(np.median(np.abs(xh - loc))) + _CALIB_EPS
    xh = (xh - loc) / scl
    xo = (xo - loc) / scl
    p, coef = _fit_ar_bic(xh)
    rh = _ar_resid_wh(xh, p, coef)
    if len(rh) < 10:
        mu = float(xh.mean())
        sd = float(xh.std()) + _CALIB_EPS
        return (xh - mu) / sd, (xo - mu) / sd
    rh_sorted = np.sort(rh)

    def _pit(resid):
        ranks = np.searchsorted(rh_sorted, resid, side="right") / (len(rh_sorted) + 1.0)
        return norm.ppf(np.clip(ranks, 1e-4, 1 - 1e-4))

    wh_h = _pit(rh)
    primed = np.concatenate([xh[-p:], xo]) if p > 0 else xo
    wh_on = _pit(_ar_resid_wh(primed, p, coef))
    return wh_h, wh_on


def wblock_feats(wh):
    wh = np.asarray(wh, dtype=np.float64)
    n = len(wh)
    if n < 16:
        return [0.0] * _WBLOCK_N
    m1 = float(wh.mean())
    v = float(wh.var()) + _CALIB_EPS
    sd = math.sqrt(v)
    zc = np.clip((wh - m1) / sd, -8.0, 8.0)
    skew = float((zc ** 3).mean())
    kurt = float((zc ** 4).mean() - 3.0)
    fifth = float((zc ** 5).mean())
    h = n // 2
    a, bb = wh[:h], wh[h:]
    va = float(a.var()) + _CALIB_EPS
    vb = float(bb.var()) + _CALIB_EPS
    lvr = float(np.clip(math.log(vb / va), -20.0, 20.0))
    tsh = float(np.clip((bb.mean() - a.mean()) / math.sqrt(vb / len(bb) + va / len(a) + _CALIB_EPS), -50.0, 50.0))
    cs = np.cumsum(wh - m1)
    cusum = float(np.max(np.abs(cs)) / (sd * math.sqrt(n) + _CALIB_EPS))
    out = [m1, float(np.clip(v - 1.0, -50.0, 50.0)), float(np.clip(skew, -20.0, 20.0)),
           float(np.clip(kurt, -50.0, 50.0)), float(np.clip(fifth, -100.0, 100.0)), lvr, tsh,
           float(np.clip(cusum, 0.0, 50.0))]
    return [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in out]


def fit_calib_ref(x_h, ctx):
    """Per-series ROBUST null band of the 79 base features from sliding pre-break windows.

    Slides `_CALIB_NULL_WINS` windows of length `_CALIB_WIN` across `x_h`, computes the SAME
    ship-wrapped base79 row on each (window-vs-full-history, exactly as online), and stores the
    per-feature robust median / MAD. CAUSAL: uses x_h (history) only, never online points.
    Mirrors the robust branch of _calib_rankerrf.fit_calib_ref numerically (median, 1.4826*MAD
    + _EPS, the >=3-rows guard, the len>=_CALIB_WIN+10 guard, evenly spaced window starts).
    """
    x = np.asarray(x_h, dtype=np.float64)
    rows = []
    if len(x) >= _CALIB_WIN + 10:
        starts = np.linspace(0, len(x) - _CALIB_WIN, _CALIB_NULL_WINS).astype(int)
        for s in starts:
            rows.append(_base79_producer(x[s:s + _CALIB_WIN], ctx))
    if len(rows) >= 3:
        Z = np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        med = np.median(Z, axis=0)
        mad = 1.4826 * np.median(np.abs(Z - med), axis=0) + _CALIB_EPS
    else:
        med = np.zeros(_CALIB_NCOL)
        mad = np.ones(_CALIB_NCOL)
    return {"med": med, "mad": mad}


def compute_features_ms(w, ctx, wh=None):
    base = np.asarray(_base79_producer(w, ctx), dtype=np.float64)   # 79 cols, UNCHANGED
    cref = ctx.get("calib")
    if cref is None:
        z = base  # safety: no calib ref -> pass base through (not hit in train/infer)
    else:
        z = (base - cref["med"]) / cref["mad"]
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    wb = np.asarray(wblock_feats(wh), dtype=np.float64) if wh is not None else np.zeros(_WBLOCK_N)
    return np.concatenate([base, z, wb])


print(f"calibration block (czrb_all79 + wblock): +{_CALIB_NCOL} robust-studentized cols +{_WBLOCK_N} whitened cols "
      f"({_CALIB_NULL_WINS} windows x len {_CALIB_WIN}) -> {2 * _CALIB_NCOL + _WBLOCK_N} total cols")


# @crunch/keep:on
# E116: append the selected full-prefix history-whitened global locator.
# The first 166 columns remain byte-identical to the 63.17 champion.
_CHAMP_NCOL = 166
_WHITE_GLOC_N = 7
_FULL_NCOL = _CHAMP_NCOL + _WHITE_GLOC_N
MODEL_FEATURE_WIDTHS = {
    "ranker": _FULL_NCOL,
    "rf": _FULL_NCOL,
    "xgb": _CHAMP_NCOL,
    "catboost": _CHAMP_NCOL,
}
_champ166_producer = compute_features_ms


def _gloc_parts(w, min_fraction=0.1):
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    if n < 24:
        return None
    edge = max(6, int(n * min_fraction))
    lo, hi = edge, n - edge
    if hi <= lo:
        return None
    csum = np.cumsum(w)
    csq = np.cumsum(w * w)
    ks = np.arange(lo, hi)
    a = ks.astype(np.float64)
    b = (n - ks).astype(np.float64)
    m1 = csum[ks - 1] / a
    m2 = (csum[-1] - csum[ks - 1]) / b
    v1 = np.maximum(csq[ks - 1] / a - m1 * m1, _CALIB_EPS)
    v2 = np.maximum((csq[-1] - csq[ks - 1]) / b - m2 * m2, _CALIB_EPS)
    return w, n, ks, m1, m2, v1, v2


def white_gloc_feats(wh_full):
    """Gaussian likelihood locator on the complete causal prefix of whitened innovations."""
    parts = _gloc_parts(wh_full)
    if parts is None:
        return [0.0] * _WHITE_GLOC_N
    w, n, ks, m1, m2, v1, v2 = parts
    var0 = max(float(w.var()), _CALIB_EPS)
    a = ks.astype(np.float64)
    b = (n - ks).astype(np.float64)
    gain = np.maximum(
        0.5 * (n * math.log(var0) - a * np.log(v1) - b * np.log(v2)),
        0.0,
    )
    ib = int(np.argmax(gain))
    kbest = int(ks[ib])
    post = w[kbest:]
    ac1 = 0.0
    if len(post) >= 5:
        centered = post - post.mean()
        ac1 = float(
            np.dot(centered[:-1], centered[1:])
            / (np.dot(centered, centered) + _CALIB_EPS)
        )
    radius = max(2, len(gain) // 20)
    keep = np.ones(len(gain), dtype=bool)
    keep[max(0, ib - radius):min(len(gain), ib + radius + 1)] = False
    second = float(gain[keep].max()) if np.any(keep) else 0.0
    peak = float(gain[ib])
    prominence = (peak - second) / (peak + _CALIB_EPS)
    lvr = np.log(v2 / v1)
    out = [
        peak / n,
        kbest / n,
        float(m2[ib] - m1[ib]),
        float(lvr[ib]),
        ac1,
        prominence,
        float(np.mean(gain >= 0.8 * peak)),
    ]
    return [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in out]


def compute_features_ms(w, ctx, wh=None, wh_full=None):
    champion = np.asarray(_champ166_producer(w, ctx, wh=wh), dtype=np.float64)
    prefix = wh if wh_full is None else wh_full
    gloc = np.asarray(white_gloc_feats(prefix), dtype=np.float64)
    return np.concatenate([champion, gloc])


print(
    f"E116 feature allocation: {_FULL_NCOL} generated cols; "
    f"ranker/RF use {_FULL_NCOL}, XGBoost/CatBoost use {_CHAMP_NCOL}"
)


import time as _time


def _hb(msg):
    """Flushed, timestamped progress line so the cloud log shows how far train() has gotten."""
    print(f"[hb {_time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rss_gb():
    """Resident memory (GB) from /proc so the log shows whether memory is ballooning (swap/OOM)."""
    try:
        with open("/proc/self/status") as _f:
            for _line in _f:
                if _line.startswith("VmRSS:"):
                    return int(_line.split()[1]) / (1024 ** 2)
    except Exception:
        return -1.0
    return -1.0


def _pointwise_ctor(name, seed):
    if name == "hgb":
        return HistGradientBoostingClassifier(early_stopping=True, random_state=seed, **HGB_PARAMS)
    if name == "rf":
        n_jobs_fit = max(1, (os.cpu_count() or 4) - 1)
        return RandomForestClassifier(n_jobs=n_jobs_fit, random_state=seed, **RF_PARAMS)
    if name == "xgb":
        n_jobs_fit = max(1, (os.cpu_count() or 4) - 1)
        return XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=n_jobs_fit,
                             random_state=seed, verbosity=0, **XGB_PARAMS)
    if name == "catboost":
        n_jobs_fit = max(1, (os.cpu_count() or 4) - 1)
        return CatBoostClassifier(random_seed=seed, thread_count=n_jobs_fit, verbose=0,
                                  allow_writing_files=False, **CAT_PARAMS)
    raise ValueError(name)


def _series_features(x_h, x_on, tau, pt_idx, rk_idx):
    """Build deterministic pointwise and ranker rows for one series."""
    with threadpool_limits(limits=1):
        ctx = hist_context(x_h, HIST_CAP)
        ctx["calib"] = fit_calib_ref(x_h, ctx)
        wh_on = whiten_pair(x_h, x_on)[1]

        pt_feats, pt_labs = [], []
        for t in pt_idx:
            lo = max(0, t - WINDOW + 1)
            pt_feats.append(compute_features_ms(
                x_on[lo:t + 1], ctx, wh=wh_on[lo:t + 1], wh_full=wh_on[:t + 1]
            ))
            pt_labs.append(1 if (tau is not None and t >= tau) else 0)

        rk_feats, rk_labs, rk_ts = [], [], []
        for t in rk_idx:
            lo = max(0, t - WINDOW + 1)
            rk_feats.append(compute_features_ms(
                x_on[lo:t + 1], ctx, wh=wh_on[lo:t + 1], wh_full=wh_on[:t + 1]
            ))
            rk_labs.append(1 if (tau is not None and t >= tau) else 0)
            rk_ts.append(int(t))
    return pt_feats, pt_labs, rk_feats, rk_labs, rk_ts


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
):
    rng = np.random.default_rng(SEED)

    # Draw indices serially in the original order so sampled rows remain unchanged.
    tasks: List[Tuple] = []
    n_used = 0
    for dataset_id, x_hist, x_online, tau in datasets:
        if n_used >= MAX_TRAIN_SERIES:
            break
        x_h = np.asarray(x_hist, dtype=np.float64)
        x_on = np.asarray(list(x_online), dtype=np.float64)
        L = len(x_on)
        if L <= MIN_N or len(x_h) < 5:
            continue
        cand = np.arange(MIN_N, L)
        kh = min(STEPS_PER_SERIES, len(cand))
        pt_idx = rng.choice(cand, size=kh, replace=False)
        kr = min(DENSE_K, len(cand))
        rk_idx = rng.choice(cand, size=kr, replace=False)
        tasks.append((x_h, x_on, tau, pt_idx, rk_idx))
        n_used += 1
        if n_used % 2000 == 0:
            _hb(f"pre-pass: {n_used} series drawn, rss={_rss_gb():.1f}GB")

    n_jobs_feat = max(1, (os.cpu_count() or 4) - 1)
    _hb(f"pre-pass done: {n_used} series; parallel feature build on {n_jobs_feat} workers; rss={_rss_gb():.1f}GB")

    results = joblib.Parallel(n_jobs=n_jobs_feat, backend="loky", verbose=0)(
        joblib.delayed(_series_features)(x_h, x_on, tau, pt_idx, rk_idx)
        for (x_h, x_on, tau, pt_idx, rk_idx) in tasks
    )
    get_reusable_executor().shutdown(wait=True)

    point_rows: List[List[float]] = []
    point_labels: List[int] = []
    rk_rows: List[List[float]] = []
    rk_labels: List[int] = []
    rk_t: List[int] = []
    for pt_feats, pt_labs, rk_feats, rk_labs, rk_ts in results:
        point_rows.extend(pt_feats)
        point_labels.extend(pt_labs)
        rk_rows.extend(rk_feats)
        rk_labels.extend(rk_labs)
        rk_t.extend(rk_ts)

    _hb(f"feature build done: {n_used} series; assembling matrices; rss={_rss_gb():.1f}GB")
    X_pt = np.asarray(point_rows, dtype=np.float64)
    y_pt = np.asarray(point_labels, dtype=np.int64)
    if X_pt.shape[1] != _FULL_NCOL:
        raise ValueError(f"expected {_FULL_NCOL} generated features, got {X_pt.shape[1]}")
    print(f"pointwise matrix {X_pt.shape} from {n_used} series, {y_pt.mean():.3f} positive")

    models, refs = {}, {}
    for name in ("rf", "xgb", "catboost"):
        X_member = X_pt[:, :MODEL_FEATURE_WIDTHS[name]]
        clf = _pointwise_ctor(name, SEED)
        _hb(f"fit {name} start on {X_member.shape[1]} cols")
        with threadpool_limits(limits=None):
            clf.fit(X_member, y_pt)
        _hb(f"fit {name} done")
        try:
            clf.set_params(n_jobs=1)
        except Exception:
            pass
        try:
            clf.set_params(thread_count=1)
        except Exception:
            pass
        models[name] = clf
        refs[name] = np.sort(clf.predict_proba(X_member)[:, 1])
        print(f"  trained {name}")

    rf_clf = models["rf"]
    X_rf = X_pt[:, :MODEL_FEATURE_WIDTHS["rf"]]
    _hb("onnx convert start")
    onx = convert_sklearn(
        rf_clf,
        initial_types=[("input", FloatTensorType([None, rf_clf.n_features_in_]))],
        options={type(rf_clf): {"zipmap": False}},
        target_opset=17,
    )
    rf_onnx_bytes = onx.SerializeToString()
    _sess = ort.InferenceSession(rf_onnx_bytes, providers=["CPUExecutionProvider"])
    _pn = _sess.get_outputs()[1].name
    _rf_pp = _sess.run([_pn], {"input": X_rf.astype(np.float32)})[0][:, 1]
    refs["rf"] = np.sort(_rf_pp.astype(np.float64))
    models.pop("rf", None)
    _hb("onnx convert done")
    print(f"  converted rf -> ONNX ({len(rf_onnx_bytes) / 1e6:.1f} MB)")

    X_rk = np.asarray(rk_rows, dtype=np.float64)
    y_rk = np.asarray(rk_labels, dtype=np.int64)
    t_rk = np.asarray(rk_t, dtype=np.int64)
    order = np.argsort(t_rk, kind="stable")
    X_rk, y_rk, t_rk = X_rk[order], y_rk[order], t_rk[order]
    X_ranker = X_rk[:, :MODEL_FEATURE_WIDTHS["ranker"]]
    _, groups = np.unique(t_rk, return_counts=True)
    print(f"ranker matrix {X_ranker.shape}, {len(groups)} query groups (online steps)")
    n_jobs_fit = max(1, (os.cpu_count() or 4) - 1)
    ranker = lgb.LGBMRanker(n_jobs=n_jobs_fit, **LGB_PARAMS)
    _hb("ranker fit start")
    with threadpool_limits(limits=1):
        ranker.fit(X_ranker, y_rk, group=groups)
    _hb("ranker fit done")
    ranker.set_params(n_jobs=1)
    models["ranker"] = ranker
    refs["ranker"] = np.sort(ranker.predict(X_ranker))

    _hb("joblib.dump start")
    joblib.dump(
        {
            "members": [("ranker", "ranker"), ("rf", "onnx"), ("xgb", "clf"),
                        ("catboost", "clf")],
            "models": models,
            "refs": refs,
            "rf_onnx": rf_onnx_bytes,
            "blend": dict(BLEND),
            "feature_widths": dict(MODEL_FEATURE_WIDTHS),
            "config": {
                "WINDOW": WINDOW,
                "MS_SCALES": list(MS_SCALES),
                "MIN_N": MIN_N,
                "HIST_CAP": HIST_CAP,
                "PREDICT_EVERY": PREDICT_EVERY,
            },
        },
        os.path.join(model_directory_path, "model.joblib"),
        compress=3,
    )
    _hb("joblib.dump done")
    print(f"saved 4 members; active blend = { {k: v for k, v in BLEND.items() if v > 0} }")


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    saved = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    models = saved["models"]
    refs = saved["refs"]
    blend = saved["blend"]
    feature_widths = saved["feature_widths"]
    cfg = saved["config"]
    window = int(cfg["WINDOW"])
    min_n = int(cfg["MIN_N"])
    hist_cap = int(cfg["HIST_CAP"])
    predict_every = int(cfg["PREDICT_EVERY"])
    global MS_SCALES
    MS_SCALES = tuple(cfg.get("MS_SCALES", MS_SCALES))

    active = []
    wsum = sum(w for w in blend.values() if w > 0)

    rf_sess = rf_out = None
    if saved.get("rf_onnx") is not None:
        _so = ort.SessionOptions()
        _so.intra_op_num_threads = 1
        _so.inter_op_num_threads = 1
        rf_sess = ort.InferenceSession(
            saved["rf_onnx"], sess_options=_so, providers=["CPUExecutionProvider"]
        )
        rf_out = rf_sess.get_outputs()[1].name
        saved["rf_onnx"] = None

    for name, kind in saved["members"]:
        weight = float(blend.get(name, 0.0))
        if weight <= 0:
            continue
        if kind == "onnx":
            model = (rf_sess, rf_out)
        else:
            model = models[name]
            try:
                model.set_params(n_jobs=1)
            except Exception:
                pass
            try:
                model.set_params(thread_count=1)
            except Exception:
                pass
        ref = np.asarray(refs[name], dtype=np.float64)
        width = int(feature_widths[name])
        active.append((kind, model, ref, len(ref), weight / wsum, width))

    yield

    with threadpool_limits(limits=1):
        for x_historical, x_online in datasets:
            x_h = np.asarray(x_historical, dtype=np.float64)
            ctx = hist_context(x_h, hist_cap)
            ctx["calib"] = fit_calib_ref(x_h, ctx)

            use_identity_wh = len(x_h) < 30
            use_fallback_wh = False
            loc = 0.0
            scl = 1.0
            mu = 0.0
            sd = 1.0
            p = 0
            coef = None
            rh_sorted = None
            pit_den = 1.0
            lag_tail = None

            if not use_identity_wh:
                loc = float(np.median(x_h))
                scl = 1.4826 * float(np.median(np.abs(x_h - loc))) + _CALIB_EPS
                xh_std = (x_h - loc) / scl
                p, coef = _fit_ar_bic(xh_std)
                rh = _ar_resid_wh(xh_std, p, coef)
                if len(rh) < 10:
                    use_fallback_wh = True
                    mu = float(xh_std.mean())
                    sd = float(xh_std.std()) + _CALIB_EPS
                else:
                    rh_sorted = np.sort(rh)
                    pit_den = len(rh_sorted) + 1.0
                    if p > 0:
                        lag_tail = deque(xh_std[-p:].tolist(), maxlen=p)

            buf = deque(maxlen=window)
            wh_buf = deque(maxlen=window)
            wh_full = []
            score = 0.0
            for idx, point in enumerate(x_online):
                x = float(point)
                buf.append(x)

                if use_identity_wh:
                    wh_val = x
                else:
                    xz = (x - loc) / scl
                    if use_fallback_wh:
                        wh_val = (xz - mu) / sd
                    else:
                        if p == 0:
                            resid = xz - float(coef[0])
                        else:
                            lags = np.fromiter(reversed(lag_tail), dtype=np.float64, count=p)
                            resid = xz - float(np.dot(lags, coef[:-1]) + coef[-1])
                            lag_tail.append(xz)
                        rank = np.searchsorted(rh_sorted, resid, side="right") / pit_den
                        wh_val = float(norm.ppf(np.clip(rank, 1e-4, 1 - 1e-4)))

                wh_buf.append(float(wh_val))
                wh_full.append(float(wh_val))
                n = idx + 1

                if n < min_n:
                    yield 0.0
                    continue

                if idx % predict_every == 0:
                    feats = np.asarray(
                        compute_features_ms(
                            np.asarray(buf, dtype=np.float64),
                            ctx,
                            wh=np.asarray(wh_buf, dtype=np.float64),
                            wh_full=np.asarray(wh_full, dtype=np.float64),
                        ),
                        dtype=np.float64,
                    ).reshape(1, -1)
                    if feats.shape[1] != _FULL_NCOL:
                        raise ValueError(
                            f"expected {_FULL_NCOL} generated features, got {feats.shape[1]}"
                        )
                    score = 0.0
                    for kind, model, ref, nref, weight, width in active:
                        member_feats = feats[:, :width]
                        if kind == "ranker":
                            raw = float(model.predict(member_feats)[0])
                        elif kind == "onnx":
                            session, output_name = model
                            raw = float(session.run(
                                [output_name],
                                {"input": member_feats.astype(np.float32)},
                            )[0][0, 1])
                        else:
                            raw = float(model.predict_proba(member_feats)[0, 1])
                        pct = float(np.searchsorted(ref, raw, side="right")) / nref
                        score += weight * pct

                yield float(score)


# @crunch/keep:on
# Each infer worker is a separate process with its own ONNX random-forest session.
# Cloud session construction can peak near 8-12 GB even though steady-state RSS is lower.
# Budget for the peak and leave RAM for Python, model blobs, and scoring infrastructure.
_PER_WORKER_GB = 12.0
_RAM_FRACTION = 0.80
_cpu = max(1, (os.cpu_count() or 4) - 1)
try:
    _ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    _mem_cap = max(1, int(_ram_gb * _RAM_FRACTION / _PER_WORKER_GB))
except (ValueError, OSError):
    _mem_cap = 1
INFER_PARALLELISM = max(1, min(_cpu, _mem_cap))
print(f"INFER_PARALLELISM={INFER_PARALLELISM} (cpu_cap={_cpu}, mem_cap={_mem_cap} @ {_PER_WORKER_GB} GB/worker)")
# INFER_PARALLELISM = 1  # set to 1 while debugging


#crunch_tools.test(
#    # force_first_train=False,
#    # no_determinism_check=True,
#)


# Compute local TS-AUC (same as the leaderboard, on the local test split).
#prediction = pd.read_parquet("prediction/prediction.parquet")
#y_test = pd.read_parquet("data/y_test.reduced.parquet")

#merged = prediction.merge(y_test, how="left", left_index=True, right_index=True)
#merged["time_online"] = merged.groupby("id").cumcount()

#weighted_auc_sum = 0.0
#total_weight = 0.0
#for t, group in merged.groupby("time_online"):
#    labels = group["target"].values
#    scores = group["prediction"].values
#    n_pos = int(labels.sum())
#    n_neg = int((1 - labels).sum())
#    if n_pos == 0 or n_neg == 0:
#        continue
#    auc_t = float(roc_auc_score(labels, scores))
#    weight = float(n_pos * n_neg)
#    weighted_auc_sum += weight * auc_t
#    total_weight += weight

#ts_auc = weighted_auc_sum / total_weight if total_weight > 0 else 0.5
#print(f"Local TS-AUC: {ts_auc:.4f}")


#Message = "4-member stack base79 + MS_SCALES 60,130,250 -> 40,100,250 (winsearch t_40_100_250, best mean +0.00076 2/3; LB gamble on prev champ 62.70%)"
#crunch_tools.submit(message=Message)
