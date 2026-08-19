"""Feature banks for the overnight big-run (deliverables a / b / c).

All features are CAUSAL: they compare the trailing online window `w` (= x_on[t-249:t+1])
against the historical pre-break segment exposed through `ctx` (hist_context) and per-series
reference fits. Break sits at online index tau; a window is "post" once t>=tau. So every
window-vs-history contrast is the causal analogue of the offline pre/post contrasts that won
the 2025 first place (Alphabot), whose features all consumed the FULL post segment.

Three banks:
  * champion base (51) -- champion_feat(45) + trail120 AR8 residual (2 spans x 3 scales = 6).
  * dependence / early-onset bank (deliverable b) -- studentized residual (EDA lever 2),
    early-onset residual CUSUM (EDA lever 1), ACF-lag window-vs-history contrasts, sup-F /
    Chow max-split scan, and an ARCH-lite BDS proxy. Targets the ma/ar_coef latency blind spot.
  * Alphabot bank (deliverable c) -- Humberto statistical tests + Mario descriptors/fusions +
    Rafael multi-view distances (primitives), then a deterministic random-composite explosion
    to reach ~1000 features, mirroring the winning stacking pipeline.

Every function returns plain python floats / a python list so joblib pickles cleanly.
"""
import math

import numpy as np
from scipy.signal import lfilter

from _ensemble_tune import champion_feat
from _exp_fast import _autocorr, _fast_ks
from _residual_reffit import _fit_ar, _ar_resid, _fit_refs as _reffit_fit, _extra_feats as _reffit_extra

_EPS = 1e-12

# --- champion base (trail120, the shipped LB champion) ----------------------
CHAMP_SPEC = dict(refs=[dict(p=8, trail=None), dict(p=8, trail=120)])
NCHAMP_BASE = 45 + 2 * 3   # 51


def fit_champ_refs(x_h):
    return _reffit_fit(x_h, CHAMP_SPEC)


def champ_base_row(w, ctx, champ_refs):
    """51-col champion base = champion_feat(45) + trail120 residual (6)."""
    return np.concatenate([
        np.asarray(champion_feat(w, ctx), dtype=np.float64),
        np.asarray(_reffit_extra(w, champ_refs, CHAMP_SPEC), dtype=np.float64),
    ])


CHAMP_NAMES = (
    [f"ch{s}_{k}" for s in (60, 130, 250)
     for k in ("mean_z", "log_var_ratio", "ks", "skew_diff", "kurt_diff", "ac1_diff",
               "ac2_diff", "iqr_ratio", "tail2", "tail3", "zero_cross", "slope_norm",
               "spec_centroid_diff", "rng_ratio", "spec_entropy")]
    + [f"ar8_full_s{s}" for s in (60, 130, 250)]
    + [f"ar8_trail120_s{s}" for s in (60, 130, 250)]
)
assert len(CHAMP_NAMES) == NCHAMP_BASE


# ===========================================================================
# Lever #2: cross-sectional calibration bank
# ---------------------------------------------------------------------------
# The TS-AUC metric ranks series against EACH OTHER at every online step, so a
# raw champion feature is only useful in so far as a fixed value means the same
# thing across series. A noisy series has a wider null band of `mean_z` /
# `log_var_ratio` than a clean one, so a fixed GBT threshold is not comparable.
# Here we build the SERIES' OWN pre-break null band of each of the 51 champion
# base features (from sliding historical windows), then emit two calibrated
# readouts of the online row: a studentized z (value minus null-mean over
# null-sd) and an empirical null-CDF rank. This generalizes the E62 studentized
# AR8-residual win (which paid +0.90 LB) to the full champion feature set.
# ===========================================================================
_CALIB_NULL_WINS = 20   # historical windows used to estimate each feature's null band
_CALIB_WIN = 250        # null-window length (matches the online buffer / champion top scale)
NCALIB = 2 * NCHAMP_BASE   # 51 studentized z + 51 null-CDF ranks = 102


def fit_calib_refs(x_h, champ_refs, ctx):
    """Per-series null band of the 51 champion-base features from pre-break windows.

    Slides `_CALIB_NULL_WINS` windows of length `_CALIB_WIN` across the history, computes the
    SAME `champ_base_row` on each (window-vs-full-history, exactly as online), and stores the
    per-feature null mean/sd plus the raw null matrix (for the empirical-CDF rank readout).
    """
    x = np.asarray(x_h, dtype=np.float64)
    rows = []
    if len(x) >= _CALIB_WIN + 10:
        starts = np.linspace(0, len(x) - _CALIB_WIN, _CALIB_NULL_WINS).astype(int)
        for s in starts:
            rows.append(champ_base_row(x[s:s + _CALIB_WIN], ctx, champ_refs))
    if len(rows) >= 3:
        Z = np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0) + _EPS
    else:
        Z = np.zeros((0, NCHAMP_BASE))
        mu = np.zeros(NCHAMP_BASE)
        sd = np.ones(NCHAMP_BASE)
    return {"mu": mu, "sd": sd, "null": Z}


def calib_feats(base_row, cref):
    """102 calibrated readouts of the champion base row: 51 studentized z + 51 null-CDF ranks.

    `base_row` is the already-computed 51-vector champion base (no recompute). z-score makes a
    fixed threshold mean 'k sigma above this series own null'; the null-CDF rank in [0,1] is the
    monotone rank-normalized analogue, robust to the null's shape.
    """
    raw = np.nan_to_num(np.asarray(base_row, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    z = (raw - cref["mu"]) / cref["sd"]
    Z = cref["null"]
    rank = (Z <= raw).mean(axis=0) if len(Z) else np.full(NCHAMP_BASE, 0.5)
    return np.concatenate([z, rank])


CALIB_NAMES = ([f"cz_{nm}" for nm in CHAMP_NAMES] + [f"crank_{nm}" for nm in CHAMP_NAMES])
assert len(CALIB_NAMES) == NCALIB


# ===========================================================================
# Lever #2 refinement (E69): robust, z-ONLY calibration
# ---------------------------------------------------------------------------
# E68 showed the studentized z on all 51 champion features is positive on 7/7 seeds but sub-threshold
# (min_t ~0), the rank readout HURTS, and the whole beats every subset. The per-seed noise (not the
# mean) is what kills t, and two things plausibly tighten it: (1) drop rank, keep z only; (2) a better
# null estimate -- more windows and a ROBUST center/scale (median / MAD) that is not dragged by the
# heavy-tailed champion features. This bank emits BOTH readouts so one run isolates the estimator:
#   z_meansd : (val - null_mean) / null_sd     (E68's readout, now at 32 windows)
#   z_robust : (val - null_median) / null_MAD  (1.4826*MAD ~ sd under normality)
# ===========================================================================
_CALIBZ_NULL_WINS = 32
NCALIBZ = 2 * NCHAMP_BASE   # z_meansd(51) + z_robust(51)


def fit_calibz_refs(x_h, champ_refs, ctx):
    """Per-series null band of the 51 champion features via `_CALIBZ_NULL_WINS` pre-break windows,
    keeping BOTH mean/sd and robust median/MAD centers/scales."""
    x = np.asarray(x_h, dtype=np.float64)
    rows = []
    if len(x) >= _CALIB_WIN + 10:
        starts = np.linspace(0, len(x) - _CALIB_WIN, _CALIBZ_NULL_WINS).astype(int)
        for s in starts:
            rows.append(champ_base_row(x[s:s + _CALIB_WIN], ctx, champ_refs))
    if len(rows) >= 3:
        Z = np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0) + _EPS
        med = np.median(Z, axis=0)
        mad = 1.4826 * np.median(np.abs(Z - med), axis=0) + _EPS
    else:
        mu = np.zeros(NCHAMP_BASE)
        sd = np.ones(NCHAMP_BASE)
        med = np.zeros(NCHAMP_BASE)
        mad = np.ones(NCHAMP_BASE)
    return {"mu": mu, "sd": sd, "med": med, "mad": mad}


def calibz_feats(base_row, cref):
    """102 z-only calibrated readouts of the champion base row: 51 mean/sd z + 51 robust median/MAD z."""
    raw = np.nan_to_num(np.asarray(base_row, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    z = (raw - cref["mu"]) / cref["sd"]
    zr = (raw - cref["med"]) / cref["mad"]
    return np.concatenate([z, zr])


CALIBZ_NAMES = ([f"czms_{nm}" for nm in CHAMP_NAMES] + [f"czr_{nm}" for nm in CHAMP_NAMES])
assert len(CALIBZ_NAMES) == NCALIBZ


# ===========================================================================
# Deliverable (b): dependence / early-onset residual bank
# ===========================================================================
_DEP_TRAIL = 120       # recent-history AR reference (matches champion's trail span)
_HIST_NULL_WINS = 24   # how many historical windows to estimate the per-series logVR null


def fit_dep_refs(x_h):
    """Per-series references for the dependence bank.

    ar8       : (coef, refvar) fit on full history -- the reference the residual lens uses.
    null      : (mean, sd) of the AR8 log-var-ratio computed on sliding historical windows
                (all pre-break => the series' own null band for the score) -> EDA lever 2.
    hist_acf  : historical ACF at lags {1,2,5,10} for the window-vs-history contrast.
    arch_hist : lag-1 autocorrelation of squared AR residuals on history (ARCH/BDS proxy).
    """
    x = np.asarray(x_h, dtype=np.float64)
    m8 = _fit_ar(x, 8)
    coef, refvar = (m8[0], m8[1]) if m8 is not None else (None, 1.0)

    logvr = []
    arch_vals = []
    if coef is not None and len(x) >= 300:
        win = 250
        starts = np.linspace(0, len(x) - win, _HIST_NULL_WINS).astype(int)
        for s in starts:
            seg = x[s:s + win]
            r = _ar_resid(seg, coef, 8)
            if r is None or len(r) < 20:
                continue
            logvr.append(math.log((float(np.var(r)) + _EPS) / refvar))
            r2 = r * r
            arch_vals.append(_autocorr(r2, 1))
    null = (float(np.mean(logvr)), float(np.std(logvr) + _EPS)) if len(logvr) >= 3 else (0.0, 1.0)
    arch_hist = float(np.mean(arch_vals)) if arch_vals else 0.0

    hist_acf = [_autocorr(x, lag) for lag in (1, 2, 5, 10)]
    return {"ar8": (coef, refvar), "null": null, "hist_acf": hist_acf, "arch_hist": arch_hist}


def _chow_supf(v):
    """Max standardized mean-contrast (Chow-like t) and max |log var-ratio| over interior splits,
    plus the normalized position of the best mean split. Scans a coarse grid for speed."""
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
        v1 = max(csq[k - 1] / k - m1 * m1, _EPS)
        v2 = max((csq[-1] - csq[k - 1]) / (n - k) - m2 * m2, _EPS)
        t = abs(m1 - m2) / math.sqrt(v1 / k + v2 / (n - k) + _EPS)
        lvr = abs(math.log(v2 / v1))
        if t > best_t:
            best_t, best_pos = t, k / n
        if lvr > best_v:
            best_v = lvr
    return best_t, best_v, best_pos


def dep_feats(w, ctx, dref):
    """Dependence / early-onset residual features (deliverable b)."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    coef, refvar = dref["ar8"]
    out = []

    # --- studentized residual variance-ratio (EDA lever 2: per-series specificity) ---
    r = _ar_resid(w, coef, 8) if coef is not None else None
    if r is not None and len(r) >= 8:
        logvr = math.log((float(np.var(r)) + _EPS) / refvar)
        m, s = dref["null"]
        z_student = (logvr - m) / s
        # early-onset residual CUSUM of standardized residual surprise |r|-scaled
        rs = r / (math.sqrt(refvar) + _EPS)
        surprise = rs * rs - 1.0                      # >0 when residual variance inflates
        cusum = np.cumsum(surprise)
        nn = len(cusum)
        cmax = float(np.max(cusum)) / math.sqrt(nn)
        cend = float(cusum[-1]) / math.sqrt(nn)
        cpos = float(np.argmax(cusum)) / nn
        cdrift = float(cusum[-1] - cusum[0]) / nn
        arch_w = _autocorr(r * r, 1)
    else:
        logvr = z_student = cmax = cend = cpos = cdrift = arch_w = 0.0
    out += [logvr, z_student, cmax, cend, cpos, cdrift]

    # --- ACF-lag window-vs-history contrasts (dependence-structure breaks) ---
    for lag, h_ac in zip((1, 2, 5, 10), dref["hist_acf"]):
        out.append(_autocorr(w, lag) - h_ac)

    # --- ARCH-lite BDS proxy: change in squared-value dependence vs history ---
    out.append(arch_w - dref["arch_hist"])

    # --- sup-F / Chow max-split scan inside the window ---
    t_sf, v_sf, pos_sf = _chow_supf(w)
    out += [t_sf, v_sf, pos_sf]
    return out


DEP_NAMES = (
    ["dep_logvr", "dep_zstudent", "dep_cusum_max", "dep_cusum_end", "dep_cusum_pos",
     "dep_cusum_drift"]
    + [f"dep_dacf{lag}" for lag in (1, 2, 5, 10)]
    + ["dep_arch_contrast", "dep_supf_t", "dep_supf_var", "dep_supf_pos"]
)
NDEP = len(DEP_NAMES)


# ===========================================================================
# Deliverable (e): causal sequential change-point detector bank
# ---------------------------------------------------------------------------
# The EDA early-detection cliff showed the champion loses ~27 AUC points from a clean window
# (0.97) to the just-after-break regime (0.71 at 10 post points), and the per-time-online metric
# lives in that low-post regime. These are online change-point statistics on the STUDENTIZED AR8
# residual, read out with recency/endpoint emphasis so they fire on FEW post-break points. All are
# vectorizable (no reset loop) and reuse the dep bank's ar8 coef + refvar (no new per-series fit).
SEQ_NAMES = ["seq_bb_max", "seq_bb_min", "seq_bb_pos", "seq_ph_max", "seq_ph_end",
             "seq_ewma_fend", "seq_ewma_fmax", "seq_ewma_send",
             "seq_rv30_z", "seq_rv60_z", "seq_rmean30_z"]
NSEQ = len(SEQ_NAMES)
_EWMA_FAST = 0.20
_EWMA_SLOW = 0.06


def seq_feats(w, ctx, dref):
    """Causal online change-point detectors on the studentized AR8 residual (deliverable e).

    bb_*     : Brownian-bridge CUSUM of the residual (mean-shift) -- max/min/argmax-recency.
    ph_*     : Page-Hinkley on the variance surprise (rs^2-1) -- max and endpoint level.
    ewma_*   : EWMA of the variance surprise at fast/slow decay -- endpoint (recent-weighted,
               fires on few points) + running max.
    rv/rmean : recency short-window residual z (last 30/60 pts) -- probes the low-post regime.
    """
    w = np.asarray(w, dtype=np.float64)
    coef, refvar = dref["ar8"]
    r = _ar_resid(w, coef, 8) if coef is not None else None
    if r is None or len(r) < 40:
        return [0.0] * NSEQ
    rs = r / (math.sqrt(refvar) + _EPS)
    n = len(rs)
    sn = math.sqrt(n)
    surprise = rs * rs - 1.0

    c = np.cumsum(rs)
    bridge = c - (np.arange(1, n + 1) / n) * c[-1]
    bb_max = float(bridge.max()) / sn
    bb_min = float(bridge.min()) / sn
    bb_pos = float(np.argmax(np.abs(bridge))) / n

    g = np.cumsum(surprise)
    ph = g - np.minimum.accumulate(g)
    ph_max = float(ph.max()) / sn
    ph_end = float(ph[-1]) / sn

    ef = lfilter([_EWMA_FAST], [1.0, -(1.0 - _EWMA_FAST)], surprise)
    es = lfilter([_EWMA_SLOW], [1.0, -(1.0 - _EWMA_SLOW)], surprise)
    ewma_fend = float(ef[-1])
    ewma_fmax = float(np.max(np.abs(ef)))
    ewma_send = float(es[-1])

    rv30_z = float(np.mean(surprise[-30:])) / math.sqrt(2.0 / 30.0)
    rv60_z = float(np.mean(surprise[-60:])) / math.sqrt(2.0 / 60.0)
    rmean30_z = float(np.mean(rs[-30:])) * math.sqrt(30.0)

    return [bb_max, bb_min, bb_pos, ph_max, ph_end,
            ewma_fend, ewma_fmax, ewma_send, rv30_z, rv60_z, rmean30_z]


# ===========================================================================
# Deliverable (b) v2: dependence-STRUCTURE bank (mined after b_no_accf shipped @ LB 62.48%)
# ---------------------------------------------------------------------------
# The shipped b_no_accf owns residual-*variance* change (studentized logVR) + accumulation
# (CUSUM) + a mean/var split scan (supF). It does NOT directly see a change in the dependence
# STRUCTURE itself (the AR/MA coefficient signature) when residual variance is ~unchanged --
# exactly the ma/ar_coef axis the EDA flagged. This v2 bank adds five orthogonal
# dependence-structure lenses, each isolable so the screen says which pay:
#   ARCOEF  : window-vs-history AR(8) coefficient-vector distance (ar_coef breaks)
#   PACF    : partial-autocorrelation direct-lag contrasts (cleaner than the noise ACF)
#   LJUNG   : Ljung-Box portmanteau of the historical-AR residual over the window (whiteness)
#   VR      : Lo-MacKinlay multi-horizon variance ratio profile (MA-vs-AR signature -> ma breaks)
#   HURST   : DFA long-memory exponent shift
# ===========================================================================
_DEP2_AR = 8
_VR_HORIZONS = (2, 4, 8)
_DFA_SCALES = (8, 16, 32, 64)


def _autocov(x, maxlag):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    x = x - x.mean()
    ac = np.empty(maxlag + 1)
    for k in range(maxlag + 1):
        ac[k] = float(np.dot(x[:n - k], x[k:])) / n if n > k else 0.0
    return ac


def _levinson(ac, p):
    """Levinson-Durbin: autocovariances ac[0..p] -> (AR coeffs a[1..p], PACF/reflection k[1..p])."""
    a = np.zeros(p)
    pacf = np.zeros(p)
    err = float(ac[0])
    if err <= _EPS:
        return a, pacf
    for i in range(1, p + 1):
        acc = ac[i] - float(np.dot(a[:i - 1], ac[i - 1:0:-1])) if i > 1 else ac[i]
        k = acc / (err + _EPS)
        pacf[i - 1] = k
        aprev = a[:i - 1].copy()
        a[i - 1] = k
        if i > 1:
            a[:i - 1] = aprev - k * aprev[::-1]
        err *= (1.0 - k * k)
        if err <= _EPS:
            break
    return a, pacf


def _dfa_hurst(x):
    """Detrended-fluctuation Hurst exponent (closed-form linear detrend per block, vectorized)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    y = np.cumsum(x - x.mean())
    logS, logF = [], []
    for s in _DFA_SCALES:
        nb = n // s
        if nb < 2 or s < 4:
            continue
        seg = y[:nb * s].reshape(nb, s)
        t = np.arange(s, dtype=np.float64)
        tm = t.mean()
        tv = float(((t - tm) ** 2).mean()) + _EPS
        segm = seg.mean(axis=1, keepdims=True)
        cov = ((seg - segm) * (t - tm)).mean(axis=1)
        detr_var = np.maximum(seg.var(axis=1) - cov * cov / tv, 0.0)
        logS.append(math.log(s))
        logF.append(math.log(math.sqrt(float(detr_var.mean()) + _EPS) + _EPS))
    if len(logS) < 2:
        return 0.5
    ls = np.asarray(logS)
    lf = np.asarray(logF)
    return float(((ls - ls.mean()) * (lf - lf.mean())).sum() / (((ls - ls.mean()) ** 2).sum() + _EPS))


def _vr_profile(acf):
    """Lo-MacKinlay variance ratios VR(q) = 1 + 2 sum_{k<q}(1-k/q) rho(k) from the ACF."""
    out = []
    for q in _VR_HORIZONS:
        out.append(1.0 + 2.0 * sum((1.0 - k / q) * acf[k] for k in range(1, q)))
    return out


def fit_dep2_refs(x_h):
    """Per-series references for the dependence-structure v2 bank."""
    x = np.asarray(x_h, dtype=np.float64)
    m8 = _fit_ar(x, _DEP2_AR)
    coef, refvar = (m8[0], m8[1]) if m8 is not None else (None, 1.0)
    ac_h = _autocov(x, _DEP2_AR)
    a_hist, pacf_hist = _levinson(ac_h, _DEP2_AR)
    norm_ahist = float(np.linalg.norm(a_hist)) + _EPS
    hurst_hist = _dfa_hurst(x)

    # historical VR-profile null band from sliding windows (all pre-break => the series' own null)
    vr_stack = []
    if len(x) >= 300:
        win = 250
        starts = np.linspace(0, len(x) - win, _HIST_NULL_WINS).astype(int)
        for s in starts:
            seg = x[s:s + win]
            acs = _autocov(seg, _DEP2_AR)
            acf = acs / (acs[0] + _EPS)
            vr_stack.append(_vr_profile(acf))
    if len(vr_stack) >= 3:
        arr = np.asarray(vr_stack)
        vr_null = (arr.mean(axis=0), arr.std(axis=0) + _EPS)
    else:
        vr_null = (np.ones(len(_VR_HORIZONS)), np.ones(len(_VR_HORIZONS)))
    return {"ar8": (coef, refvar), "a_hist": a_hist, "norm_ahist": norm_ahist,
            "pacf_hist": pacf_hist, "hurst_hist": hurst_hist, "vr_null": vr_null}


def dep2_feats(w, ctx, dref2):
    """Dependence-structure v2 features: [ARCOEF(3), PACF(3), LJUNG(3), VR(4), HURST(1)]."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    out = []
    if n >= 24:
        ac = _autocov(w, _DEP2_AR)
        acf = ac / (ac[0] + _EPS)
        a_win, pacf_win = _levinson(ac, _DEP2_AR)
    else:
        ac = None
        a_win = np.zeros(_DEP2_AR)
        pacf_win = np.zeros(_DEP2_AR)
        acf = np.zeros(_DEP2_AR + 1)

    # --- ARCOEF: AR(8) coefficient-vector change (targets ar_coef breaks) ---
    a_hist = dref2["a_hist"]
    diff = a_win - a_hist
    l2 = float(np.linalg.norm(diff)) / dref2["norm_ahist"]
    maxshift = float(np.max(np.abs(diff))) if len(diff) else 0.0
    denom = (float(np.linalg.norm(a_win)) * dref2["norm_ahist"]) + _EPS
    cos_dist = 1.0 - float(np.dot(a_win, a_hist)) / denom
    out += [l2, maxshift, cos_dist]

    # --- PACF: partial-autocorrelation direct-lag contrasts (lags 1,2,3) ---
    ph = dref2["pacf_hist"]
    out += [float(pacf_win[j] - ph[j]) for j in range(3)]

    # --- LJUNG: portmanteau whiteness of the historical-AR residual over the window ---
    coef, _refvar = dref2["ar8"]
    r = _ar_resid(w, coef, _DEP2_AR) if coef is not None else None
    if r is not None and len(r) >= 20:
        nn = len(r)
        rho = np.asarray([_autocorr(r, k) for k in range(1, 11)])
        lb5 = nn * (nn + 2) * float(np.sum(rho[:5] ** 2 / (nn - np.arange(1, 6))))
        lb10 = nn * (nn + 2) * float(np.sum(rho[:10] ** 2 / (nn - np.arange(1, 11))))
        absum = float(np.sum(np.abs(rho)))
    else:
        lb5 = lb10 = absum = 0.0
    out += [lb5, lb10, absum]

    # --- VR: Lo-MacKinlay multi-horizon variance ratio (MA-vs-AR signature) ---
    vr = _vr_profile(acf) if ac is not None else list(np.ones(len(_VR_HORIZONS)))
    vm, vs = dref2["vr_null"]
    vr_z = [(vr[i] - vm[i]) / vs[i] for i in range(len(_VR_HORIZONS))]
    vr_slope = vr_z[-1] - vr_z[0]
    out += [vr_z[0], vr_z[1], vr_z[2], vr_slope]

    # --- HURST: DFA long-memory exponent shift ---
    out.append(_dfa_hurst(w) - dref2["hurst_hist"])
    return out


DEP2_NAMES = (
    ["dep2_arcoef_l2", "dep2_arcoef_maxshift", "dep2_arcoef_cos"]
    + [f"dep2_dpacf{lag}" for lag in (1, 2, 3)]
    + ["dep2_ljung5", "dep2_ljung10", "dep2_resid_acf_absum"]
    + ["dep2_vr2_z", "dep2_vr4_z", "dep2_vr8_z", "dep2_vr_slope"]
    + ["dep2_hurst_shift"]
)
NDEP2 = len(DEP2_NAMES)


# ===========================================================================
# Deliverable (b) v3: sup-F / Chow SCAN refinement (sharpen the one structure feature that PAID)
# ---------------------------------------------------------------------------
# The shipped b_no_accf keeps a COARSE (~40-point grid) sup-F/Chow scan on the RAW window only,
# returning (best mean-contrast t, max |log var-ratio|, best mean-split position). This bank
# sharpens that same detector three ways, each isolable:
#   RAW   : a FINE full-grid scan + the variance-split position the coarse one omits
#   RESID : the same Chow scan run on the AR(8) RESIDUAL (the productive residual formulation
#           applied to the split-scan -> localizes a residual-level/dependence break)
#   AGG   : Andrews-Ploberger ave-F / exp-F aggregations of the F-profile (more powerful than
#           the sup/max against diffuse or gradual breaks)
# ===========================================================================
def _supf2_scan(v):
    """One fine-grid Chow scan -> (max mean-contrast t, max |log var-ratio|, mean-split pos,
    var-split pos, ave-F, exp-F). Vectorized over interior split points."""
    n = len(v)
    if n < 24:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    v = np.asarray(v, dtype=np.float64)
    csum = np.cumsum(v)
    csq = np.cumsum(v * v)
    tot = csum[-1]
    totsq = csq[-1]
    lo, hi = max(6, n // 8), n - max(6, n // 8)
    k = np.arange(lo, hi)
    if len(k) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    m1 = csum[k - 1] / k
    m2 = (tot - csum[k - 1]) / (n - k)
    v1 = np.maximum(csq[k - 1] / k - m1 * m1, _EPS)
    v2 = np.maximum((totsq - csq[k - 1]) / (n - k) - m2 * m2, _EPS)
    t = np.abs(m1 - m2) / np.sqrt(v1 / k + v2 / (n - k) + _EPS)
    lvr = np.abs(np.log(v2 / v1))
    f = t * t
    it, iv = int(np.argmax(t)), int(np.argmax(lvr))
    avef = float(np.mean(f))
    expf = float(np.log(np.mean(np.exp(np.clip(f / 2.0, -30.0, 30.0))) + _EPS))
    return (float(t[it]), float(lvr[iv]), float(k[it]) / n, float(k[iv]) / n, avef, expf)


def fit_supf2_refs(x_h):
    x = np.asarray(x_h, dtype=np.float64)
    m8 = _fit_ar(x, 8)
    coef = m8[0] if m8 is not None else None
    return {"ar8_coef": coef}


def supf2_feats(w, ctx, sref):
    """sup-F/Chow scan refinement: [RAW(3), RESID(3), AGG(2)]."""
    w = np.asarray(w, dtype=np.float64)
    raw_t, raw_v, raw_mpos, raw_vpos, avef, expf = _supf2_scan(w)
    coef = sref["ar8_coef"]
    r = _ar_resid(w, coef, 8) if coef is not None else None
    if r is not None and len(r) >= 24:
        res_t, res_v, res_mpos, _res_vpos, _a, _e = _supf2_scan(r)
    else:
        res_t = res_v = res_mpos = 0.0
    return [raw_t, raw_mpos, raw_vpos, res_t, res_v, res_mpos, avef, expf]


SUPF2_NAMES = [
    "supf2_raw_t", "supf2_raw_meanpos", "supf2_raw_varpos",
    "supf2_resid_t", "supf2_resid_var", "supf2_resid_pos",
    "supf2_avef", "supf2_expf",
]
NSUPF2 = len(SUPF2_NAMES)


# ===========================================================================
# Deliverable (c): Alphabot-style wide bank (primitives + composites)
# ===========================================================================
def _safe_div(a, b):
    return a / b if abs(b) > _EPS else 0.0


def _moments(v):
    n = len(v)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    mu = float(np.mean(v))
    var = float(np.var(v))
    sd = math.sqrt(var + _EPS)
    d = v - mu
    sk = float(np.mean(d ** 3)) / (sd ** 3 + _EPS)
    ku = float(np.mean(d ** 4)) / (var * var + _EPS) - 3.0
    return mu, sd, sk, ku


def _financial(v):
    """Sharpe, Sortino, profit-factor, CVaR5, max-drawdown, Gini -- Mario's risk metrics."""
    n = len(v)
    if n < 4:
        return [0.0] * 6
    mu = float(np.mean(v))
    sd = float(np.std(v)) + _EPS
    downside = v[v < 0]
    dstd = float(np.std(downside)) + _EPS if len(downside) else _EPS
    pos = float(v[v > 0].sum())
    neg = float(-v[v < 0].sum()) + _EPS
    q05 = float(np.quantile(v, 0.05))
    tail = v[v <= q05]
    cvar = float(np.mean(tail)) if len(tail) else q05
    cs = np.cumsum(v)
    peak = np.maximum.accumulate(cs)
    maxdd = float(np.max(peak - cs))
    sv = np.sort(v)
    idx = np.arange(1, n + 1)
    gini = float((2.0 * (idx * sv).sum()) / (n * sv.sum() + _EPS) - (n + 1.0) / n) if sv.sum() > _EPS else 0.0
    return [_safe_div(mu, sd), _safe_div(mu, dstd), _safe_div(pos, neg), cvar, maxdd, gini]


_QS = (0.10, 0.25, 0.50, 0.75, 0.90)


def _describe(v):
    """~18 descriptors on a single view (Mario stage-1)."""
    mu, sd, sk, ku = _moments(v)
    if len(v) >= 2:
        qs = np.quantile(v, _QS)
        mn, mx = float(np.min(v)), float(np.max(v))
    else:
        qs = np.zeros(len(_QS))
        mn = mx = 0.0
    ac1 = _autocorr(v, 1)
    ac2 = _autocorr(v, 2)
    ac5 = _autocorr(v, 5)
    iqr = float(qs[3] - qs[1])
    return [mu, sd, sk, ku, *[float(q) for q in qs], mn, mx, ac1, ac2, ac5, iqr, _safe_div(mu, sd)]


_DESC_LEN = 18


def _welch_t_abs(a, b):
    aa, bb = np.abs(a), np.abs(b)
    na, nb = len(aa), len(bb)
    if na < 4 or nb < 4:
        return 0.0
    va, vb = float(np.var(aa)) / na, float(np.var(bb)) / nb
    denom = math.sqrt(va + vb + _EPS)
    return (float(np.mean(aa)) - float(np.mean(bb))) / denom


def _entropy(v, bins=32):
    if len(v) < 8:
        return 0.0
    h, _ = np.histogram(v, bins=bins)
    p = h.astype(np.float64)
    s = p.sum()
    if s < _EPS:
        return 0.0
    p = p[p > 0] / s
    return float(-(p * np.log(p)).sum())


def _jsd_hellinger(a, b, bins=40):
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if hi - lo < _EPS:
        return 0.0, 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, bins=edges)[0].astype(np.float64)
    pb = np.histogram(b, bins=edges)[0].astype(np.float64)
    pa = pa / (pa.sum() + _EPS)
    pb = pb / (pb.sum() + _EPS)
    m = 0.5 * (pa + pb)
    def _kl(p, q):
        mask = p > 0
        return float((p[mask] * np.log(p[mask] / (q[mask] + _EPS))).sum())
    jsd = 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)
    hell = float(math.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(pa * pb)))))
    return jsd, hell


def _distances(vw, vh):
    """Sorted-tail distances between window and history (Rafael)."""
    n = min(len(vw), len(vh))
    if n < 8:
        return [0.0] * 5
    a = np.sort(vw)[-n:]
    b = np.sort(vh)[-n:]
    canberra = float(np.sum(np.abs(a - b) / (np.abs(a) + np.abs(b) + _EPS)))
    bray = float(np.sum(np.abs(a - b)) / (np.sum(np.abs(a + b)) + _EPS))
    city = float(np.sum(np.abs(a - b)))
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + _EPS))
    wass = float(np.mean(np.abs(a - b)))            # equal-length sorted -> 1-Wasserstein proxy
    return [canberra, bray, city, cos, wass]


def alpha_primitives(w, ctx):
    """Humberto + Mario + Rafael primitive features (causal window-vs-history)."""
    w = np.asarray(w, dtype=np.float64)
    h = ctx["sample_ordered"]
    aw = np.abs(w)
    dw = np.diff(w) if len(w) > 1 else w
    out = []

    # --- Mario: descriptors on 3 window views + cross-regime fusion vs history ---
    dv_w = _describe(w)
    dv_h = _describe(h)
    out += dv_w
    for a, b in zip(dv_w, dv_h):                    # fusions: diff / abs-diff / ratio
        out += [a - b, abs(a - b), _safe_div(a, b)]
    out += _describe(aw)
    out += _describe(dw)

    # --- Mario: financial metrics on window + fusion vs history ---
    fw, fh = _financial(w), _financial(h)
    out += fw
    for a, b in zip(fw, fh):
        out += [a - b, _safe_div(a, b)]

    # --- Humberto: statistical tests window-vs-history ---
    L = len(w)
    for span in (50, 100):
        s = min(span, L, len(h))
        out.append(_welch_t_abs(w[-s:], h[-s:]))
    out.append(_fast_ks(np.sort(w), np.sort(h)))    # KS D full
    vw, vh = float(np.var(w)) + _EPS, float(np.var(h)) + _EPS
    out.append(math.log(vw / vh))                   # log F variance ratio
    out.append(_entropy(w) - _entropy(h))           # entropy diff
    jsd, hell = _jsd_hellinger(w, h)
    out += [jsd, hell]
    # central-band skew shift (Humberto f21)
    if L >= 20 and len(h) >= 20:
        qw = np.quantile(w, [0.05, 0.95])
        qh = np.quantile(h, [0.05, 0.95])
        cw = w[(w >= qw[0]) & (w <= qw[1])]
        ch = h[(h >= qh[0]) & (h <= qh[1])]
        out.append(_moments(cw)[2] - _moments(ch)[2])
    else:
        out.append(0.0)

    # --- Rafael: multi-view distances (window vs history) ---
    out += _distances(w, h)
    out += _distances(aw, np.abs(h))
    out += _distances(np.cumsum(w), np.cumsum(h))
    return out


def _n_primitives():
    """Compute the primitive dimension once from a dummy input."""
    ctx = {"sample_ordered": np.random.default_rng(0).standard_normal(400)}
    return len(alpha_primitives(np.random.default_rng(1).standard_normal(250), ctx))


NPRIM = _n_primitives()

# --- deterministic composite explosion (Mario/Rafael 287+477 composites) ----
NCOMP = 900
_OPS = ("add", "sub", "mul", "div", "adiff")
_NLS = ("id", "tanh", "log1p", "sqrtabs", "sin")


def _build_recipes(nprim, ncomp, seed=98765):
    rng = np.random.default_rng(seed)
    recipes = []
    for _ in range(ncomp):
        k = int(rng.integers(2, 4))                 # combine 2-3 primitives
        idx = rng.integers(0, nprim, size=k)
        op = _OPS[int(rng.integers(len(_OPS)))]
        nl = _NLS[int(rng.integers(len(_NLS)))]
        coef = rng.uniform(0.5, 2.0, size=k)
        recipes.append((idx, coef, op, nl))
    return recipes


_RECIPES = _build_recipes(NPRIM, NCOMP)


def _apply_nl(x, nl):
    if nl == "tanh":
        return math.tanh(x)
    if nl == "log1p":
        return math.log1p(abs(x))
    if nl == "sqrtabs":
        return math.copysign(math.sqrt(abs(x)), x)
    if nl == "sin":
        return math.sin(x)
    return x


def composites(prim):
    """Expand primitives into NCOMP deterministic non-linear composites."""
    p = prim
    out = []
    for idx, coef, op, nl in _RECIPES:
        vals = [coef[j] * p[idx[j]] for j in range(len(idx))]
        if op == "add":
            x = sum(vals)
        elif op == "sub":
            x = vals[0] - sum(vals[1:])
        elif op == "mul":
            x = 1.0
            for v in vals:
                x *= v
        elif op == "div":
            x = _safe_div(vals[0], vals[1] if len(vals) > 1 else 1.0)
        else:  # adiff
            x = abs(vals[0] - vals[1]) if len(vals) > 1 else abs(vals[0])
        out.append(_apply_nl(x, nl))
    return out


ALPHA_PRIM_NAMES = [f"aprim_{i}" for i in range(NPRIM)]
ALPHA_COMP_NAMES = [f"acomp_{i}" for i in range(NCOMP)]


def alpha_full(w, ctx):
    """Primitives + composites (Alphabot bank)."""
    prim = alpha_primitives(w, ctx)
    return prim + composites(prim)


NALPHA = NPRIM + NCOMP
ALPHA_NAMES = ALPHA_PRIM_NAMES + ALPHA_COMP_NAMES


# --- block index helpers (column ranges within the full wide matrix) --------
def wide_names():
    return CHAMP_NAMES + DEP_NAMES + ALPHA_NAMES


def wide_blocks():
    c = NCHAMP_BASE
    d = c + NDEP
    a = d + NALPHA
    return {
        "champ": np.arange(0, c),
        "dep": np.arange(c, d),
        "alpha_prim": np.arange(d, d + NPRIM),
        "alpha_comp": np.arange(d + NPRIM, a),
        "alpha_all": np.arange(d, a),
        "champ_dep": np.arange(0, d),
        "all": np.arange(0, a),
    }


if __name__ == "__main__":
    print(f"NCHAMP_BASE={NCHAMP_BASE} NDEP={NDEP} NPRIM={NPRIM} NCOMP={NCOMP} "
          f"NALPHA={NALPHA} WIDE={NCHAMP_BASE + NDEP + NALPHA}")
    assert len(wide_names()) == NCHAMP_BASE + NDEP + NALPHA
    print("names ok; sample dep:", DEP_NAMES)
