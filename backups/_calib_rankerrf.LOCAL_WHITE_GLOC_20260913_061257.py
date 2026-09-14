"""PER-SERIES CALIBRATION probe on the CURRENT (E78) ranker+rf 79-col SHIP base.

Ports the E68/E69 "studentize every base feature against the series' OWN pre-break null" idea to
the current base79 champion. E68/E69 were only ever measured on the OLD base61 (pre window-switch);
this is a faithful RE-MEASUREMENT on base79 -- not a new idea. E68 showed the mean/sd studentized z
on all champion base cols was positive on 7/7 seeds but sub-threshold under the OLD t>=2.5 bar; E69
added a robust median/MAD variant. Here we re-test both under the CURRENT relaxed believe-rule
(any arm mean_d>0, all-seeds-pos), NOT the retired t>=2.5 gate.

DESIGN (mirrors _night_banks.fit_calibz_refs, generalized from the 51-col champ row to the full
base79 row):
  Slide _CALIB_NULL_WINS windows of length _CALIB_WIN across the pre-break history x_h, compute the
  SAME _base79_row on each (window-vs-full-history, exactly as online), and store the per-feature
  null mean/sd (E68) and robust median/MAD (E69). The online base79 row is then studentized against
  that per-series null: z = (raw - mu)/sd (or (raw - med)/mad). The TS-AUC metric ranks series
  against each other every step, so a fixed threshold on a raw feature is not comparable across
  series (a noisy series has a wider null band); the studentized readout makes a value mean
  "k sigma above THIS series' own null".

BASE (always in, = the E78 champion 79, byte-identical to _nldep/_wloss, SHIP_SCALES=40,100,250):
  champ51 (AR8 full + AR8 trail120) + dep10 + ship12 + [AR8 trail60, AR8 trail200] x 3 scales.
CANDIDATE arms (each a contiguous studentized block appended after base79; base79 REUSED, no recompute):
  czms_champ51  mean/sd studentize the 51 champion-base cols only  (closest analog to E68 z_all)  w51
  czms_all79    mean/sd studentize all 79 base cols                (base79 generalization)         w79
  czrb_all79    robust median/MAD studentize all 79 base cols       (E69 variant)                   w79
  calib_all     all three arms together
Each arm is tested in ISOLATION on base79 (baseline = base79 alone; cand_cols = base79 + the arm block).

AFFINE-INVARIANT to machine precision: every base79 feature is affine-invariant, and the per-series
null is built with the SAME _base79_row, so raw and (mu,sd)/(med,mad) transform identically under
x -> a*x+b -> the studentized z cancels a and b exactly. Validated by the smoke affine-check (~0).

CAUSAL: the null band is built from x_h (pre-break history) ONLY; the online row uses the causal
trailing window. No future information enters either.

Same seeds / gate / recipe / harness as _nldep_rankerrf.py.

Run:
  SMOKE=1 .venv/bin/python -u _calib_rankerrf.py           # tiny self-test + affine check
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _calib_rankerrf.py \
      > _calib_rankerrf.out 2>&1 & disown
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

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, _subwin, load_pool_strat, select_picks
from _exp_fast import HIST_CAP, hist_context, split_series
from _overnight import MIN_N, N_JOBS
from _residual_overnight import _active, pct
from _residual_reffit import _fit_ar, _ar_resid
from _night_banks import (NCHAMP_BASE, fit_champ_refs, champ_base_row,
                          fit_dep_refs, dep_feats)
from _lambdarank import sort_into_groups, fit_ranker
from _whiten import whiten_pair

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]
if os.environ.get("SEEDS"):                        # fresh-confirm override
    SEARCH_SEEDS = [int(s) for s in os.environ["SEEDS"].split(",")]

RF_TREES = 120 if SMOKE else 500
DENSE_K = 20 if SMOKE else 40
W_RANK, W_RF = 0.45, 0.55
RF_KW = dict(max_features=0.5, min_samples_leaf=20)

_SS_ENV = os.environ.get("SHIP_SCALES")            # multi-scale set (baked into cache -> isolate per value)
SHIP_SCALES = tuple(int(s) for s in _SS_ENV.split(",")) if _SS_ENV else (40, 100, 250)

# training-step sampling: uniform (default) vs metric (sample t ~ per-step metric weight pos*neg)
# vs bimodal (oversample steps far from the break onset). STEP_BIAS -> ranker rows (0.45 of blend);
# PICK_BIAS -> rf/pointwise rows (0.55). Isolated RNG so biasing one leaves the other byte-identical.
_STEP_BIAS = os.environ.get("STEP_BIAS", "uniform")
_PICK_BIAS = os.environ.get("PICK_BIAS", "uniform")

# WHITEN feature bet (forum "oracle ceilings" lead): compute the base79 family on a Rosenblatt-
# whitened stream (AR(p<=12 BIC) innovations PIT->normal) instead of the raw values. Whitens BOTH
# history and online with one history-fit whitener so window-vs-history features stay consistent.
_WHITEN = os.environ.get("WHITEN") == "1"

# LOCATOR bet (forum "oracle ceilings": ~84% of the reachable ceiling is LOCATING tau, not the
# statistic). Scan splits inside the trailing window, locate the argmax mean-contrast break, then
# emit a MATCHED statistic at that split -- crucially the HIGHER-moment (skew/kurt/5th) shift of
# post-vs-pre that the champion's existing supF (_chow_supf: mean-contrast, var-ratio, position)
# does NOT capture. Appended as an isolated candidate arm (base79 vs base79+locator).
_LOCATOR = os.environ.get("LOCATOR") == "1"
NLOC = 7

# GLOCATOR bet (E111): the FULL-BUFFER analog of the window-local locator (E110, NULL). E110 scanned
# only the trailing <=WINDOW window, so a tau older than WINDOW steps was invisible (the whole window
# is post-break). The oracle says ~84% of the reachable ceiling is LOCATING tau -- but that is a
# GLOBAL location across the entire online buffer, not within a 250-pt window. Here we scan the
# WHOLE online prefix x_on[:t+1] with the same argmax mean-contrast locator and emit the same 7
# affine-invariant readouts at the located global break. Requires plumbing the full prefix to _row.
_GLOC = os.environ.get("GLOC") == "1"
NGLOC = NLOC

# Six-arm refinement screen for the full-prefix locator. All arms share one feature cache and are
# compared against the current local baseline (base79 + robust calibration + WBLOCK).
_GLOC6 = os.environ.get("GLOC6") == "1"
NWHITE_GLOC = 7
NAGREE_GLOC = 3
NASYM_GLOC = NLOC
NVAR_GLOC = 7
NHORIZON_GLOC = 9

# WBLOCK bet (E112): ADDITIVE whitened readout. E109 (WHITEN) REPLACED the raw stream with the AR-
# innovation-whitened stream and recomputed the whole base79 family -> NULL. Here we KEEP raw base79
# and APPEND a compact break-signal block computed on the whitened trailing window (moments that
# measure deviation from the whitened null + half/half contrasts + a CUSUM magnitude). The oracle
# showed whitening a naked moment statistic lifts it ~1.85x; this lets the model pick up that
# incremental whitened signal WITHOUT losing the raw contrasts. Whitening is affine-invariant.
_WBLOCK = os.environ.get("WBLOCK") == "1"
WBLOCK_N = 8

# shipped dep block keeps 10 of dep_feats' 14 cols (drops ACF-lag dacf1/2/5/10) -> b_no_accf.
_DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]

HERE = os.path.dirname(os.path.abspath(__file__))
_NW_ENV = os.environ.get("CALIB_NULL_WINS")   # calib null_wins baked into cache -> isolate per value
_CW_ENV = os.environ.get("CALIB_WIN")
CACHE_DIR = os.path.join(HERE, "_calib_cache"
                         + (f"_hc{HIST_CAP}" if HIST_CAP != 200 else "")
                         + (f"_nw{_NW_ENV}" if _NW_ENV and _NW_ENV != "32" else "")
                         + (f"_cw{_CW_ENV}" if _CW_ENV and _CW_ENV != "250" else "")
                         + (f"_ss{'_'.join(map(str, SHIP_SCALES))}" if SHIP_SCALES != (40, 100, 250) else "")
                         + (f"_sb{_STEP_BIAS}" if _STEP_BIAS != "uniform" else "")
                         + (f"_pb{_PICK_BIAS}" if _PICK_BIAS != "uniform" else "")
                         + ("_whiten" if _WHITEN else "")
                         + ("_locator" if _LOCATOR else "")
                         + ("_gloc" if _GLOC else "")
                         + ("_gloc6" if _GLOC6 else "")
                         + ("_wblock" if _WBLOCK else ""))
RAW = os.path.join(HERE, "_calib_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_calib_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_calib_rankerrf.log")

# --- base AR8 spans (SHIPPED, byte-identical to _nldep/_wloss) --------------
BASE_SPAN_SPECS = [
    ("ar8_t60",  dict(p=8, trail=60)),
    ("ar8_t200", dict(p=8, trail=200)),
]

# --- per-series calibration null settings (E68/E69) -------------------------
_CALIB_WIN = int(os.environ.get("CALIB_WIN", 250))        # null-window length (env-sweepable)
_CALIB_NULL_WINS = int(os.environ.get("CALIB_NULL_WINS", 32))  # n null windows (env-sweepable)
_EPS = 1e-9

SPAN_W = len(SHIP_SCALES)                          # 3 cols per AR8 base span
NBASE = (NCHAMP_BASE + len(_DEP_KEEP) + 4 * len(SHIP_SCALES)
         + len(BASE_SPAN_SPECS) * SPAN_W)          # champ51 + dep10 + ship12 + base-spans6 = 79

# --- calibration candidate arms: (name, estimator, base79 slice) ------------
_CHAMP_SLICE = slice(0, NCHAMP_BASE)          # first 51 cols = champ base
_DEP28_SLICE = slice(NCHAMP_BASE, NBASE)      # last 28 cols = dep10 + ship12 + span6
CALIB_ARMS = [
    ("czms_all79",   "meansd", slice(0, NBASE)),
    ("czrb_all79",   "robust", slice(0, NBASE)),   # shipped arm (control)
    ("czrb_champ51", "robust", _CHAMP_SLICE),       # localize: champ block only
    ("czrb_dep28",   "robust", _DEP28_SLICE),       # localize: dependence/ship/span block (EDA hypothesis)
]
_ARM_W = {name: len(range(*sl.indices(NBASE))) for name, _e, sl in CALIB_ARMS}
_CAND_W = sum(_ARM_W.values())                     # 51 + 79 + 79 = 209
NCOL = NBASE + _CAND_W                              # 288
BASE_COLS = np.arange(NBASE)

CAND_SLICE = {}
_off = NBASE
for _name, _est, _sl in CALIB_ARMS:
    CAND_SLICE[_name] = np.arange(_off, _off + _ARM_W[_name])
    _off += _ARM_W[_name]


def cand_cols(cand_names):
    return np.concatenate([BASE_COLS] + [CAND_SLICE[n] for n in cand_names])


CAND_SINGLES = [a[0] for a in CALIB_ARMS]
CANDIDATES = [(n, [n]) for n in CAND_SINGLES] + [
    ("czmix", ["czms_all79", "czrb_all79"]),   # both 7/7 winners combined
    ("calib_all", CAND_SINGLES),
]

if _LOCATOR:                                       # directed change-point locator arm (isolated)
    CAND_SLICE["locator"] = np.arange(NCOL, NCOL + NLOC)
    NCOL = NCOL + NLOC
    CANDIDATES = CANDIDATES + [
        ("locator", ["locator"]),                          # base79 + locator (isolated)
        ("czrb_locator", ["czrb_all79", "locator"]),       # SHIPPED arm + locator (incremental test)
    ]

if _GLOC:                                          # FULL-BUFFER global change-point locator (E111)
    CAND_SLICE["gloc"] = np.arange(NCOL, NCOL + NGLOC)
    NCOL = NCOL + NGLOC
    CANDIDATES = CANDIDATES + [
        ("gloc", ["gloc"]),                                # base79 + global-locator (isolated)
        ("czrb_gloc", ["czrb_all79", "gloc"]),             # SHIPPED arm + global-locator (incremental)
    ]

if _GLOC6:
    for _name, _width in [
        ("gloc", NGLOC),
        ("white_gloc", NWHITE_GLOC),
        ("agree_gloc", NAGREE_GLOC),
        ("asym_gloc", NASYM_GLOC),
        ("var_gloc", NVAR_GLOC),
        ("horizon_gloc", NHORIZON_GLOC),
    ]:
        CAND_SLICE[_name] = np.arange(NCOL, NCOL + _width)
        NCOL += _width

if _WBLOCK:                                         # ADDITIVE whitened readout block (E112)
    CAND_SLICE["wblock"] = np.arange(NCOL, NCOL + WBLOCK_N)
    NCOL = NCOL + WBLOCK_N
    CANDIDATES = CANDIDATES + [
        ("wblock", ["wblock"]),                            # base79 + whitened block (isolated)
        ("czrb_wblock", ["czrb_all79", "wblock"]),         # SHIPPED arm + whitened block (incremental)
    ]

if _GLOC6:
    _champ = ["czrb_all79", "wblock"]
    CANDIDATES = [
        ("raw_gloc", _champ + ["gloc"]),
        ("white_gloc", _champ + ["white_gloc"]),
        ("dual_gloc", _champ + ["gloc", "white_gloc", "agree_gloc"]),
        ("asym_gloc", _champ + ["asym_gloc"]),
        ("var_gloc", _champ + ["var_gloc"]),
        ("horizon_gloc", _champ + ["horizon_gloc"]),
    ]

if os.environ.get("CANDIDATES"):
    _selected = set(os.environ["CANDIDATES"].split(","))
    CANDIDATES = [(name, arms) for name, arms in CANDIDATES if name in _selected]
    if not CANDIDATES:
        raise ValueError(f"CANDIDATES matched no available arms: {sorted(_selected)}")


# --- AR8 base-span fit / residual-var-ratio feats (verbatim from _nldep) -----
def _fit_span(x_h, spec):
    x = np.asarray(x_h, dtype=np.float64)
    if spec.get("trail"):
        x = x[-spec["trail"]:]
    p = spec["p"]
    return p, _fit_ar(x, p)


def _span_feats(w, p, m):
    w = np.asarray(w, dtype=np.float64)
    out = []
    for s in SHIP_SCALES:
        ws = w if s >= len(w) else w[-s:]
        r = _ar_resid(ws, m[0], p) if m is not None else None
        out.append(0.0 if (r is None or len(r) < 5) else math.log((float(np.var(r)) + 1e-12) / m[1]))
    return out


# --- ship feature funcs (verbatim from the shipped notebook) ---------------
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


def _ship_block(ws, ctx):
    return _ship_perm_entropy4(ws) + _ship_max_split_lr(ws) + _ship_mosum(ws, ctx)


_LOC_FACT = np.array([1.0, 2.0, 6.0, 24.0, 120.0])   # k! null-variance scaling for moments 1..5


def _locator_parts(w, min_fraction=0.1):
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
    v1 = np.maximum(csq[ks - 1] / a - m1 * m1, _EPS)
    v2 = np.maximum((csq[-1] - csq[ks - 1]) / b - m2 * m2, _EPS)
    tstat = np.abs(m1 - m2) / np.sqrt(v1 / a + v2 / b + _EPS)
    lvr = np.log(v2 / v1)
    return w, n, ks, m1, m2, v1, v2, tstat, lvr


def locator_feats(w, ctx, min_fraction=0.1):
    """Directed change-point locator on the trailing window (all cols affine-invariant).

    Scan interior splits, locate the argmax standardized mean-contrast break kbest, then emit a
    matched readout at that split. Beyond the supF the champion already has (peak-t, var-ratio,
    position), this adds the HIGHER-moment (skew/kurt/5th) shift of the located post-segment vs the
    located pre-segment -- the part _chow_supf does not measure.
    """
    parts = _locator_parts(w, min_fraction)
    if parts is None:
        return [0.0] * NLOC
    w, n, ks, _m1, _m2, _v1, _v2, tstat, lvr = parts
    it = int(np.argmax(tstat))
    il = int(np.argmax(np.abs(lvr)))
    kbest = int(ks[it])
    pre = w[:kbest]
    post = w[kbest:]
    mu_pre = pre.mean()
    sd_pre = pre.std() + _EPS
    z = np.clip((post - mu_pre) / sd_pre, -8.0, 8.0)   # clip guards z**5 float32 overflow on ~constant pre
    m = np.array([z.mean(), (z * z).mean() - 1.0, (z ** 3).mean(),
                  (z ** 4).mean() - 3.0, (z ** 5).mean()])
    hm_mag = float(np.sum(m[2:] * m[2:] / _LOC_FACT[2:]))   # skew+kurt+5th effect size (matched)
    out = [float(min(tstat[it], 50.0)), kbest / n, float(min(abs(lvr[il]), 20.0)),
           float(ks[il] / n), hm_mag, float(m[2]), float(m[3])]
    return [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in out]


def white_gloc_feats(wh):
    """Gaussian likelihood locator on history-whitened full-prefix innovations."""
    parts = _locator_parts(wh)
    if parts is None:
        return [0.0] * NWHITE_GLOC
    w, n, ks, m1, m2, v1, v2, _tstat, lvr = parts
    var0 = max(float(w.var()), _EPS)
    a = ks.astype(np.float64)
    b = (n - ks).astype(np.float64)
    gain = np.maximum(0.5 * (n * math.log(var0) - a * np.log(v1) - b * np.log(v2)), 0.0)
    ib = int(np.argmax(gain))
    kbest = int(ks[ib])
    post = w[kbest:]
    ac1 = 0.0
    if len(post) >= 5:
        centered = post - post.mean()
        ac1 = float(np.dot(centered[:-1], centered[1:]) / (np.dot(centered, centered) + _EPS))
    radius = max(2, len(gain) // 20)
    keep = np.ones(len(gain), dtype=bool)
    keep[max(0, ib - radius):min(len(gain), ib + radius + 1)] = False
    second = float(gain[keep].max()) if np.any(keep) else 0.0
    peak = float(gain[ib])
    prominence = (peak - second) / (peak + _EPS)
    out = [peak / n, kbest / n, float(m2[ib] - m1[ib]), float(lvr[ib]), ac1,
           prominence, float(np.mean(gain >= 0.8 * peak))]
    return [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in out]


def agree_gloc_feats(raw, white):
    raw_pos, white_pos = raw[1], white[1]
    return [abs(raw_pos - white_pos), min(raw[0], white[0]), raw_pos * white_pos]


def var_gloc_feats(w):
    """Read out the full prefix at the split selected by maximum variance contrast."""
    parts = _locator_parts(w)
    if parts is None:
        return [0.0] * NVAR_GLOC
    w, n, ks, m1, m2, v1, v2, tstat, lvr = parts
    ib = int(np.argmax(np.abs(lvr)))
    kbest = int(ks[ib])
    pre, post = w[:kbest], w[kbest:]
    sp, sq = pre.std() + _EPS, post.std() + _EPS
    skew_pre = float(np.mean(((pre - pre.mean()) / sp) ** 3))
    skew_post = float(np.mean(((post - post.mean()) / sq) ** 3))
    mean_shift = float((m2[ib] - m1[ib]) / math.sqrt(0.5 * (v1[ib] + v2[ib]) + _EPS))
    out = [abs(float(lvr[ib])), kbest / n, float(tstat[ib]), mean_shift,
           skew_post - skew_pre, float(v2[ib] / (v1[ib] + _EPS)), len(post) / n]
    return [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in out]


def horizon_gloc_feats(w):
    """Mean, variance and lag-1 dependence shifts over three post-location horizons."""
    parts = _locator_parts(w)
    if parts is None:
        return [0.0] * NHORIZON_GLOC
    w, _n, ks, _m1, _m2, _v1, _v2, tstat, _lvr = parts
    kbest = int(ks[int(np.argmax(tstat))])
    pre = w[:kbest]
    mu, sd = float(pre.mean()), float(pre.std()) + _EPS
    pre_z = (pre - mu) / sd
    pre_ac1 = float(np.dot(pre_z[:-1], pre_z[1:]) / (np.dot(pre_z, pre_z) + _EPS)) if len(pre_z) >= 5 else 0.0
    out = []
    for width in (32, 96, len(w) - kbest):
        post = w[max(kbest, len(w) - width):]
        z = (post - mu) / sd
        ac1 = float(np.dot(z[:-1] - z.mean(), z[1:] - z.mean()) /
                    (np.dot(z - z.mean(), z - z.mean()) + _EPS)) if len(z) >= 5 else 0.0
        out.extend([float(z.mean()), float(np.log(z.var() + _EPS)), ac1 - pre_ac1])
    return [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in out]


def wblock_feats(wh, ctx):
    """Compact break-signal readout on the WHITENED trailing window (additive alongside raw base79).

    Whitening (AR(p<=12 BIC) innovation PIT->N(0,1)) is affine-invariant, so every col here is
    affine-invariant. Under the null the whitened window is ~N(0,1); a break shows as deviation.
    8 cols: mean, var-1, skew, kurt-3, 5th std moment, half/half log-var-ratio, half/half mean-shift
    t-stat, and a standardized CUSUM magnitude (max |running mean-deviation| / (sd*sqrt(n))).
    """
    wh = np.asarray(wh, dtype=np.float64)
    n = len(wh)
    if n < 16:
        return [0.0] * WBLOCK_N
    m1 = float(wh.mean())
    v = float(wh.var()) + _EPS
    sd = math.sqrt(v)
    zc = np.clip((wh - m1) / sd, -8.0, 8.0)   # clip guards zc**5 float32 overflow
    skew = float((zc ** 3).mean())
    kurt = float((zc ** 4).mean() - 3.0)
    fifth = float((zc ** 5).mean())
    h = n // 2
    a, bb = wh[:h], wh[h:]
    va, vb = float(a.var()) + _EPS, float(bb.var()) + _EPS
    lvr = float(np.clip(math.log(vb / va), -20.0, 20.0))
    tsh = float(np.clip((bb.mean() - a.mean()) / math.sqrt(vb / len(bb) + va / len(a) + _EPS), -50.0, 50.0))
    cs = np.cumsum(wh - m1)
    cusum = float(np.max(np.abs(cs)) / (sd * math.sqrt(n) + _EPS))
    out = [m1, float(np.clip(v - 1.0, -50.0, 50.0)), float(np.clip(skew, -20.0, 20.0)),
           float(np.clip(kurt, -50.0, 50.0)), float(np.clip(fifth, -100.0, 100.0)), lvr, tsh,
           float(np.clip(cusum, 0.0, 50.0))]
    return [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in out]


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# --- base79 row builder (base79 columns byte-identical to _nldep/_wloss) ----
def _base79_row(w, ctx, refs):
    w = np.asarray(w, dtype=np.float64)
    dep_full = dep_feats(w, ctx, refs["dep"])
    parts = [np.asarray(champ_base_row(w, ctx, refs["champ"]), dtype=np.float64),
             np.asarray([dep_full[i] for i in _DEP_KEEP], dtype=np.float64)]
    ship = []
    for s in SHIP_SCALES:
        ship.extend(_ship_block(_subwin(w, s), ctx))
    parts.append(np.asarray(ship, dtype=np.float64))
    base_span = []
    for name, _spec in BASE_SPAN_SPECS:
        p, m = refs["base_span"][name]
        base_span.extend(_span_feats(w, p, m))
    parts.append(np.asarray(base_span, dtype=np.float64))
    return np.concatenate(parts)   # length NBASE=79


# --- per-series calibration null over the full base79 row (E68/E69) ---------
def fit_calib_ref(x_h, ctx, refs):
    """Per-series null band of the 79 base features from sliding pre-break windows.

    Slides `_CALIB_NULL_WINS` windows of length `_CALIB_WIN` across `x_h`, computes the SAME
    `_base79_row` on each (window-vs-full-history, exactly as online), stores per-feature
    mean/sd (E68) and robust median/MAD (E69).
    """
    x = np.asarray(x_h, dtype=np.float64)
    rows = []
    if len(x) >= _CALIB_WIN + 10:
        starts = np.linspace(0, len(x) - _CALIB_WIN, _CALIB_NULL_WINS).astype(int)
        for s in starts:
            rows.append(_base79_row(x[s:s + _CALIB_WIN], ctx, refs))
    if len(rows) >= 3:
        Z = np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0) + _EPS
        med = np.median(Z, axis=0)
        mad = 1.4826 * np.median(np.abs(Z - med), axis=0) + _EPS
    else:
        mu = np.zeros(NBASE)
        sd = np.ones(NBASE)
        med = np.zeros(NBASE)
        mad = np.ones(NBASE)
    return {"mu": mu, "sd": sd, "med": med, "mad": mad}


def _calib_block(base_row, cref, estimator, sl):
    raw = np.nan_to_num(np.asarray(base_row, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if estimator == "meansd":
        z = (raw - cref["mu"]) / cref["sd"]
    else:
        z = (raw - cref["med"]) / cref["mad"]
    return z[sl]


def _cand_feats(base_row, refs):
    """Studentize the ALREADY-COMPUTED base79 row against the per-series null (no recompute)."""
    cref = refs["calib"]
    out = []
    for _name, est, sl in CALIB_ARMS:
        out.extend(_calib_block(base_row, cref, est, sl))
    return out


# --- feature row build -----------------------------------------------------
def _refs(x_h, ctx):
    base = {"champ": fit_champ_refs(x_h), "dep": fit_dep_refs(x_h),
            "base_span": {name: _fit_span(x_h, spec) for name, spec in BASE_SPAN_SPECS}}
    base["calib"] = fit_calib_ref(x_h, ctx, base)   # needs champ/dep/base_span above
    return base


def _row(w, ctx, refs, full=None, wh=None, wh_full=None):
    w = np.asarray(w, dtype=np.float64)
    base = _base79_row(w, ctx, refs)
    feats = [base, np.asarray(_cand_feats(base, refs), dtype=np.float64)]
    if _LOCATOR:
        feats.append(np.asarray(locator_feats(w, ctx), dtype=np.float64))
    if _GLOC:                                          # scan the ENTIRE online prefix, not just w
        seq = w if full is None else np.asarray(full, dtype=np.float64)
        feats.append(np.asarray(locator_feats(seq, ctx), dtype=np.float64))
    if _GLOC6:
        seq = w if full is None else np.asarray(full, dtype=np.float64)
        wseq = seq if wh_full is None else np.asarray(wh_full, dtype=np.float64)
        raw = locator_feats(seq, ctx)
        white = white_gloc_feats(wseq)
        feats.extend([
            np.asarray(raw, dtype=np.float64),
            np.asarray(white, dtype=np.float64),
            np.asarray(agree_gloc_feats(raw, white), dtype=np.float64),
            np.asarray(locator_feats(seq, ctx, min_fraction=0.0), dtype=np.float64),
            np.asarray(var_gloc_feats(seq), dtype=np.float64),
            np.asarray(horizon_gloc_feats(seq), dtype=np.float64),
        ])
    if _WBLOCK:                                         # whitened-window readout (additive)
        wseq = w if wh is None else np.asarray(wh, dtype=np.float64)
        feats.append(np.asarray(wblock_feats(wseq, ctx), dtype=np.float64))
    return np.concatenate(feats)


def _san(a):
    if not len(a):
        return np.zeros((0, NCOL))
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _train_series(sid, x_h, x_on, tau, steps):
    if _WHITEN:
        x_h, x_on = whiten_pair(x_h, x_on)
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return np.zeros((0, NCOL)), [], [], []
    refs = _refs(x_h, ctx)
    wh_on = whiten_pair(x_h, x_on)[1] if (_WBLOCK or _GLOC6) else None
    base = active[0]
    rows, labels, tvals = [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        lo = max(0, t - WINDOW + 1)
        rows.append(_row(x_on[lo:t + 1], ctx, refs,
                         full=x_on[:t + 1] if (_GLOC or _GLOC6) else None,
                         wh=wh_on[lo:t + 1] if _WBLOCK else None,
                         wh_full=wh_on[:t + 1] if _GLOC6 else None))
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return _san(rows), labels, tvals, [int(sid)] * len(labels)


def _val_series(sid, x_h, x_on, oi):
    if _WHITEN:
        x_h, x_on = whiten_pair(x_h, x_on)
    ctx = hist_context(x_h, HIST_CAP)
    refs = _refs(x_h, ctx)
    active = _active(x_on)
    wh_on = whiten_pair(x_h, x_on)[1] if (_WBLOCK or _GLOC6) else None
    rows = [_row(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs,
                  full=x_on[:t + 1] if (_GLOC or _GLOC6) else None,
                  wh=wh_on[max(0, t - WINDOW + 1):t + 1] if _WBLOCK else None,
                  wh_full=wh_on[:t + 1] if _GLOC6 else None)
            for t in active]
    feat = _san(rows)
    midx = pd.MultiIndex.from_product([[sid], oi], names=["id", "time"])
    return midx, active, feat


def build_train(X, yi, sid_to_steps):
    jobs = []
    for sid, steps in sid_to_steps.items():
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64),
                     int(yi.loc[sid, "tau_index"]), np.asarray(steps)))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_train_series)(sid, x_h, x_on, tau, steps) for (sid, x_h, x_on, tau, steps) in jobs)
    rows = [r for r, _l, _t, _s in parts if len(r)]
    lab = np.concatenate([l for _r, l, _t, _s in parts if len(l)]) if rows else np.zeros(0, int)
    tv = np.concatenate([t for _r, _l, t, _s in parts if len(t)]) if rows else np.zeros(0, int)
    sids = np.concatenate([s for _r, _l, _t, s in parts if len(s)]) if rows else np.zeros(0, int)
    feat = np.vstack(rows) if rows else np.zeros((0, NCOL))
    return feat, lab.astype(np.int64), tv.astype(np.int64), sids.astype(np.int64)


def build_val(X, fold_ids):
    jobs = [(sid, np.asarray(split_series(X.loc[sid])[0], dtype=np.float64),
             split_series(X.loc[sid])[1].to_numpy(dtype=np.float64),
             split_series(X.loc[sid])[1].index) for sid in fold_ids]
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series)(sid, x_h, x_on, oi) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, active_all, offset = [], [], [], 0
    for midx, active, feat in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, NCOL)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCOL))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


def prepare_seed(seed):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"seed_{seed}{'_smoke' if SMOKE else ''}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    ET.SEED = seed
    X, yi, y_val, train_ids, folds = load_pool_strat()
    rng = np.random.default_rng(seed + 202)          # ranker-row RNG (unchanged stream)
    rng_pick = np.random.default_rng(seed + 303)     # separate rf-pick RNG (only drawn when biasing)
    rank_steps = {}
    pick_steps = {}
    Ls = {sid: len(split_series(X.loc[sid])[1].to_numpy()) for sid in train_ids}
    wcurve = None
    if "metric" in (_STEP_BIAS, _PICK_BIAS):
        # exact per-online-step metric weight over the train pool: w(t) = n_pos(t) * n_neg(t)
        maxL = max(Ls.values())
        alive = np.zeros(maxL + 1, dtype=np.float64)
        pos = np.zeros(maxL + 1, dtype=np.float64)
        for sid in train_ids:
            L = Ls[sid]
            alive[:L] += 1.0
            tau = int(yi.loc[sid, "tau_index"])
            if tau >= 0:
                pos[max(tau, 0):L] += 1.0
        wcurve = pos * (alive - pos)

    def _bias_p(mode, cand, sid, k):
        """Sampling prob over `cand` for the given bias mode, or None for uniform."""
        if mode == "metric" and wcurve is not None:
            pc = wcurve[cand[0]:cand[-1] + 1].astype(np.float64)
            s = pc.sum()
            if s > 0 and np.count_nonzero(pc) >= min(k, len(cand)):
                return pc / s
        elif mode == "bimodal":
            # E105 lesson inverted: oversample CLEAN steps far from the break onset.
            tau = int(yi.loc[sid, "tau_index"])
            if tau >= 0:
                pc = 1.0 + np.abs(cand.astype(np.float64) - float(tau))
                s = pc.sum()
                if s > 0 and np.count_nonzero(pc) >= min(k, len(cand)):
                    return pc / s
        return None

    for sid in train_ids:
        L = Ls[sid]
        if L <= MIN_N:
            continue
        cand = np.arange(MIN_N, L)
        rank_steps[sid] = rng.choice(cand, size=min(DENSE_K, len(cand)), replace=False,
                                     p=_bias_p(_STEP_BIAS, cand, sid, DENSE_K))
        if _PICK_BIAS != "uniform":
            kp = min(ET.STEPS_PER_SERIES, len(cand))
            pick_steps[sid] = rng_pick.choice(cand, size=kp, replace=False,
                                              p=_bias_p(_PICK_BIAS, cand, sid, ET.STEPS_PER_SERIES))
    picks = pick_steps if _PICK_BIAS != "uniform" else select_picks(X, train_ids)
    Xtr, ytr, Ttr, Str = build_train(X, yi, picks)
    Rr, Rl, Rt, Rs = build_train(X, yi, rank_steps)
    caches = [build_val(X, f) for f in folds]
    sd = dict(Xtr=Xtr, ytr=ytr, Rr=Rr, Rl=Rl, Rt=Rt, caches=caches, y_val=y_val)
    with open(path, "wb") as f:
        pickle.dump(sd, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  [cache] seed {seed} built in {time.time()-t0:.0f}s  Xtr{Xtr.shape} Rr{Rr.shape} -> {path}")
    return sd


# --- scoring ---------------------------------------------------------------
def _proba(m, X):
    return m.predict_proba(X)[:, 1]


def _auc(lab, sc):
    r = pd.Series(sc).rank(method="average").to_numpy()
    npos = float(lab.sum())
    nneg = float(len(lab) - npos)
    if npos == 0 or nneg == 0:
        return None
    return (r[lab == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def make_folds_meta(sd):
    metas = []
    for c in sd["caches"]:
        idx = c["index"]
        L = len(idx)
        df = pd.Series(np.arange(L), index=idx).to_frame("row")
        merged = df.merge(sd["y_val"], how="left", left_index=True, right_index=True)
        merged["time_online"] = merged.groupby("id").cumcount()
        groups = []
        for _, g in merged.groupby("time_online"):
            lab = g["target"].to_numpy(dtype=np.float64)
            npos = int(lab.sum())
            nneg = int(len(lab) - npos)
            if npos == 0 or nneg == 0:
                continue
            groups.append((g["row"].to_numpy(), lab, float(npos * nneg)))
        metas.append((L, c["active"], groups))
    return metas


def score_cols(sd, metas, cols, seed):
    Xtr, ytr = sd["Xtr"], sd["ytr"]
    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=N_JOBS, random_state=seed, **RF_KW)
    rf.fit(Xtr[:, cols], ytr)
    ref_rf = np.sort(_proba(rf, Xtr[:, cols]))
    rows, lab, grp = sort_into_groups(sd["Rr"][:, cols], sd["Rl"], sd["Rt"])
    rk = fit_ranker("rank_xendcg", rows, lab, grp)
    ref_rk = np.sort(_proba(rk, sd["Rr"][:, cols]))

    aucs = []
    for (L, active, groups), c in zip(metas, sd["caches"]):
        srf = np.zeros(L)
        srk = np.zeros(L)
        if len(c["feat"]):
            srf[active] = _proba(rf, c["feat"][:, cols])
            srk[active] = _proba(rk, c["feat"][:, cols])
        blend = W_RANK * pct(srk, ref_rk) + W_RF * pct(srf, ref_rf)
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
    """Candidate calibration cols must be invariant to x -> a*x + b (a>0)."""
    rng = np.random.default_rng(0)
    xh = rng.standard_normal(600) * 3.0 + 5.0
    for i in range(2, len(xh)):                        # give it mild AR structure
        xh[i] += 0.3 * xh[i - 1] - 0.1 * xh[i - 2]
    w = xh[-260:].copy()
    ctx = hist_context(xh, HIST_CAP)
    wh1 = whiten_pair(xh, xh)[1] if (_WBLOCK or _GLOC6) else None
    r1 = _row(w, ctx, _refs(xh, ctx),
              full=xh if (_GLOC or _GLOC6) else None,
              wh=wh1[-len(w):] if _WBLOCK else None,
              wh_full=wh1 if _GLOC6 else None)
    a, b = 7.3, -2.1
    xh2, w2 = a * xh + b, a * w + b
    ctx2 = hist_context(xh2, HIST_CAP)
    wh2 = whiten_pair(xh2, xh2)[1] if (_WBLOCK or _GLOC6) else None
    r2 = _row(w2, ctx2, _refs(xh2, ctx2),
              full=xh2 if (_GLOC or _GLOC6) else None,
              wh=wh2[-len(w2):] if _WBLOCK else None,
              wh_full=wh2 if _GLOC6 else None)
    cand_idx = np.arange(NBASE, NCOL)
    md = float(np.max(np.abs(r1[cand_idx] - r2[cand_idx]))) if len(cand_idx) else 0.0
    nz = int(np.sum(np.abs(r1[cand_idx]) > 1e-9))
    log(f"[affine-check] cand cols maxdiff under x->{a}x+{b}: {md:.2e}  (nonzero cand cols {nz}/{len(cand_idx)})")


def main():
    open(LOG, "w").close()
    log(f"=== PER-SERIES CALIB probe on ranker+rf E78 base79  {datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF}")
    log(f"NBASE={NBASE} NCOL={NCOL} ship_scales={SHIP_SCALES}  calib_win={_CALIB_WIN} null_wins={_CALIB_NULL_WINS}  whiten={_WHITEN}")
    log(f"arms={[(n, e, _ARM_W[n]) for n, e, _sl in CALIB_ARMS]}")
    log(f"candidates={[n for n, _ in CANDIDATES]}")
    _affine_check()

    if not os.path.exists(RAW):
        with open(RAW, "w") as f:
            f.write("seed\tcandidate\tbase_auc\tcand_auc\tdelta\n")

    per_seed = {}
    for seed in SEARCH_SEEDS:
        sd = prepare_seed(seed)
        metas = make_folds_meta(sd)
        t0 = time.time()
        baseline_names = os.environ.get("BASELINE", "").split(",") if os.environ.get("BASELINE") else []
        baseline_cols = cand_cols(baseline_names) if baseline_names else BASE_COLS
        base_auc = score_cols(sd, metas, baseline_cols, seed)
        baseline_label = "+".join(baseline_names) if baseline_names else "base79"
        log(f"[seed {seed}] {baseline_label} ranker+rf = {base_auc:.4f}  ({time.time()-t0:.0f}s)")
        res = {}
        for name, cand_names in CANDIDATES:
            tc = time.time()
            cols = cand_cols(cand_names)
            auc = score_cols(sd, metas, cols, seed)
            d = auc - base_auc
            res[name] = d
            with open(RAW, "a") as f:
                f.write(f"{seed}\t{name}\t{base_auc:.6f}\t{auc:.6f}\t{d:+.6f}\n")
            log(f"[seed {seed}] {name:<14} {auc:.4f}  d={d:+.4f}  ({time.time()-tc:.0f}s)")
        per_seed[seed] = res

    names = [n for n, _ in CANDIDATES]
    agg = []
    for n in names:
        ds = [per_seed[s][n] for s in SEARCH_SEEDS if n in per_seed[s]]
        if not ds:
            continue
        md = float(np.mean(ds))
        npos = sum(1 for d in ds if d > 0)
        agg.append((n, md, min(ds), max(ds), npos, len(ds)))
    agg.sort(key=lambda r: -r[1])
    with open(SUMMARY, "w") as f:
        f.write(f"# per-series calibration probe on ranker+rf E78 base79  {datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF}\n")
        f.write(f"# base79 = E78 champion (ship_scales={SHIP_SCALES}); arms = studentize base79 vs\n")
        f.write(f"# per-series pre-break null (E68/E69 port), calib_win={_CALIB_WIN} null_wins={_CALIB_NULL_WINS}\n")
        f.write(f"# czms_champ51 / czms_all79 (mean/sd) ; czrb_all79 (robust med/MAD) ; ISOLATION vs base79\n")
        f.write(f"# RELAXED gate: any mean_d>0 is a candidate; rank by mean_d. NOT the retired t>=2.5 bar.\n\n")
        f.write(f"{'candidate':<14}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for n, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md > 0 and npos == tot) else ""
            f.write(f"{n:<14}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
