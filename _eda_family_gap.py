"""Extended EDA — where is the shipped 61-col champion weakest, and what would fix it?

Part A  GAP MAP: rank the break families by how well the CHAMPION representation already
        detects them. For each family generate break + no-break synthetic series (standardised
        exactly like production), build the shipped 61-col champion feature vector on the final
        online window, out-of-fold-score with the champion HGB, and report per-family AUC
        (break vs no-break). LOW auc = the family with the most head-room.

Part B  STATISTIC / MODEL BATTERY: for every family, score a battery of EXTRA descriptors that
        are NOT in the champion — distributional two-sample tests, L-moments, tail/shape,
        dependence-model distances (AR-coef, PACF, log-spectrum), volatility-clustering — two
        ways:  (single) single-feature break-vs-no-break AUC within the family, and
        (+champ)  AUC of a 2-feature logistic on [champion OOF score, the descriptor].
        (+champ) - (champ) = the ORTHOGONAL lift the descriptor would add on that family.

Usage:  .venv/bin/python -u _eda_family_gap.py
"""
import math
import warnings

import numpy as np
from scipy.stats import (anderson_ksamp, cramervonmises_2samp, energy_distance,
                         wasserstein_distance)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from _ensemble_tune import make_hgb
from _exp_fast import HIST_CAP, hist_context
from _night_banks import champ_base_row, dep_feats, fit_champ_refs, fit_dep_refs
from _synth_gen import PARAM_FAMILIES, gen_break, gen_nobreak

warnings.filterwarnings("ignore")

WINDOW = 250
HIST_N = 1200
TAU = 40           # break lands early so the final window is fully post-break
POST = 320         # post length > WINDOW  -> last-250 window is pure post
K = 260            # series per class per family
SEED = 0
_DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]   # shipped b_no_accf (drops dacf1/2/5/10)


# --------------------------------------------------------------------------- champion 61-col
def champ61(w, ctx, cref, dref):
    base = champ_base_row(w, ctx, cref)              # 51
    df = dep_feats(w, ctx, dref)                     # 14
    return np.concatenate([base, np.asarray([df[i] for i in _DEP_KEEP], dtype=np.float64)])


def make_series(rng, fam, broken):
    if broken:
        pre, post = gen_break(rng, fam, HIST_N + TAU, POST)
        seg = np.concatenate([pre, post])
        ref = pre
    else:
        seg = gen_nobreak(rng, fam, HIST_N + TAU + POST)
        ref = seg[:HIST_N + TAU]
    mu, sd = float(ref.mean()), float(ref.std())
    seg = (seg - mu) / (sd if sd > 1e-6 else 1.0)
    x_h = seg[:HIST_N]
    w = seg[-WINDOW:]
    return x_h, w


# --------------------------------------------------------------------------- extra descriptors
def _levinson_pacf(x, p):
    x = np.asarray(x, float) - x.mean()
    n = len(x)
    ac = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(p + 1)])
    if ac[0] < 1e-12:
        return np.zeros(p), np.zeros(p)
    a = np.zeros(p)
    pacf = np.zeros(p)
    e = ac[0]
    for k in range(p):
        acc = ac[k + 1] - np.dot(a[:k], ac[k:0:-1])
        kref = acc / e if e > 1e-12 else 0.0
        pacf[k] = kref
        a_new = a.copy()
        a_new[k] = kref
        for j in range(k):
            a_new[j] = a[j] - kref * a[k - 1 - j]
        a = a_new
        e *= max(1.0 - kref * kref, 1e-9)
    return a, pacf


def _lmoments(a):
    x = np.sort(np.asarray(a, float))
    n = len(x)
    if n < 4:
        return 0.0, 0.0
    j = np.arange(1, n + 1, dtype=float)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((j - 1) * (j - 2) * (j - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    if abs(l2) < 1e-9:
        return 0.0, 0.0
    return l3 / l2, l4 / l2          # L-skew, L-kurt


def _bowley(a):
    q = np.quantile(a, [0.25, 0.5, 0.75])
    return float((q[2] + q[0] - 2 * q[1]) / (q[2] - q[0] + 1e-9))


def _tailratio(a):
    q = np.quantile(a, [0.05, 0.25, 0.75, 0.95])
    return float((q[3] - q[0]) / (q[2] - q[1] + 1e-9))


def _logspec(a):
    a = np.asarray(a, float) - a.mean()
    p = np.abs(np.fft.rfft(a)) ** 2
    p = p[1:]                          # drop DC
    return np.log(p + 1e-12)


def _autocorr_sq(a, lag=1):
    r = np.asarray(a, float)
    r = r * r
    r = r - r.mean()
    d = np.dot(r, r)
    return float(np.dot(r[:-lag], r[lag:]) / d) if d > 1e-12 else 0.0


def _dfa_hurst(x):
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    scales = (8, 16, 32, 64)
    F = []
    for s in scales:
        m = len(y) // s
        if m < 2:
            F.append(np.nan)
            continue
        yy = y[:m * s].reshape(m, s)
        t = np.arange(s)
        A = np.vstack([t, np.ones(s)]).T
        coef, *_ = np.linalg.lstsq(A, yy.T, rcond=None)
        fit = A @ coef
        F.append(np.sqrt(np.mean((yy.T - fit) ** 2)))
    F = np.asarray(F, float)
    ok = np.isfinite(F) & (F > 0)
    if ok.sum() < 2:
        return 0.5
    sl = np.polyfit(np.log(np.asarray(scales)[ok]), np.log(F[ok]), 1)[0]
    return float(sl)


def _arcoef(x, p=4):
    x = np.asarray(x, float)
    n = len(x)
    cols = [x[p - lag:n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
    return coef[:p]


def _mmd_rbf(x, y, gamma=0.5, cap=160):
    rng = np.random.default_rng(0)
    if len(x) > cap:
        x = rng.choice(x, cap, replace=False)
    if len(y) > cap:
        y = rng.choice(y, cap, replace=False)
    def k(a, b):
        d = a[:, None] - b[None, :]
        return np.exp(-gamma * d * d)
    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def descriptors(w, x_h):
    """Battery of EXTRA descriptors (window vs history). Returns dict name->float."""
    hs = x_h[-WINDOW:] if len(x_h) >= WINDOW else x_h        # length-matched hist window
    hsamp = x_h if len(x_h) <= 400 else x_h[:: max(1, len(x_h) // 400)][:400]
    d = {}
    # --- distribution two-sample tests (shape / tails) ---
    try:
        d["ad"] = float(anderson_ksamp([w, hsamp]).statistic)
    except Exception:
        d["ad"] = 0.0
    try:
        d["cvm"] = float(cramervonmises_2samp(w, hsamp).statistic)
    except Exception:
        d["cvm"] = 0.0
    d["wass"] = float(wasserstein_distance(w, hsamp))
    d["energy"] = float(energy_distance(w, hsamp))
    d["mmd"] = _mmd_rbf(w, hsamp)
    # --- moment / L-moment shape ---
    ls_w, lk_w = _lmoments(w)
    ls_h, lk_h = _lmoments(x_h)
    d["lskew_d"] = ls_w - ls_h
    d["lkurt_d"] = lk_w - lk_h
    d["bowley_d"] = _bowley(w) - _bowley(x_h)
    d["tailratio_d"] = _tailratio(w) - _tailratio(x_h)
    # --- dependence-model distances ---
    aw, pw = _levinson_pacf(w, 5)
    ah, ph = _levinson_pacf(hs, 5)
    d["pacf_dist"] = float(np.linalg.norm(pw - ph))
    d["arcoef_dist"] = float(np.linalg.norm(_arcoef(w, 4) - _arcoef(hs, 4)))
    sw, sh = _logspec(w), _logspec(hs)
    L = min(len(sw), len(sh))
    d["spec_dist"] = float(np.linalg.norm(sw[:L] - sh[:L]) / math.sqrt(L))
    d["hurst_d"] = _dfa_hurst(w) - _dfa_hurst(x_h)
    # --- volatility clustering ---
    d["archac_d"] = _autocorr_sq(w, 1) - _autocorr_sq(x_h, 1)
    d["voratio"] = math.log((float(np.var(np.diff(w))) + 1e-12) / (float(np.var(np.diff(x_h))) + 1e-12))
    return d


DESC_NAMES = ["ad", "cvm", "wass", "energy", "mmd", "lskew_d", "lkurt_d", "bowley_d",
              "tailratio_d", "pacf_dist", "arcoef_dist", "spec_dist", "hurst_d",
              "archac_d", "voratio"]


# --------------------------------------------------------------------------- main
def main():
    print(f"=== EDA family gap map | {K}/class/family | window={WINDOW} | 61-col champion ===\n",
          flush=True)
    rng = np.random.default_rng(SEED)

    X, y, fam_arr, D = [], [], [], []
    for fam in PARAM_FAMILIES:
        for broken in (True, False):
            for _ in range(K):
                x_h, w = make_series(rng, fam, broken)
                ctx = hist_context(x_h, HIST_CAP)
                cref = fit_champ_refs(x_h)
                dref = fit_dep_refs(x_h)
                X.append(champ61(w, ctx, cref, dref))
                D.append(descriptors(w, x_h))
                y.append(1 if broken else 0)
                fam_arr.append(fam)
        print(f"  built {fam}", flush=True)
    X = np.nan_to_num(np.vstack(X), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y)
    fam_arr = np.asarray(fam_arr)

    # --- Part A: champion OOF score, per-family AUC ---
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    strat = np.array([f"{f}_{t}" for f, t in zip(fam_arr, y)])
    for tr, te in skf.split(X, strat):
        m = make_hgb()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

    print("\n===== PART A — champion per-family detection AUC (low = most head-room) =====",
          flush=True)
    gaps = []
    for fam in PARAM_FAMILIES:
        mask = fam_arr == fam
        auc = roc_auc_score(y[mask], oof[mask])
        gaps.append((fam, auc))
    for fam, auc in sorted(gaps, key=lambda z: z[1]):
        bar = "#" * int((auc - 0.5) * 100)
        print(f"  {fam:<12} champ_auc={auc:.3f}  {bar}", flush=True)

    # --- Part B: descriptor battery per family ---
    Dmat = {nm: np.nan_to_num(np.asarray([row[nm] for row in D]), nan=0.0, posinf=0.0, neginf=0.0)
            for nm in DESC_NAMES}
    print("\n===== PART B — per-family descriptor value =====", flush=True)
    print("  (single) single-feature AUC   (+champ) AUC of logistic[champ_oof, desc]   "
          "lift = (+champ) - champ_auc\n", flush=True)
    weak = [f for f, _ in sorted(gaps, key=lambda z: z[1])]
    champ_auc = {f: a for f, a in gaps}
    for fam in weak:
        mask = fam_arr == fam
        yf = y[mask]
        base = champ_auc[fam]
        rows = []
        for nm in DESC_NAMES:
            col = Dmat[nm][mask]
            try:
                s_auc = roc_auc_score(yf, col)
                s_auc = max(s_auc, 1 - s_auc)
            except Exception:
                s_auc = 0.5
            # combined logistic [champ_oof, desc], 5-fold within family
            feats = np.column_stack([oof[mask], col])
            comb = np.zeros(len(yf))
            try:
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(feats, yf):
                    lr = LogisticRegression(max_iter=500)
                    lr.fit(StandardScaler().fit_transform(feats[tr]), yf[tr])
                    comb[te] = lr.predict_proba(
                        StandardScaler().fit(feats[tr]).transform(feats[te]))[:, 1]
                c_auc = roc_auc_score(yf, comb)
            except Exception:
                c_auc = base
            rows.append((nm, s_auc, c_auc, c_auc - base))
        rows.sort(key=lambda z: z[3], reverse=True)
        print(f"--- {fam}  (champ_auc={base:.3f}) ---", flush=True)
        for nm, s_auc, c_auc, lift in rows[:6]:
            flag = "  <== " if lift > 0.02 else ""
            print(f"    {nm:<12} single={s_auc:.3f}  +champ={c_auc:.3f}  lift={lift:+.3f}{flag}",
                  flush=True)
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
