"""Shape/tail feature hunt on synthetic data: what catches the residual champion's blind spots?

The residual variance ratio owns dynamics breaks but is BLIND to pure shape breaks (skew, fat tails).
This diagnoses candidate SHAPE/TAIL features on the two weakest families (skew, tdist) + all others:
  residual higher-moments (skew/exkurt of the AR8 residuals vs the historical residuals) -- our lens, 3rd/4th moment,
  robust contrasts on the raw window vs history: bowley skewness, tail-heaviness ratio (q95-q05)/(q75-q25),
  plus the champion's own shape feats (skew_diff, kurt_diff, tail3, ks) to check if they already cover it.

Per-family detectability AUC = magnitude-of-change (|feature - unbroken baseline|) separating a POST-break
window from a PRE-break window. High AUC on skew/tdist for a NEW feat that the champion feats miss = a
real complementary lever to harness-test.

Usage:  .venv/bin/python -u _synth_diag.py
"""
import math

import numpy as np
from sklearn.metrics import roc_auc_score

from _synth_gen import PARAM_FAMILIES, gen_break

WINDOW = 250
N_PER_FAMILY = 400
HIST_N = 1200


def _fit_ar(x, p=8):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    cols = [x[p - lag:n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
    resid = x[p:] - Xd @ coef
    return coef, float(np.var(resid)) + 1e-12, resid


def _ar_resid(w, coef, p=8):
    n = len(w)
    cols = [w[p - lag:n - lag] for lag in range(1, p + 1)]
    return w[p:] - np.column_stack(cols + [np.ones(n - p)]) @ coef


def _skew(a):
    a = np.asarray(a, dtype=np.float64)
    s = a.std()
    return 0.0 if s < 1e-9 else float(np.mean(((a - a.mean()) / s) ** 3))


def _exkurt(a):
    a = np.asarray(a, dtype=np.float64)
    s = a.std()
    return 0.0 if s < 1e-9 else float(np.mean(((a - a.mean()) / s) ** 4)) - 3.0


def _bowley(a):
    q = np.quantile(a, [0.25, 0.5, 0.75])
    return float((q[2] + q[0] - 2 * q[1]) / (q[2] - q[0] + 1e-9))


def _tailratio(a):
    q = np.quantile(a, [0.05, 0.25, 0.75, 0.95])
    return float((q[3] - q[0]) / (q[2] - q[1] + 1e-9))


def _ks(w_sorted, hist_sorted):
    allv = np.concatenate([hist_sorted, w_sorted])
    ch = np.searchsorted(hist_sorted, allv, side="right") / len(hist_sorted)
    cw = np.searchsorted(w_sorted, allv, side="right") / len(w_sorted)
    return float(np.max(np.abs(ch - cw)))


FEATS = ["resid_vr", "resid_skew", "resid_kurt", "skew_diff", "kurt_diff", "tail3",
         "bowley_d", "tailratio_d", "ks"]


def _features(win, coef, rv, rskew, rkurt, hd):
    r = _ar_resid(win, coef)
    dev = np.abs(win - hd["mu"]) / hd["sd"]
    return {
        "resid_vr": math.log((float(np.var(r)) + 1e-12) / rv),
        "resid_skew": _skew(r) - rskew,
        "resid_kurt": _exkurt(r) - rkurt,
        "skew_diff": _skew(win) - hd["skew"],
        "kurt_diff": _exkurt(win) - hd["kurt"],
        "tail3": float((dev > 3.0).mean()),
        "bowley_d": _bowley(win) - hd["bowley"],
        "tailratio_d": _tailratio(win) - hd["tailratio"],
        "ks": _ks(np.sort(win), hd["sorted"]),
    }


def main():
    rng = np.random.default_rng(0)
    print(f"=== shape/tail feature hunt | {N_PER_FAMILY}/family | window={WINDOW} ===", flush=True)
    print("family        " + "".join(f"{f:>12}" for f in FEATS), flush=True)
    for fam in PARAM_FAMILIES:
        cols = {f: ([], []) for f in FEATS}
        for _ in range(N_PER_FAMILY):
            pre, post = gen_break(rng, fam, HIST_N + WINDOW + 20, WINDOW + 20)
            hist = pre[:HIST_N]
            unbroken, broken = pre[HIST_N:HIST_N + WINDOW], post[:WINDOW]
            if len(unbroken) < WINDOW or len(broken) < WINDOW:
                continue
            coef, rv, hres = _fit_ar(hist)
            rskew, rkurt = _skew(hres), _exkurt(hres)
            hd = {"mu": float(hist.mean()), "sd": float(hist.std()) + 1e-9,
                  "skew": _skew(hist), "kurt": _exkurt(hist), "bowley": _bowley(hist),
                  "tailratio": _tailratio(hist), "sorted": np.sort(hist)}
            fu = _features(unbroken, coef, rv, rskew, rkurt, hd)
            fb = _features(broken, coef, rv, rskew, rkurt, hd)
            for f in FEATS:
                cols[f][0].append(fu[f])
                cols[f][1].append(fb[f])
        row, aucs = f"{fam:<14}", {}
        for f in FEATS:
            pre = np.nan_to_num(np.asarray(cols[f][0]), nan=0.0, posinf=0.0, neginf=0.0)
            post = np.nan_to_num(np.asarray(cols[f][1]), nan=0.0, posinf=0.0, neginf=0.0)
            y = np.r_[np.zeros(len(pre)), np.ones(len(post))]
            x = np.abs(np.r_[pre, post] - float(np.median(pre)))
            aucs[f] = float(roc_auc_score(y, x))
            row += f"{aucs[f]:>12.3f}"
        mark = "  <<<" if fam in ("skew", "tdist") else ""
        print(row + mark, flush=True)
    print("\n(skew/tdist = residual blind spots. NEW feats = resid_skew/resid_kurt/bowley_d/tailratio_d; "
          "champion already has skew_diff/kurt_diff/tail3/ks.)", flush=True)


if __name__ == "__main__":
    main()
