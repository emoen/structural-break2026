"""Shippable feature drop-in: champion features + residual-variance-ratio block.

Confirmed lever (Exp C' / residual-lens ablation, 7/7 seeds): a structural break shows up as the
REFERENCE model's residuals blowing up. Fit AR(p) on the historical (pre-break) segment, apply it
forward onto the current online window, and the log ratio of window-residual variance to
reference-residual variance is the single strongest add-on feature we have found -- larger than the
champion's own multiscale edge and robust on fresh seeds.

Two shippable variants (choose per LB):
  * champion_feat_min  (45 + 4)  -- only the residual variance ratio at AR{1,2,3,4}. Parsimonious;
                                    strongest+cleanest local signal (mean_d +0.0191, min_t +4.13).
  * champion_feat_rich (45 + 24) -- all six residual statistics x AR{1,2,3,4}. Local tie with min
                                    (+0.0185); more features -> let the LB adjudicate rich vs min.

Integration (one change): build the context with `make_context(x_h)` instead of `hist_context(x_h)`
so the reference AR fits ride along on the ctx, then pass `champion_feat_min`/`champion_feat_rich`
wherever `champion_feat` is used. Signature is unchanged: feat_fn(window, ctx) -> list[float].

Also recommended (independent, additive training-time lever): STEPS_PER_SERIES = 30.
"""
import math

import numpy as np

from _ensemble_tune import champion_feat
from _exp_fast import HIST_CAP, hist_context

AR_ORDERS = (1, 2, 3, 4)


def _fit_ar(x, p):
    """AR(p) least-squares fit on an ordered series -> (coef[p+1], residual variance)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < p + 8:
        return None
    cols = [x[p - lag: n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    coef, *_ = np.linalg.lstsq(Xd, x[p:], rcond=None)
    return coef, float(np.var(x[p:] - Xd @ coef)) + 1e-12


def _ar_resid(w, coef, p):
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    if n < p + 3:
        return None
    cols = [w[p - lag: n - lag] for lag in range(1, p + 1)]
    Xd = np.column_stack(cols + [np.ones(n - p)])
    return w[p:] - Xd @ coef


def _acf(x, k):
    if len(x) <= k:
        return 0.0
    x0 = x - x.mean()
    d = float(np.dot(x0, x0)) + 1e-12
    return float(np.dot(x0[k:], x0[:-k])) / d


def fit_ar_ref(x_h):
    """Reference AR fits on the historical segment, one per order in AR_ORDERS."""
    x = np.asarray(x_h, dtype=np.float64)
    return {p: _fit_ar(x, p) for p in AR_ORDERS}


def make_context(x_h):
    """hist_context(x_h) augmented with reference AR fits under ctx['ar_ref']."""
    ctx = hist_context(x_h, HIST_CAP)
    ctx["ar_ref"] = fit_ar_ref(x_h)
    return ctx


def _varratio(win, ar_ref):
    out = []
    for p in AR_ORDERS:
        m = ar_ref.get(p)
        r = _ar_resid(win, m[0], p) if m is not None else None
        if r is None or len(r) < 5:
            out.append(0.0)
        else:
            out.append(math.log((float(np.var(r)) + 1e-12) / m[1]))
    return out


def _all_stats(win, ar_ref):
    out = []
    for p in AR_ORDERS:
        m = ar_ref.get(p)
        r = _ar_resid(win, m[0], p) if m is not None else None
        if r is None or len(r) < 5:
            out += [0.0] * 6
            continue
        sig = math.sqrt(m[1])
        rv = float(np.var(r)) + 1e-12
        n = len(r)
        q = 0.0
        for k in (1, 2, 3):
            if n - k > 0:
                ak = _acf(r, k)
                q += ak * ak / (n - k)
        cs = np.cumsum(r) / (sig * math.sqrt(n))
        out += [math.log(rv / m[1]), float(np.mean(r)) / math.sqrt(rv), _acf(r, 1),
                n * (n + 2) * q, float(np.max(np.abs(cs))), float(np.mean(np.abs(r) > 3.0 * sig))]
    return out


def champion_feat_min(win, ctx):
    """45 champion features + 4 residual-variance-ratio features (AR 1-4). ctx from make_context."""
    return list(champion_feat(win, ctx)) + _varratio(win, ctx["ar_ref"])


def champion_feat_rich(win, ctx):
    """45 champion features + 24 residual statistics (6 x AR 1-4). ctx from make_context."""
    return list(champion_feat(win, ctx)) + _all_stats(win, ctx["ar_ref"])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x_h = rng.normal(size=400)
    ctx = make_context(x_h)
    win = rng.normal(size=250)
    fm, fr = champion_feat_min(win, ctx), champion_feat_rich(win, ctx)
    assert len(fm) == 49 and len(fr) == 69, (len(fm), len(fr))
    # break-like window (variance inflation) must lift the residual-variance-ratio features
    broken = np.concatenate([rng.normal(size=125), rng.normal(scale=3.0, size=125)])
    vr = _varratio(broken, ctx["ar_ref"])
    print("min dims:", len(fm), "rich dims:", len(fr))
    print("varratio (calm window):", [round(v, 3) for v in _varratio(win, ctx["ar_ref"])])
    print("varratio (broken window):", [round(v, 3) for v in vr])
    assert max(vr) > 0.5, "variance-inflation break should raise a residual-variance ratio"
    print("OK")
