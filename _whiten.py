"""Rosenblatt-style stream whitening (WHITEN feature bet, forum "oracle ceilings" lead).

The champion computes its base79 family on the RAW value stream (it only PARTIALLY whitens
internally via AR8 residual + dependence blocks). The ADIA forum "oracle ceilings" post, and our
own `_eda_oracle_run.py` per-age check, both show a properly whitened stream carries markedly more
break signal (our EDA: +0.03..+0.07 TS-AUC on the raw 5-moment statistic at every online age).

This module reproduces the EXACT transform that produced that EDA signal (AR(p<=12 by BIC)
innovations, PIT through the history-residual empirical CDF into a standard normal) and exposes a
`whiten_pair(x_h, x_on)` that whitens BOTH the history and the online stream with the SAME whitener
so the champion's window-vs-history features live in one consistent whitened space.

Length is preserved on the online side (primed with the history tail) so tau / online-step indexing
is unchanged. The history side loses its first p (<=12) points to the AR lag window -- negligible for
the null-band fit.

AFFINE-INVARIANT: `fit_ar_bic` carries an intercept, so AR residuals are scale-equivariant and
shift-invariant (r -> a*r under x -> a*x+b); the PIT is rank-based -> monotone-invariant. The whitened
stream is therefore invariant to x -> a*x+b, matching the base pipeline's guarantee.
"""
import numpy as np
from scipy.stats import norm

_EPS = 1e-9


def fit_ar_bic(x, pmax=12):
    """AR order 0..pmax chosen by BIC (intercept included); returns (p, coef). Verbatim from the EDA."""
    x = np.asarray(x, float)
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


def _ar_resid(seq, p, coef):
    """AR residuals of `seq` given (p, coef) with intercept. Verbatim from the EDA."""
    seq = np.asarray(seq, float)
    n = len(seq)
    if p == 0:
        return seq - coef[0]
    if n <= p:
        return np.array([])
    Xd = np.column_stack([seq[p - l:n - l] for l in range(1, p + 1)] + [np.ones(n - p)])
    return seq[p:] - Xd @ coef


def whiten_pair(x_h, x_on):
    """Whiten BOTH history and online with one AR-innovation PIT->normal whitener fit on the history.

    Returns (wh_h, wh_on):
      wh_h  : whitened history, length len(x_h) - p  (feeds refs / null bands)
      wh_on : whitened online, length len(x_on)       (index-aligned to x_on -> tau/step preserved)
    Falls back to a plain history z-score (or passthrough) when the series is too short to whiten.
    """
    xh = np.asarray(x_h, float)
    xo = np.asarray(x_on, float)
    if len(xh) < 30 or len(xo) < 3:
        return xh, xo
    # Standardize BOTH streams by a history-based robust location/scale BEFORE fitting. Under
    # x -> a*x+b (a>0): loc -> a*loc+b and scl -> a*scl, so the standardized series is IDENTICAL for
    # x and its affine image -> every downstream step (BIC order, AR coef, residuals, PIT) is
    # computed on identical inputs -> the whitened output is affine-invariant to float precision.
    # (fit_ar_bic's BIC has an (n-p)*log(a^2) scale leak that otherwise flips the selected order.)
    loc = float(np.median(xh))
    scl = 1.4826 * float(np.median(np.abs(xh - loc))) + _EPS
    xh = (xh - loc) / scl
    xo = (xo - loc) / scl
    p, coef = fit_ar_bic(xh)
    rh = _ar_resid(xh, p, coef)
    if len(rh) < 10:
        mu, sd = xh.mean(), xh.std() + _EPS
        return (xh - mu) / sd, (xo - mu) / sd
    rh_sorted = np.sort(rh)

    def pit(resid):
        ranks = np.searchsorted(rh_sorted, resid, side="right") / (len(rh_sorted) + 1.0)
        return norm.ppf(np.clip(ranks, 1e-4, 1 - 1e-4))

    wh_h = pit(rh)
    primed = np.concatenate([xh[-p:], xo]) if p > 0 else xo
    wh_on = pit(_ar_resid(primed, p, coef))
    return wh_h, wh_on
