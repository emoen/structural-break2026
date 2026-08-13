"""Feature-family builders for E30 CV re-validation (importable so joblib/loky
workers can un-pickle them — functions defined in __main__ cannot be shipped to
worker processes).

Verbatim family stats from E25 (exp_bank_selection). Each `feat_*` builder returns
the champion base15-per-scale vector plus the named extra family stat(s) per scale.
"""
import math

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata, anderson_ksamp

from _feat_lib import _feat

BASE_KEYS = ("spec_entropy",)
SCALES = (60, 130, 250)
AR_ORDER = 4
SAMPEN_CAP = 128
EPS = 1e-12


def _subwin(w, s):
    n = len(w)
    return w if s >= n else w[-s:]


def _hurst_rs(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 16:
        return 0.5
    y = x - x.mean()
    z = np.cumsum(y)
    R = float(z.max() - z.min())
    S = float(x.std())
    if S < EPS or R < EPS:
        return 0.5
    return math.log(R / S) / math.log(n)


def _ar_coefs(x, p=AR_ORDER):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n <= p + 1:
        return [0.0] * p
    r = np.array([float(np.dot(x[:n - k], x[k:])) / n for k in range(p + 1)])
    if r[0] < EPS:
        return [0.0] * p
    R = np.empty((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            R[i, j] = r[abs(i - j)]
    try:
        phi = np.linalg.solve(R, r[1:p + 1])
    except np.linalg.LinAlgError:
        phi = np.zeros(p)
    return [float(v) for v in phi]


def _anderson(ws, h_sub):
    try:
        return float(anderson_ksamp([np.asarray(ws), np.asarray(h_sub)]).statistic)
    except Exception:
        return 0.0


def _levene_bf(ws, h_dev):
    za = np.abs(ws - np.median(ws))
    zb = h_dev
    n1, n2 = len(za), len(zb)
    N = n1 + n2
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = float(za.mean()), float(zb.mean())
    m = (za.sum() + zb.sum()) / N
    num = (N - 2) * (n1 * (m1 - m) ** 2 + n2 * (m2 - m) ** 2)
    den = float(((za - m1) ** 2).sum() + ((zb - m2) ** 2).sum())
    return float(num / den) if den > EPS else 0.0


def _fligner(ws, h_dev):
    za = np.abs(ws - np.median(ws))
    zb = h_dev
    n1, n2 = len(za), len(zb)
    N = n1 + n2
    if n1 < 2 or n2 < 2:
        return 0.0
    a = ndtri(0.5 + rankdata(np.concatenate([za, zb])) / (2.0 * (N + 1)))
    abar = float(a.mean())
    v = float(a.var())
    if v <= EPS:
        return 0.0
    a1 = a[:n1]
    return (n1 * (float(a1.mean()) - abar) ** 2 +
            n2 * (float(a[n1:].mean()) - abar) ** 2) / v


def _mood_median(ws, h_sorted):
    med = float(np.median(np.concatenate([ws, h_sorted])))
    n1, n2 = len(ws), len(h_sorted)
    N = n1 + n2
    T = int(np.sum(ws > med) + np.sum(h_sorted > med))
    if T == 0 or T == N:
        return 0.0
    a_above = int(np.sum(ws > med))
    exp_a = n1 * T / N
    var = n1 * n2 * T * (N - T) / (N * N * (N - 1))
    return (a_above - exp_a) ** 2 / var if var > EPS else 0.0


def _sampen(x, m=2):
    x = np.asarray(x, dtype=np.float64)
    if len(x) > SAMPEN_CAP:
        x = x[-SAMPEN_CAP:]
    n = len(x)
    if n < m + 2:
        return 0.0
    r = 0.2 * float(x.std())
    if r <= 0:
        return 0.0

    def _count(mm):
        tmpl = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        c = 0
        for i in range(len(tmpl)):
            d = np.max(np.abs(tmpl - tmpl[i]), axis=1)
            c += int(np.sum(d <= r)) - 1
        return c / 2.0

    B = _count(m)
    A = _count(m + 1)
    if B <= 0 or A <= 0:
        return 0.0
    return -math.log(A / B)


def _base(ws, ctx):
    return list(_feat(ws, ctx, keys=BASE_KEYS))


# --- named builders (picklable, no partials) ------------------------------
def champion_feat(w, ctx):
    out = []
    for s in SCALES:
        out.extend(_base(_subwin(w, s), ctx))
    return out


def feat_ar(w, ctx):
    out = []
    for s in SCALES:
        ws = _subwin(w, s)
        out.extend(_base(ws, ctx))
        out.extend(_ar_coefs(ws))
    return out


def feat_hurst(w, ctx):
    out = []
    for s in SCALES:
        ws = _subwin(w, s)
        out.extend(_base(ws, ctx))
        out.append(_hurst_rs(ws))
    return out


def feat_ar_hurst(w, ctx):
    out = []
    for s in SCALES:
        ws = _subwin(w, s)
        out.extend(_base(ws, ctx))
        out.append(_hurst_rs(ws))
        out.extend(_ar_coefs(ws))
    return out


def feat_levene(w, ctx):
    h = ctx["sample_sorted"]
    h_dev = np.abs(h - np.median(h))
    out = []
    for s in SCALES:
        ws = _subwin(w, s)
        out.extend(_base(ws, ctx))
        out.append(_levene_bf(ws, h_dev))
    return out


def feat_fligner(w, ctx):
    h = ctx["sample_sorted"]
    h_dev = np.abs(h - np.median(h))
    out = []
    for s in SCALES:
        ws = _subwin(w, s)
        out.extend(_base(ws, ctx))
        out.append(_fligner(ws, h_dev))
    return out


def feat_mood(w, ctx):
    h = ctx["sample_sorted"]
    out = []
    for s in SCALES:
        ws = _subwin(w, s)
        out.extend(_base(ws, ctx))
        out.append(_mood_median(ws, h))
    return out


def feat_sampen(w, ctx):
    out = []
    for s in SCALES:
        ws = _subwin(w, s)
        out.extend(_base(ws, ctx))
        out.append(_sampen(ws))
    return out


def feat_bank_all(w, ctx):
    h = ctx["sample_sorted"]
    h_dev = np.abs(h - np.median(h))
    h_sub = h if len(h) <= 100 else h[:: max(1, len(h) // 100)][:100]
    out = []
    for s in SCALES:
        ws = _subwin(w, s)
        out.extend(_base(ws, ctx))
        out.append(_hurst_rs(ws))
        out.extend(_ar_coefs(ws))
        out.append(_anderson(ws, h_sub))
        out.append(_levene_bf(ws, h_dev))
        out.append(_fligner(ws, h_dev))
        out.append(_mood_median(ws, h))
        out.append(_sampen(ws))
    return out
