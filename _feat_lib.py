"""Feature-battery library for the overnight run.

Each candidate is base14 (compute_features) + one or more EXTRA feature groups. The
groups are cheap, vectorised, history-independent-or-vs-history scalars, guarded against
NaN/inf. Candidates are built as functools.partial(_feat, keys=(...)) so they pickle
cleanly for joblib workers (no closures).

A group fn has signature (w, ctx) -> list[float]; w is the trailing window (>=15 pts),
ctx is the hist_context dict from _exp_fast (mu, sd, var, skew, kurt, ac1, ac2, iqr, rng,
sc, sample_sorted). Robust hist stats are cached on ctx["_ol"].
"""
import math
from functools import partial

import numpy as np

from _exp_fast import compute_features as base_features, _autocorr, _fast_ks


def _hist(ctx):
    r = ctx.get("_ol")
    if r is None:
        s = ctx["sample_sorted"]
        med = float(np.median(s))
        absd = np.abs(s - med)
        mad = max(float(np.median(absd)) * 1.4826, 1e-12)
        r = {
            "s": s, "med": med, "mad": mad,
            "q10": float(np.quantile(s, 0.10)), "q90": float(np.quantile(s, 0.90)),
        }
        ctx["_ol"] = r
    return r


def _sd(ctx):
    return max(ctx["sd"], 1e-8)


# --- location / scale ------------------------------------------------------
def f_median_shift(w, ctx):
    h = _hist(ctx)
    n = len(w)
    return [(float(np.median(w)) - h["med"]) / max(h["mad"] / math.sqrt(n), 1e-8)]


def f_mad_ratio(w, ctx):
    h = _hist(ctx)
    med = float(np.median(w))
    mad = max(float(np.median(np.abs(w - med))) * 1.4826, 1e-12)
    return [math.log(mad / h["mad"])]


def f_trimmed(w, ctx):
    ws = np.sort(w)
    n = len(ws)
    k = max(1, int(0.1 * n))
    tm = float(ws[k:n - k].mean()) if n - 2 * k > 0 else float(ws.mean())
    return [(tm - ctx["mu"]) / max(_sd(ctx) / math.sqrt(n), 1e-8)]


def f_qshift(w, ctx):
    h = _hist(ctx)
    q = np.quantile(w, [0.1, 0.5, 0.9])
    sd = _sd(ctx)
    return [(q[0] - h["q10"]) / sd, (q[1] - h["med"]) / sd, (q[2] - h["q90"]) / sd]


# --- distribution distance -------------------------------------------------
def f_wasserstein(w, ctx):
    h = _hist(ctx)
    qs = np.linspace(0.0, 1.0, 21)
    a = np.quantile(w, qs)
    b = np.quantile(h["s"], qs)
    return [float(np.mean(np.abs(a - b))) / _sd(ctx)]


def f_cvm(w, ctx):
    h = _hist(ctx)
    ws = np.sort(w)
    s = h["s"]
    allv = np.concatenate([s, ws])
    cdf_h = np.searchsorted(s, allv, side="right") / len(s)
    cdf_w = np.searchsorted(ws, allv, side="right") / len(ws)
    return [float(np.mean((cdf_h - cdf_w) ** 2))]


def f_hist_entropy(w, ctx):
    lo, hi = ctx["mu"] - 3 * ctx["sd"], ctx["mu"] + 3 * ctx["sd"]
    if hi <= lo:
        return [0.0]
    c, _ = np.histogram(w, bins=10, range=(lo, hi))
    p = c / max(c.sum(), 1)
    p = p[p > 0]
    return [float(-(p * np.log(p)).sum()) / math.log(10)]


# --- change-point / cumulative --------------------------------------------
def f_cusum_max(w, ctx):
    c = np.cumsum(w - ctx["mu"])
    return [float(np.max(np.abs(c))) / max(_sd(ctx) * math.sqrt(len(w)), 1e-8)]


def f_cusum_pos(w, ctx):
    c = np.abs(np.cumsum(w - ctx["mu"]))
    return [float(np.argmax(c)) / max(len(w) - 1, 1)]


def f_thirds(w, ctx):
    n = len(w)
    t = n // 3
    if t < 3:
        return [0.0]
    return [(float(w[-t:].mean()) - float(w[:t].mean())) / max(_sd(ctx) / math.sqrt(t), 1e-8)]


def f_slope_change(w, ctx):
    n = len(w)
    h = n // 2
    if h < 3 or n - h < 3:
        return [0.0]

    def slope(a):
        i = np.arange(len(a), dtype=np.float64)
        d = len(a) * (i * i).sum() - i.sum() ** 2
        return (len(a) * (i * a).sum() - i.sum() * a.sum()) / d if abs(d) > 1e-12 else 0.0

    return [(slope(w[h:]) - slope(w[:h])) * n / _sd(ctx)]


# --- volatility / curvature ------------------------------------------------
def f_abschange(w, ctx):
    return [float(np.mean(np.abs(np.diff(w)))) / _sd(ctx)] if len(w) > 1 else [0.0]


def f_rv_ratio(w, ctx):
    rv = math.sqrt(float(np.sum(np.diff(w) ** 2)) / max(len(w) - 1, 1)) if len(w) > 1 else 0.0
    return [rv / _sd(ctx)]


def f_curv(w, ctx):
    if len(w) < 3:
        return [0.0]
    return [float(np.mean(np.abs(np.diff(w, 2)))) / _sd(ctx)]


def f_var_of_var(w, ctx):
    q = np.array_split(w, 4)
    vs = np.array([seg.var() for seg in q if len(seg) > 1])
    if len(vs) < 2 or ctx["var"] < 1e-12:
        return [0.0]
    return [float(vs.std()) / ctx["var"]]


def f_down_up_vol(w, ctx):
    d = np.diff(w)
    if len(d) < 2:
        return [0.0]
    up = d[d > 0]
    dn = d[d < 0]
    vu = float(up.var()) if len(up) > 1 else 0.0
    vd = float(dn.var()) if len(dn) > 1 else 0.0
    return [math.log((vd + 1e-12) / (vu + 1e-12))]


# --- temporal dependence ---------------------------------------------------
def f_acmore(w, ctx):
    return [_autocorr(w, 3), _autocorr(w, 5)]


def f_acf_decay(w, ctx):
    a1, a2 = _autocorr(w, 1), _autocorr(w, 2)
    return [a1 - a2]


def f_ljung(w, ctx):
    return [sum(_autocorr(w, k) ** 2 for k in range(1, 6))]


def f_arch(w, ctx):
    """Volatility-clustering / ARCH effect: autocorrelation of SQUARED residuals.

    All existing variance features measure the LEVEL of variance (window-vs-history or a
    one-shot variance changepoint). This measures whether volatility is AUTOCORRELATED
    (GARCH-style persistence) -- a distinct DGP property that can change at a break.
    Returns [lag-1 autocorr of r^2, Ljung-Box-style sum of squared autocorrs of r^2 over
    lags 1..5]. Autocorrelation is scale-free so no history normalisation is needed.
    """
    r = w - w.mean()
    r2 = r * r
    ac1 = _autocorr(r2, 1)
    lb = sum(_autocorr(r2, k) ** 2 for k in range(1, 6))
    return [ac1, lb]



def f_turning(w, ctx):
    if len(w) < 3:
        return [0.0]
    d = np.diff(w)
    tp = int(np.sum(np.sign(d[:-1]) != np.sign(d[1:])))
    exp = 2 * (len(w) - 2) / 3
    return [(float(tp) - exp) / max(math.sqrt(len(w)), 1e-8)]


def f_runs(w, ctx):
    med = float(np.median(w))
    s = np.sign(w - med)
    s = s[s != 0]
    if len(s) < 2:
        return [0.0]
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    return [(runs - len(s) / 2) / max(math.sqrt(len(s)), 1e-8)]


def f_longest_run(w, ctx):
    med = float(np.median(w))
    best = cur = 0
    prev = None
    for v in (w > med):
        cur = cur + 1 if v == prev else 1
        prev = v
        if cur > best:
            best = cur
    return [best / max(len(w), 1)]


def f_median_cross(w, ctx):
    med = float(np.median(w))
    s = np.sign(w - med)
    return [float((np.diff(s) != 0).mean())] if len(w) > 1 else [0.0]


# --- spectral / complexity -------------------------------------------------
def f_spec_entropy(w, ctx):
    a = w - w.mean()
    ps = np.abs(np.fft.rfft(a)) ** 2
    s = ps.sum()
    if s < 1e-12 or len(ps) < 2:
        return [0.0]
    p = ps / s
    p = p[p > 0]
    return [float(-(p * np.log(p)).sum()) / math.log(len(ps))]


def f_spec_bands(w, ctx):
    a = w - w.mean()
    ps = np.abs(np.fft.rfft(a)) ** 2
    m = max(len(ps) // 2, 1)
    lo = ps[:m].sum()
    hi = ps[m:].sum()
    return [math.log((lo + 1e-12) / (hi + 1e-12))]


def f_hjorth(w, ctx):
    d1 = np.diff(w)
    d2 = np.diff(d1)
    v0 = float(w.var())
    v1 = float(d1.var()) if len(d1) else 0.0
    v2 = float(d2.var()) if len(d2) else 0.0
    mob = math.sqrt(v1 / v0) if v0 > 1e-12 else 0.0
    comp = (math.sqrt(v2 / v1) / mob) if v1 > 1e-12 and mob > 1e-12 else 0.0
    return [mob, comp]


def f_hurst(w, ctx):
    x = w - w.mean()
    z = np.cumsum(x)
    r = float(z.max() - z.min())
    s = float(w.std())
    if s < 1e-12 or len(w) < 2:
        return [0.0]
    return [math.log((r / s) + 1e-12) / math.log(len(w))]


def f_perm_entropy(w, ctx):
    n = len(w)
    if n < 5:
        return [0.0]
    counts = {}
    for i in range(n - 2):
        pat = tuple(np.argsort(w[i:i + 3]))
        counts[pat] = counts.get(pat, 0) + 1
    p = np.array(list(counts.values()), dtype=np.float64)
    p /= p.sum()
    return [float(-(p * np.log(p)).sum()) / math.log(6)]


# --- extremes / short-scale ------------------------------------------------
def f_maxabs_z(w, ctx):
    return [float(np.max(np.abs(w - ctx["mu"]))) / _sd(ctx)]


def f_tail_asym(w, ctx):
    z = (w - ctx["mu"]) / _sd(ctx)
    return [float((z > 1.5).mean()) - float((z < -1.5).mean())]


def f_short_scale(w, ctx):
    k = min(25, len(w))
    wr = w[-k:]
    vr = max(float(wr.var()), 1e-12)
    return [math.log(vr / ctx["var"]),
            (float(wr.mean()) - ctx["mu"]) / max(_sd(ctx) / math.sqrt(k), 1e-8)]


def f_short_ks(w, ctx):
    h = _hist(ctx)
    k = min(25, len(w))
    return [_fast_ks(np.sort(w[-k:]), h["s"])]


def f_short_shape(w, ctx):
    k = min(25, len(w))
    wr = w[-k:]
    m = float(wr.mean())
    v = max(float(wr.var()), 1e-12)
    d = wr - m
    sk = float(np.mean(d ** 3)) / (v ** 1.5)
    ku = float(np.mean(d ** 4)) / (v ** 2) - 3.0
    return [sk - ctx["skew"], ku - ctx["kurt"]]


def f_range_recent(w, ctx):
    k = min(25, len(w))
    wr = w[-k:]
    return [float(wr.max() - wr.min()) / max(ctx["rng"], 1e-8)]


# --- E14: extra spectral / complexity groups (tested on top of spec_entropy) ---
def f_spec_flatness(w, ctx):
    ps = np.abs(np.fft.rfft(w - w.mean())) ** 2
    ps = ps[1:]  # drop DC
    if len(ps) < 2 or ps.sum() < 1e-12:
        return [0.0]
    gm = math.exp(float(np.mean(np.log(ps + 1e-20))))
    return [gm / (float(ps.mean()) + 1e-20)]


def f_spec_rolloff(w, ctx):
    ps = np.abs(np.fft.rfft(w - w.mean())) ** 2
    tot = ps.sum()
    if tot < 1e-12 or len(ps) < 2:
        return [0.0]
    idx = int(np.searchsorted(np.cumsum(ps), 0.85 * tot))
    return [idx / (len(ps) - 1)]


def f_spec_spread(w, ctx):
    a = w - w.mean()
    sp = np.abs(np.fft.rfft(a))
    freqs = np.fft.rfftfreq(len(a))
    s = sp.sum()
    if s < 1e-12:
        return [0.0]
    centroid = float((freqs * sp).sum() / s)
    return [math.sqrt(float(((freqs - centroid) ** 2 * sp).sum() / s))]


def f_spec_peak(w, ctx):
    ps = np.abs(np.fft.rfft(w - w.mean())) ** 2
    ps = ps[1:]
    tot = ps.sum()
    if tot < 1e-12 or len(ps) < 1:
        return [0.0]
    return [float(ps.max()) / tot]


def _spec_entropy_of(a):
    ps = np.abs(np.fft.rfft(a - a.mean())) ** 2
    s = ps.sum()
    if s < 1e-12 or len(ps) < 2:
        return 0.0
    p = ps / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) / math.log(len(ps))


def f_spec_entropy_diff(w, ctx):
    if len(w) < 3:
        return [0.0]
    return [_spec_entropy_of(np.diff(w))]


def f_spec_entropy_short(w, ctx):
    k = min(40, len(w))
    return [_spec_entropy_of(w[-k:])]


def f_perm_entropy4(w, ctx):
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


def f_higuchi(w, ctx):
    n = len(w)
    kmax = 6
    if n < kmax * 2:
        return [0.0]
    lnl, lnk = [], []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            lm.append(float(np.abs(np.diff(w[idx])).sum()) * (n - 1)
                      / ((len(idx) - 1) * k))
        if lm:
            lnl.append(math.log(float(np.mean(lm)) + 1e-20))
            lnk.append(math.log(1.0 / k))
    if len(lnl) < 2:
        return [0.0]
    return [float(np.polyfit(lnk, lnl, 1)[0])]


def f_katz(w, ctx):
    n = len(w) - 1
    if n < 2:
        return [0.0]
    ln = float(np.abs(np.diff(w)).sum())
    d = float(np.abs(w - w[0]).max())
    if ln < 1e-12 or d < 1e-12:
        return [0.0]
    lnn = math.log(n)
    denom = lnn + math.log(d / ln)
    if abs(denom) < 1e-12:
        return [0.0]
    return [lnn / denom]


def f_lzc(w, ctx):
    n = len(w)
    if n < 4:
        return [0.0]
    med = float(np.median(w))
    s = "".join("1" if v > med else "0" for v in w)
    seen, phrase, c = set(), "", 0
    for ch in s:
        phrase += ch
        if phrase not in seen:
            seen.add(phrase)
            phrase = ""
            c += 1
    return [c / (n / math.log2(n))]


# --- Lever B: within-online max-split changepoint localisation -------------
# Scan candidate split points k inside the trailing window; take the MAX standardized
# contrast between the two sides (self-normalized within the window, NOT vs history).
# Fires when a break lies INSIDE the recent window -> complements vs-history features
# (which fire once the break is entirely before the window). MIN_SEG guards unstable tails.
_MSPLIT_MIN_SEG = 5


def _msplit_prefix(w):
    """Return (n, ks, S1L, S1R, SSL, SSR) arrays over valid split points k=nl.

    ks[i] = number of points on the left; segments are w[:k] and w[k:].
    SSL/SSR are within-segment sums of squared deviations (numerically stable prefix form).
    """
    n = len(w)
    lo, hi = _MSPLIT_MIN_SEG, n - _MSPLIT_MIN_SEG
    if hi <= lo:
        return None
    c1 = np.cumsum(w)
    c2 = np.cumsum(w * w)
    tot1 = c1[-1]
    tot2 = c2[-1]
    ks = np.arange(lo, hi + 1, dtype=np.float64)          # nl
    kidx = ks.astype(np.int64) - 1                        # index into cumsum for left seg
    s1l = c1[kidx]
    s2l = c2[kidx]
    nl = ks
    nr = n - ks
    s1r = tot1 - s1l
    s2r = tot2 - s1l  # placeholder, fixed below
    s2r = tot2 - s2l
    ssl = s2l - s1l * s1l / nl
    ssr = s2r - s1r * s1r / nr
    ssl = np.maximum(ssl, 0.0)
    ssr = np.maximum(ssr, 0.0)
    return n, nl, nr, s1l, s1r, ssl, ssr


def f_max_split_t(w, ctx):
    """Max over split k of |standardized mean contrast| (Chow / max-t changepoint stat)."""
    pf = _msplit_prefix(w)
    if pf is None:
        return [0.0]
    n, nl, nr, s1l, s1r, ssl, ssr = pf
    meanl = s1l / nl
    meanr = s1r / nr
    pooled_var = (ssl + ssr) / np.maximum(n - 2.0, 1.0)
    se = np.sqrt(np.maximum(pooled_var, 1e-12) * (1.0 / nl + 1.0 / nr))
    tstat = np.abs(meanr - meanl) / np.maximum(se, 1e-12)
    return [float(np.max(tstat))]


def f_max_split_pos(w, ctx):
    """Fraction position (k/n) of the max mean-contrast split -> localises the break."""
    pf = _msplit_prefix(w)
    if pf is None:
        return [0.0]
    n, nl, nr, s1l, s1r, ssl, ssr = pf
    meanl = s1l / nl
    meanr = s1r / nr
    pooled_var = (ssl + ssr) / np.maximum(n - 2.0, 1.0)
    se = np.sqrt(np.maximum(pooled_var, 1e-12) * (1.0 / nl + 1.0 / nr))
    tstat = np.abs(meanr - meanl) / np.maximum(se, 1e-12)
    return [float(nl[int(np.argmax(tstat))]) / n]


def f_max_split_var(w, ctx):
    """Max over split k of |log(var_right / var_left)| -> variance changepoint."""
    pf = _msplit_prefix(w)
    if pf is None:
        return [0.0]
    n, nl, nr, s1l, s1r, ssl, ssr = pf
    varl = np.maximum(ssl / nl, 1e-12)
    varr = np.maximum(ssr / nr, 1e-12)
    return [float(np.max(np.abs(np.log(varr / varl))))]


def f_max_split_lr(w, ctx):
    """Max normal log-LR over split k for a mean+variance change (Gaussian max-LR stat)."""
    pf = _msplit_prefix(w)
    if pf is None:
        return [0.0]
    n, nl, nr, s1l, s1r, ssl, ssr = pf
    # full-window variance (single-regime MLE), computed directly as a scalar
    var0 = max(float(np.sum(w * w) / n - (float(np.sum(w)) / n) ** 2), 1e-12)
    varl = np.maximum(ssl / nl, 1e-12)
    varr = np.maximum(ssr / nr, 1e-12)
    # log-LR = 0.5*[ n*log(var0) - nl*log(varl) - nr*log(varr) ]
    llr = 0.5 * (n * math.log(var0) - nl * np.log(varl) - nr * np.log(varr))
    return [float(np.max(np.maximum(llr, 0.0)))]


# --- registry --------------------------------------------------------------
EXTRA = {
    "median_shift": f_median_shift, "mad_ratio": f_mad_ratio, "trimmed": f_trimmed,
    "qshift": f_qshift, "wasserstein": f_wasserstein, "cvm": f_cvm,
    "hist_entropy": f_hist_entropy, "cusum_max": f_cusum_max, "cusum_pos": f_cusum_pos,
    "thirds": f_thirds, "slope_change": f_slope_change, "abschange": f_abschange,
    "rv_ratio": f_rv_ratio, "curv": f_curv, "var_of_var": f_var_of_var,
    "down_up_vol": f_down_up_vol, "acmore": f_acmore, "acf_decay": f_acf_decay,
    "ljung": f_ljung, "turning": f_turning, "runs": f_runs, "longest_run": f_longest_run,
    "median_cross": f_median_cross, "spec_entropy": f_spec_entropy,
    "spec_bands": f_spec_bands, "hjorth": f_hjorth, "hurst": f_hurst, "arch": f_arch,
    "perm_entropy": f_perm_entropy, "maxabs_z": f_maxabs_z, "tail_asym": f_tail_asym,
    "short_scale": f_short_scale, "short_ks": f_short_ks, "short_shape": f_short_shape,
    "range_recent": f_range_recent,
    # E14 spectral / complexity
    "spec_flatness": f_spec_flatness, "spec_rolloff": f_spec_rolloff,
    "spec_spread": f_spec_spread, "spec_peak": f_spec_peak,
    "spec_entropy_diff": f_spec_entropy_diff, "spec_entropy_short": f_spec_entropy_short,
    "perm_entropy4": f_perm_entropy4, "higuchi": f_higuchi, "katz": f_katz, "lzc": f_lzc,
    # Lever B: within-online max-split changepoint localisation
    "max_split_t": f_max_split_t, "max_split_pos": f_max_split_pos,
    "max_split_var": f_max_split_var, "max_split_lr": f_max_split_lr,
}

# Combos of related groups.
COMBOS = {
    "combo_changepoint": ("cusum_max", "cusum_pos", "thirds", "slope_change"),
    "combo_distance": ("wasserstein", "cvm", "qshift"),
    "combo_temporal": ("acmore", "acf_decay", "turning", "runs", "ljung"),
    "combo_spectral": ("spec_entropy", "spec_bands", "hjorth"),
    "combo_shortscale": ("short_scale", "short_ks", "short_shape"),
    "combo_volatility": ("abschange", "rv_ratio", "curv", "down_up_vol"),
}


def _feat(w, ctx, keys):
    feats = base_features(w, ctx)
    for k in keys:
        feats.extend(EXTRA[k](w, ctx))
    return feats


def build_candidates():
    """Ordered dict {name: feature_fn}. 'base14' first (baseline)."""
    cands = {"base14": base_features}
    for name in EXTRA:
        cands[name] = partial(_feat, keys=(name,))
    for name, keys in COMBOS.items():
        cands[name] = partial(_feat, keys=keys)
    return cands
