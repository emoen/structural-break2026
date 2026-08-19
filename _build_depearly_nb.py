"""Build streaming_detector_depearly.ipynb (E62 = b_no_accf) from the trail120 champion.

Clones the shipped trail120 champion notebook and inserts ONE dependence/early-onset feature
cell after the residual block. The dependence features are inlined byte-for-byte from
_night_banks.dep_feats / fit_dep_refs (with _fit_ar/_ar_resid/_autocorr copied from
_residual_reffit/_exp_fast) so the deployed features == the fresh-confirmed b_no_accf features.
Kept indices [0,1,2,3,4,5,10,11,12,13] drop the noisy ACF-lag block (dacf1/2/5/10).
"""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

SRC = "streaming_detector_trail120.ipynb"
DST = "streaming_detector_depearly.ipynb"

DEP_MD = """## Dependence / early-onset residual block (E62 = `b_no_accf`)

Fresh-confirmed **+0.0066 (4/4 seeds)** on top of the trail120 champion — the best additive
lead since trail120. Targets the two axes the EDA flagged as the residual lens's blind spots:

- **Latency** on pure dependence-structure breaks (`ma` / `ar_coef`, detected ~120–190 pts late)
  → an **early-onset residual CUSUM** (accumulate residual surprise from the first post-break point).
- **Specificity leak** (strong-AR quiet series false-positive) → the **AR-studentized residual**
  (z-score the window log-var-ratio against the series' *own* sliding-window historical null).

Plus an ARCH-lite squared-residual contrast and a sup-F / Chow max-split scan. The ACF-lag
contrasts from the full 14-feature bank were pure noise (`b_accf`) and are **dropped** here → the
10-feature `b_no_accf` set. Rides along everywhere via the same `hist_context` / `compute_features_ms`
wrapping the residual block uses, so `train()` / `infer()` need no change."""

DEP_CODE = r'''# @crunch/keep:on
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
'''


def main():
    nb = nbformat.read(SRC, as_version=4)
    # locate the residual-block code cell
    ridx = next(i for i, c in enumerate(nb.cells)
                if "Residual-variance-ratio feature block" in c.source)
    # retitle
    nb.cells[0].source = (
        "# Rank + Tuned Ensemble + Multi-scale Residual Lens + Dependence/Early-onset Lens "
        "(E62 = `b_no_accf`)\n\n"
        "Extends the shipped **trail120** champion (LB 61.58%) with a 10-feature "
        "dependence / early-onset residual bank (fresh-confirmed **+0.0066, 4/4 seeds**): "
        "AR-studentized residual + early-onset CUSUM + ARCH contrast + sup-F scan. The ACF-lag "
        "contrasts were noise and are dropped. Everything else (2-member fixed-CDF blend, "
        "`w=0.6`, tuned HGB + `rank_xendcg`) is unchanged.\n\n"
        + nb.cells[0].source.split("\n", 1)[1]
    )
    dep_md = new_markdown_cell(DEP_MD)
    dep_code = new_code_cell(DEP_CODE)
    for c in (dep_md, dep_code):
        c.metadata = {}
    nb.cells[ridx + 1:ridx + 1] = [dep_md, dep_code]
    nbformat.write(nb, DST)
    print(f"wrote {DST} ({len(nb.cells)} cells); dep block inserted after cell {ridx}")


if __name__ == "__main__":
    main()
