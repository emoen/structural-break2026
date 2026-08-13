"""E23: SERIES-TRANSFORM two-sample features (trailing window vs history) on top of the
E17 champion. This is the untried "big lever" from the 2nd-place playbook: instead of
adding yet another marginal distance on the RAW values (E21/E22 -> saturated, LB
reversed), we apply a transform to BOTH the trailing sub-window and the historical
sample, then compare the transformed VIEWS. A transform can expose a break that is
invisible in raw marginal stats:

  - cumsum : integrates the standardized series -> a small persistent mean shift becomes
             a large drift (classic CUSUM amplification).
  - diff   : first difference -> roughness / step-change / short-range dependence.
  - mstd   : rolling std (k=10) -> LOCAL volatility, i.e. volatility clustering that raw
             global variance misses.
  - absdev : |z| -> tail / dispersion changes, robust.

Everything is computed on the history-standardized series z = (x - mu_h)/sd_h so the
transformed views are scale-comparable across series. For each transform we append the
champion's 3 core discriminators between transform(window) and transform(history):
mean_z, log_var_ratio, KS -> 3 feats x 3 scales (60/130/250) = 9 feats per candidate,
on top of the 45-dim champion multiscale vector.

Ordered history comes from ctx["sample_ordered"] (added to _exp_fast.hist_context;
additive, notebook has its own copy so submission is unaffected).

*** HARDENED SHIP BAR (post E22a wass reversal) ***: we are 0-for-2 on t<2 features.
Only features with CV t>=~2.5 (ideally >=3) have ever transferred to LB. Do NOT spend an
LB slot on anything below that. This harness is CV-only, no LB risk.

Baseline = champion ms_60_130_250 @250. 10 folds, paired.

Usage:
  SMOKE=1 .venv/bin/python -u _transforms.py
  nohup .venv/bin/python -u _transforms.py > _transforms.log 2>&1 & disown
"""
import os
import time
from datetime import datetime
from functools import partial

import numpy as np

from _overnight import (
    load_pool, select_picks, evaluate, paired, SMOKE,
)
from _feat_lib import _feat

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "_transforms_results.tsv")
SUMMARY = os.path.join(HERE, "_transforms_summary.txt")

BASE_KEYS = ("spec_entropy",)     # base15 per scale
BASE_SCALES = (60, 130, 250)      # E17 champion
BASE_WINDOW = 250
EPS = 1e-12


def _subwin(w, s):
    n = len(w)
    return w if s >= n else w[-s:]


# --- transforms: input is history-standardized z, output a 1-D view -----------
def _t_cumsum(z):
    n = len(z)
    if n == 0:
        return z
    return np.cumsum(z) / np.sqrt(n)          # normalize drift by sqrt(len)


def _t_diff(z):
    return np.diff(z) if len(z) > 1 else z


def _t_mstd(z, k=10):
    n = len(z)
    if n < k:
        return np.array([float(z.std())]) if n else z
    # rolling std via cumulative sums, length n-k+1
    c1 = np.cumsum(np.insert(z, 0, 0.0))
    c2 = np.cumsum(np.insert(z * z, 0, 0.0))
    s1 = c1[k:] - c1[:-k]
    s2 = c2[k:] - c2[:-k]
    var = np.maximum(s2 / k - (s1 / k) ** 2, 0.0)
    return np.sqrt(var)


def _t_absdev(z):
    return np.abs(z)


TRANSFORMS = {
    "cumsum": _t_cumsum,
    "diff":   _t_diff,
    "mstd":   _t_mstd,
    "absdev": _t_absdev,
}


def _cmp3(tw, th):
    """champion's 3 core discriminators between two transformed views."""
    if len(tw) < 2 or len(th) < 2:
        return [0.0, 0.0, 0.0]
    mh = float(th.mean())
    sh = float(th.std())
    sh = sh if sh > EPS else EPS
    mean_z = (float(tw.mean()) - mh) / sh
    log_var_ratio = float(np.log((tw.var() + EPS) / (th.var() + EPS)))
    # KS between the two transformed samples
    a = np.sort(tw)
    b = np.sort(th)
    allv = np.concatenate([a, b])
    allv.sort()
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    ks = float(np.max(np.abs(ca - cb)))
    return [mean_z, log_var_ratio, ks]


def tf_feat(w, ctx, tkeys):
    """champion base15 per scale + transform-comparison feats per scale."""
    out = []
    mu = ctx["mu"]
    sd = ctx["sd"] if ctx["sd"] > EPS else EPS
    zh_full = (ctx["sample_ordered"] - mu) / sd     # standardized ordered history
    for s in BASE_SCALES:
        ws = _subwin(w, s)
        out.extend(_feat(ws, ctx, keys=BASE_KEYS))
        if tkeys:
            zw = (ws - mu) / sd
            for k in tkeys:
                fn = TRANSFORMS[k]
                out.extend(_cmp3(fn(zw), fn(zh_full)))
    return out


CANDIDATES = {
    "tf_cumsum": ("cumsum",),
    "tf_diff":   ("diff",),
    "tf_mstd":   ("mstd",),
    "tf_absdev": ("absdev",),
    "tf_all":    ("cumsum", "diff", "mstd", "absdev"),
}
if SMOKE:
    CANDIDATES = {"tf_cumsum": ("cumsum",), "tf_mstd": ("mstd",)}


def write_row(fh, name, nfeat, aucs, base_aucs, secs):
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1))
    if base_aucs is None:
        md = dsd = t = float("nan")
        verdict = "baseline(ms_60_130_250@250)"
    else:
        md, dsd, t, _ = paired(aucs, base_aucs)
        # HARDENED bar: only t>=2.5 is worth an LB slot
        if md > 0 and t >= 2.5:
            verdict = "SHIP(hard)"
        elif md > 0 and t >= 2.0:
            verdict = "marginal(HOLD)"
        elif md > 0 and t >= 1.5:
            verdict = "weak"
        else:
            verdict = "noise"
    fh.write("\t".join([
        name, str(nfeat), ",".join(f"{v:.4f}" for v in aucs),
        f"{mean:.4f}", f"{sd:.4f}", f"{md:+.4f}", f"{dsd:.4f}", f"{t:+.2f}",
        verdict, f"{secs:.0f}", datetime.now().strftime("%H:%M:%S"),
    ]) + "\n")
    fh.flush()
    os.fsync(fh.fileno())
    print(f"{name:<12} nf={nfeat:<2} mean={mean:.4f} d={md:+.4f} t={t:+.2f} "
          f"{verdict} ({secs:.0f}s)", flush=True)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== E23 series-transform two-sample ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {len(folds[0])}", flush=True)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("name\tn_feat\tfold_aucs\tmean\tstd\tdelta\tdelta_std\tt\t"
             "verdict\tsecs\ttime\n")
    fh.flush()

    ts = time.time()
    base_fn = partial(tf_feat, tkeys=())
    base, nf = evaluate(base_fn, BASE_WINDOW, X, yi, y_val, picks, folds)
    write_row(fh, "ms_60_130_250", nf, base, None, time.time() - ts)

    for name, keys in CANDIDATES.items():
        ts = time.time()
        try:
            fn = partial(tf_feat, tkeys=keys)
            aucs, nf = evaluate(fn, BASE_WINDOW, X, yi, y_val, picks, folds)
            write_row(fh, name, nf, aucs, base, time.time() - ts)
        except Exception as e:  # noqa: BLE001 - keep the run alive
            fh.write(f"{name}\tERR\t\t\t\t\t\t\tERROR:{type(e).__name__}\t"
                     f"{time.time()-ts:.0f}\t{datetime.now():%H:%M:%S}\n")
            fh.flush()
            print(f"{name} FAILED: {type(e).__name__}: {e}", flush=True)

    fh.close()
    _summary()
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)


def _summary():
    import pandas as pd
    df = pd.read_csv(RESULTS, sep="\t")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(f"E23 series-transform two-sample, 10 folds, paired vs champion "
                f"ms_60_130_250@250 ({datetime.now():%Y-%m-%d %H:%M})\n")
        f.write("HARDENED bar: only t>=2.5 worth an LB slot (0-for-2 on t<2).\n\n")
        f.write(df.sort_values("t", ascending=False)[
            ["name", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
