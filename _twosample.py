"""E21: nonparametric TWO-SAMPLE distances (trailing window vs history) on top of the
E17 champion. We hit a representation ceiling (E19: every scale variant ~0) using only
KS + moment diffs. These are the statistically-principled "the distribution changed"
detectors we never tried: Wasserstein-1, energy distance, Cramer-von Mises, MMD-RBF.
Each is computed at the 3 champion scales (60/130/250) between the trailing sub-window
and the historical sample, and appended to the champion's 45-dim multi-scale vector.

All pure-numpy, reusing ctx["sample_sorted"] (sorted history sample) + ctx["sd"] so no
extra sorting of history per step. Scale-comparable across series (W1/energy divided by
history sd; CvM/MMD rank/standardized). Baseline = champion ms_60_130_250 @250.
10 folds, paired. Ship if |t|>=2 & delta>0 (t>=1.9 = LB-testable).

Usage:
  SMOKE=1 .venv/bin/python -u _twosample.py
  nohup .venv/bin/python -u _twosample.py > _twosample.log 2>&1 &
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
RESULTS = os.path.join(HERE, "_twosample_results.tsv")
SUMMARY = os.path.join(HERE, "_twosample_summary.txt")

BASE_KEYS = ("spec_entropy",)     # base15 per scale
BASE_SCALES = (60, 130, 250)      # E17 champion
BASE_WINDOW = 250


def _subwin(w, s):
    n = len(w)
    return w if s >= n else w[-s:]


def _w1(a_sorted, b_sorted, sd):
    """1D Wasserstein-1 = integral|F_a - F_b|, standardized by history sd."""
    allv = np.concatenate([a_sorted, b_sorted])
    allv.sort()
    d = np.diff(allv)
    ca = np.searchsorted(a_sorted, allv[:-1], side="right") / len(a_sorted)
    cb = np.searchsorted(b_sorted, allv[:-1], side="right") / len(b_sorted)
    return float(np.sum(np.abs(ca - cb) * d)) / sd


def _energy(a_sorted, b_sorted, sd):
    """1D energy distance = 2*integral(F_a - F_b)^2, standardized by history sd."""
    allv = np.concatenate([a_sorted, b_sorted])
    allv.sort()
    d = np.diff(allv)
    ca = np.searchsorted(a_sorted, allv[:-1], side="right") / len(a_sorted)
    cb = np.searchsorted(b_sorted, allv[:-1], side="right") / len(b_sorted)
    return float(2.0 * np.sum((ca - cb) ** 2 * d)) / sd


def _cvm(a_sorted, b_sorted):
    """Cramer-von Mises-style 2-sample statistic: mean squared CDF gap (rank-based, scale-free)."""
    allv = np.concatenate([a_sorted, b_sorted])
    allv.sort()
    ca = np.searchsorted(a_sorted, allv, side="right") / len(a_sorted)
    cb = np.searchsorted(b_sorted, allv, side="right") / len(b_sorted)
    return float(np.mean((ca - cb) ** 2))


def _sub(x, k=96):
    if len(x) <= k:
        return x
    st = len(x) // k
    return x[::st][:k]


def _mmd(a, b, sd):
    """Squared MMD with RBF kernel (gamma=0.5 on sd-standardized values), subsampled to <=96."""
    a = _sub(np.asarray(a, dtype=np.float64) / sd)
    b = _sub(np.asarray(b, dtype=np.float64) / sd)

    def kmean(x, y):
        d2 = (x[:, None] - y[None, :]) ** 2
        return float(np.exp(-0.5 * d2).mean())

    return kmean(a, a) - 2.0 * kmean(a, b) + kmean(b, b)


def ts_feat(w, ctx, dist_keys):
    """champion base15 per scale (60/130/250) + selected two-sample distances per scale."""
    out = []
    h = ctx["sample_sorted"]
    sd = ctx["sd"]
    for s in BASE_SCALES:
        ws = _subwin(w, s)
        out.extend(_feat(ws, ctx, keys=BASE_KEYS))
        if dist_keys:
            ws_sorted = np.sort(ws)
            if "wass" in dist_keys:
                out.append(_w1(ws_sorted, h, sd))
            if "energy" in dist_keys:
                out.append(_energy(ws_sorted, h, sd))
            if "cvm" in dist_keys:
                out.append(_cvm(ws_sorted, h))
            if "mmd" in dist_keys:
                out.append(_mmd(ws, h, sd))
    return out


# name -> distance families appended at each of the 3 scales
CANDIDATES = {
    "ts_wass":   ("wass",),
    "ts_energy": ("energy",),
    "ts_cvm":    ("cvm",),
    "ts_mmd":    ("mmd",),
    "ts_all":    ("wass", "energy", "cvm", "mmd"),
}
if SMOKE:
    CANDIDATES = {"ts_wass": ("wass",), "ts_all": ("wass", "energy", "cvm", "mmd")}


def write_row(fh, name, nfeat, aucs, base_aucs, secs):
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1))
    if base_aucs is None:
        md = dsd = t = float("nan")
        verdict = "baseline(ms_60_130_250@250)"
    else:
        md, dsd, t, _ = paired(aucs, base_aucs)
        verdict = "SHIP" if abs(t) >= 2 and md > 0 else (
            "promising" if t >= 1.5 and md > 0 else "noise")
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
    print(f"=== E21 two-sample distances ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {len(folds[0])}", flush=True)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("name\tn_feat\tfold_aucs\tmean\tstd\tdelta\tdelta_std\tt\t"
             "verdict\tsecs\ttime\n")
    fh.flush()

    ts = time.time()
    base_fn = partial(ts_feat, dist_keys=())
    base, nf = evaluate(base_fn, BASE_WINDOW, X, yi, y_val, picks, folds)
    write_row(fh, "ms_60_130_250", nf, base, None, time.time() - ts)

    for name, keys in CANDIDATES.items():
        ts = time.time()
        try:
            fn = partial(ts_feat, dist_keys=keys)
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
        f.write(f"E21 two-sample distances, 10 folds, paired vs champion "
                f"ms_60_130_250@250 ({datetime.now():%Y-%m-%d %H:%M})\n\n")
        f.write(df.sort_values("t", ascending=False)[
            ["name", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
