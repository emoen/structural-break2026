"""E24: VARIANCE two-sample tests (trailing window vs history) on top of the E17
champion. From both 2nd/4th-place writeups the variance-change detectors are F-test,
Levene, Fligner-Killeen. Our champion already has `log_var_ratio` == the log F-stat
(non-robust), so the F-test adds ~nothing. The genuinely NEW detectors are the ROBUST
ones, which fire on a variance break even under heavy tails / non-normality where a raw
variance ratio is unreliable:

  - Levene (Brown-Forsythe): one-way stat on median-centered absolute deviations. Pure
    numpy, cheap -> preferred for the production notebook.
  - Fligner-Killeen: rank-based (normal-scores of |x-median|), the most robust; needs
    scipy.special.ndtri + rankdata.

Each is computed at the 3 champion scales (60/130/250) between the trailing sub-window
and the historical sample and appended to the 45-dim champion vector. Baseline = champion
ms_60_130_250 @250. 10 folds, paired. HARDENED bar: only t>=~2.5 is worth an LB slot.

Usage:
  SMOKE=1 .venv/bin/python -u _vartests.py
  nohup .venv/bin/python -u _vartests.py > _vartests.log 2>&1 & disown
"""
import os
import time
from datetime import datetime
from functools import partial

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from _overnight import (
    load_pool, select_picks, evaluate, paired, SMOKE,
)
from _feat_lib import _feat

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "_vartests_results.tsv")
SUMMARY = os.path.join(HERE, "_vartests_summary.txt")

BASE_KEYS = ("spec_entropy",)     # base15 per scale
BASE_SCALES = (60, 130, 250)      # E17 champion
BASE_WINDOW = 250
EPS = 1e-12


def _subwin(w, s):
    n = len(w)
    return w if s >= n else w[-s:]


def _ftest(ws, h_sorted):
    """log variance ratio window/history (== log F-stat; sanity, ~ champion's log_var_ratio)."""
    vw = float(np.var(ws, ddof=1)) if len(ws) > 1 else EPS
    vh = float(np.var(h_sorted, ddof=1)) if len(h_sorted) > 1 else EPS
    return float(np.log((vw + EPS) / (vh + EPS)))


def _levene_bf(ws, h_dev):
    """Brown-Forsythe Levene 2-sample statistic (median-centered abs deviations).

    h_dev = precomputed |history - median(history)|. Pure numpy.
    """
    za = np.abs(ws - np.median(ws))
    zb = h_dev
    n1, n2 = len(za), len(zb)
    N = n1 + n2
    if n1 < 2 or n2 < 2:
        return 0.0
    m1 = float(za.mean())
    m2 = float(zb.mean())
    m = (za.sum() + zb.sum()) / N
    num = (N - 2) * (n1 * (m1 - m) ** 2 + n2 * (m2 - m) ** 2)
    den = float(((za - m1) ** 2).sum() + ((zb - m2) ** 2).sum())
    return float(num / den) if den > EPS else 0.0


def _fligner(ws, h_dev):
    """Fligner-Killeen 2-sample statistic (rank-based normal-scores of |x-median|)."""
    za = np.abs(ws - np.median(ws))
    zb = h_dev
    n1, n2 = len(za), len(zb)
    N = n1 + n2
    if n1 < 2 or n2 < 2:
        return 0.0
    d = np.concatenate([za, zb])
    r = rankdata(d)
    a = ndtri(0.5 + r / (2.0 * (N + 1)))
    abar = float(a.mean())
    v = float(a.var())  # population variance of normal scores
    if v <= EPS:
        return 0.0
    a1 = a[:n1]
    stat = (n1 * (float(a1.mean()) - abar) ** 2 +
            n2 * (float(a[n1:].mean()) - abar) ** 2) / v
    return float(stat)


def vt_feat(w, ctx, vkeys):
    """champion base15 per scale + selected variance-test stats per scale."""
    out = []
    h = ctx["sample_sorted"]
    h_dev = np.abs(h - np.median(h))
    for s in BASE_SCALES:
        ws = _subwin(w, s)
        out.extend(_feat(ws, ctx, keys=BASE_KEYS))
        if vkeys:
            if "ftest" in vkeys:
                out.append(_ftest(ws, h))
            if "levene" in vkeys:
                out.append(_levene_bf(ws, h_dev))
            if "fligner" in vkeys:
                out.append(_fligner(ws, h_dev))
    return out


CANDIDATES = {
    "vt_ftest":   ("ftest",),      # sanity: ~ existing log_var_ratio
    "vt_levene":  ("levene",),
    "vt_fligner": ("fligner",),
    "vt_lev_fli": ("levene", "fligner"),
}
if SMOKE:
    CANDIDATES = {"vt_levene": ("levene",), "vt_fligner": ("fligner",)}


def write_row(fh, name, nfeat, aucs, base_aucs, secs):
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1))
    if base_aucs is None:
        md = dsd = t = float("nan")
        verdict = "baseline(ms_60_130_250@250)"
    else:
        md, dsd, t, _ = paired(aucs, base_aucs)
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
    print(f"=== E24 variance two-sample tests ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {len(folds[0])}", flush=True)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("name\tn_feat\tfold_aucs\tmean\tstd\tdelta\tdelta_std\tt\t"
             "verdict\tsecs\ttime\n")
    fh.flush()

    ts = time.time()
    base_fn = partial(vt_feat, vkeys=())
    base, nf = evaluate(base_fn, BASE_WINDOW, X, yi, y_val, picks, folds)
    write_row(fh, "ms_60_130_250", nf, base, None, time.time() - ts)

    for name, keys in CANDIDATES.items():
        ts = time.time()
        try:
            fn = partial(vt_feat, vkeys=keys)
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
        f.write(f"E24 variance two-sample tests, 10 folds, paired vs champion "
                f"ms_60_130_250@250 ({datetime.now():%Y-%m-%d %H:%M})\n")
        f.write("HARDENED bar: only t>=2.5 worth an LB slot.\n\n")
        f.write(df.sort_values("t", ascending=False)[
            ["name", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
