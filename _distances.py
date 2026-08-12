"""E20: distance-family sweep. E-PEF paper nugget generalized -> a window's base45
multi-scale feature vector can be scored by how far it sits from the NO-BREAK (label-0)
feature distribution. Mahalanobis is one metric; test a family at once, each added as
an extra feature on top of the champion ms_60_130_250 (45 feats).

Reference is fit on the fold's TRAINING pool only (mu0/Sigma0 from label-0 rows, plus
mu1/Sigma1 for the discriminative QDA log-LR), frozen, then the distance is computed per
window in both train and stream -> no leakage (val folds are disjoint series). 10 folds,
paired vs champion. Ship if |t|>=2 & delta>0 (t>=1.9 = LB-testable).

Distances (on the standardized/whitened 45-vec unless noted):
  maha           full Mahalanobis  sqrt((f-mu0) Sig0^-1 (f-mu0))
  maha_perscale  Mahalanobis per 15-dim scale block (3 scalars)
  seuclid        diagonal-only (standardized Euclidean) -> does covariance matter?
  l1             sum |z|            (robust)
  linf           max |z|           (single most-deviant feature)
  qda_llr        0.5(maha0^2 - maha1^2) + 0.5(logdet0 - logdet1)  (discriminative)
  family         maha + seuclid + l1 + linf + qda_llr together

Usage:
  SMOKE=1 .venv/bin/python -u _distances.py
  nohup .venv/bin/python -u _distances.py > _distances.log 2>&1 &
"""
import math
import os
import time
from datetime import datetime

import numpy as np

from _overnight import (
    load_pool, select_picks, evaluate, paired, build_training, SMOKE,
)
from _feat_lib import _feat

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "_distances_results.tsv")
SUMMARY = os.path.join(HERE, "_distances_summary.txt")

BASE_KEYS = ("spec_entropy",)   # base15 per scale
SCALES = (60, 130, 250)         # E17/E19 champion
WINDOW = 250
K = 15                          # base15 features per scale
EPS = 1e-9


def ms_feat(w, ctx, scales=SCALES):
    """base15 features on each nested trailing sub-window w[-s:] (or full w if s>=n)."""
    out = []
    n = len(w)
    for s in scales:
        ws = w if s >= n else w[-s:]
        out.extend(_feat(ws, ctx, keys=BASE_KEYS))
    return out


def _class_ref(Xc):
    mu = Xc.mean(0)
    cov = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(Xc.shape[1])
    cinv = np.linalg.pinv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if not np.isfinite(logdet):
        logdet = 0.0
    sd = Xc.std(0) + EPS
    return {"mu": mu, "cinv": cinv, "logdet": float(logdet), "sd": sd}


def fit_ref(Xb, yb):
    ref = {0: _class_ref(Xb[yb == 0]), 1: _class_ref(Xb[yb == 1])}
    Xc0 = Xb[yb == 0]
    blocks = []
    for b in range(len(SCALES)):
        sl = slice(b * K, (b + 1) * K)
        blocks.append((sl, _class_ref(Xc0[:, sl])))
    ref["blocks"] = blocks
    return ref


def _q(d, cinv):
    return float(d @ cinv @ d)


# --- extra-feature functions: fa = base45 vector (np.array), ref = fitted dict ---
def x_maha(fa, ref):
    d = fa - ref[0]["mu"]
    return [math.sqrt(max(0.0, _q(d, ref[0]["cinv"])))]


def x_maha_ps(fa, ref):
    out = []
    for sl, r in ref["blocks"]:
        d = fa[sl] - r["mu"]
        out.append(math.sqrt(max(0.0, _q(d, r["cinv"]))))
    return out


def x_seuc(fa, ref):
    z = (fa - ref[0]["mu"]) / ref[0]["sd"]
    return [float(np.sqrt((z * z).sum()))]


def x_l1(fa, ref):
    z = (fa - ref[0]["mu"]) / ref[0]["sd"]
    return [float(np.abs(z).sum())]


def x_linf(fa, ref):
    z = (fa - ref[0]["mu"]) / ref[0]["sd"]
    return [float(np.abs(z).max())]


def x_qda(fa, ref):
    d0, d1 = fa - ref[0]["mu"], fa - ref[1]["mu"]
    llr = 0.5 * (_q(d0, ref[0]["cinv"]) - _q(d1, ref[1]["cinv"])) \
        + 0.5 * (ref[0]["logdet"] - ref[1]["logdet"])
    return [float(llr)]


def x_family(fa, ref):
    return x_maha(fa, ref) + x_seuc(fa, ref) + x_l1(fa, ref) + x_linf(fa, ref) + x_qda(fa, ref)


CANDIDATES = {
    "maha":          x_maha,
    "maha_perscale": x_maha_ps,
    "seuclid":       x_seuc,
    "l1":            x_l1,
    "linf":          x_linf,
    "qda_llr":       x_qda,
    "family":        x_family,
}
if SMOKE:
    CANDIDATES = {"maha": x_maha, "qda_llr": x_qda}


def make_feat(extra_fn, ref):
    def fn(w, ctx):
        base = ms_feat(w, ctx)
        fa = np.asarray(base, dtype=np.float64)
        return base + extra_fn(fa, ref)
    return fn


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
    print(f"{name:<16} nf={nfeat:<2} mean={mean:.4f} d={md:+.4f} t={t:+.2f} "
          f"{verdict} ({secs:.0f}s)", flush=True)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== E20 distance-family sweep ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {len(folds[0])}", flush=True)

    # fit reference on the training pool's base45 vectors (frozen for all candidates)
    Xb, yb = build_training(X, yi, picks, ms_feat, WINDOW)
    ref = fit_ref(Xb, yb)
    print(f"ref fit: base {Xb.shape}, label1 frac {yb.mean():.3f}", flush=True)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("name\tn_feat\tfold_aucs\tmean\tstd\tdelta\tdelta_std\tt\t"
             "verdict\tsecs\ttime\n")
    fh.flush()

    ts = time.time()
    base, nf = evaluate(ms_feat, WINDOW, X, yi, y_val, picks, folds)
    write_row(fh, "ms_60_130_250", nf, base, None, time.time() - ts)

    for name, extra_fn in CANDIDATES.items():
        ts = time.time()
        try:
            fn = make_feat(extra_fn, ref)
            aucs, nf = evaluate(fn, WINDOW, X, yi, y_val, picks, folds)
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
        f.write(f"E20 distance-family sweep, 10 folds, paired vs champion "
                f"ms_60_130_250@250 ({datetime.now():%Y-%m-%d %H:%M})\n\n")
        f.write(df.sort_values("t", ascending=False)[
            ["name", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
