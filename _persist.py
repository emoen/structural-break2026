"""Causal TEMPORAL PERSISTENCE post-processing of the per-step score trajectory.
The correct form of "reduce chatter / finalize sequentially" for a RANKING metric (TS-AUC):
a structural break is an ABSORBING state, so the ideal per-step score is monotone-nondecreasing
after tau. Lever A tested per-series DE-MEANING (negative). This tests the orthogonal axis:
temporal smoothing / evidence-accumulation, all CAUSAL (deployable: carry per-series state).

Variants (per-series, causal, applied to the raw score trajectory s_t):
  raw          : baseline (no post-processing)
  cummax       : running maximum (hard absorbing-state enforcement)
  ewma10/30    : EWMA(alpha) smoothing (variance reduction, no monotone bias)
  expmean      : causal expanding mean (heavy smoothing)
  ph_eviden    : Page-Hinkley cumulative positive evidence  cumsum(max(s - running_mean, 0))
  half_cummax  : 0.5*raw + 0.5*cummax (partial persistence)
  half_ewma30  : 0.5*raw + 0.5*ewma30
Ship rule stays |t|>=2 & delta>0 over folds. Reuses the Lever A base16 streaming harness as a
CHEAP first screen for the IDEA; if any variant clears, port to the current 61-col champion.
"""
import os, time
from datetime import datetime
from functools import partial
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from _overnight import BASE, MIN_N, load_pool, select_picks, build_training, paired
from _exp_fast import ts_auc_from
from _feat_lib import _feat
from _lever_a import WINDOW, FEAT, fold_raw_series  # reuse exact base16 streaming

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "_persist_summary.txt")
VARIANTS = ["raw", "cummax", "ewma10", "ewma30", "expmean", "ph_eviden",
            "half_cummax", "half_ewma30"]


def _ewma(v, alpha):
    out = np.empty_like(v); acc = v[0]
    out[0] = acc
    for i in range(1, len(v)):
        acc = alpha * v[i] + (1 - alpha) * acc
        out[i] = acc
    return out


def apply_variant(s, name):
    v = s.to_numpy().astype(np.float64)
    if name == "raw":
        out = v
    elif name == "cummax":
        out = np.maximum.accumulate(v)
    elif name == "ewma10":
        out = _ewma(v, 0.10)
    elif name == "ewma30":
        out = _ewma(v, 0.30)
    elif name == "expmean":
        out = np.cumsum(v) / np.arange(1, len(v) + 1)
    elif name == "ph_eviden":
        rm = np.cumsum(v) / np.arange(1, len(v) + 1)      # causal running mean
        out = np.cumsum(np.maximum(v - rm, 0.0))
    elif name == "half_cummax":
        out = 0.5 * v + 0.5 * np.maximum.accumulate(v)
    elif name == "half_ewma30":
        out = 0.5 * v + 0.5 * _ewma(v, 0.30)
    else:
        out = v
    return pd.Series(out, index=s.index)


def main():
    t0 = time.time()
    mode = "SMOKE" if os.environ.get("SMOKE") else "FULL"
    print(f"=== Persistence post-proc @W={WINDOW} ({mode}) {datetime.now():%H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    Xtr, ytr = build_training(X, yi, picks, FEAT, WINDOW)
    model = HistGradientBoostingClassifier(**BASE)
    with threadpool_limits(limits=1):
        model.fit(Xtr, ytr)
    print(f"trained base16 {Xtr.shape}; {len(folds)} folds x {len(folds[0])}", flush=True)

    aucs = {v: [] for v in VARIANTS}
    for fi, fold_ids in enumerate(folds):
        raw = fold_raw_series(model, X, fold_ids)
        for v in VARIANTS:
            pred = pd.concat([apply_variant(s, v) for s in raw]).to_frame("prediction")
            aucs[v].append(ts_auc_from(pred, y_val))
        print(f"fold {fi}: " + "  ".join(f"{v}={aucs[v][-1]:.4f}" for v in VARIANTS), flush=True)

    base = np.asarray(aucs["raw"])
    lines = [f"Persistence @W={WINDOW}  folds={len(base)}  raw mean={base.mean():.4f}",
             f"{'variant':<12} {'mean':>8} {'delta':>9} {'t':>7}  verdict"]
    rows = []
    for v in VARIANTS:
        a = np.asarray(aucs[v])
        if v == "raw":
            rows.append((9e9, f"{v:<12} {a.mean():>8.4f} {'--':>9} {'--':>7}  baseline")); continue
        md, _, t, _ = paired(a, base)
        verdict = "SHIP" if abs(t) >= 2 and md > 0 else ("promising" if t >= 1.5 and md > 0 else "noise")
        rows.append((t, f"{v:<12} {a.mean():>8.4f} {md:>+9.4f} {t:>+7.2f}  {verdict}"))
    for _, ln in sorted(rows, key=lambda r: -r[0]):
        lines.append(ln)
    text = "\n".join(lines)
    open(SUMMARY, "w").write(text + "\n")
    print("\n" + text, flush=True)
    print(f"=== DONE {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
