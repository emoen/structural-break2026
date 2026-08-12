"""E19: scale-set sweep. E17 proved multi-scale is the winning representation but only
tried 5 hand-picked sets; (60,130,250) won. This searches around that winner: does a
4th trailing scale, denser spacing, or a longer top-scale beat the champion?

Mechanism identical to E17 (_multiscale.ms_feat): base15 on nested trailing sub-windows
w[-s:], concatenated. harness_window must be >= max(scales). Baseline = champion
ms_60_130_250 @250. 10 folds, paired. Ship if |t|>=2 & delta>0 (t>=1.9 = LB-testable).

Usage:
  SMOKE=1 .venv/bin/python -u _scalesweep.py
  nohup .venv/bin/python -u _scalesweep.py > _scalesweep.log 2>&1 &
"""
import os
import time
from datetime import datetime
from functools import partial

from _overnight import (
    load_pool, select_picks, evaluate, paired, SMOKE,
)
from _feat_lib import _feat

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "_scalesweep_results.tsv")
SUMMARY = os.path.join(HERE, "_scalesweep_summary.txt")

BASE_KEYS = ("spec_entropy",)   # base15 per scale
BASE_SCALES = (60, 130, 250)    # E17 champion
BASE_WINDOW = 250


def ms_feat(w, ctx, scales):
    """base15 features on each nested trailing sub-window w[-s:] (or full w if s>=n)."""
    out = []
    n = len(w)
    for s in scales:
        ws = w if s >= n else w[-s:]
        out.extend(_feat(ws, ctx, keys=BASE_KEYS))
    return out


# name -> (scales, harness_window). harness_window must be >= max(scales).
CANDIDATES = {
    "ms_40_60_130_250":   ((40, 60, 130, 250), 250),   # add a short 40
    "ms_60_90_130_250":   ((60, 90, 130, 250), 250),   # add a mid 90
    "ms_60_130_190_250":  ((60, 130, 190, 250), 250),  # add a long-mid 190
    "ms_60_100_160_250":  ((60, 100, 160, 250), 250),  # even 4-scale spacing
    "ms_40_90_160_250":   ((40, 90, 160, 250), 250),   # geometric-ish 4-scale
    "ms_60_130_250_350":  ((60, 130, 250, 350), 350),  # extend the long view to 350
}
if SMOKE:
    CANDIDATES = {"ms_40_60_130_250": ((40, 60, 130, 250), 250)}


def write_row(fh, name, window, nfeat, aucs, base_aucs, secs):
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1))
    if base_aucs is None:
        md = dsd = t = float("nan")
        verdict = "baseline(ms_60_130_250@250)"
    else:
        md, dsd, t, _ = paired(aucs, base_aucs)
        verdict = "SHIP" if abs(t) >= 2 and md > 0 else (
            "promising" if t >= 1.5 and md > 0 else "noise")
    fh.write("\t".join([
        name, str(window), str(nfeat), ",".join(f"{v:.4f}" for v in aucs),
        f"{mean:.4f}", f"{sd:.4f}", f"{md:+.4f}", f"{dsd:.4f}", f"{t:+.2f}",
        verdict, f"{secs:.0f}", datetime.now().strftime("%H:%M:%S"),
    ]) + "\n")
    fh.flush()
    os.fsync(fh.fileno())
    print(f"{name:<20} W={window:<4} nf={nfeat:<2} mean={mean:.4f} d={md:+.4f} t={t:+.2f} "
          f"{verdict} ({secs:.0f}s)", flush=True)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== E19 scale-set sweep ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {len(folds[0])}", flush=True)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("name\twindow\tn_feat\tfold_aucs\tmean\tstd\tdelta\tdelta_std\tt\t"
             "verdict\tsecs\ttime\n")
    fh.flush()

    ts = time.time()
    base_fn = partial(ms_feat, scales=BASE_SCALES)
    base, nf = evaluate(base_fn, BASE_WINDOW, X, yi, y_val, picks, folds)
    write_row(fh, "ms_60_130_250", BASE_WINDOW, nf, base, None, time.time() - ts)

    for name, (scales, window) in CANDIDATES.items():
        ts = time.time()
        try:
            fn = partial(ms_feat, scales=scales)
            aucs, nf = evaluate(fn, window, X, yi, y_val, picks, folds)
            write_row(fh, name, window, nf, aucs, base, time.time() - ts)
        except Exception as e:  # noqa: BLE001 - keep the run alive
            fh.write(f"{name}\t{window}\tERR\t\t\t\t\t\t\tERROR:{type(e).__name__}\t"
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
        f.write(f"E19 scale-set sweep, 10 folds, paired vs champion "
                f"ms_60_130_250@250 ({datetime.now():%Y-%m-%d %H:%M})\n\n")
        f.write(df.sort_values("t", ascending=False)[
            ["name", "window", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
