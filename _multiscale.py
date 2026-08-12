"""E17: multi-scale features — compute the base15 feature set at several nested window
lengths and concatenate, so the model sees short (responsive) + medium (champion 130) +
long (slow) views of the same series-vs-history contrast at once.

Baseline = base15 @ W=130 (E13 champion). Each candidate computes _feat(keys=spec_entropy)
on nested trailing sub-windows w[-s:] for s in `scales` (guarded when s>len(w)). The 130-scale
sub-window reproduces the baseline's features exactly, so a candidate STRICTLY adds scales
-> clean nested paired comparison. 10 folds. Ship if |t|>=2 & delta>0 (t>=1.9 = LB-testable).

Note: evaluate() trains+infers at the harness `window`, which must be >= the largest scale.
So ms candidates that include a 250 scale run at window=250; the 130-only-extra ones at 130.

Usage:
  SMOKE=1 .venv/bin/python -u _multiscale.py
  nohup .venv/bin/python -u _multiscale.py > _multiscale.log 2>&1 &
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
RESULTS = os.path.join(HERE, "_multiscale_results.tsv")
SUMMARY = os.path.join(HERE, "_multiscale_summary.txt")

BASE_KEYS = ("spec_entropy",)   # base15 per scale
BASE_WINDOW = 130               # champion


def ms_feat(w, ctx, scales):
    """base15 features computed on each nested trailing sub-window w[-s:], concatenated."""
    out = []
    n = len(w)
    for s in scales:
        ws = w if s >= n else w[-s:]
        out.extend(_feat(ws, ctx, keys=BASE_KEYS))
    return out


# name -> (scales, harness_window). harness_window must be >= max(scales).
CANDIDATES = {
    "ms_40_130":       ((40, 130), 130),
    "ms_20_60_130":    ((20, 60, 130), 130),
    "ms_130_250":      ((130, 250), 250),
    "ms_40_130_250":   ((40, 130, 250), 250),
    "ms_60_130_250":   ((60, 130, 250), 250),
}
if SMOKE:
    CANDIDATES = {"ms_40_130": ((40, 130), 130)}


def write_row(fh, name, window, nfeat, aucs, base_aucs, secs):
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1))
    if base_aucs is None:
        md = dsd = t = float("nan")
        verdict = f"baseline(base15@{BASE_WINDOW})"
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
    print(f"{name:<16} W={window:<4} nf={nfeat:<2} mean={mean:.4f} d={md:+.4f} t={t:+.2f} "
          f"{verdict} ({secs:.0f}s)", flush=True)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== E17 multi-scale features ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {len(folds[0])}", flush=True)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("name\twindow\tn_feat\tfold_aucs\tmean\tstd\tdelta\tdelta_std\tt\tverdict\tsecs\ttime\n")
    fh.flush()

    ts = time.time()
    base_fn = partial(_feat, keys=BASE_KEYS)
    base, nf = evaluate(base_fn, BASE_WINDOW, X, yi, y_val, picks, folds)
    write_row(fh, "base15", BASE_WINDOW, nf, base, None, time.time() - ts)

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
        f.write(f"E17 multi-scale features (base15 per nested scale), 10 folds, "
                f"paired vs base15@{BASE_WINDOW} ({datetime.now():%Y-%m-%d %H:%M})\n\n")
        f.write(df.sort_values("t", ascending=False)[
            ["name", "window", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))
    print(f"summary -> {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
