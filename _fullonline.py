"""E18: full-online-vs-history. The champion (E17 ms_60_130_250) still caps every
view at a 250-point trailing window, so as the online segment grows (up to ~1000
points) it discards most of the post-break evidence. This adds an UNBOUNDED scale:
compute the base15 series-vs-history contrast over the ENTIRE online-so-far, which
sharpens as more post-break points accumulate.

Mechanism: reuse the multi-scale feat builder. A candidate whose harness `window` is
huge means the streaming buffer is never truncated -> the feature fn receives the full
online prefix x_on[:t+1]. A sentinel FULL scale (> any series length) makes the guard
`s >= n` reuse that whole prefix as one extra base15 block. Finite scales still slice
trailing sub-windows off the same prefix, so the champion's 60/130/250 views are
reproduced exactly and the candidate STRICTLY adds the full-online block -> clean
nested paired comparison.

Baseline = E17 champion ms_60_130_250 (scales (60,130,250) @ window 250).
10 folds. Ship if |t|>=2 & delta>0 (t>=1.9 = LB-testable).

Usage:
  SMOKE=1 .venv/bin/python -u _fullonline.py
  nohup .venv/bin/python -u _fullonline.py > _fullonline.log 2>&1 &
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
RESULTS = os.path.join(HERE, "_fullonline_results.tsv")
SUMMARY = os.path.join(HERE, "_fullonline_summary.txt")

BASE_KEYS = ("spec_entropy",)   # base15 per scale
FULL = 10 ** 9                  # sentinel scale: >= any online length -> full prefix
BIG = 10 ** 7                   # harness window big enough to never truncate the buffer

# E17 champion baseline: base15 on nested trailing sub-windows 60/130/250 @ window 250.
BASE_SCALES = (60, 130, 250)
BASE_WINDOW = 250


def ms_feat(w, ctx, scales):
    """base15 features on each nested trailing sub-window w[-s:] (or full w if s>=n)."""
    out = []
    n = len(w)
    for s in scales:
        ws = w if s >= n else w[-s:]
        out.extend(_feat(ws, ctx, keys=BASE_KEYS))
    return out


# name -> (scales, harness_window). BIG window => buffer holds the full online prefix.
CANDIDATES = {
    "ms_60_130_250_full": ((60, 130, 250, FULL), BIG),  # champion + full-online block
    "ms_60_130_full":     ((60, 130, FULL), BIG),        # full-online replaces the 250 view
    "ms_130_full":        ((130, FULL), BIG),            # champion 130 + full-online only
}
if SMOKE:
    CANDIDATES = {"ms_60_130_250_full": ((60, 130, 250, FULL), BIG)}


def write_row(fh, name, window, nfeat, aucs, base_aucs, secs):
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1))
    if base_aucs is None:
        md = dsd = t = float("nan")
        verdict = "baseline(ms_60_130_250@250)"
    else:
        md, dsd, t, _ = paired(aucs, base_aucs)
        verdict = "SHIP" if abs(t) >= 2 and md > 0 else (
            "promising" if t >= 1.5 and md > 0 else "noise")
    wtxt = "BIG" if window >= BIG else str(window)
    fh.write("\t".join([
        name, wtxt, str(nfeat), ",".join(f"{v:.4f}" for v in aucs),
        f"{mean:.4f}", f"{sd:.4f}", f"{md:+.4f}", f"{dsd:.4f}", f"{t:+.2f}",
        verdict, f"{secs:.0f}", datetime.now().strftime("%H:%M:%S"),
    ]) + "\n")
    fh.flush()
    os.fsync(fh.fileno())
    print(f"{name:<20} W={wtxt:<4} nf={nfeat:<2} mean={mean:.4f} d={md:+.4f} t={t:+.2f} "
          f"{verdict} ({secs:.0f}s)", flush=True)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== E18 full-online-vs-history ({mode}) "
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
            fh.write(f"{name}\tBIG\tERR\t\t\t\t\t\t\tERROR:{type(e).__name__}\t"
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
        f.write(f"E18 full-online-vs-history, 10 folds, paired vs champion "
                f"ms_60_130_250@250 ({datetime.now():%Y-%m-%d %H:%M})\n\n")
        f.write(df.sort_values("t", ascending=False)[
            ["name", "window", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
