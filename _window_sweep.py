"""E16: WINDOW length re-sweep on the E13 champion (base15 = base14 + spec_entropy).

E11 established WINDOW 75->130 was our biggest single jump (+0.69pt LB). The proxy plateaued
~160-250, so 130 may be sub-optimal. This tests larger windows paired vs W=130 on 10 folds
with the SHIPPED feature set (base15). Ship a larger window only if |t|>=2 & delta>0
(t>=1.9 = testable on LB). Incremental crash-safe TSV.

Usage:
  SMOKE=1 .venv/bin/python -u _window_sweep.py
  nohup .venv/bin/python -u _window_sweep.py > _window_sweep.log 2>&1 &
"""
import os
import time
from datetime import datetime
from functools import partial

from _overnight import (
    load_pool, select_picks, evaluate, paired, SMOKE, WINDOW_CANDIDATE,
)
from _feat_lib import _feat

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "_window_sweep_results.tsv")
SUMMARY = os.path.join(HERE, "_window_sweep_summary.txt")

BASE_KEYS = ("spec_entropy",)          # base15 = shipped E13 config
BASE_WINDOW = WINDOW_CANDIDATE          # 130 (current champion)
CANDIDATE_WINDOWS = [160, 180, 200, 250]
if SMOKE:
    CANDIDATE_WINDOWS = [160]


def write_row(fh, window, nfeat, aucs, base_aucs, secs):
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1))
    if base_aucs is None:
        md = dsd = t = float("nan")
        verdict = f"baseline(W={BASE_WINDOW})"
    else:
        md, dsd, t, _ = paired(aucs, base_aucs)
        verdict = "SHIP" if abs(t) >= 2 and md > 0 else (
            "promising" if t >= 1.5 and md > 0 else "noise")
    fh.write("\t".join([
        str(window), str(nfeat), ",".join(f"{v:.4f}" for v in aucs),
        f"{mean:.4f}", f"{sd:.4f}", f"{md:+.4f}", f"{dsd:.4f}", f"{t:+.2f}",
        verdict, f"{secs:.0f}", datetime.now().strftime("%H:%M:%S"),
    ]) + "\n")
    fh.flush()
    os.fsync(fh.fileno())
    print(f"W={window:<4} nf={nfeat:<2} mean={mean:.4f} d={md:+.4f} t={t:+.2f} "
          f"{verdict} ({secs:.0f}s)", flush=True)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== E16 WINDOW re-sweep (base15) ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {len(folds[0])}", flush=True)

    feat_fn = partial(_feat, keys=BASE_KEYS)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("window\tn_feat\tfold_aucs\tmean\tstd\tdelta\tdelta_std\tt\tverdict\tsecs\ttime\n")
    fh.flush()

    ts = time.time()
    base, nf = evaluate(feat_fn, BASE_WINDOW, X, yi, y_val, picks, folds)
    write_row(fh, BASE_WINDOW, nf, base, None, time.time() - ts)

    for w in CANDIDATE_WINDOWS:
        ts = time.time()
        try:
            aucs, nf = evaluate(feat_fn, w, X, yi, y_val, picks, folds)
            write_row(fh, w, nf, aucs, base, time.time() - ts)
        except Exception as e:  # noqa: BLE001 - keep the run alive
            fh.write(f"{w}\tERR\t\t\t\t\t\t\tERROR:{type(e).__name__}\t"
                     f"{time.time()-ts:.0f}\t{datetime.now():%H:%M:%S}\n")
            fh.flush()
            print(f"W={w} FAILED: {type(e).__name__}: {e}", flush=True)

    fh.close()
    _summary()
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)


def _summary():
    import pandas as pd
    df = pd.read_csv(RESULTS, sep="\t")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(f"E16 WINDOW re-sweep on base15, 10 folds, paired vs W={BASE_WINDOW} "
                f"({datetime.now():%Y-%m-%d %H:%M})\n\n")
        f.write(df.sort_values("window")[
            ["window", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))
    print(f"summary -> {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
