"""Lever B: within-online max-split changepoint localisation features on top of base15 (W130).

Baseline = base15 = base14 + spec_entropy (the E13 shipped champion). Each candidate adds ONE
max-split changepoint statistic ON TOP of spec_entropy, so t measures the *marginal* value
beyond what we already ship. These scan candidate split points INSIDE the trailing online
window and take the max standardized contrast between the two sides -> fire when a break lies
inside the recent window, complementing the vs-history features. 10 folds at WINDOW=130.
Ship rule: |t|>=2 & delta>0 (t>=1.9 = testable on LB). Incremental crash-safe TSV.

Usage:
  SMOKE=1 .venv/bin/python -u _lever_b.py
  nohup .venv/bin/python -u _lever_b.py > _lever_b.log 2>&1 &
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
RESULTS = os.path.join(HERE, "_lever_b_results.tsv")
SUMMARY = os.path.join(HERE, "_lever_b_summary.txt")
WINDOW = WINDOW_CANDIDATE  # 130

BASE_KEYS = ("spec_entropy",)  # base15 = shipped E13 config

SINGLES = [
    "max_split_t",     # Chow / max-t mean-contrast changepoint
    "max_split_lr",    # Gaussian max log-LR (mean+variance change)
    "max_split_var",   # max |log var ratio| variance changepoint
    "max_split_pos",   # localised position (k*/n) of the best split
]
COMBOS = {
    "combo_maxsplit": ("max_split_t", "max_split_lr", "max_split_var", "max_split_pos"),
}


def candidate_list():
    tests = [(k, partial(_feat, keys=BASE_KEYS + (k,))) for k in SINGLES]
    tests += [(n, partial(_feat, keys=BASE_KEYS + ks)) for n, ks in COMBOS.items()]
    if SMOKE:
        tests = tests[:2]
    return tests


def write_row(fh, name, nfeat, aucs, base_aucs, secs):
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1))
    if base_aucs is None:
        md = dsd = t = float("nan")
        verdict = "baseline(base15)"
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
    print(f"{name:<20} nf={nfeat:<2} mean={mean:.4f} d={md:+.4f} t={t:+.2f} "
          f"{verdict} ({secs:.0f}s)", flush=True)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== Lever B max-split changepoint search @W={WINDOW} ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {len(folds[0])}", flush=True)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("name\tn_feat\tfold_aucs\tmean\tstd\tdelta\tdelta_std\tt\tverdict\tsecs\ttime\n")
    fh.flush()

    ts = time.time()
    base_fn = partial(_feat, keys=BASE_KEYS)
    base, nf = evaluate(base_fn, WINDOW, X, yi, y_val, picks, folds)
    write_row(fh, "base15", nf, base, None, time.time() - ts)

    for name, fn in candidate_list():
        ts = time.time()
        try:
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
        f.write(f"Lever B max-split changepoint on top of spec_entropy @W={WINDOW} "
                f"({datetime.now():%Y-%m-%d %H:%M})\n\n")
        f.write(df.sort_values("t", ascending=False)[
            ["name", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))
    print(f"summary -> {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
