"""E13: re-confirm top E12 feature groups at the shipped WINDOW=130 (10 folds).

The E12 battery ranked features at WINDOW=75; features can interact with window length,
so before shipping any we re-test the winners at WINDOW=130 with 10-fold power. base14@130
is the reference (== E11's shipped config). Paired t over the same 10 folds.

Reuses the overnight harness (load_pool/select_picks/evaluate/paired) and _feat_lib.
Writes incrementally to _feat130_results.tsv so partial progress survives.

Usage:
  SMOKE=1 .venv/bin/python -u _confirm_feats130.py    # ~1-2 min self-test
  nohup .venv/bin/python -u _confirm_feats130.py > _feat130.log 2>&1 &   # real run
"""
import os
import time
from datetime import datetime
from functools import partial

from _overnight import (
    load_pool, select_picks, evaluate, paired, SMOKE, WINDOW_CANDIDATE,
)
from _feat_lib import base_features, build_candidates, _feat

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "_feat130_results.tsv")
SUMMARY = os.path.join(HERE, "_feat130_summary.txt")
WINDOW = WINDOW_CANDIDATE  # 130

# Distinct-family stacks (combo_volatility already contains rv_ratio, so don't stack them).
STACK_RUNS_SPEC = partial(_feat, keys=("runs", "spec_entropy"))
STACK_SHIP = partial(_feat, keys=("runs", "spec_entropy", "combo_volatility"))


def candidate_list():
    cands = build_candidates()
    names = ["runs", "spec_entropy", "combo_volatility", "rv_ratio", "perm_entropy"]
    tests = [(n, cands[n]) for n in names]
    tests += [("stack_runs_spec", STACK_RUNS_SPEC), ("stack_ship", STACK_SHIP)]
    if SMOKE:
        tests = tests[:2]
    return tests


def write_row(fh, name, nfeat, aucs, base_aucs, secs):
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1))
    if base_aucs is None:
        md = dsd = t = float("nan")
        verdict = "baseline"
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
    print(f"{name:<18} nf={nfeat:<2} mean={mean:.4f} d={md:+.4f} t={t:+.2f} "
          f"{verdict} ({secs:.0f}s)", flush=True)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== E13 feature confirm @WINDOW={WINDOW} ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    X, yi, y_val, train_ids, folds = load_pool()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {len(folds[0])}", flush=True)

    fh = open(RESULTS, "w", encoding="utf-8")
    fh.write("name\tn_feat\tfold_aucs\tmean\tstd\tdelta\tdelta_std\tt\tverdict\tsecs\ttime\n")
    fh.flush()

    ts = time.time()
    base, nf = evaluate(base_features, WINDOW, X, yi, y_val, picks, folds)
    write_row(fh, "base14@130", nf, base, None, time.time() - ts)

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
        f.write(f"E13 feature confirm @WINDOW={WINDOW} ({datetime.now():%Y-%m-%d %H:%M})\n\n")
        f.write(df.sort_values("t", ascending=False)[
            ["name", "n_feat", "mean", "delta", "t", "verdict"]].to_string(index=False))
    print(f"summary -> {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
