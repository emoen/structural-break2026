"""Reproduce the E17 champion (ms_60_130_250, 45 feats, W=250) under the E30 STRATIFIED
20%-val harness (8000 train / 2000 val, 10 folds x 200, break-rate balanced).

Purpose: confirm we recover the champion baseline recorded on this harness (~0.5674 in
E30/E31) on the rebuilt env + freshly downloaded data. Scores ONLY the champion HGB with a
single feature pass, so it is much faster than _ensemble_tune (no LR/blend/grid).

Everything (pool split, feature fn, feature cache, model, TS-AUC) is imported from
_ensemble_tune so this measures the exact same pipeline those experiments used.

Usage:
  SMOKE=1 .venv/bin/python -u _repro_champion_strat.py     # ~2 min plumbing check
  .venv/bin/python -u _repro_champion_strat.py             # full 10-fold repro
"""
import time
from datetime import datetime

from _ensemble_tune import (
    FOLD_SIZE, SMOKE, WINDOW,
    champion_feat, eval_model, load_pool_strat, make_hgb,
    precompute_fold, select_picks,
)
from _overnight import build_training

EXPECTED = 0.5674  # champion baseline recorded on this harness (README E30/E31)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== reproduce champion @ stratified 20%-val ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)

    X, yi, y_val, train_ids, folds = load_pool_strat()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {FOLD_SIZE} "
          f"(20% val, stratified)", flush=True)

    Xtr, ytr = build_training(X, yi, picks, champion_feat, WINDOW)
    print(f"train matrix {Xtr.shape}, positives {ytr.mean():.3f}", flush=True)

    tf = time.time()
    caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
    print(f"val feature cache built in {(time.time() - tf) / 60:.1f} min", flush=True)

    aucs, _ = eval_model(make_hgb(), Xtr, ytr, caches, y_val)
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1))
    print("\nfold TS-AUC: " + " ".join(f"{v:.4f}" for v in aucs), flush=True)
    print(f"champion mean={mean:.4f} std={sd:.4f} "
          f"[{aucs.min():.4f},{aucs.max():.4f}]", flush=True)
    if not SMOKE:
        print(f"expected ~{EXPECTED:.4f} (E30/E31 recorded) | diff {mean - EXPECTED:+.4f}",
              flush=True)
    print(f"=== DONE in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
