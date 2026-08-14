"""Confirm the multi-seed search winner on FRESH seeds (not used for selection).

The winner was chosen as best-of-48 by mean delta over seeds {42,7,123,2024}, so its
reported +0.0038 is still mildly inflated on THOSE seeds. This re-tests it vs the champion
on seeds the search never saw. If it stays positive across fresh seeds, it's a real (small)
robust win and safe to ship; if it collapses, it was residual selection luck.

Usage:
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _confirm_winner.py \
      > _confirm_winner.log 2>&1 & disown
"""
import time
from datetime import datetime

import numpy as np

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, eval_model, load_pool_strat, make_hgb,
    precompute_fold, select_picks, WINDOW,
)
from _overnight import build_training, paired

WINNER = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
              min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)
FRESH_SEEDS = [31337, 2025, 555, 99]   # disjoint from the search seeds {42,7,123,2024}
SHIP_MEAN = 0.0015


def run_seed(seed):
    ET.SEED = seed
    X, yi, y_val, train_ids, folds = load_pool_strat()
    picks = select_picks(X, train_ids)
    Xtr, ytr = build_training(X, yi, picks, champion_feat, WINDOW)
    caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
    del X
    base, _ = eval_model(make_hgb(), Xtr, ytr, caches, y_val)
    cand, _ = eval_model(make_hgb(**WINNER), Xtr, ytr, caches, y_val)
    md, _, t, _ = paired(cand, base)
    return base.mean(), cand.mean(), md, t


def main():
    print(f"=== confirm search winner on FRESH seeds {datetime.now():%Y-%m-%d %H:%M:%S} ===",
          flush=True)
    print(f"WINNER={WINNER}", flush=True)
    print(f"FRESH_SEEDS={FRESH_SEEDS} (search used 42/7/123/2024)", flush=True)
    deltas, ts = [], []
    for s in FRESH_SEEDS:
        t0 = time.time()
        bm, cm, md, t = run_seed(s)
        deltas.append(md)
        ts.append(t)
        print(f"  seed={s:<6} champion={bm:.4f} winner={cm:.4f} delta={md:+.4f} t={t:+.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    deltas = np.asarray(deltas)
    print("\n=== SUMMARY ===", flush=True)
    print(f"  mean={deltas.mean():+.4f}  min={deltas.min():+.4f}  max={deltas.max():+.4f}  "
          f"positive {int((deltas>0).sum())}/{len(FRESH_SEEDS)}", flush=True)
    ship = (deltas > 0).all() and deltas.mean() >= SHIP_MEAN
    print(f"  VERDICT: {'CONFIRMED — ship it' if ship else 'NOT ROBUST — champion stands'}",
          flush=True)


if __name__ == "__main__":
    main()
