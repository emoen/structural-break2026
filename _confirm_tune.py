"""Confirm the E31 joint-search HGB winner (hgb#11) on INDEPENDENT fold seeds.

hgb#11 (max_features=0.5, l2=5.0, max_iter=300, min_samples_leaf=40, lr=0.047) scored
+0.0035 t=+3.51 vs champion on the SEED=42 stratified folds. Because it was the best of a
24-config search (multiple comparisons), re-validate the SAME two models on several fresh
pool seeds (different train membership + different val-fold composition, model random_state
held at 42). If the delta stays positive and t stays healthy across independent seeds, the
gain is real (not fold luck); if it bounces around 0, it was a lucky draw.

Usage:
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _confirm_tune.py \
       > _confirm_tune.log 2>&1 & disown
"""
import time
from datetime import datetime

import numpy as np

import _ensemble_tune as ET
from _ensemble_tune import champion_feat, eval_model, make_hgb, precompute_fold, WINDOW
from _overnight import build_training, paired

WINNER = dict(
    learning_rate=0.047, max_iter=300, max_leaf_nodes=31,
    min_samples_leaf=40, l2_regularization=5.0, max_features=0.5,
)
# 42 reproduces the original t=+3.51 (sanity); the rest are independent.
SEEDS = [42, 7, 123, 2024, 31337]


def run_seed(seed):
    ET.SEED = seed  # load_pool_strat/select_picks read this module global at call time
    X, yi, y_val, train_ids, folds = ET.load_pool_strat()
    picks = ET.select_picks(X, train_ids)
    Xtr, ytr = build_training(X, yi, picks, champion_feat, WINDOW)
    caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
    base_aucs, _ = eval_model(make_hgb(), Xtr, ytr, caches, y_val)
    cand_aucs, _ = eval_model(make_hgb(**WINNER), Xtr, ytr, caches, y_val)
    md, sd, t, _ = paired(cand_aucs, base_aucs)
    return base_aucs.mean(), cand_aucs.mean(), md, t


def main():
    print(f"=== confirm hgb#11 start {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"WINNER over champion: {WINNER}", flush=True)
    deltas, ts = [], []
    for s in SEEDS:
        t0 = time.time()
        bm, cm, md, t = run_seed(s)
        deltas.append(md)
        ts.append(t)
        verdict = "SHIP(hard)" if (md > 0 and t >= 2.5) else (
            "marginal" if (md > 0 and t >= 2.0) else ("weak+" if md > 0 else "neg"))
        print(f"  seed={s:<6} champ={bm:.4f} cand={cm:.4f} delta={md:+.4f} "
              f"t={t:+.2f} -> {verdict}  ({time.time()-t0:.0f}s)", flush=True)
    deltas = np.asarray(deltas)
    print(f"\n=== SUMMARY over {len(SEEDS)} seeds ===", flush=True)
    print(f"  mean delta={deltas.mean():+.4f}  min={deltas.min():+.4f}  max={deltas.max():+.4f}", flush=True)
    print(f"  positive on {int((deltas > 0).sum())}/{len(SEEDS)} seeds; "
          f"t>=2.5 on {int((np.asarray(ts) >= 2.5).sum())}/{len(SEEDS)}", flush=True)
    verdict = ("CONFIRMED (robust across seeds)" if (deltas > 0).all() and deltas.mean() > 0.0015
               else "NOT ROBUST (likely fold/search luck)")
    print(f"  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
