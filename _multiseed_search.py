"""Multi-seed joint HGB search — the principled fix for selection bias.

The E31 tuning "winner" (hgb#11) looked great because it was the best of 24 configs
on ONE fold draw, then evaporated on independent seeds. This search removes that bias
by construction: every candidate is scored by its MEAN paired-delta vs the champion
across several INDEPENDENT pool seeds, and only configs that are positive on ALL seeds
(with a mean above threshold) are flagged shippable.

Efficiency: the feature precompute (caches + train matrix) is config-independent, so we
build it ONCE per pool seed, then fit/score every config against it. Only one seed's data
is held in memory at a time. HGB random_state is FIXED (42) so the only variation per
config is the hyperparameters; the only variation per seed is the data split.

Usage (detached, this laptop is a 15-job harness — run only one at a time):
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _multiseed_search.py \
      > _multiseed_search.log 2>&1 & disown
  # knobs: N_CONFIGS (default 48), SMOKE=1 for a quick shape check
"""
import os
import time
from datetime import datetime

import numpy as np

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, eval_model, load_pool_strat, make_hgb,
    precompute_fold, select_picks, WINDOW,
)
from _overnight import build_training, paired

SMOKE = os.environ.get("SMOKE") == "1"
N_CONFIGS = int(os.environ.get("N_CONFIGS", "8" if SMOKE else "48"))
POOL_SEEDS = [42, 7] if SMOKE else [42, 7, 123, 2024]
CFG_SEED = 20260813          # fixes which configs are sampled (reproducible)
SHIP_MEAN = 0.0015           # min mean delta to consider shipping
SHIP_MIN = 0.0               # every seed must clear this (i.e. positive on all)


def sample_hgb(rng):
    """Joint random hyperparameter draw (same space as the E31 search)."""
    return dict(
        learning_rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.15)))),
        max_iter=int(rng.choice([150, 200, 300, 400])),
        max_leaf_nodes=int(rng.choice([15, 31, 47, 63, 95])),
        min_samples_leaf=int(rng.choice([10, 20, 40, 80])),
        l2_regularization=float(rng.choice([0.0, 0.5, 1.0, 2.0, 5.0])),
        max_features=float(rng.choice([0.5, 0.7, 1.0])),
    )


def fmt_cfg(c):
    return (f"lr={c['learning_rate']:.3f} it={c['max_iter']} leaves={c['max_leaf_nodes']} "
            f"msl={c['min_samples_leaf']} l2={c['l2_regularization']} mf={c['max_features']}")


def main():
    print(f"=== multi-seed HGB search {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"N_CONFIGS={N_CONFIGS}  POOL_SEEDS={POOL_SEEDS}  ship: pos on all & mean>={SHIP_MEAN}",
          flush=True)

    rng = np.random.default_rng(CFG_SEED)
    configs = [sample_hgb(rng) for _ in range(N_CONFIGS)]

    deltas = np.full((N_CONFIGS, len(POOL_SEEDS)), np.nan)
    tstats = np.full((N_CONFIGS, len(POOL_SEEDS)), np.nan)

    for si, seed in enumerate(POOL_SEEDS):
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        picks = select_picks(X, train_ids)
        Xtr, ytr = build_training(X, yi, picks, champion_feat, WINDOW)
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
        del X  # free the raw parquet subset; caches + Xtr are all we need now

        base_aucs, _ = eval_model(make_hgb(), Xtr, ytr, caches, y_val)
        print(f"\n[seed {seed}] champion mean={base_aucs.mean():.4f}  "
              f"(prep {time.time()-t0:.0f}s)", flush=True)

        for ci, cfg in enumerate(configs):
            aucs, _ = eval_model(make_hgb(**cfg), Xtr, ytr, caches, y_val)
            md, _, t, _ = paired(aucs, base_aucs)
            deltas[ci, si] = md
            tstats[ci, si] = t
        print(f"[seed {seed}] {N_CONFIGS} configs done ({time.time()-t0:.0f}s total)", flush=True)
        del Xtr, ytr, caches, base_aucs

    mean_d = deltas.mean(1)
    min_d = deltas.min(1)
    mean_t = tstats.mean(1)
    pos_all = (deltas > SHIP_MIN).all(1)

    order = np.argsort(-mean_d)
    print("\n=== ranked by mean delta (top 12) ===", flush=True)
    print(f"{'rank':>4} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all  config", flush=True)
    for r, ci in enumerate(order[:12]):
        flag = "YES" if pos_all[ci] else " no"
        print(f"{r+1:>4} {mean_d[ci]:+8.4f} {min_d[ci]:+8.4f} {mean_t[ci]:+7.2f}   {flag}   "
              f"{fmt_cfg(configs[ci])}", flush=True)

    winners = [ci for ci in order if pos_all[ci] and mean_d[ci] >= SHIP_MEAN]
    print("\n=== VERDICT ===", flush=True)
    if winners:
        best = winners[0]
        print(f"  ROBUST WINNER: mean_d={mean_d[best]:+.4f} min_d={min_d[best]:+.4f} "
              f"mean_t={mean_t[best]:+.2f} positive on all {len(POOL_SEEDS)} seeds", flush=True)
        print(f"  per-seed deltas: "
              + "  ".join(f"{POOL_SEEDS[s]}:{deltas[best,s]:+.4f}(t{tstats[best,s]:+.1f})"
                          for s in range(len(POOL_SEEDS))), flush=True)
        print(f"  WIRE THIS: {fmt_cfg(configs[best])}", flush=True)
        print(f"  full dict: {configs[best]}", flush=True)
    else:
        print(f"  NO ROBUST WINNER — no config is positive on all {len(POOL_SEEDS)} seeds "
              f"with mean>={SHIP_MEAN}. Champion stands (ceiling confirmed rigorously).", flush=True)


if __name__ == "__main__":
    main()
