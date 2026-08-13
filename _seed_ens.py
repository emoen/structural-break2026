"""Validate a SEED-ENSEMBLE champion: average K HGB champions (different random_state).

Principled variance reduction (no selection bias, no new features / code-path). The only
randomness in the champion HGB is its internal early-stopping validation split, so averaging
K seeds smooths that out. We check it on SEVERAL independent pool seeds (not one fold draw)
so the verdict is not itself fold luck.

For each pool seed: single champion (random_state=42) vs K-model probability average, paired-t
over the 10 stratified folds. Ship only if the ensemble is >= champion across seeds.

Usage:
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _seed_ens.py > _seed_ens.log 2>&1 & disown
"""
import time
from datetime import datetime

import numpy as np

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, make_hgb, precompute_fold, score_fold, WINDOW,
)
from _exp_fast import ts_auc_from
from _overnight import build_training, paired
from threadpoolctl import threadpool_limits

MODEL_SEEDS = [42, 7, 123, 2024, 31337]   # K=5 ensemble members
POOL_SEEDS = [42, 7, 123]                  # independent data splits to test on


def fit_models(Xtr, ytr, seeds):
    models = []
    for s in seeds:
        m = make_hgb(random_state=s)
        with threadpool_limits(limits=1):
            m.fit(Xtr, ytr)
        models.append(m)
    return models


def fold_aucs(series_list_per_model, caches, y_val):
    """series_list_per_model: list (len=K) of [Series per fold]. Average then TS-AUC."""
    aucs = []
    for fi in range(len(caches)):
        blended = sum(series_list_per_model[k][fi] for k in range(len(series_list_per_model)))
        blended = blended / len(series_list_per_model)
        aucs.append(ts_auc_from(blended.to_frame("prediction"), y_val))
    return np.asarray(aucs)


def run_seed(pool_seed):
    ET.SEED = pool_seed
    X, yi, y_val, train_ids, folds = ET.load_pool_strat()
    picks = ET.select_picks(X, train_ids)
    Xtr, ytr = build_training(X, yi, picks, champion_feat, WINDOW)
    caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

    models = fit_models(Xtr, ytr, MODEL_SEEDS)
    per_model = [[score_fold(m, c) for c in caches] for m in models]

    single = np.asarray([ts_auc_from(per_model[0][fi].to_frame("prediction"), y_val)
                         for fi in range(len(caches))])  # member 0 == random_state=42 == champion
    ens = fold_aucs(per_model, caches, y_val)
    md, sd, t, _ = paired(ens, single)
    return single.mean(), ens.mean(), md, t


def main():
    print(f"=== seed-ensemble champion (K={len(MODEL_SEEDS)}) {datetime.now():%Y-%m-%d %H:%M:%S} ===",
          flush=True)
    print(f"member seeds={MODEL_SEEDS}  pool seeds={POOL_SEEDS}", flush=True)
    deltas, ts = [], []
    for ps in POOL_SEEDS:
        t0 = time.time()
        sm, em, md, t = run_seed(ps)
        deltas.append(md)
        ts.append(t)
        verdict = ("better" if md > 0 and t >= 2.0 else
                   ("lean+" if md > 0 else ("flat" if abs(md) < 1e-4 else "worse")))
        print(f"  pool_seed={ps:<5} single={sm:.4f} ensemble={em:.4f} "
              f"delta={md:+.4f} t={t:+.2f} -> {verdict}  ({time.time()-t0:.0f}s)", flush=True)
    deltas = np.asarray(deltas)
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  mean delta={deltas.mean():+.4f}  min={deltas.min():+.4f}  max={deltas.max():+.4f}  "
          f"positive on {int((deltas > 0).sum())}/{len(POOL_SEEDS)}", flush=True)
    ship = (deltas >= -1e-4).all() and deltas.mean() > 0
    print(f"  VERDICT: {'SHIP (>= champion on all seeds, mean+)' if ship else 'HOLD (not a clear improvement)'}",
          flush=True)


if __name__ == "__main__":
    main()
