"""Does the discarded 98% of online steps carry signal the champion isn't capturing?

The training set samples only STEPS_PER_SERIES=10 steps/series -> ~79k rows = 1.96% of the ~4.0M
online steps in the 8000-series train pool. This sweeps STEPS_PER_SERIES for the tuned HGB member
and re-scores the FULL champion blend (HGB + fixed shipped ranker, 0.6/0.4 train-CDF) on the same
val folds, paired-t vs the S=10 baseline across pool seeds.

The ranker (member 2) is held fixed at DENSE_K -> we isolate the effect of feeding the pointwise
HGB member more online steps. Reports HGB-standalone AUC and blend AUC per S so we see whether the
member improves even when the blend doesn't.

Usage:
  SMOKE=1 .venv/bin/python -u _steps_sweep.py
  SEEDS=42,7,123 STEPS=10,30,60,120,240 .venv/bin/python -u _steps_sweep.py
"""
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import _ensemble_tune as ET
from _ensemble_tune import (
    WINDOW, champion_feat, load_pool_strat, make_hgb, precompute_fold, score_fold, select_picks,
)
from _exp_fast import ts_auc_from
from _lambdarank import build_grouped, sort_into_groups, fit_ranker
from _overnight import build_training, paired

SMOKE = os.environ.get("SMOKE") == "1"
POOL_SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7" if SMOKE else "42,7,123").split(",")]
DENSE_K = 30 if SMOKE else 60
STEPS = [int(s) for s in os.environ.get("STEPS", "5,10,20" if SMOKE else "10,30,60,120,240").split(",")]

TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def main():
    print(f"=== STEPS_PER_SERIES sweep {datetime.now():%F %T} | seeds={POOL_SEEDS} "
          f"| steps={STEPS} ===", flush=True)
    blend_store = {s: [] for s in STEPS}     # per-seed list of best blend mean AUC
    d_store = {s: [] for s in STEPS}         # per-seed mean delta vs S=STEPS[0]
    t_store = {s: [] for s in STEPS}
    hgb_store = {s: [] for s in STEPS}       # HGB-standalone mean AUC
    nrows_store = {s: [] for s in STEPS}

    base_s = STEPS[0]
    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

        # fixed member 2: shipped ranker (DENSE_K), built once per seed
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr_, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr_, gr)
        ref_rank = np.sort(ranker.r.predict(rr))
        rank_series = [score_fold(ranker, c) for c in caches]
        fr = [pct(s.to_numpy(), ref_rank) for s in rank_series]
        print(f"\n[seed {seed}] ranker built ({time.time()-t0:.0f}s)", flush=True)

        base_aucs_ref = None
        for S in STEPS:
            ts = time.time()
            ET.STEPS_PER_SERIES = S
            picks = select_picks(X, train_ids)
            Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)
            hgb = make_hgb(**TUNED, random_state=seed)
            with threadpool_limits(limits=1):
                hgb.fit(Xtr_c, ytr_c)
            ref_hgb = np.sort(hgb.predict_proba(Xtr_c)[:, 1])
            hgb_series = [score_fold(hgb, c) for c in caches]
            fh = [pct(s.to_numpy(), ref_hgb) for s in hgb_series]

            blend_aucs = np.asarray([
                ts_auc_from(pd.Series(0.6 * a + 0.4 * b, index=s.index).to_frame("prediction"), y_val)
                for a, b, s in zip(fh, fr, hgb_series)])
            hgb_aucs = np.asarray([ts_auc_from(s.to_frame("prediction"), y_val) for s in hgb_series])

            if S == base_s:
                base_aucs_ref = blend_aucs
                md, t = 0.0, 0.0
            else:
                md, _sd, t, _ = paired(blend_aucs, base_aucs_ref)
            blend_store[S].append(float(blend_aucs.mean()))
            d_store[S].append(float(md))
            t_store[S].append(float(t))
            hgb_store[S].append(float(hgb_aucs.mean()))
            nrows_store[S].append(int(Xtr_c.shape[0]))
            print(f"  S={S:<4} rows={Xtr_c.shape[0]:>7} hgb_solo={hgb_aucs.mean():.4f} "
                  f"blend={blend_aucs.mean():.4f} d_vs_S{base_s}={md:+.4f} t={t:+.2f} "
                  f"({time.time()-ts:.0f}s)", flush=True)
        del X, caches, rank_series

    print(f"\n=== SUMMARY (champion blend vs S={base_s} baseline) ===", flush=True)
    for S in STEPS:
        ds, tsv = d_store[S], t_store[S]
        npos = sum(1 for x in ds if x > 0)
        tag = "baseline" if S == base_s else (
            "HELPS" if npos == len(ds) and np.mean(ds) >= 0.0015 and min(tsv) >= 2.5 else (
                "up-noise" if np.mean(ds) > 0 else "flat/down"))
        print(f"S={S:<4} rows~{int(np.mean(nrows_store[S])):>7} hgb_solo {np.mean(hgb_store[S]):.4f} "
              f"blend {np.mean(blend_store[S]):.4f} mean_d {np.mean(ds):+.4f} min_t {min(tsv):+.2f} "
              f"pos {npos}/{len(ds)} -> {tag}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
