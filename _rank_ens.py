"""rank + tuned ENSEMBLE (rank-average) harness.

Standalone rank lost head-to-head vs the tuned champion (E34). But TS-AUC is a pure
within-step ORDERING metric and the LGBM listwise ranker is a DECORRELATED view of the
same 45 champion features (listwise vs the champion's pointwise proba). So the only live
use of the rank signal is a rank-AVERAGE ensemble: it can net-lift the combined per-step
ordering via variance reduction even though it loses alone.

Method
------
Both models are scored on the SAME champion feature cache (score_fold -> per-fold Series):
  * tuned champion HGB (E32 params) -> proba
  * LGBM rank_xendcg, dense-60      -> raw score (via RankerProba)
Per fold, convert each model's scores to PERCENTILE RANKS (global, per-fold) and blend
  blended = w * rank(hgb) + (1-w) * rank(ranker)
then TS-AUC. A global rank transform is monotone => within-step order preserved, so w=1.0
must EXACTLY reproduce the champion AUC (built-in sanity check). Weights swept; paired-t vs
the tuned champion on the SAME folds, multi-seed.

Gate: a blend positive on ALL pool seeds with mean_d>=0.0015 (hard t>=2.5) -> fresh-seed
confirm before an LB slot.

Usage (one 15-job harness at a time):
  SMOKE=1 .venv/bin/python -u _rank_ens.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _rank_ens.py > _rank_ens.log 2>&1 & disown
"""
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, eval_model, load_pool_strat, make_hgb,
    precompute_fold, score_fold, select_picks, WINDOW,
)
from _exp_fast import ts_auc_from
from _lambdarank import build_grouped, sort_into_groups, fit_ranker
from _overnight import build_training, paired

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    POOL_SEEDS = [42]
    DENSE_K = 30
else:
    POOL_SEEDS = [42, 7, 123, 2024]
    DENSE_K = 60

# E32 tuned champion params (LB 57.72%). early_stopping=True from HGB_BASE.
TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)

# weight on the HGB champion; 1.0 == pure champion (sanity), lower = more ranker
WEIGHTS = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def prank(s):
    """Global per-fold percentile rank of a score Series (monotone => within-step safe)."""
    return s.rank(method="average").to_numpy() / len(s)


def main():
    print(f"=== rank+tuned ensemble {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS} dense_k={DENSE_K}  weights(on HGB)={WEIGHTS}", flush=True)

    deltas = {w: [] for w in WEIGHTS}
    tstats = {w: [] for w in WEIGHTS}
    sanity = []

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

        # tuned champion HGB (pointwise) -> aucs + per-fold series
        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)
        base_aucs, hgb_series = eval_model(make_hgb(**TUNED, random_state=seed),
                                           Xtr_c, ytr_c, caches, y_val)

        # LGBM rank_xendcg (listwise), dense-60 -> per-fold series
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr, gr)
        rank_series = [score_fold(ranker, c) for c in caches]
        del X

        # blend per fold at each weight
        per_w = {w: [] for w in WEIGHTS}
        for hs, rs_ in zip(hgb_series, rank_series):
            ph = prank(hs)
            pr = prank(rs_)
            for w in WEIGHTS:
                blended = pd.Series(w * ph + (1.0 - w) * pr, index=hs.index)
                per_w[w].append(ts_auc_from(blended.to_frame("prediction"), y_val))

        for w in WEIGHTS:
            a = np.asarray(per_w[w])
            md, _sd, t, _ = paired(a, base_aucs)
            deltas[w].append(md)
            tstats[w].append(t)
            tag = "  <- champion (sanity)" if w == 1.0 else ""
            verdict = ("SHIP(hard)" if (md > 0 and t >= 2.5) else
                       "marginal" if (md > 0 and t >= 2.0) else
                       "weak" if (md > 0 and t >= 1.5) else "noise")
            print(f"  w_hgb={w:.1f}  mean={a.mean():.4f}  d={md:+.4f}  t={t:+.2f} "
                  f"-> {verdict}{tag}", flush=True)
        sanity.append(deltas[1.0][-1])
        print(f"[seed {seed}] champion mean={base_aucs.mean():.4f}  "
              f"(sanity w=1.0 d={deltas[1.0][-1]:+.5f})  ({time.time()-t0:.0f}s)", flush=True)
        del caches, hgb_series, rank_series, Xtr_c, ytr_c, rd, ld, td

    print("\n=== SUMMARY (mean over pool seeds, vs tuned champion) ===", flush=True)
    print(f"{'w_hgb':>6} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all", flush=True)
    best = None
    for w in WEIGHTS:
        d = np.asarray(deltas[w]); t = np.asarray(tstats[w])
        pos = "YES" if (d > 0).all() else " no"
        flag = ""
        if w != 1.0 and (d > 0).all() and d.mean() >= 0.0015 and t.mean() >= 2.5:
            flag = "  *** CONFIRM ***"
        print(f"{w:>6.1f} {d.mean():+8.4f} {d.min():+8.4f} {t.mean():+7.2f}   {pos}{flag}",
              flush=True)
        if w != 1.0 and (best is None or d.mean() > best[1]):
            best = (w, float(d.mean()))
    s = np.asarray(sanity)
    print(f"\nsanity w=1.0 |d| max = {np.abs(s).max():.2e} (should be ~0: global rank is "
          f"within-step monotone)", flush=True)
    if best:
        print(f"best blend: w_hgb={best[0]:.1f} mean_d={best[1]:+.4f}", flush=True)


if __name__ == "__main__":
    main()
