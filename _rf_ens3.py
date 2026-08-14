"""RF as a 3rd member of the SHIPPED CDF-blend ensemble (E36 -> E39 candidate).

The shipped champion (LB 58.78%) = tuned HGB + LGBM rank_xendcg, FIXED train-CDF blend
  emit  0.6*F_hgb(proba) + 0.4*F_rank(score).
RF is confirmed 8/8 seeds as a decorrelated member vs HGB-ALONE, but that used raw proba-avg.
The real ship question: does RF add ON TOP of the ranker in the faithful percentile blend?

This mirrors _confirm_rank_ens.py EXACTLY (same fixed-CDF `pct`, same ranker plumbing) and adds
a 3rd fixed-CDF reference for RF. Baseline = the shipped 2-member blend (0.6/0.4/0.0). We sweep
3-member weight sets and paired-t each vs that baseline (NOT vs HGB-alone) across POOL_SEEDS.

Gate: positive vs the 2-member champion on ALL seeds AND mean_d>=0.0015 AND mean_t>=2.5
-> fresh-seed confirm, then build a 3-member notebook.

Usage (one 15-job harness at a time):
  SMOKE=1 .venv/bin/python -u _rf_ens3.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _rf_ens3.py > _rf_ens3.log 2>&1 & disown
  SEEDS=31337,2025,555,99 nohup ... (fresh-seed confirm)
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
from sklearn.ensemble import RandomForestClassifier

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, load_pool_strat, make_hgb,
    precompute_fold, score_fold, select_picks, WINDOW,
)
from _exp_fast import ts_auc_from
from _lambdarank import build_grouped, sort_into_groups, fit_ranker
from _overnight import build_training, paired

SMOKE = os.environ.get("SMOKE") == "1"
_SEEDS_ENV = os.environ.get("SEEDS", "")
if _SEEDS_ENV:
    POOL_SEEDS = [int(s) for s in _SEEDS_ENV.split(",") if s.strip()]
    DENSE_K = 60
elif SMOKE:
    POOL_SEEDS = [42]
    DENSE_K = 30
else:
    POOL_SEEDS = [42, 7, 123, 2024]
    DENSE_K = 60
RF_JOBS = 4 if SMOKE else 14

TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)

# (w_hgb, w_rank, w_rf); first row = shipped 2-member baseline (sanity ~0 vs itself).
BASE_W = (0.6, 0.4, 0.0)
WEIGHT_SETS = [
    BASE_W,
    (0.50, 0.33, 0.17),
    (0.48, 0.32, 0.20),
    (0.45, 0.30, 0.25),
    (0.42, 0.28, 0.30),
    (0.36, 0.34, 0.30),
    (0.34, 0.33, 0.33),
]


def rf_params(seed):
    return dict(n_estimators=(80 if SMOKE else 400), max_features="sqrt",
                min_samples_leaf=5, n_jobs=RF_JOBS, random_state=seed)


def pct(x, ref):
    """Fixed-reference percentile: fraction of ref <= x. ref must be pre-sorted."""
    return np.searchsorted(ref, x, side="right") / len(ref)


def _key(w):
    return f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"


def main():
    print(f"=== RF 3rd-member CDF ensemble {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS} dense_k={DENSE_K}  base=shipped 2-member {_key(BASE_W)}  "
          f"gate: pos_all & mean_d>=0.0015 & mean_t>=2.5", flush=True)

    cand = [w for w in WEIGHT_SETS if w != BASE_W]
    d_store = {_key(w): [] for w in cand}
    t_store = {_key(w): [] for w in cand}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)

        # member 1: tuned HGB
        model_hgb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            model_hgb.fit(Xtr_c, ytr_c)
        ref_hgb = np.sort(model_hgb.predict_proba(Xtr_c)[:, 1])
        hgb_series = [score_fold(model_hgb, c) for c in caches]

        # member 2: rank_xendcg dense ranker
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr, gr)
        ref_rank = np.sort(ranker.r.predict(rr))
        rank_series = [score_fold(ranker, c) for c in caches]

        # member 3: RandomForest
        rf = RandomForestClassifier(**rf_params(seed))
        rf.fit(Xtr_c, ytr_c)
        ref_rf = np.sort(rf.predict_proba(Xtr_c)[:, 1])
        rf_series = [score_fold(rf, c) for c in caches]
        del X

        # per-fold fixed-CDF percentiles, then blend under each weight set
        per = {_key(w): [] for w in WEIGHT_SETS}
        for hs, rs_, fs in zip(hgb_series, rank_series, rf_series):
            fh = pct(hs.to_numpy(), ref_hgb)
            fr = pct(rs_.to_numpy(), ref_rank)
            ff = pct(fs.to_numpy(), ref_rf)
            for w in WEIGHT_SETS:
                b = pd.Series(w[0] * fh + w[1] * fr + w[2] * ff, index=hs.index)
                per[_key(w)].append(ts_auc_from(b.to_frame("prediction"), y_val))
        base_aucs = np.asarray(per[_key(BASE_W)])

        print(f"\n[seed {seed}] 2-member base mean={base_aucs.mean():.4f} "
              f"train={Xtr_c.shape} ({time.time()-t0:.0f}s)", flush=True)
        for w in cand:
            a = np.asarray(per[_key(w)])
            md, _s, t, _ = paired(a, base_aucs)
            d_store[_key(w)].append(md); t_store[_key(w)].append(t)
            v = "SHIP(hard)" if (md > 0 and t >= 2.5) else (
                "marginal" if (md > 0 and t >= 2.0) else ("weak" if (md > 0 and t >= 1.5) else "noise"))
            print(f"  {_key(w):<16} mean={a.mean():.4f} d={md:+.4f} t={t:+.2f} -> {v}", flush=True)
        del caches, hgb_series, rank_series, rf_series, Xtr_c, ytr_c, rd, ld, td

    print("\n=== SUMMARY (mean over seeds, vs shipped 2-member blend) ===", flush=True)
    print(f"{'w_hgb/rank/rf':<16} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all  verdict", flush=True)
    for w in cand:
        d = np.asarray(d_store[_key(w)]); t = np.asarray(t_store[_key(w)])
        pos = (d > 0).all()
        confirm = pos and d.mean() >= 0.0015 and t.mean() >= 2.5
        v = "CONFIRM" if confirm else ("promising" if (pos and d.mean() > 0) else "noise")
        print(f"{_key(w):<16} {d.mean():+8.4f} {d.min():+8.4f} {t.mean():+7.2f}   "
              f"{'YES' if pos else ' no'}     {v}", flush=True)
    print("\nA weight set positive on ALL seeds with mean_d>=0.0015 & mean_t>=2.5 BEATS the shipped "
          "2-member champion -> fresh-seed confirm, then build the 3-member notebook.", flush=True)


if __name__ == "__main__":
    main()
