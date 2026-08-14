"""Fine 2-member blend-weight sweep (ranker-tuning conclusion -> ship candidate).

The ranker hyperparameter search (_ranker_tune.py) found NO parameter config beats the shipped
champion -- the ranker is already near-optimal at its defaults, and aggressive params overfit.
The one robust signal was the BLEND WEIGHT: the default ranker at 0.5 beat the shipped 0.6 on all
4 pool seeds (mean_d +0.0014, t +2.31), and every "promising" config picked 0.5. That says the
ranker deserves MORE weight than 0.6.

This harness pins the optimum: it fits the two shipped members (tuned gradient-boosting + default
LightGBM rank_xendcg ranker) ONCE per seed, then re-blends their fixed train-CDF percentiles at a
fine grid of weights (no refitting -- the sweep is essentially free). Baseline = shipped weight
0.60. Paired-t each weight vs 0.60 across ALL 8 seeds (4 pool + 4 fresh) for a definitive read.

  emit  w * F_gb(proba) + (1-w) * F_rank(score)   ; larger w = more gradient-boosting.

Usage (one 15-job harness at a time):
  SMOKE=1 .venv/bin/python -u _weight_sweep.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _weight_sweep.py > _weight_sweep.log 2>&1 & disown
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
    champion_feat, load_pool_strat, make_hgb,
    precompute_fold, score_fold, select_picks, WINDOW,
)
from _exp_fast import ts_auc_from
from _lambdarank import build_grouped, sort_into_groups, fit_ranker
from _overnight import build_training, paired

SMOKE = os.environ.get("SMOKE") == "1"
_SEEDS_ENV = os.environ.get("SEEDS", "")
if _SEEDS_ENV:
    SEEDS = [int(s) for s in _SEEDS_ENV.split(",") if s.strip()]
    DENSE_K = 60
elif SMOKE:
    SEEDS = [42]
    DENSE_K = 30
else:
    SEEDS = [42, 7, 123, 2024, 31337, 2025, 555, 99]
    DENSE_K = 60

TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)

BASE_W = 0.60  # shipped
WEIGHTS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def main():
    print(f"=== 2-member blend-weight sweep {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"SEEDS={SEEDS} dense_k={DENSE_K}  base=shipped w={BASE_W}  weights={WEIGHTS}\n"
          f"w = gradient-boosting weight; (1-w) = ranker weight. paired-t vs w={BASE_W}.", flush=True)

    store = {w: [] for w in WEIGHTS}  # per-weight list of per-seed fold-auc arrays

    for seed in SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
        idx_folds = [c["index"] for c in caches]

        # member 1: tuned gradient boosting
        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)
        gb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            gb.fit(Xtr_c, ytr_c)
        ref_gb = np.sort(gb.predict_proba(Xtr_c)[:, 1])
        fh_folds = [pct(score_fold(gb, c).to_numpy(), ref_gb) for c in caches]

        # member 2: default rank_xendcg ranker (shipped)
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr, gr)
        ref_rk = np.sort(ranker.r.predict(rr))
        fr_folds = [pct(score_fold(ranker, c).to_numpy(), ref_rk) for c in caches]
        del X

        line = [f"[seed {seed}] ({time.time()-t0:.0f}s)"]
        base_aucs = None
        seed_aucs = {}
        for w in WEIGHTS:
            aucs = np.asarray([
                ts_auc_from(pd.Series(w * fh + (1 - w) * fr, index=ix).to_frame("prediction"), y_val)
                for fh, fr, ix in zip(fh_folds, fr_folds, idx_folds)])
            seed_aucs[w] = aucs
            store[w].append(aucs)
            if abs(w - BASE_W) < 1e-9:
                base_aucs = aucs
        for w in WEIGHTS:
            md, _s, t, _ = paired(seed_aucs[w], base_aucs)
            line.append(f"w{w:.2f}:{seed_aucs[w].mean():.4f}(d{md:+.4f})")
        print("  ".join(line), flush=True)
        del caches, Xtr_c, ytr_c, rr, lr, gr, rd, ld, td

    print(f"\n=== SUMMARY (vs shipped w={BASE_W}, over {len(SEEDS)} seeds) ===", flush=True)
    print(f"{'w':>5} {'mean_auc':>9} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all", flush=True)
    # paired-t per seed then aggregate across seeds
    base_stack = np.concatenate(store[BASE_W])
    for w in WEIGHTS:
        per_seed_d, per_seed_t = [], []
        for aucs, base in zip(store[w], store[BASE_W]):
            md, _s, t, _ = paired(aucs, base)
            per_seed_d.append(md); per_seed_t.append(t)
        d = np.asarray(per_seed_d); t = np.asarray(per_seed_t)
        pos = (d > 0).all()
        mean_auc = np.concatenate(store[w]).mean()
        star = "  <-- shipped" if abs(w - BASE_W) < 1e-9 else (
            "  *BEST*" if (pos and d.mean() > 0) else "")
        print(f"{w:>5.2f} {mean_auc:>9.4f} {d.mean():>+8.4f} {d.min():>+8.4f} {t.mean():>+7.2f}   "
              f"{'YES' if pos else ' no'}{star}", flush=True)
    print("\nPick the weight with the best mean_d that is positive on ALL seeds with a solid t; "
          "if it beats w=0.60 robustly, ship it as a one-line BLEND_W change in the notebook.",
          flush=True)


if __name__ == "__main__":
    main()
