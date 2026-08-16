"""Steps-per-series sweep for the tabular gradient-boosting member (utilize more training data).

The champion's gradient-boosting member trains on only STEPS_PER_SERIES=10 randomly sampled online
snapshots per training series (~80k rows over 8000 series) -- a tiny slice of the ~4.9M available
online steps. The ranker already uses dense-60 groups; the GRU eats the full trajectory. This sweep
asks the direct question for the DEPLOYED tabular member: does sampling MORE steps/series lift the
shipped 2-member CDF-blend, or does the extra (temporally-correlated) supervision just add noise?

(E7 earlier found dense sampling HURT a stand-alone HGB via correlated rows; this re-tests it
properly IN the shipped blend, vs the CURRENT champion, across seeds.)

Mechanics: STEPS_PER_SERIES is a module global read by _ensemble_tune.select_picks, so we monkeypatch
ET.STEPS_PER_SERIES per config, rebuild picks + training rows, refit the TUNED gradient-boosting, and
re-blend against the SAME fixed rank_xendcg member (ranker + val caches computed ONCE per seed, since
neither depends on the gb sampling density). Fixed-CDF blend at the shipped weight 0.6/0.4.

Baseline = STEPS=10 (the shipped champion). paired-t each STEPS vs that baseline.
Gate: positive on ALL seeds AND mean_d>=0.0015 AND mean_t>=2.5
  -> fresh-seed confirm on [31337,2025,555,99], then bump STEPS_PER_SERIES in the notebook.

Usage (ONE 15-job harness at a time):
  SMOKE=1 .venv/bin/python -u _steps_sweep.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _steps_sweep.py > _steps_sweep.log 2>&1 & disown
  SEEDS=31337,2025,555,99 nohup ... (fresh-seed confirm)
  STEPS=10,20,30,50 to override the grid
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
    POOL_SEEDS = [int(s) for s in _SEEDS_ENV.split(",") if s.strip()]
    DENSE_K = 60
elif SMOKE:
    POOL_SEEDS = [42]
    DENSE_K = 30
else:
    POOL_SEEDS = [int(s) for s in os.environ.get("POOL_SEEDS", "42,7,123,2024").split(",")]
    DENSE_K = 60

_STEPS_ENV = os.environ.get("STEPS", "")
if _STEPS_ENV:
    STEPS_GRID = [int(s) for s in _STEPS_ENV.split(",") if s.strip()]
elif SMOKE:
    STEPS_GRID = [10, 20, 40]
else:
    STEPS_GRID = [10, 15, 20, 30, 50]
BASE_STEPS = 10                      # shipped champion sampling density
if BASE_STEPS not in STEPS_GRID:
    STEPS_GRID = [BASE_STEPS] + STEPS_GRID

BLEND_W = 0.6                        # shipped fixed-CDF weight on gb (0.4 on ranker)
TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)


def pct(x, ref):
    """Fixed-reference percentile: fraction of ref <= x. ref must be pre-sorted."""
    return np.searchsorted(ref, x, side="right") / len(ref)


def main():
    print(f"=== steps-per-series sweep (gb member) {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS} dense_k={DENSE_K} grid={STEPS_GRID} base_steps={BASE_STEPS} "
          f"blend_w={BLEND_W}  gate: pos_all & mean_d>=0.0015 & mean_t>=2.5", flush=True)

    cand = [s for s in STEPS_GRID if s != BASE_STEPS]
    d_store = {s: [] for s in cand}
    t_store = {s: [] for s in cand}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

        # member 2: rank_xendcg dense ranker — independent of gb sampling density, fit ONCE.
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr, gr)
        ref_rank = np.sort(ranker.r.predict(rr))
        rank_series = [score_fold(ranker, c) for c in caches]
        fr_folds = [pct(rs.to_numpy(), ref_rank) for rs in rank_series]

        print(f"\n[seed {seed}] ranker + caches ready ({time.time()-t0:.0f}s)", flush=True)
        per = {}
        for steps in STEPS_GRID:
            ts = time.time()
            ET.STEPS_PER_SERIES = steps
            picks = select_picks(X, train_ids)
            Xtr, ytr = build_training(X, yi, picks, champion_feat, WINDOW)
            model_gb = make_hgb(**TUNED, random_state=seed)
            with threadpool_limits(limits=1):
                model_gb.fit(Xtr, ytr)
            ref_gb = np.sort(model_gb.predict_proba(Xtr)[:, 1])
            aucs = []
            for c, fr in zip(caches, fr_folds):
                fh = pct(score_fold(model_gb, c).to_numpy(), ref_gb)
                b = pd.Series(BLEND_W * fh + (1 - BLEND_W) * fr, index=c["index"])
                aucs.append(ts_auc_from(b.to_frame("prediction"), y_val))
            per[steps] = np.asarray(aucs)
            print(f"  steps={steps:<3} rows={Xtr.shape[0]:>7} mean={per[steps].mean():.4f} "
                  f"({time.time()-ts:.0f}s)", flush=True)
        del X

        base_aucs = per[BASE_STEPS]
        for steps in cand:
            md, _s, t, _ = paired(per[steps], base_aucs)
            d_store[steps].append(md); t_store[steps].append(t)
            v = "SHIP(hard)" if (md > 0 and t >= 2.5) else (
                "marginal" if (md > 0 and t >= 2.0) else ("weak" if (md > 0 and t >= 1.5) else "noise"))
            print(f"  steps={steps:<3} d={md:+.4f} t={t:+.2f} -> {v}", flush=True)
        del caches, rank_series, rd, ld, td

    print("\n=== SUMMARY (mean over seeds, vs shipped STEPS=10 champion) ===", flush=True)
    print(f"{'steps':<7} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all  verdict", flush=True)
    for steps in cand:
        d = np.asarray(d_store[steps]); t = np.asarray(t_store[steps])
        pos = (d > 0).all()
        confirm = pos and d.mean() >= 0.0015 and t.mean() >= 2.5
        v = "CONFIRM" if confirm else ("promising" if (pos and d.mean() > 0) else "noise")
        print(f"{steps:<7} {d.mean():+8.4f} {d.min():+8.4f} {t.mean():+7.2f}   "
              f"{'YES' if pos else ' no'}     {v}", flush=True)
    print("\nA steps value positive on ALL seeds with mean_d>=0.0015 & mean_t>=2.5 BEATS the shipped "
          "champion -> fresh-seed confirm, then bump STEPS_PER_SERIES in the notebook.", flush=True)


if __name__ == "__main__":
    main()
