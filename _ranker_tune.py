"""Tune the LightGBM ranker member of the shipped ensemble (E36 -> E40 candidate).

The shipped champion (leaderboard 58.78%) = tuned gradient-boosting member + a LightGBM
`rank_xendcg` ranker, blended by fixed training-score percentiles: emit
    0.6 * F_gb(proba) + 0.4 * F_rank(score).
The ranker was NEVER tuned (default LightGBM params) yet it is the source of the whole +1.06pt
jump, so it has the most headroom. This harness randomly searches ranker hyperparameters,
evaluates the faithful 2-member percentile blend (same computation as _confirm_rank_ens), and
paired-t's each candidate vs the shipped champion (default ranker at weight 0.6). Each candidate
is scored at BOTH blend weights 0.5 and 0.6 (this also settles the weight follow-up).

Fixed to isolate the parameter effect: dense_k = 60 steps/series (the confirmed value), the
tuned gradient-boosting member, the 45 champion features, the same stratified folds.

Gate: positive on ALL pool seeds AND mean_d >= 0.0015 AND mean_t >= 2.5 (at its best weight)
-> fresh-seed confirm, then rebuild the ensemble notebook with the tuned ranker.

Usage (one 15-job harness at a time):
  SMOKE=1 .venv/bin/python -u _ranker_tune.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _ranker_tune.py > _ranker_tune.log 2>&1 & disown
  SEEDS=31337,2025,555,99 nohup ...   (fresh-seed confirm of the winner)
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

import lightgbm as lgb

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, load_pool_strat, make_hgb,
    precompute_fold, score_fold, select_picks, WINDOW,
)
from _exp_fast import ts_auc_from
from _lambdarank import build_grouped, sort_into_groups, RankerProba
from _overnight import build_training, paired, N_JOBS

SMOKE = os.environ.get("SMOKE") == "1"
_SEEDS_ENV = os.environ.get("SEEDS", "")
if _SEEDS_ENV:
    POOL_SEEDS = [int(s) for s in _SEEDS_ENV.split(",") if s.strip()]
elif SMOKE:
    POOL_SEEDS = [42]
else:
    POOL_SEEDS = [42, 7, 123, 2024]

DENSE_K = 30 if SMOKE else 60          # steps/series (fixed at the confirmed value)
N_CONFIGS = 6 if SMOKE else 24
WEIGHTS = [0.5, 0.6]                    # blend weight on the gradient-boosting member
RANK_SEED = 42

# tuned gradient-boosting member (the shipped pointwise member).
TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)

# shipped default ranker params (baseline == champion).
DEFAULT_RANKER = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
    reg_lambda=1.0, subsample=1.0, subsample_freq=0, colsample_bytree=0.7,
)


def sample_config(rng):
    subsample = float(rng.choice([0.7, 0.85, 1.0]))
    return dict(
        n_estimators=int(rng.choice([200, 300, 400, 600, 800])),
        learning_rate=float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        num_leaves=int(rng.choice([15, 31, 47, 63])),
        min_child_samples=int(rng.choice([10, 20, 40, 80])),
        reg_lambda=float(rng.choice([0.0, 1.0, 5.0])),
        colsample_bytree=float(rng.choice([0.5, 0.7, 1.0])),
        subsample=subsample,
        subsample_freq=(1 if subsample < 1.0 else 0),
    )


def fit_ranker(params, rows, labels, groups):
    r = lgb.LGBMRanker(objective="rank_xendcg", random_state=RANK_SEED,
                       n_jobs=N_JOBS, verbose=-1, **params)
    with threadpool_limits(limits=1):
        r.fit(rows, labels, group=groups)
    return RankerProba(r)


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def short(params):
    return (f"it={params['n_estimators']} lr={params['learning_rate']:.2f} "
            f"lv={params['num_leaves']} mcs={params['min_child_samples']} "
            f"l2={params['reg_lambda']:.0f} cs={params['colsample_bytree']:.1f} "
            f"ss={params['subsample']:.2f}")


def main():
    print(f"=== ranker tuning {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS} dense_k={DENSE_K} N_CONFIGS={N_CONFIGS} weights={WEIGHTS}\n"
          f"base = shipped champion (default ranker, weight 0.6); gate pos_all & mean_d>=0.0015 "
          f"& mean_t>=2.5", flush=True)

    rng_cfg = np.random.default_rng(20260814)
    configs = [("default", DEFAULT_RANKER)]
    seen = {tuple(sorted(DEFAULT_RANKER.items()))}
    while len(configs) < N_CONFIGS:
        p = sample_config(rng_cfg)
        key = tuple(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append((f"cfg{len(configs)}", p))

    # per (config,weight) accumulators over seeds
    store = {(name, w): {"d": [], "t": []} for name, _ in configs for w in WEIGHTS}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

        # tuned gradient-boosting member (shared): fold percentiles are fixed across configs
        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)
        gb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            gb.fit(Xtr_c, ytr_c)
        ref_gb = np.sort(gb.predict_proba(Xtr_c)[:, 1])
        fh_folds = [pct(score_fold(gb, c).to_numpy(), ref_gb) for c in caches]
        idx_folds = [c["index"] for c in caches]

        # grouped ranker training data (shared across configs)
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        del X

        # baseline = shipped champion (default ranker at weight 0.6)
        base_ranker = fit_ranker(DEFAULT_RANKER, rr, lr, gr)
        ref_b = np.sort(base_ranker.r.predict(rr))
        base_aucs = None
        base_fr = [pct(score_fold(base_ranker, c).to_numpy(), ref_b) for c in caches]
        base_list = []
        for fh, fr, ix in zip(fh_folds, base_fr, idx_folds):
            b = pd.Series(0.6 * fh + 0.4 * fr, index=ix)
            base_list.append(ts_auc_from(b.to_frame("prediction"), y_val))
        base_aucs = np.asarray(base_list)

        print(f"\n[seed {seed}] champion(base) mean={base_aucs.mean():.4f} "
              f"train={Xtr_c.shape} rank_rows={rr.shape[0]} ({time.time()-t0:.0f}s prep)",
              flush=True)

        for name, params in configs:
            tc = time.time()
            ranker = fit_ranker(params, rr, lr, gr)
            ref_r = np.sort(ranker.r.predict(rr))
            fr_folds = [pct(score_fold(ranker, c).to_numpy(), ref_r) for c in caches]
            line = []
            for w in WEIGHTS:
                aucs = []
                for fh, fr, ix in zip(fh_folds, fr_folds, idx_folds):
                    b = pd.Series(w * fh + (1.0 - w) * fr, index=ix)
                    aucs.append(ts_auc_from(b.to_frame("prediction"), y_val))
                aucs = np.asarray(aucs)
                md, _s, t, _ = paired(aucs, base_aucs)
                store[(name, w)]["d"].append(md)
                store[(name, w)]["t"].append(t)
                line.append(f"w{w}: d={md:+.4f} t={t:+.2f}")
            print(f"  {name:<7} [{short(params)}]  " + "  ".join(line)
                  + f"  ({time.time()-tc:.0f}s)", flush=True)
        del caches, Xtr_c, ytr_c, rr, lr, gr, rd, ld, td

    # ---- summary: best weight per config, ranked by mean_d ----
    print("\n=== SUMMARY (best weight per config, vs shipped champion) ===", flush=True)
    print(f"{'config':<7} {'w*':>4} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all  verdict  "
          f"params", flush=True)
    rows_out = []
    for name, params in configs:
        best = None
        for w in WEIGHTS:
            d = np.asarray(store[(name, w)]["d"]); t = np.asarray(store[(name, w)]["t"])
            cand = (d.mean(), w, d, t)
            if best is None or cand[0] > best[0]:
                best = cand
        md, w, d, t = best
        pos = (d > 0).all()
        confirm = pos and md >= 0.0015 and t.mean() >= 2.5
        verdict = "CONFIRM" if confirm else ("promising" if (pos and md > 0) else "noise")
        rows_out.append((md, name, w, d, t, pos, verdict, params))
    rows_out.sort(key=lambda r: -r[0])
    for md, name, w, d, t, pos, verdict, params in rows_out:
        print(f"{name:<7} {w:>4.1f} {md:>+8.4f} {d.min():>+8.4f} {t.mean():>+7.2f}   "
              f"{'YES' if pos else ' no':>3}    {verdict:<9} {short(params)}", flush=True)
    print("\nA config positive on ALL seeds with mean_d>=0.0015 & mean_t>=2.5 beats the shipped "
          "champion -> fresh-seed confirm (SEEDS=31337,2025,555,99), then rebuild the notebook.",
          flush=True)


if __name__ == "__main__":
    main()
