"""Tune the random-forest 3rd member of the CDF-blend ensemble (E39 follow-up -> E41 candidate).

E39 showed the DEFAULT random forest is sub-threshold on top of the ranker (+0.0014, t~2.0,
below the gate) because most of its value overlaps with the ranker. This harness asks whether a
TUNED random forest can clear the bar: it randomly searches random-forest hyperparameters,
adds the forest as a 3rd fixed-CDF member, and paired-t's the 3-member blend vs the shipped
2-member champion. Mirrors _rf_ens3's faithful percentile blend exactly.

The gradient-boosting member and the (default) ranker member are fit ONCE per seed and shared
across all forest configs; only the forest is re-fit. Each config is scored at a few 3-member
weight schemes (near-equal, which E39 found best); the best is reported.

Gate: positive on ALL pool seeds AND mean_d >= 0.0015 AND mean_t >= 2.5 (at its best weights)
-> fresh-seed confirm, then build a 3-member notebook.

Usage (one 15-job harness at a time; run AFTER _ranker_tune finishes):
  SMOKE=1 .venv/bin/python -u _forest_tune.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _forest_tune.py > _forest_tune.log 2>&1 & disown
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
FOREST_JOBS = 4 if SMOKE else 14
N_CONFIGS = 6 if SMOKE else 20

TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)

# 3-member weight schemes (gb, rank, forest); E39 found near-equal best.
WEIGHT_SETS = [(0.40, 0.30, 0.30), (0.36, 0.34, 0.30), (0.34, 0.33, 0.33), (0.34, 0.30, 0.36)]


def sample_forest(rng, seed):
    n_estimators = 80 if SMOKE else int(rng.choice([300, 500, 800]))
    return dict(
        n_estimators=n_estimators,
        max_depth=(None if (d := rng.choice(["none", "12", "20", "30"])) == "none" else int(d)),
        max_features=str(rng.choice(["sqrt", "log2"])) if rng.random() < 0.5
                     else float(rng.choice([0.3, 0.5])),
        min_samples_leaf=int(rng.choice([1, 3, 5, 10])),
        min_samples_split=int(rng.choice([2, 5, 10])),
        max_samples=(None if (m := rng.choice(["none", "0.6", "0.8"])) == "none" else float(m)),
        class_weight=(None if rng.random() < 0.5 else "balanced_subsample"),
        n_jobs=FOREST_JOBS, random_state=seed,
    )


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def short(p):
    mf = p["max_features"]
    return (f"n={p['n_estimators']} depth={p['max_depth']} mf={mf} "
            f"msl={p['min_samples_leaf']} mss={p['min_samples_split']} "
            f"maxs={p['max_samples']} cw={'bal' if p['class_weight'] else 'none'}")


def _key(w):
    return f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"


def main():
    print(f"=== random-forest 3rd-member tuning {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS} dense_k={DENSE_K} N_CONFIGS={N_CONFIGS}\n"
          f"base = shipped 2-member champion (0.6/0.4); gate pos_all & mean_d>=0.0015 & mean_t>=2.5",
          flush=True)

    rng_cfg = np.random.default_rng(20260814)
    # sample forest param templates once (random_state filled per seed)
    templates = [sample_forest(rng_cfg, 0) for _ in range(N_CONFIGS)]
    store = {i: {"d": [], "t": [], "w": None} for i in range(N_CONFIGS)}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
        idx_folds = [c["index"] for c in caches]

        # member 1: tuned gradient boosting (shared)
        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)
        gb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            gb.fit(Xtr_c, ytr_c)
        ref_gb = np.sort(gb.predict_proba(Xtr_c)[:, 1])
        fh_folds = [pct(score_fold(gb, c).to_numpy(), ref_gb) for c in caches]

        # member 2: default ranker (shared)
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr, gr)
        ref_rk = np.sort(ranker.r.predict(rr))
        fr_folds = [pct(score_fold(ranker, c).to_numpy(), ref_rk) for c in caches]
        del X

        # baseline = shipped 2-member blend 0.6/0.4
        base_aucs = np.asarray([
            ts_auc_from(pd.Series(0.6 * fh + 0.4 * fr, index=ix).to_frame("prediction"), y_val)
            for fh, fr, ix in zip(fh_folds, fr_folds, idx_folds)])

        print(f"\n[seed {seed}] 2-member base mean={base_aucs.mean():.4f} "
              f"train={Xtr_c.shape} ({time.time()-t0:.0f}s prep)", flush=True)

        for i, tmpl in enumerate(templates):
            tc = time.time()
            params = {**tmpl, "random_state": seed}
            forest = RandomForestClassifier(**params)
            forest.fit(Xtr_c, ytr_c)
            ref_f = np.sort(forest.predict_proba(Xtr_c)[:, 1])
            ff_folds = [pct(score_fold(forest, c).to_numpy(), ref_f) for c in caches]

            best = None
            for w in WEIGHT_SETS:
                aucs = np.asarray([
                    ts_auc_from(pd.Series(w[0]*fh + w[1]*fr + w[2]*ff, index=ix)
                                .to_frame("prediction"), y_val)
                    for fh, fr, ff, ix in zip(fh_folds, fr_folds, ff_folds, idx_folds)])
                md, _s, t, _ = paired(aucs, base_aucs)
                if best is None or md > best[0]:
                    best = (md, t, w)
            md, t, w = best
            store[i]["d"].append(md); store[i]["t"].append(t); store[i]["w"] = w
            print(f"  cfg{i:<2} [{short(params)}]  best {_key(w)} d={md:+.4f} t={t:+.2f} "
                  f"({time.time()-tc:.0f}s)", flush=True)
        del caches, Xtr_c, ytr_c, rr, lr, gr, rd, ld, td

    print("\n=== SUMMARY (best weights per forest config, vs shipped 2-member champion) ===",
          flush=True)
    print(f"{'cfg':<5} {'weights':<16} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all  verdict  "
          f"params", flush=True)
    rows_out = []
    for i, tmpl in enumerate(templates):
        d = np.asarray(store[i]["d"]); t = np.asarray(store[i]["t"])
        pos = (d > 0).all()
        confirm = pos and d.mean() >= 0.0015 and t.mean() >= 2.5
        verdict = "CONFIRM" if confirm else ("promising" if (pos and d.mean() > 0) else "noise")
        rows_out.append((d.mean(), i, store[i]["w"], d, t, pos, verdict, {**tmpl, "random_state": 0}))
    rows_out.sort(key=lambda r: -r[0])
    for md, i, w, d, t, pos, verdict, params in rows_out:
        print(f"cfg{i:<2} {_key(w):<16} {md:>+8.4f} {d.min():>+8.4f} {t.mean():>+7.2f}   "
              f"{'YES' if pos else ' no':>3}    {verdict:<9} {short(params)}", flush=True)
    print("\nA config positive on ALL seeds with mean_d>=0.0015 & mean_t>=2.5 beats the shipped "
          "champion -> fresh-seed confirm, then build a 3-member notebook. Else the random forest "
          "is confirmed redundant with the ranker even after tuning.", flush=True)


if __name__ == "__main__":
    main()
