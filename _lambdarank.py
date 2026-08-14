"""LambdaMART / rank-objective experiment.

Motivation
----------
Every model we have ever trained (HGB, LR, LGBM-swap E29) optimises a POINTWISE loss
(log-loss / BCE). But the leaderboard metric TS-AUC is a per-`time_online` CROSS-SECTIONAL
RANKING score: at each online step it computes a weighted rank-AUC over the series present
at that step. That is a train/metric mismatch. This harness trains LightGBM with a LISTWISE
rank objective (`lambdarank`) whose query group == a single `time_online` step, so the model
optimises exactly the within-step ordering the metric rewards.

Controls (to attribute any gain correctly):
  * champion HGB (pointwise, the incumbent)             -> baseline
  * LGBM pointwise (binary)                             -> isolates the TOOLCHAIN (E29 said noise)
  * LGBM lambdarank @ 10 steps/series                   -> the OBJECTIVE change
  * LGBM lambdarank @ dense steps/series                -> objective + the "more-data" lever
  * LGBM rank_xentropy @ dense                          -> alt listwise objective

If a rank objective clears the hardened bar (positive on ALL pool seeds, mean delta >= 0.0015,
ideally t>=2.5) it is a STRUCTURAL win a feature nibble never could be. Scored apples-to-apples
on the SAME stratified folds / feature cache as every other E-series experiment.

Group key: in training, a sampled step `t` is the online index; TS-AUC's `time_online` is the
online cumcount == that same index -> group by `t`.

Usage (one 15-job harness at a time):
  SMOKE=1 .venv/bin/python -u _lambdarank.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _lambdarank.py > _lambdarank.log 2>&1 & disown
"""
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from threadpoolctl import threadpool_limits

import lightgbm as lgb

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, eval_model, load_pool_strat, make_hgb,
    precompute_fold, score_fold, WINDOW,
)
from _exp_fast import HIST_CAP, hist_context, split_series, ts_auc_from
from _overnight import MIN_N, N_JOBS, paired

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    POOL_SEEDS = [42]
    DENSE_K = 30
else:
    POOL_SEEDS = [42, 7, 123, 2024]
    DENSE_K = 60          # steps/series for the "dense" (more-data) variants

SPARSE_K = 10             # matches the champion training density
RANK_SEED = 42            # fixed model seed so only data varies across pool seeds


# --- training-row builder that also records the group key (online step t) ---
def build_grouped(X, yi, train_ids, feat_fn, window, k, rng):
    """Return (rows, labels, tvals) sampling k online steps/series (recording the step)."""
    rows, labels, tvals = [], [], []
    for sid in train_ids:
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        L = len(x_on)
        if L <= MIN_N:
            continue
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        cand = np.arange(MIN_N, L)
        kk = min(k, len(cand))
        picks = rng.choice(cand, size=kk, replace=False)
        for t in picks:
            lo = max(0, t - window + 1)
            rows.append(feat_fn(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
            tvals.append(int(t))
    return (np.asarray(rows, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
            np.asarray(tvals, dtype=np.int64))


def sort_into_groups(rows, labels, tvals):
    """Sort rows so each query group (== one online step t) is contiguous; return group sizes."""
    order = np.argsort(tvals, kind="stable")
    rows, labels, tvals = rows[order], labels[order], tvals[order]
    _, counts = np.unique(tvals, return_counts=True)
    return rows, labels, counts


class RankerProba:
    """Adapt an LGBMRanker (raw scores) to the predict_proba(...)[:,1] interface used by
    score_fold. TS-AUC only uses the within-step ordering, so a monotone raw score suffices."""
    def __init__(self, ranker):
        self.r = ranker

    def predict_proba(self, Xf):
        s = self.r.predict(Xf)
        return np.column_stack([-s, s])


LGB_COMMON = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, reg_lambda=1.0, subsample=1.0,
    colsample_bytree=0.7, random_state=RANK_SEED, n_jobs=N_JOBS, verbose=-1,
)


def fit_ranker(objective, rows, labels, groups):
    r = lgb.LGBMRanker(objective=objective, **LGB_COMMON)
    with threadpool_limits(limits=1):
        r.fit(rows, labels, group=groups)
    return RankerProba(r)


def fit_clf(rows, labels):
    c = lgb.LGBMClassifier(objective="binary", **LGB_COMMON)
    with threadpool_limits(limits=1):
        c.fit(rows, labels)
    return c


def score_all(model, caches, y_val):
    series = [score_fold(model, c) for c in caches]
    return np.asarray([ts_auc_from(s.to_frame("prediction"), y_val) for s in series])


def main():
    print(f"=== lambdarank experiment {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS}  sparse_k={SPARSE_K} dense_k={DENSE_K}  "
          f"ship: pos on all seeds & mean>=0.0015 (hard t>=2.5)", flush=True)

    members = ["lgbm_pointwise", "rank_sparse10", "rank_dense", "rankxent_dense"]
    deltas = {m: [] for m in members}
    tstats = {m: [] for m in members}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

        # champion HGB baseline (pointwise, 10 steps/series via ET.select_picks + build_training)
        from _ensemble_tune import select_picks
        from _overnight import build_training as bt_champ
        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = bt_champ(X, yi, picks, champion_feat, WINDOW)
        base_aucs, _ = eval_model(make_hgb(), Xtr_c, ytr_c, caches, y_val)

        rng = np.random.default_rng(seed + 101)
        rs, ls, ts_ = build_grouped(X, yi, train_ids, champion_feat, WINDOW, SPARSE_K, rng)
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        del X

        # LGBM pointwise control (dense rows, binary) -> isolates toolchain
        m = fit_clf(rd, ld)
        a = score_all(m, caches, y_val)
        _log("lgbm_pointwise", a, base_aucs, deltas, tstats)

        # rank objective @ sparse (10 steps, matched density)
        rr, lr, gr = sort_into_groups(rs, ls, ts_)
        m = fit_ranker("lambdarank", rr, lr, gr)
        a = score_all(m, caches, y_val)
        _log("rank_sparse10", a, base_aucs, deltas, tstats)

        # rank objective @ dense (the more-data lever)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        m = fit_ranker("lambdarank", rr, lr, gr)
        a = score_all(m, caches, y_val)
        _log("rank_dense", a, base_aucs, deltas, tstats)

        # alt listwise objective @ dense
        m = fit_ranker("rank_xendcg", rr, lr, gr)
        a = score_all(m, caches, y_val)
        _log("rankxent_dense", a, base_aucs, deltas, tstats)

        print(f"[seed {seed}] champion mean={base_aucs.mean():.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        del caches, Xtr_c, ytr_c, rs, ls, ts_, rd, ld, td

    print("\n=== SUMMARY (mean over pool seeds) ===", flush=True)
    print(f"{'member':<18} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all", flush=True)
    for m in members:
        d = np.asarray(deltas[m]); t = np.asarray(tstats[m])
        pos = "YES" if (d > 0).all() else " no"
        print(f"{m:<18} {d.mean():+8.4f} {d.min():+8.4f} {t.mean():+7.2f}   {pos}", flush=True)
    print("\nRank objectives that are positive on ALL seeds with mean_d>=0.0015 are shippable "
          "(build a lambdarank notebook + confirm on fresh seeds before an LB slot).", flush=True)


def _log(name, aucs, base, deltas, tstats):
    md, sd, t, _ = paired(aucs, base)
    deltas[name].append(md)
    tstats[name].append(t)
    verdict = "SHIP(hard)" if (md > 0 and t >= 2.5) else (
        "marginal" if (md > 0 and t >= 2.0) else ("weak" if (md > 0 and t >= 1.5) else "noise"))
    print(f"  {name:<18} mean={aucs.mean():.4f} delta={md:+.4f} t={t:+.2f} -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
