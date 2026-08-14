"""Diversity ensemble — beyond seed-averaging.

The seed-ensemble (K champions with different random_state) was the ONLY thing that moved
the LB this arc (+0.13pt). Seeds only perturb the tree-growth RNG. This harness tests whether
STRONGER decorrelation sources add more variance reduction on top of the tuned config:

  * seed_ens5        : 5 champions, random_state {42,7,123,2024,31337}      (reference)
  * scale_ens        : one HGB per window-scale VIEW (s60 | s130 | s250), averaged
                       -> genuinely different decision boundaries (different feature subspace)
  * scale_plus_full  : {full-45, s60, s130, s250} averaged
  * bag_series5      : 5 champions each trained on a 70% random subset of TRAIN SERIES
                       -> data-level decorrelation (bag over series, not rows)
  * tuned_seed_ens3  : 3 E32-TUNED models, rs {42,7,123}  -> STACK both proven wins
  * tuned_bag5       : 5 E32-TUNED models on 70% series subsets

Each ensemble = equal-weight average of member predict_proba, scored TS-AUC on the SAME
stratified folds, paired-t vs a SINGLE champion HGB, across POOL_SEEDS. Hardened bar t>=2.5.
The champion feature cache (45 cols) is reused; scale members just slice columns
(champion_feat concatenates scales in order: s60=0:15, s130=15:30, s250=30:45), so nothing
is recomputed.

Usage (one 15-job harness at a time):
  SMOKE=1 .venv/bin/python -u _diversity_ens.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _diversity_ens.py > _diversity_ens.log 2>&1 & disown
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
    champion_feat, eval_model, load_pool_strat, make_hgb,
    precompute_fold, select_picks, WINDOW,
)
from _exp_fast import HIST_CAP, hist_context, split_series, ts_auc_from
from _overnight import MIN_N, paired

SMOKE = os.environ.get("SMOKE") == "1"
POOL_SEEDS = [42] if SMOKE else [42, 7, 123, 2024]

# E32 robust tuned winner (confirmed positive on 8/8 independent seeds).
TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)

SCALE_COLS = {"s60": slice(0, 15), "s130": slice(15, 30), "s250": slice(30, 45)}
ENS_SEEDS = [42, 7, 123, 2024, 31337]


def build_rows(X, yi, train_ids, feat_fn, window):
    """Champion 10-step training rows + the source series id per row (for series-bagging)."""
    picks = select_picks(X, train_ids)
    rows, labels, sids = [], [], []
    for sid, ts in picks.items():
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        for t in ts:
            lo = max(0, t - window + 1)
            rows.append(feat_fn(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
            sids.append(sid)
    return (np.asarray(rows, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
            np.asarray(sids))


def score_fold_cols(model, cache, cols):
    full = np.zeros(len(cache["index"]))
    if len(cache["feat"]):
        feat = cache["feat"] if cols is None else cache["feat"][:, cols]
        full[cache["active"]] = model.predict_proba(feat)[:, 1]
    return pd.Series(full, index=cache["index"])


def fit_one(params, Xtr, ytr, cols=None):
    m = make_hgb(**params)
    with threadpool_limits(limits=1):
        m.fit(Xtr if cols is None else Xtr[:, cols], ytr)
    return m


def ensemble_aucs(members, caches, y_val):
    """members = list of (fitted_model, cols). Equal-weight proba average, per-fold TS-AUC."""
    out = []
    for c in caches:
        acc = None
        for model, cols in members:
            s = score_fold_cols(model, c, cols)
            acc = s if acc is None else acc + s
        acc = acc / len(members)
        out.append(ts_auc_from(acc.to_frame("prediction"), y_val))
    return np.asarray(out)


def build_seed_ens(params, Xtr, ytr, seeds):
    return [(fit_one({**params, "random_state": s}, Xtr, ytr), None) for s in seeds]


def build_scale_ens(Xtr, ytr, include_full):
    mem = []
    if include_full:
        mem.append((fit_one({}, Xtr, ytr), None))
    for name, sl in SCALE_COLS.items():
        cols = np.arange(sl.start, sl.stop)
        mem.append((fit_one({}, Xtr, ytr, cols=cols), cols))
    return mem


def build_bag_series(params, Xtr, ytr, sids, k, frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sids)
    mem = []
    for i in range(k):
        keep = set(rng.choice(uniq, size=int(len(uniq) * frac), replace=False).tolist())
        mask = np.fromiter((s in keep for s in sids), bool, len(sids))
        mem.append((fit_one({**params, "random_state": ENS_SEEDS[i]}, Xtr[mask], ytr[mask]), None))
    return mem


def _log(name, aucs, base, store):
    md, sd, t, _ = paired(aucs, base)
    store["d"][name].append(md)
    store["t"][name].append(t)
    verdict = "SHIP(hard)" if (md > 0 and t >= 2.5) else (
        "marginal" if (md > 0 and t >= 2.0) else ("weak" if (md > 0 and t >= 1.5) else "noise"))
    print(f"  {name:<18} mean={aucs.mean():.4f} delta={md:+.4f} t={t:+.2f} -> {verdict}", flush=True)


def main():
    print(f"=== diversity ensemble {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS}  ship: pos on all seeds & mean>=0.0015 (hard t>=2.5)", flush=True)

    names = ["seed_ens5", "scale_ens", "scale_plus_full", "bag_series5",
             "tuned_seed_ens3", "tuned_bag5"]
    store = {"d": {n: [] for n in names}, "t": {n: [] for n in names}}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        Xtr, ytr, sids = build_rows(X, yi, train_ids, champion_feat, WINDOW)
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
        del X

        base_aucs, _ = eval_model(make_hgb(), Xtr, ytr, caches, y_val)
        print(f"[seed {seed}] champion mean={base_aucs.mean():.4f}", flush=True)

        _log("seed_ens5", ensemble_aucs(build_seed_ens({}, Xtr, ytr, ENS_SEEDS), caches, y_val),
             base_aucs, store)
        _log("scale_ens", ensemble_aucs(build_scale_ens(Xtr, ytr, False), caches, y_val),
             base_aucs, store)
        _log("scale_plus_full", ensemble_aucs(build_scale_ens(Xtr, ytr, True), caches, y_val),
             base_aucs, store)
        _log("bag_series5", ensemble_aucs(
            build_bag_series({}, Xtr, ytr, sids, 5, 0.7, seed + 7), caches, y_val), base_aucs, store)
        _log("tuned_seed_ens3", ensemble_aucs(
            build_seed_ens(TUNED, Xtr, ytr, [42, 7, 123]), caches, y_val), base_aucs, store)
        _log("tuned_bag5", ensemble_aucs(
            build_bag_series(TUNED, Xtr, ytr, sids, 5, 0.7, seed + 11), caches, y_val),
            base_aucs, store)

        print(f"[seed {seed}] done ({time.time()-t0:.0f}s)\n", flush=True)
        del Xtr, ytr, sids, caches, base_aucs

    print("=== SUMMARY (mean over pool seeds) ===", flush=True)
    print(f"{'ensemble':<18} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all", flush=True)
    for n in names:
        d = np.asarray(store["d"][n]); t = np.asarray(store["t"][n])
        pos = "YES" if (d > 0).all() else " no"
        print(f"{n:<18} {d.mean():+8.4f} {d.min():+8.4f} {t.mean():+7.2f}   {pos}", flush=True)
    print("\nAn ensemble positive on ALL seeds with mean_d>=0.0015 is a ship candidate "
          "(build notebook + confirm on fresh seeds). tuned_* stack the two proven wins.", flush=True)


if __name__ == "__main__":
    main()
