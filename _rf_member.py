"""RandomForest member test — does a bagged-tree model add decorrelated signal?

The shipped ensemble = tuned HGB (pointwise) + LGBM ranker (rank_xendcg), CDF-blended.
User wants a RandomForest model added. RF is a *bagged* tree ensemble (variance reduction
by averaging deep decorrelated trees) vs HGB which is *boosted* (bias reduction). They may
correlate (both axis-aligned trees on the same 45 champion features), but bagging is a
genuinely different estimator, so it gets a fair multi-seed screen.

Members scored on the SAME stratified folds, per-fold TS-AUC, paired-t vs the TUNED HGB
champion (the shipped pointwise member — NOT the untuned base), across POOL_SEEDS:

  * rf_solo        : RandomForest alone                              (weak prior; reference)
  * hgb_rf         : avg(tuned-HGB proba, RF proba)   equal weight
  * hgb_rf_30      : 0.7*tuned-HGB + 0.3*RF           (RF as a light decorrelator)

Gate: positive on ALL seeds AND mean_d>=0.0015 AND mean_t>=2.5 -> CONFIRM
(then fresh-seed confirm + CDF-blend faithful test before any LB slot).

Usage (one 15-job harness at a time — run AFTER the ARCH test finishes):
  SMOKE=1 .venv/bin/python -u _rf_member.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _rf_member.py > _rf_member.log 2>&1 & disown
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
    champion_feat, eval_model, load_pool_strat, make_hgb,
    precompute_fold, select_picks, WINDOW,
)
from _exp_fast import HIST_CAP, hist_context, split_series, ts_auc_from
from _overnight import paired

SMOKE = os.environ.get("SMOKE") == "1"
_SEEDS_ENV = os.environ.get("SEEDS", "")
POOL_SEEDS = ([int(s) for s in _SEEDS_ENV.split(",") if s.strip()] if _SEEDS_ENV
              else ([42] if SMOKE else [42, 7, 123, 2024]))
RF_JOBS = 4 if SMOKE else 14

# E32 robust tuned winner (the shipped pointwise member).
TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)


def rf_params(seed):
    return dict(n_estimators=(80 if SMOKE else 400), max_features="sqrt",
                min_samples_leaf=5, n_jobs=RF_JOBS, random_state=seed)


def build_rows(X, yi, train_ids, feat_fn, window):
    picks = select_picks(X, train_ids)
    rows, labels = [], []
    for sid, ts in picks.items():
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        for t in ts:
            lo = max(0, t - window + 1)
            rows.append(feat_fn(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def score_fold(model, cache):
    full = np.zeros(len(cache["index"]))
    if len(cache["feat"]):
        full[cache["active"]] = model.predict_proba(cache["feat"])[:, 1]
    return pd.Series(full, index=cache["index"])


def blend_aucs(members, caches, y_val):
    """members = list of (fitted_model, weight). Weighted proba blend, per-fold TS-AUC."""
    wsum = sum(w for _, w in members)
    out = []
    for c in caches:
        acc = None
        for model, w in members:
            s = score_fold(model, c) * (w / wsum)
            acc = s if acc is None else acc + s
        out.append(ts_auc_from(acc.to_frame("prediction"), y_val))
    return np.asarray(out)


def fit_hgb(params, Xtr, ytr, seed):
    m = make_hgb(**params, random_state=seed)
    with threadpool_limits(limits=1):
        m.fit(Xtr, ytr)
    return m


def fit_rf(Xtr, ytr, seed):
    m = RandomForestClassifier(**rf_params(seed))
    m.fit(Xtr, ytr)  # RF parallelises via n_jobs; leave BLAS unthrottled here
    return m


def _log(name, aucs, base, store):
    md, sd, t, _ = paired(aucs, base)
    store["d"][name].append(md)
    store["t"][name].append(t)
    verdict = "SHIP(hard)" if (md > 0 and t >= 2.5) else (
        "marginal" if (md > 0 and t >= 2.0) else ("weak" if (md > 0 and t >= 1.5) else "noise"))
    print(f"  {name:<12} mean={aucs.mean():.4f} delta={md:+.4f} t={t:+.2f} -> {verdict}", flush=True)


def main():
    print(f"=== RF member test {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS}  base=TUNED HGB  gate: pos_all & mean_d>=0.0015 & mean_t>=2.5",
          flush=True)

    names = ["rf_solo", "hgb_rf", "hgb_rf_30"]
    store = {"d": {n: [] for n in names}, "t": {n: [] for n in names}}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        Xtr, ytr = build_rows(X, yi, train_ids, champion_feat, WINDOW)
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
        del X

        hgb = fit_hgb(TUNED, Xtr, ytr, seed)
        rf = fit_rf(Xtr, ytr, seed)
        base_aucs = blend_aucs([(hgb, 1.0)], caches, y_val)
        print(f"[seed {seed}] tuned-HGB mean={base_aucs.mean():.4f} "
              f"train={Xtr.shape} (prep {time.time()-t0:.0f}s)", flush=True)

        _log("rf_solo", blend_aucs([(rf, 1.0)], caches, y_val), base_aucs, store)
        _log("hgb_rf", blend_aucs([(hgb, 0.5), (rf, 0.5)], caches, y_val), base_aucs, store)
        _log("hgb_rf_30", blend_aucs([(hgb, 0.7), (rf, 0.3)], caches, y_val), base_aucs, store)

        print(f"[seed {seed}] done ({time.time()-t0:.0f}s)\n", flush=True)
        del Xtr, ytr, caches, base_aucs, hgb, rf

    print("=== SUMMARY (mean over pool seeds, vs TUNED HGB) ===", flush=True)
    print(f"{'member':<12} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all  verdict", flush=True)
    for n in names:
        d = np.asarray(store["d"][n]); t = np.asarray(store["t"][n])
        pos = (d > 0).all()
        confirm = pos and d.mean() >= 0.0015 and t.mean() >= 2.5
        v = "CONFIRM" if confirm else ("promising" if (pos and d.mean() > 0) else "noise")
        print(f"{n:<12} {d.mean():+8.4f} {d.min():+8.4f} {t.mean():+7.2f}   "
              f"{'YES' if pos else ' no'}     {v}", flush=True)
    print("\nrf_solo weak is expected (bagged < boosted). The ship question is whether "
          "hgb_rf / hgb_rf_30 BEAT tuned-HGB-alone -> RF adds decorrelated signal.", flush=True)


if __name__ == "__main__":
    main()
