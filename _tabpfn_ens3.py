"""TabPFN as a 3rd MEMBER of the shipped champion blend (the CORRECT test).

Champion (LB 58.78%) = tuned HGB + LGBM rank_xendcg, fixed train-CDF blend 0.6*F_hgb+0.4*F_rank.
E39/RF and the MLP/GRU search both showed a member can look great vs HGB-alone yet collapse on top
of the ranker (the ranker already captures the decorrelation gain a tree/MLP provides).

TabPFN is a PRE-TRAINED tabular transformer (in-context learning, no gradient training on our data).
It is a fundamentally different function class from both tree members AND from a from-scratch MLP,
so it is the strongest remaining "orthogonal member" candidate. Our data (~8000 rows x 45 champion
features, binary) sits right in TabPFN's native regime.

Each config is scored as the 3rd member vs the shipped 2-member baseline, fixed-train-CDF percentile
blend, weight-set sweep, paired-t across pool seeds.
Gate: positive on ALL seeds AND mean_d>=0.0015 AND mean_t>=2.5 (same hardened bar as the laptop).

Usage:
  SMOKE=1 .venv/bin/python -u _tabpfn_ens3.py
  SEEDS=42,7,123 TABPFN_CONFIGS=tabpfn_n8,tabpfn_n8_bal .venv/bin/python -u _tabpfn_ens3.py
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

import torch
from tabpfn import TabPFNClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE") == "1"
POOL_SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7" if SMOKE else "42,7,123").split(",")]
DENSE_K = 30 if SMOKE else 60

TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)
BASE_W = (0.6, 0.4, 0.0)                      # shipped 2-member champion
WEIGHT_SETS = [BASE_W, (0.50, 0.33, 0.17), (0.45, 0.30, 0.25), (0.40, 0.35, 0.25),
               (0.36, 0.34, 0.30)]


# --- TabPFN member ---------------------------------------------------------
def make_tabpfn(cfg, seed):
    return TabPFNClassifier(
        device=DEVICE,
        n_estimators=cfg["n_estimators"],
        balance_probabilities=cfg.get("balance", False),
        ignore_pretraining_limits=True,
        memory_saving_mode=True,   # batch inference -> cap host RAM (prev run was OOM-SIGKILLed)
        random_state=seed,
    )


CONFIGS = [
    dict(name="tabpfn_n4",     n_estimators=4),
    dict(name="tabpfn_n8",     n_estimators=8),
    dict(name="tabpfn_n8_bal", n_estimators=8, balance=True),
    dict(name="tabpfn_n16",    n_estimators=16),
]
_sel = os.environ.get("TABPFN_CONFIGS")
if _sel:
    keep = set(_sel.split(","))
    CONFIGS = [c for c in CONFIGS if c["name"] in keep]
if SMOKE:
    CONFIGS = [dict(name="tabpfn_n2_smoke", n_estimators=2)]


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def eval_config(cfg, Xtr_c, ytr_c, caches, fh, fr, base_aucs, y_val, seed):
    """Fit one TabPFN member + score it as the 3rd fixed-CDF blend member."""
    t0 = time.time()
    clf = make_tabpfn(cfg, seed)
    with threadpool_limits(limits=1):
        clf.fit(Xtr_c, ytr_c)
    ref_tp = np.sort(clf.predict_proba(Xtr_c)[:, 1])
    tp_series = [score_fold(clf, c) for c in caches]
    sa, per = [], {w: [] for w in WEIGHT_SETS}
    for a, b, ts in zip(fh, fr, tp_series):
        fn = pct(ts.to_numpy(), ref_tp)
        sa.append(ts_auc_from(ts.to_frame("prediction"), y_val))
        for w in WEIGHT_SETS:
            blend = w[0] * a + w[1] * b + w[2] * fn
            per[w].append(ts_auc_from(pd.Series(blend, index=ts.index).to_frame("prediction"), y_val))
    best = (BASE_W, -1e9, 0.0)
    for w in WEIGHT_SETS:
        if w == BASE_W:
            continue
        md, _s, t, _ = paired(np.asarray(per[w]), base_aucs)
        if md > best[1]:
            best = (w, md, t)
    return cfg["name"], best[0], best[1], best[2], float(np.mean(sa)), time.time() - t0


def main():
    print(f"=== TabPFN 3rd-member search {datetime.now():%F %T} | seeds={POOL_SEEDS} "
          f"| {len(CONFIGS)} configs | device={DEVICE} ===", flush=True)
    d_store = {c["name"]: [] for c in CONFIGS}
    t_store = {c["name"]: [] for c in CONFIGS}
    standalone = {c["name"]: [] for c in CONFIGS}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)

        # member 1: tuned HGB
        hgb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            hgb.fit(Xtr_c, ytr_c)
        ref_hgb = np.sort(hgb.predict_proba(Xtr_c)[:, 1])
        hgb_series = [score_fold(hgb, c) for c in caches]

        # member 2: LGBM rank_xendcg (shipped ranker)
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr_, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr_, gr)
        ref_rank = np.sort(ranker.r.predict(rr))
        rank_series = [score_fold(ranker, c) for c in caches]

        # fixed-train-CDF percentiles for the two members + 2-member base AUC
        fh = [pct(s.to_numpy(), ref_hgb) for s in hgb_series]
        fr = [pct(s.to_numpy(), ref_rank) for s in rank_series]
        base_aucs = np.asarray([
            ts_auc_from(pd.Series(0.6 * a + 0.4 * b, index=s.index).to_frame("prediction"), y_val)
            for a, b, s in zip(fh, fr, hgb_series)])
        print(f"\n[seed {seed}] 2-member base mean={base_aucs.mean():.4f} ({time.time()-t0:.0f}s)",
              flush=True)

        for cfg in CONFIGS:
            name, bw, bd, bt, sa, dt = eval_config(cfg, Xtr_c, ytr_c, caches, fh, fr,
                                                   base_aucs, y_val, seed)
            d_store[name].append(bd)
            t_store[name].append(bt)
            standalone[name].append(sa)
            print(f"  {name:<16} standalone={sa:.4f} best_w={bw} d={bd:+.4f} t={bt:+.2f} ({dt:.0f}s)",
                  flush=True)
        del X, caches, hgb_series, rank_series

    print("\n=== SUMMARY (3rd member vs shipped 2-member champion) ===", flush=True)
    for name in sorted(d_store, key=lambda n: -np.mean(d_store[n])):
        ds, ts_ = d_store[name], t_store[name]
        npos = sum(1 for x in ds if x > 0)
        gate = "SHIP(hard)" if npos == len(ds) and np.mean(ds) >= 0.0015 and min(ts_) >= 2.5 else (
            "promising" if npos == len(ds) and np.mean(ds) > 0 else "noise")
        print(f"{name:<16} mean_d {np.mean(ds):+.4f} min_d {min(ds):+.4f} pos {npos}/{len(ds)} "
              f"min_t {min(ts_):+.2f} standalone {np.mean(standalone[name]):.4f} -> {gate}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
