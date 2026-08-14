"""Scale-specialization diversity as a 3rd member of the CDF-blend (2nd follow-up -> E42 candidate).

Follow-up (b) to the leaderboard win. The champion features are 45 columns = 15 base statistics
across 3 window scales (s60=0:15, s130=15:30, s250=30:45). A gradient-boosting model trained on a
SINGLE scale's 15-column view learns a genuinely different decision boundary than the full-45
pointwise member and than the ranker -- a different decorrelation axis than "another tree model"
(the random forest, which E39 found ~80% redundant with the ranker).

This harness builds a scale-specialist 3rd member (average proba of the 3 single-scale
gradient-boosting models, tuned params) and, as a variant, a full+scales average, adds it as a
fixed-CDF 3rd member, and paired-t's the 3-member blend vs the shipped 2-member champion. Faithful
percentile blend, exactly like _rf_ens3 / _confirm_rank_ens.

Gate: positive on ALL pool seeds AND mean_d>=0.0015 AND mean_t>=2.5 (at its best weights)
-> fresh-seed confirm, then build a 3-member notebook.

Usage (one 15-job harness at a time; run AFTER _forest_tune finishes):
  SMOKE=1 .venv/bin/python -u _scale_ens3.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _scale_ens3.py > _scale_ens3.log 2>&1 & disown
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

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, load_pool_strat, make_hgb,
    precompute_fold, score_fold, select_picks, WINDOW,
)
from _exp_fast import ts_auc_from
from _diversity_ens import SCALE_COLS, score_fold_cols
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

TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)

# 3-member weight schemes (gb, rank, scale-member); near-equal region (E39 finding).
WEIGHT_SETS = [(0.40, 0.30, 0.30), (0.36, 0.34, 0.30), (0.34, 0.33, 0.33), (0.34, 0.30, 0.36)]

# member variants: which column-views to average into the scale-specialist member.
VARIANTS = {
    "scale_avg": [np.arange(sl.start, sl.stop) for sl in SCALE_COLS.values()],
    "scale_plus_full": [None] + [np.arange(sl.start, sl.stop) for sl in SCALE_COLS.values()],
}


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def _key(w):
    return f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"


def fit_scale_member(Xtr, ytr, col_views, seed):
    """Fit one tuned gradient-boosting model per column-view; return the list of (model, cols)."""
    members = []
    for cols in col_views:
        m = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            m.fit(Xtr if cols is None else Xtr[:, cols], ytr)
        members.append((m, cols))
    return members


def member_train_ref(members, Xtr):
    """Fixed-CDF reference = sort of the averaged training proba across the member's views."""
    acc = np.zeros(len(Xtr))
    for m, cols in members:
        acc = acc + m.predict_proba(Xtr if cols is None else Xtr[:, cols])[:, 1]
    return np.sort(acc / len(members))


def member_fold_score(members, cache):
    """Averaged proba Series for one fold across the member's views."""
    acc = None
    for m, cols in members:
        s = score_fold_cols(m, cache, cols)
        acc = s if acc is None else acc + s
    return acc / len(members)


def main():
    print(f"=== scale-specialist 3rd-member CDF ensemble {datetime.now():%Y-%m-%d %H:%M:%S} ===",
          flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS} dense_k={DENSE_K}  variants={list(VARIANTS)}\n"
          f"base = shipped 2-member champion (0.6/0.4); gate pos_all & mean_d>=0.0015 & mean_t>=2.5",
          flush=True)

    store = {name: {"d": [], "t": [], "w": None} for name in VARIANTS}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
        idx_folds = [c["index"] for c in caches]

        # member 1: tuned gradient boosting (full 45), shared
        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)
        gb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            gb.fit(Xtr_c, ytr_c)
        ref_gb = np.sort(gb.predict_proba(Xtr_c)[:, 1])
        fh_folds = [pct(score_fold(gb, c).to_numpy(), ref_gb) for c in caches]

        # member 2: default ranker, shared
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr, gr)
        ref_rk = np.sort(ranker.r.predict(rr))
        fr_folds = [pct(score_fold(ranker, c).to_numpy(), ref_rk) for c in caches]
        del X

        base_aucs = np.asarray([
            ts_auc_from(pd.Series(0.6 * fh + 0.4 * fr, index=ix).to_frame("prediction"), y_val)
            for fh, fr, ix in zip(fh_folds, fr_folds, idx_folds)])
        print(f"\n[seed {seed}] 2-member base mean={base_aucs.mean():.4f} "
              f"train={Xtr_c.shape} ({time.time()-t0:.0f}s prep)", flush=True)

        for name, col_views in VARIANTS.items():
            tc = time.time()
            members = fit_scale_member(Xtr_c, ytr_c, col_views, seed)
            ref_m = member_train_ref(members, Xtr_c)
            fm_folds = [pct(member_fold_score(members, c).to_numpy(), ref_m) for c in caches]

            best = None
            for w in WEIGHT_SETS:
                aucs = np.asarray([
                    ts_auc_from(pd.Series(w[0]*fh + w[1]*fr + w[2]*fm, index=ix)
                                .to_frame("prediction"), y_val)
                    for fh, fr, fm, ix in zip(fh_folds, fr_folds, fm_folds, idx_folds)])
                md, _s, t, _ = paired(aucs, base_aucs)
                if best is None or md > best[0]:
                    best = (md, t, w)
            md, t, w = best
            store[name]["d"].append(md); store[name]["t"].append(t); store[name]["w"] = w
            print(f"  {name:<16} best {_key(w)} d={md:+.4f} t={t:+.2f} ({time.time()-tc:.0f}s)",
                  flush=True)
        del caches, Xtr_c, ytr_c, rr, lr, gr, rd, ld, td

    print("\n=== SUMMARY (best weights per variant, vs shipped 2-member champion) ===", flush=True)
    print(f"{'variant':<16} {'weights':<16} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all  verdict",
          flush=True)
    rows_out = []
    for name in VARIANTS:
        d = np.asarray(store[name]["d"]); t = np.asarray(store[name]["t"])
        pos = (d > 0).all()
        confirm = pos and d.mean() >= 0.0015 and t.mean() >= 2.5
        verdict = "CONFIRM" if confirm else ("promising" if (pos and d.mean() > 0) else "noise")
        rows_out.append((d.mean(), name, store[name]["w"], d, t, pos, verdict))
    rows_out.sort(key=lambda r: -r[0])
    for md, name, w, d, t, pos, verdict in rows_out:
        print(f"{name:<16} {_key(w):<16} {md:>+8.4f} {d.min():>+8.4f} {t.mean():>+7.2f}   "
              f"{'YES' if pos else ' no'}     {verdict}", flush=True)
    print("\nA variant positive on ALL seeds with mean_d>=0.0015 & mean_t>=2.5 beats the shipped "
          "champion -> fresh-seed confirm, then build a 3-member notebook. Else scale-specialization "
          "diversity is confirmed redundant with the existing 2-member blend.", flush=True)


if __name__ == "__main__":
    main()
