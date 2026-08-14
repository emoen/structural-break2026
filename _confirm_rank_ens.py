"""Fresh-seed confirm of the SHIP-REALISTIC rank+tuned ensemble (E36).

_rank_ens.py showed a blockbuster lift but used a LIVE per-fold global rank (not available
at inference). The shippable blend replaces the live rank with each model's FIXED empirical
CDF estimated OFFLINE from TRAINING scores:  emit  w*F_hgb(proba) + (1-w)*F_rank(score).
This harness validates that EXACT computation on FRESH disjoint seeds vs the tuned champion,
and prints the live-rank variant side-by-side to show the (small) gap.

Decision = the FIXED variant (what we can actually deploy). Gate: positive on ALL fresh seeds
with mean_d>=0.0015 (hard t>=2.5) -> build the ensemble notebook + ship.

Usage:
  SMOKE=1 .venv/bin/python -u _confirm_rank_ens.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _confirm_rank_ens.py \
       > _confirm_rank_ens.log 2>&1 & disown
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
if SMOKE:
    FRESH_SEEDS = [31337]
    DENSE_K = 30
else:
    FRESH_SEEDS = [31337, 2025, 555, 99]
    DENSE_K = 60

TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)
WEIGHTS = [0.4, 0.5, 0.6]


def pct(x, ref):
    """Fixed-reference percentile: fraction of ref <= x. ref must be pre-sorted."""
    return np.searchsorted(ref, x, side="right") / len(ref)


def main():
    print(f"=== confirm rank+tuned ensemble (SHIP-REALISTIC) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"FRESH_SEEDS={FRESH_SEEDS} dense_k={DENSE_K} weights={WEIGHTS} "
          f"decision=FIXED-train-CDF blend", flush=True)

    d_fixed = {w: [] for w in WEIGHTS}
    t_fixed = {w: [] for w in WEIGHTS}
    d_live = {w: [] for w in WEIGHTS}
    t_live = {w: [] for w in WEIGHTS}

    for seed in FRESH_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

        # tuned champion HGB -> model, train-score reference CDF, val series
        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)
        model_hgb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            model_hgb.fit(Xtr_c, ytr_c)
        ref_hgb = np.sort(model_hgb.predict_proba(Xtr_c)[:, 1])
        hgb_series = [score_fold(model_hgb, c) for c in caches]
        base_aucs = np.asarray([ts_auc_from(s.to_frame("prediction"), y_val)
                                for s in hgb_series])

        # rank_xendcg dense ranker -> train-score reference CDF, val series
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr, gr)
        ref_rank = np.sort(ranker.r.predict(rr))
        rank_series = [score_fold(ranker, c) for c in caches]
        del X

        per_fixed = {w: [] for w in WEIGHTS}
        per_live = {w: [] for w in WEIGHTS}
        for hs, rs_ in zip(hgb_series, rank_series):
            hv = hs.to_numpy()
            rv = rs_.to_numpy()
            # FIXED train-CDF percentile (shippable)
            fh = pct(hv, ref_hgb)
            fr = pct(rv, ref_rank)
            # LIVE per-fold global rank (idealized, not shippable)
            lh = hs.rank(method="average").to_numpy() / len(hs)
            lr_ = rs_.rank(method="average").to_numpy() / len(rs_)
            for w in WEIGHTS:
                bf = pd.Series(w * fh + (1.0 - w) * fr, index=hs.index)
                bl = pd.Series(w * lh + (1.0 - w) * lr_, index=hs.index)
                per_fixed[w].append(ts_auc_from(bf.to_frame("prediction"), y_val))
                per_live[w].append(ts_auc_from(bl.to_frame("prediction"), y_val))

        print(f"\n[seed {seed}] champion mean={base_aucs.mean():.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        for w in WEIGHTS:
            af = np.asarray(per_fixed[w]); al = np.asarray(per_live[w])
            mdf, _s, tf, _ = paired(af, base_aucs)
            mdl, _s2, tl, _ = paired(al, base_aucs)
            d_fixed[w].append(mdf); t_fixed[w].append(tf)
            d_live[w].append(mdl); t_live[w].append(tl)
            vf = "SHIP" if (mdf > 0 and tf >= 2.5) else ("+" if mdf > 0 else "-")
            print(f"  w={w:.1f}  FIXED mean={af.mean():.4f} d={mdf:+.4f} t={tf:+.2f} [{vf}]"
                  f"   | live d={mdl:+.4f} t={tl:+.2f}", flush=True)
        del caches, hgb_series, rank_series, Xtr_c, ytr_c, rd, ld, td

    print("\n=== SUMMARY vs tuned champion (FRESH seeds) ===", flush=True)
    print(f"{'w':>4} {'FIX_mean_d':>10} {'FIX_min_d':>10} {'FIX_mean_t':>10} {'pos':>5}  "
          f"{'live_mean_d':>11}", flush=True)
    for w in WEIGHTS:
        df = np.asarray(d_fixed[w]); tf = np.asarray(t_fixed[w]); dl = np.asarray(d_live[w])
        pos = int((df > 0).sum())
        flag = ""
        if pos == len(FRESH_SEEDS) and df.mean() >= 0.0015 and tf.mean() >= 2.5:
            flag = "  *** CONFIRMED (ship) ***"
        print(f"{w:>4.1f} {df.mean():>+10.4f} {df.min():>+10.4f} {tf.mean():>+10.2f} "
              f"{pos:>3}/{len(FRESH_SEEDS)}  {dl.mean():>+11.4f}{flag}", flush=True)
    print("\nCONFIRMED (FIXED) -> build ensemble notebook: tuned HGB + LGBMRanker + 2 sorted "
          "ref arrays; stream both, emit w*pct_hgb+(1-w)*pct_rank.", flush=True)


if __name__ == "__main__":
    main()
