"""Confirm the E33 lambdarank winner on FRESH disjoint seeds, re-baselined vs the TUNED champion.

E33 (_lambdarank.py) found rank_xendcg@dense positive on all 4 SEARCH seeds {42,7,123,2024} vs the
OLD HGB champion. Two things must hold before spending an LB slot (E32 discipline):
  1. it must beat the CURRENT champion, which is now the TUNED HGB (LB 57.72%), not the old one;
  2. it must reproduce on seeds NOT used in the search.

This re-tests rank_xendcg@dense (winner) and lambdarank@dense on FRESH seeds {31337,2025,555,99},
paired vs BOTH the old champion and the tuned champion. Ship only if it stays positive vs the TUNED
champion across fresh seeds.

Usage:
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _confirm_rank.py > _confirm_rank.log 2>&1 & disown
"""
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from threadpoolctl import threadpool_limits

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, eval_model, load_pool_strat, make_hgb, precompute_fold, select_picks, WINDOW,
)
from _lambdarank import build_grouped, sort_into_groups, fit_ranker, score_all
from _overnight import build_training as bt_champ, paired

FRESH_SEEDS = [31337, 2025, 555, 99]
DENSE_K = 60

# E32 robust tuned winner = the CURRENT champion (LB 57.72%).
TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)


def verdict(md, t):
    return "SHIP(hard)" if (md > 0 and t >= 2.5) else (
        "marginal" if (md > 0 and t >= 2.0) else ("weak" if (md > 0 and t >= 1.5) else "noise"))


def main():
    print(f"=== confirm rank winner (fresh seeds) {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"FRESH_SEEDS={FRESH_SEEDS} dense_k={DENSE_K}  decision = vs TUNED champion", flush=True)

    cands = ["rankxend_dense", "lambdarank_dense"]
    vs_old = {c: [] for c in cands}
    vs_tuned = {c: [] for c in cands}
    vs_tuned_t = {c: [] for c in cands}

    for seed in FRESH_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

        picks = select_picks(X, train_ids)
        Xc, yc = bt_champ(X, yi, picks, champion_feat, WINDOW)
        old_aucs, _ = eval_model(make_hgb(), Xc, yc, caches, y_val)
        tuned_aucs, _ = eval_model(make_hgb(**TUNED), Xc, yc, caches, y_val)

        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        del X
        rr, lr, gr = sort_into_groups(rd, ld, td)

        print(f"\n[seed {seed}] old_champ={old_aucs.mean():.4f} tuned_champ={tuned_aucs.mean():.4f} "
              f"(prep {time.time()-t0:.0f}s)", flush=True)
        for name, obj in (("rankxend_dense", "rank_xendcg"), ("lambdarank_dense", "lambdarank")):
            a = score_all(fit_ranker(obj, rr, lr, gr), caches, y_val)
            mo, _, to, _ = paired(a, old_aucs)
            mt, _, tt, _ = paired(a, tuned_aucs)
            vs_old[name].append(mo)
            vs_tuned[name].append(mt); vs_tuned_t[name].append(tt)
            print(f"  {name:<16} mean={a.mean():.4f}  vs_old d={mo:+.4f}(t{to:+.2f})  "
                  f"vs_tuned d={mt:+.4f}(t{tt:+.2f}) -> {verdict(mt, tt)}", flush=True)
        del caches, Xc, yc, rd, ld, td

    print("\n=== SUMMARY vs TUNED champion (fresh seeds) ===", flush=True)
    print(f"{'candidate':<16} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all", flush=True)
    for c in cands:
        d = np.asarray(vs_tuned[c]); t = np.asarray(vs_tuned_t[c])
        pos = "YES" if (d > 0).all() else " no"
        print(f"{c:<16} {d.mean():+8.4f} {d.min():+8.4f} {t.mean():+7.2f}   {pos}", flush=True)
    print("\nCONFIRMED if positive on ALL fresh seeds vs the TUNED champion (mean_d>=0.0015). "
          "Combined with E33 search seeds that = robust across 8 splits -> build a lambdarank "
          "notebook (LGBMRanker, infer() calls .predict()) and submit.", flush=True)


if __name__ == "__main__":
    main()
