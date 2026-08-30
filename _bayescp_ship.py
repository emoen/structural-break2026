"""Ship-gate: does the confirmed bayescp bank (batch Bayesian change-point, 3 feats) lift the
FULL streaming champion (not just the OOF-AUC screen)? Fits the champion members (tuned HGB +
LGBMRanker) on champ61 vs champ61+bayescp, both with the SHIPPED 0.6/0.4 percentile blend, and
scores per-val-series TS-AUC. Ship gate vs the 61-col champion: pos_all & mean_d>=0.0015 &
min_t>=2.0. Mirrors _stack_champ member fits exactly. Set FRESH=1 for the fresh-seed confirm.
"""
import os, time
from datetime import datetime
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from _night_stack import SEARCH_SEEDS, FRESH_SEEDS, NCHAMP_BASE, prepare_seed, paired
from _night_banks import NDEP
from _residual_overnight import _score, pct, TUNED
from _ensemble_tune import make_hgb
from _lambdarank import fit_ranker, sort_into_groups
from _exp_fast import ts_auc_from

C = NCHAMP_BASE
DEP_KEEP = [C + i for i in (0, 1, 2, 3, 4, 5, 10, 11, 12, 13)]
BASE61 = np.concatenate([np.arange(C), np.asarray(DEP_KEEP, dtype=int)])       # b_no_accf
BCP_COLS = [C + NDEP + i for i in range(3)]                                     # 65,66,67
CAND64 = np.concatenate([BASE61, np.asarray(BCP_COLS, dtype=int)])
W = 0.60
EPS = 1e-6


def score_cols(cols, S, seed):
    """Per-val-series TS-AUC of the shipped 0.6/0.4 HGB+ranker blend fit on `cols`."""
    Xtr, ytr, Rr, Rl, Rt = S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"]
    caches, y_val = S["caches"], S["y_val"]

    hgb = make_hgb(**TUNED, random_state=seed)
    with threadpool_limits(limits=1):
        hgb.fit(Xtr[:, cols], ytr)
    ref_h = np.sort(hgb.predict_proba(Xtr[:, cols])[:, 1])

    rrows, rlab, rgrp = sort_into_groups(Rr[:, cols], Rl, Rt)
    ranker = fit_ranker("rank_xendcg", rrows, rlab, rgrp)
    ref_r = np.sort(ranker.r.predict(Rr[:, cols]))

    hs = [_score(hgb, c, cols) for c in caches]
    rs = [_score(ranker, c, cols) for c in caches]
    out = []
    for a_s, r_s in zip(hs, rs):
        ph = pct(a_s.to_numpy(), ref_h)
        pr = pct(r_s.to_numpy(), ref_r)
        b = pd.Series(W * ph + (1.0 - W) * pr, index=a_s.index).to_frame("prediction")
        out.append(ts_auc_from(b, y_val))
    return np.asarray(out)


def main():
    seeds = FRESH_SEEDS if os.environ.get("FRESH") else SEARCH_SEEDS
    print(f"=== bayescp ship-gate on champion  {datetime.now():%F %T}  seeds={seeds} ===", flush=True)
    store, basemeans, candmeans = [], [], []
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_bayescp")
        base = score_cols(BASE61, S, seed)
        cand = score_cols(CAND64, S, seed)
        md, _s, t, _ = paired(cand, base)
        store.append((md, t))
        basemeans.append(base.mean()); candmeans.append(cand.mean())
        print(f"  [seed {seed}] champ61={base.mean():.4f} +bcp={cand.mean():.4f} "
              f"d{md:+.4f} t{t:+.2f} ({time.time()-t0:.0f}s)", flush=True)
        del S

    ds = np.array([d for d, _ in store]); ts = np.array([t for _, t in store])
    npos = int((ds > 0).sum())
    ok = (ds > 0).all() and ds.mean() >= 0.0015 and ts.min() >= 2.0
    print(f"\n=== verdict (champ61 mean {np.mean(basemeans):.4f} -> +bcp {np.mean(candmeans):.4f}, "
          f"{len(seeds)} seeds) ===", flush=True)
    print(f"champ61+bayescp  mean_d={ds.mean():+.4f}  min_d={ds.min():+.4f}  npos={npos}/{len(ds)}  "
          f"min_t={ts.min():+.2f}  {'SHIP-CANDIDATE' if ok else 'below-bar'}", flush=True)


if __name__ == "__main__":
    main()
