"""Tune the STACK/COMBINER (not the members) on the CURRENT 61-col champion (b_no_accf).

The shipped blend is a FIXED linear percentile combine  score = 0.6*pct_hgb + 0.4*pct_rank.
The 0.6/0.4 weight was tuned only against the OLD 58.78% ensemble and carried forward; the
current champion added residual/dep features (changed the HGB member) and the combiner was
never re-tuned. This screen re-tunes the combiner on the current champion:
  * blend-weight sweep  w in {.30..80}   (baseline = shipped .60)
  * learned LOGISTIC meta-stacker on [pct_hgb, pct_rank] (learns the optimal linear combine
    from training percentiles; deployable as 2 fixed coefficients + intercept)
  * GEOMETRIC combine  sqrt(pct_hgb*pct_rank)

Everything is a fixed monotone/param transform of the two train-CDF percentiles => deployable
in infer() with no cross-section. Mirrors _persist_champ._eval exactly for the member fits.
Gate: pos_all & mean_d>=0.0015 & min_t>=2.0 vs the shipped w=.60. Set FRESH=1 for fresh seeds.
"""
import os, time
from datetime import datetime
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression

from _night_stack import SEARCH_SEEDS, FRESH_SEEDS, NCHAMP_BASE, prepare_seed, paired
from _residual_overnight import _score, pct, TUNED
from _ensemble_tune import make_hgb
from _lambdarank import fit_ranker, sort_into_groups
from _exp_fast import ts_auc_from

C = NCHAMP_BASE
DEP_KEEP = [C + i for i in (0, 1, 2, 3, 4, 5, 10, 11, 12, 13)]
BASE_COLS = np.concatenate([np.arange(C), np.asarray(DEP_KEEP, dtype=int)])   # 61 = b_no_accf

WEIGHTS = {f"w{int(round(w*100)):03d}": round(w, 2)
           for w in (0.40, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55, 0.60)}
EXTRA = []                            # combiner forms (logit/geo) already closed; fine weight grid only
ORDER = ["w040", "w044", "w046", "w048", "w050", "w052", "w055", "w060"]
BASELINE = "w055"                     # the NEWLY shipped combiner (2026-08-21)
EPS = 1e-6


def eval_stack(cols, S, seed):
    Xtr, ytr, Rr, Rl, Rt = S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"]
    caches, y_val = S["caches"], S["y_val"]

    hgb = make_hgb(**TUNED, random_state=seed)
    with threadpool_limits(limits=1):
        hgb.fit(Xtr[:, cols], ytr)
    proba_tr = hgb.predict_proba(Xtr[:, cols])[:, 1]
    ref_h = np.sort(proba_tr)

    rrows, rlab, rgrp = sort_into_groups(Rr[:, cols], Rl, Rt)
    ranker = fit_ranker("rank_xendcg", rrows, rlab, rgrp)
    rank_tr = ranker.r.predict(Rr[:, cols])
    ref_r = np.sort(rank_tr)

    lr = None
    if "logit" in EXTRA:
        ph_tr = pct(proba_tr, ref_h)
        pr_tr = pct(ranker.r.predict(Xtr[:, cols]), ref_r)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(np.c_[ph_tr, pr_tr], ytr)

    hs = [_score(hgb, c, cols) for c in caches]
    rs = [_score(ranker, c, cols) for c in caches]
    out = {k: [] for k in ORDER}
    for a_s, r_s in zip(hs, rs):
        idx = a_s.index
        ph = pct(a_s.to_numpy(), ref_h)
        pr = pct(r_s.to_numpy(), ref_r)
        scores = {k: WEIGHTS[k] * ph + (1.0 - WEIGHTS[k]) * pr for k in WEIGHTS}
        if "logit" in EXTRA:
            scores["logit"] = lr.decision_function(np.c_[ph, pr])   # monotone w/ proba -> ranking-equiv
        if "geo50" in EXTRA:
            scores["geo50"] = np.sqrt(np.clip(ph, EPS, None) * np.clip(pr, EPS, None))
        for k in ORDER:
            b = pd.Series(scores[k], index=idx).to_frame("prediction")
            out[k].append(ts_auc_from(b, y_val))
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    seeds = FRESH_SEEDS if os.environ.get("FRESH") else SEARCH_SEEDS
    print(f"=== STACK tuning on 61-col champion  {datetime.now():%F %T}  seeds={seeds} ===", flush=True)
    store = {k: [] for k in ORDER if k != BASELINE}
    basemeans = []
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_dep")
        res = eval_stack(BASE_COLS, S, seed)
        basemeans.append(res[BASELINE].mean())
        line = [f"[seed {seed}] {BASELINE}={res[BASELINE].mean():.4f} ({time.time()-t0:.0f}s)"]
        for k in ORDER:
            if k == BASELINE:
                continue
            md, _s, t, _ = paired(res[k], res[BASELINE])
            store[k].append((md, t))
            line.append(f"{k}={res[k].mean():.4f} d{md:+.4f} t{t:+.2f}")
        print("  " + "  ".join(line), flush=True)
        del S

    print(f"\n=== verdict (baseline {BASELINE} mean {np.mean(basemeans):.4f}, {len(seeds)} seeds) ===", flush=True)
    for k in ORDER:
        if k == BASELINE:
            continue
        ds = np.array([d for d, _ in store[k]]); ts = np.array([t for _, t in store[k]])
        npos = int((ds > 0).sum())
        ok = (ds > 0).all() and ds.mean() >= 0.0015 and ts.min() >= 2.0
        print(f"{k:<8} mean_d={ds.mean():+.4f}  min_d={ds.min():+.4f}  npos={npos}/{len(ds)}  "
              f"min_t={ts.min():+.2f}  {'SHIP-CANDIDATE' if ok else 'below-bar'}", flush=True)


if __name__ == "__main__":
    main()
