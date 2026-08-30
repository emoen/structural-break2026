"""Decorrelation diagnostic for the AR-MDN learned member (the E-probe that came back NULL).

Question this answers: the minimal residual-shape MLP member was net-NEGATIVE in the blend at every
weight (mean_d -0.0012 @ w=0.10). Was that because it is (a) REDUNDANT -- carries no ranking signal
the ranker+RF don't already have (=> this EMBEDDING is the dead end, a bigger net on the same 32
quantiles won't help), or (b) DECORRELATED-BUT-MISCOMBINED -- it has orthogonal signal the crude
fixed-CDF percentile blend double-counts / mis-weights (=> a better combiner + a stronger model is
worth building)?

Reuses the already-built _armdn_cache seeds and refits the 3 members exactly as the probe does
(prep_scores), then measures everything IN METRIC SPACE (per online-step cross-sectional weighted
rank-AUC), because standalone Pearson/accuracy are irrelevant to TS-AUC.

Per seed it reports:
  1. standalone TS-AUC of each member: mlp / ranker / rf / 2-member blend  (how good is mlp ALONE?)
  2. pooled Spearman rank-corr of the mlp percentile vs blend / ranker / rf   (redundancy)
  3. ORTHOGONAL signal: residualize the mlp percentile off [ranker, rf] percentiles by pooled OLS,
     then TS-AUC of the residual alone  (>0.52 => real new ranking signal; ~0.50 => none)
  4. better-combiner probe: blend = (1-w)*base2 + w*pct(orthogonal residual), small w grid
     (does USING the decorrelated part -- instead of the raw correlated member -- actually help?)

Read-only diagnostic. Does NOT touch the champion, the caches, or any submission. ~250s/seed (refit).
Run:
  PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _armdn_diag.py > _armdn_diag.out 2>&1
"""
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _armdn_rankerrf import prepare_seed, prep_scores
from _wloss_rankerrf import make_folds_meta, _auc, W_RANK, W_RF

SEEDS = [42, 7, 123]
ORTH_W = [0.05, 0.10, 0.20, 0.30]

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "_armdn_diag.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def _tsauc(groups, score_full):
    """Per-online-step weighted cross-sectional rank-AUC on a full-length score array."""
    wsum = tot = 0.0
    for r, lb, w in groups:
        a = _auc(lb, score_full[r])
        if a is None:
            continue
        wsum += w * a
        tot += w
    return wsum / tot if tot > 0 else 0.5


def _mean_tsauc(folds, pick):
    """Average TS-AUC over folds where pick(prk,prf,pmdn) -> the full-length score array."""
    return float(np.mean([_tsauc(groups, pick(prk, prf, pmdn))
                          for (_a, groups, prk, prf, pmdn) in folds]))


def _pct_self(vals):
    """Rank-normalize a 1-D array to [0,1] (self-CDF), NaN-safe."""
    if len(vals) == 0:
        return vals
    return (pd.Series(vals).rank(method="average").to_numpy() - 1.0) / max(len(vals) - 1, 1)


def main():
    open(LOG, "w").close()
    log(f"=== AR-MDN DECORRELATION DIAGNOSTIC  {datetime.now():%F %T} ===")
    log(f"seeds={SEEDS}  base2 = {W_RANK}*pct(ranker)+{W_RF}*pct(rf)  orth_w grid={ORTH_W}")
    log("q: is the learned member REDUNDANT (embedding dead) or DECORRELATED-MISCOMBINED (build bigger)?\n")

    for seed in SEEDS:
        t0 = time.time()
        sd = prepare_seed(seed)                 # cache hit
        metas = make_folds_meta(sd)
        folds = prep_scores(sd, metas, seed)    # refit ranker+rf+mlp (the expensive part)

        base2 = _mean_tsauc(folds, lambda rk, rf, md: W_RANK * rk + W_RF * rf)
        a_mlp = _mean_tsauc(folds, lambda rk, rf, md: md)
        a_rk = _mean_tsauc(folds, lambda rk, rf, md: rk)
        a_rf = _mean_tsauc(folds, lambda rk, rf, md: rf)

        # pool the scored (grouped) rows across folds for correlation + OLS residualization
        idx_all, mlp_all, rk_all, rf_all = [], [], [], []
        for (_a, groups, prk, prf, pmdn) in folds:
            rows = np.concatenate([r for r, _l, _w in groups]) if groups else np.zeros(0, int)
            idx_all.append(rows)
            mlp_all.append(pmdn[rows]); rk_all.append(prk[rows]); rf_all.append(prf[rows])
        mlp_p = np.concatenate(mlp_all); rk_p = np.concatenate(rk_all); rf_p = np.concatenate(rf_all)
        blend_p = W_RANK * rk_p + W_RF * rf_p

        c_blend = pd.Series(mlp_p).corr(pd.Series(blend_p), method="spearman")
        c_rk = pd.Series(mlp_p).corr(pd.Series(rk_p), method="spearman")
        c_rf = pd.Series(mlp_p).corr(pd.Series(rf_p), method="spearman")

        # pooled OLS: mlp ~ 1 + ranker + rf ; residual = the part of mlp NOT explained by the members
        A = np.column_stack([np.ones_like(rk_p), rk_p, rf_p])
        coef, *_ = np.linalg.lstsq(A, mlp_p, rcond=None)
        resid_p = mlp_p - A @ coef
        r2 = 1.0 - resid_p.var() / mlp_p.var() if mlp_p.var() > 0 else float("nan")

        # orthogonal-signal TS-AUC: score every row by the mlp residual (apply same coef per fold)
        def _orth_full(prk, prf, pmdn):
            return pmdn - (coef[0] + coef[1] * prk + coef[2] * prf)
        a_orth = _mean_tsauc(folds, _orth_full)

        # better-combiner probe: blend in the orthogonal residual as its OWN self-CDF percentile
        orth_pct_p = _pct_self(resid_p)   # for reference; per-fold pct applied below
        grid = []
        for w in ORTH_W:
            def _combo(prk, prf, pmdn, w=w):
                resid = pmdn - (coef[0] + coef[1] * prk + coef[2] * prf)
                # self-CDF percentile of the residual on this fold's scored rows only
                full = np.zeros_like(resid)
                # rank only the finite scored positions; inactive stay 0 (harmless, not in any group)
                order_vals = resid
                full = _pct_self_fullsafe(order_vals)
                return (1.0 - w) * (W_RANK * prk + W_RF * prf) + w * full
            grid.append((w, _mean_tsauc(folds, _combo)))

        log(f"[seed {seed}]  ({time.time()-t0:.0f}s)")
        log(f"  standalone TS-AUC : mlp={a_mlp:.4f}  ranker={a_rk:.4f}  rf={a_rf:.4f}  base2={base2:.4f}")
        log(f"  redundancy Spearman(mlp, .): blend={c_blend:+.3f}  ranker={c_rk:+.3f}  rf={c_rf:+.3f}"
            f"   (OLS mlp~rk+rf  R^2={r2:.3f})")
        log(f"  ORTHOGONAL residual TS-AUC = {a_orth:.4f}   (0.50 => no new ranking signal)")
        gtxt = "  ".join(f"w={w:.2f}:{a:.4f}({a-base2:+.4f})" for w, a in grid)
        log(f"  better-combiner (base2 + w*pct(orth)):  base2={base2:.4f}  {gtxt}")
        best_w, best_a = max(grid, key=lambda z: z[1])
        verdict = ("DECORRELATED-MISCOMBINED (orth signal usable -> bigger build worth it)"
                   if (a_orth > 0.52 or best_a - base2 >= 0.0015)
                   else "REDUNDANT on this embedding (change the INPUT, not just the model)")
        log(f"  => {verdict}\n")

    log("done")


def _pct_self_fullsafe(vals):
    """Rank-normalize to [0,1] over the whole array (used inside the fold combiner)."""
    n = len(vals)
    if n == 0:
        return vals
    return (pd.Series(vals).rank(method="average").to_numpy() - 1.0) / max(n - 1, 1)


if __name__ == "__main__":
    main()
