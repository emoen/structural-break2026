"""TWO-HEAD EARLY/LATE probe on the CURRENT (E78) ranker+rf base79 champion.

Origin: 2026-08-28 2-agent ideation #3 (proposer idea #1, critic KEEP #2). The single generalist
classifier is a compromise: it sees ~DENSE_K steps/series roughly uniform in offset, so most rows
are early/easy and the decision surface is pulled toward the transition edge (per-offset AUC peaks
~0.75 at offset 5-10) instead of the persistent-regime late tail (~0.64 by offset 200). A two-head
split trains ONE member specialized on EARLY rows and ONE on LATE rows, so each fits its own regime's
decision boundary instead of averaging them.

MECHANISM (no new features -- base79 byte-identical to the E78 champion; shares _struct79_cache with
_wloss_rankerrf.py): split TRAINING rows by post-break offset off = t - tau:
    early head: negatives (off<0) + positive rows with 0 <= off < K
    late  head: negatives (off<0) + positive rows with off >= K
Each head = the same ranker+rf(fixed-CDF) blend as the champion, fit on its subset. At inference the
offset is unknown, so BOTH heads score every step and are blended:  lam*head_early + (1-lam)*head_late.
Baseline = the single generalist (all rows) == the champion (reused from _wloss_rankerrf.score_gamma
at gamma=0). ARMS = a small (K, lam) grid.

Negatives go to BOTH heads (a negative has no offset and must be ranked below positives at every step).

CRITIC NOTES: (a) the late regime is near its ceiling, so a late-only member is unlikely (E63 model-
diversity ship-gate was negative); the two-head form hedges by keeping the early specialist where the
E72 offset-elbow says most separable head-room sits. (b) The real risk: if the late signal is at the
info floor, the late head just fits noise on fewer rows and adds variance -> the blend is a wash or
worse. BELIEVE-RULE before an LB slot: an arm must clear mean_d >= 0.0015 AND be positive on all
screen seeds, then confirm on fresh seeds. Prior ~12%.

RUN ONE AT A TIME (memory); shares the base79 cache with _wloss_rankerrf.py. Run:
  SMOKE=1 .venv/bin/python -u _twohead_rankerrf.py
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _twohead_rankerrf.py \
      > _twohead_rankerrf.out 2>&1 & disown
"""
import os
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestClassifier

from _overnight import N_JOBS
from _residual_overnight import pct
from _lambdarank import sort_into_groups, fit_ranker
# reuse the byte-identical base79 build + shared cache + generalist baseline
from _wloss_rankerrf import (
    prepare_seed, make_folds_meta, score_gamma, _proba, _auc,
    W_RANK, W_RF, RF_TREES, RF_KW, SHIP_SCALES,
)

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]

# (K = early/late offset split, lam = weight on the EARLY head). base = single generalist.
ARMS = ([(40, 0.5)] if SMOKE
        else [(40, 0.4), (40, 0.6), (60, 0.4), (60, 0.6)])

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "_twohead_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_twohead_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_twohead_rankerrf.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def _fit_head(sd, mask_tr, mask_r, seed):
    Xtr, ytr = sd["Xtr"], sd["ytr"]
    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=N_JOBS, random_state=seed, **RF_KW)
    rf.fit(Xtr[mask_tr], ytr[mask_tr])
    ref_rf = np.sort(_proba(rf, Xtr[mask_tr]))
    rows, lab, grp = sort_into_groups(sd["Rr"][mask_r], sd["Rl"][mask_r], sd["Rt"][mask_r])
    rk = fit_ranker("rank_xendcg", rows, lab, grp)
    ref_rk = np.sort(_proba(rk, sd["Rr"][mask_r]))
    return rf, ref_rf, rk, ref_rk


def score_twohead(sd, metas, K, lam, seed):
    otr, orr = sd["off_tr"], sd["off_r"]
    neg_tr, neg_r = otr < 0, orr < 0
    e_tr = neg_tr | ((otr >= 0) & (otr < K))
    l_tr = neg_tr | (otr >= K)
    e_r = neg_r | ((orr >= 0) & (orr < K))
    l_r = neg_r | (orr >= K)
    # degenerate-head guard: a head needs both classes with >=2 positives, else
    # RF predict_proba has one column / the ranker is meaningless -> skip this arm.
    ytr = sd["ytr"]
    for m in (e_tr, l_tr):
        if int(ytr[m].sum()) < 2 or int((ytr[m] == 0).sum()) < 2:
            return float("nan")
    rf_e, ref_rf_e, rk_e, ref_rk_e = _fit_head(sd, e_tr, e_r, seed)
    rf_l, ref_rf_l, rk_l, ref_rk_l = _fit_head(sd, l_tr, l_r, seed)

    aucs = []
    for (L, active, groups), c in zip(metas, sd["caches"]):
        srf_e = np.zeros(L); srk_e = np.zeros(L)
        srf_l = np.zeros(L); srk_l = np.zeros(L)
        if len(c["feat"]):
            fe = c["feat"]
            srf_e[active] = _proba(rf_e, fe); srk_e[active] = _proba(rk_e, fe)
            srf_l[active] = _proba(rf_l, fe); srk_l[active] = _proba(rk_l, fe)
        head_e = W_RANK * pct(srk_e, ref_rk_e) + W_RF * pct(srf_e, ref_rf_e)
        head_l = W_RANK * pct(srk_l, ref_rk_l) + W_RF * pct(srf_l, ref_rf_l)
        blend = lam * head_e + (1.0 - lam) * head_l
        wsum = tot = 0.0
        for r, lb, w in groups:
            a = _auc(lb, blend[r])
            if a is None:
                continue
            wsum += w * a
            tot += w
        aucs.append(wsum / tot if tot > 0 else 0.5)
    return float(np.mean(aucs))


def main():
    open(LOG, "w").close()
    log(f"=== TWO-HEAD EARLY/LATE probe on ranker+rf E78 base79  {datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} blend={W_RANK}/{W_RF} ship_scales={SHIP_SCALES}")
    log(f"arms (K,lam)={ARMS}  (lam = weight on EARLY head; base = single generalist)")

    if not os.path.exists(RAW):
        with open(RAW, "w") as f:
            f.write("seed\tarm\tbase_auc\tcand_auc\tdelta\n")

    per_seed = {}
    for seed in SEARCH_SEEDS:
        sd = prepare_seed(seed)
        metas = make_folds_meta(sd)
        pos_tr = int((sd["off_tr"] >= 0).sum())
        n_tr = len(sd["off_tr"])
        for K in sorted({K for K, _ in ARMS}):
            late_pos = int((sd["off_tr"] >= K).sum())
            early_pos = int(((sd["off_tr"] >= 0) & (sd["off_tr"] < K)).sum())
            log(f"[seed {seed}] K={K} head pos: early={early_pos} late={late_pos}")
        t0 = time.time()
        base_auc = score_gamma(sd, metas, 0.0, seed)   # generalist == champion
        log(f"[seed {seed}] base79 generalist ranker+rf = {base_auc:.4f}  "
            f"(pos_rows {pos_tr}/{n_tr})  ({time.time()-t0:.0f}s)")
        res = {}
        for K, lam in ARMS:
            name = f"K{K}_l{lam}"
            tc = time.time()
            auc = score_twohead(sd, metas, K, lam, seed)
            if np.isnan(auc):
                res[name] = float("nan")
                with open(RAW, "a") as f:
                    f.write(f"{seed}\t{name}\t{base_auc:.6f}\tnan\tnan\t# DEGENERATE\n")
                log(f"[seed {seed}] {name:<10} DEGENERATE (skipped)")
                continue
            d = auc - base_auc
            res[name] = d
            with open(RAW, "a") as f:
                f.write(f"{seed}\t{name}\t{base_auc:.6f}\t{auc:.6f}\t{d:+.6f}\n")
            log(f"[seed {seed}] {name:<10} {auc:.4f}  d={d:+.4f}  ({time.time()-tc:.0f}s)")
        per_seed[seed] = res

    names = [f"K{K}_l{lam}" for K, lam in ARMS]
    agg = []
    skipped = []
    for n in names:
        ds = [per_seed[s][n] for s in SEARCH_SEEDS if n in per_seed[s]]
        ds = [d for d in ds if not np.isnan(d)]
        if not ds:
            skipped.append(n)
            continue
        md = float(np.mean(ds))
        npos = sum(1 for d in ds if d > 0)
        agg.append((n, md, min(ds), max(ds), npos, len(ds)))
    agg.sort(key=lambda r: -r[1])
    with open(SUMMARY, "w") as f:
        f.write(f"# two-head early/late probe on ranker+rf E78 base79  {datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} blend={W_RANK}/{W_RF} ship={SHIP_SCALES}\n")
        f.write(f"# early head: neg + pos(off<K); late head: neg + pos(off>=K); blend lam*early+(1-lam)*late.\n")
        f.write(f"# base = single generalist (all rows) == champion.\n")
        f.write(f"# BELIEVE only if mean_d>=+0.0015 AND all-seeds-positive, then confirm on fresh seeds.\n\n")
        f.write(f"{'arm':<12}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for n, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md > 0 and npos == tot) else ""
            f.write(f"{n:<12}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
        for n in skipped:
            f.write(f"{n:<12}{'SKIPPED':>10}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
