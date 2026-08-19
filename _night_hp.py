"""E70: hyperparameter + blend-weight search on the champion, augmented with E69's top calib block.

The shipped detector is a FIXED 0.6*F_hgb + 0.4*F_rank blend with HGB=TUNED (E32) and a default
rank_xendcg ranker. E68/E69 surface a weak-but-real per-series z-calibration signal. This run asks:
given the augmented feature set (shipped 61 cols + the best E69 calib block), can we do better by
(a) re-tuning the HGB and (b) re-choosing the blend weight w? To separate "re-tune helps" from
"calib helps", the SAME grid is scored on two column sets:
    base61 : the shipped 61 columns          -> isolates pure re-tune value
    aug    : base61 + E69 top calib block     -> re-tune + calibration together
For each column set the ranker is fit ONCE (independent of HGB params); each HGB config is fit once
and then blended at every w in WSWEEP for free (fixed train-CDF percentiles). Every (colset,config,w)
is paired vs the SHIPPED champion (base61, TUNED, w=0.6) so any win is directly LB-relevant.
Screen {42,7,123} -> fresh {31337,2025,555,99}. Results append to _night_hp_results.log.

NOTE: set AUG_CALIB_COLS from the E69 FRESH winner before running. Default = robust-z all block.
"""
import os
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from _night_stack import (
    FRESH_SEEDS, SEARCH_SEEDS, NCHAMP_BASE,
    make_logger, paired, prepare_seed, summarize, verdict,
)
from _night_banks import NDEP
from _residual_overnight import _score, pct, TUNED
from _ensemble_tune import make_hgb
from _lambdarank import fit_ranker, sort_into_groups
from _exp_fast import ts_auc_from

log = make_logger("_night_hp_results.log")

C = NCHAMP_BASE
CDEP = C
CMS = C + NDEP                 # z_meansd block (champ_calibz) 65..115
CRB = CMS + NCHAMP_BASE        # z_robust block 116..166

KEPT_DEP = [CDEP + i for i in (0, 1, 2, 3, 4, 5, 10, 11, 12, 13)]
BASE61 = np.concatenate([np.arange(C), np.asarray(KEPT_DEP, dtype=int)])

# ---- E69 winner block (UPDATE from _night_calibz_results.log FRESH summary) ----
# E69 result: robust median/MAD z is DEAD (all g_zr_* noise); the only survivor is z_meansd (mean/sd
# z), and even it is sub-threshold (3/4 fresh). Use z_meansd cols 65..115 as the "top" calib readout;
# calibration is not expected to ship -- the real E70 question is the base61 pure-retune colset.
AUG_CALIB_COLS = list(range(CMS, CMS + NCHAMP_BASE))

AUG = np.concatenate([BASE61, np.asarray(AUG_CALIB_COLS, dtype=int)]) if AUG_CALIB_COLS else BASE61
COLSETS = {"base61": BASE61, "aug": AUG}

WSWEEP = [0.50, 0.55, 0.60, 0.65, 0.70]


def hgb_grid():
    """TUNED plus one-knob perturbations around it (kept small so the run fits one harness slot)."""
    g = [dict(name="tuned", **TUNED)]
    def var(name, **ov):
        p = dict(TUNED); p.update(ov); g.append(dict(name=name, **p))
    var("lr015", learning_rate=0.015)
    var("lr030", learning_rate=0.030)
    var("iter600", max_iter=600)
    var("leaf95", max_leaf_nodes=95)
    var("leaf31", max_leaf_nodes=31)
    var("msl20", min_samples_leaf=20)
    var("l2_3", l2_regularization=3.0)
    var("mf07", max_features=0.7)
    return g


def ranker_cache(cols, S):
    rrows, rlab, rgrp = sort_into_groups(S["Rr"][:, cols], S["Rl"], S["Rt"])
    ranker = fit_ranker("rank_xendcg", rrows, rlab, rgrp)
    ref_r = np.sort(ranker.r.predict(S["Rr"][:, cols]))
    rs = [_score(ranker, c, cols) for c in S["caches"]]
    idxs = [s.index for s in rs]
    fr = [pct(s.to_numpy(), ref_r) for s in rs]
    return fr, idxs


def hgb_cache(cols, S, seed, params):
    p = {k: v for k, v in params.items() if k != "name"}
    hgb = make_hgb(**p, random_state=seed)
    with threadpool_limits(limits=1):
        hgb.fit(S["Xtr"][:, cols], S["ytr"])
    ref_h = np.sort(hgb.predict_proba(S["Xtr"][:, cols])[:, 1])
    hs = [_score(hgb, c, cols) for c in S["caches"]]
    return [pct(s.to_numpy(), ref_h) for s in hs]


def blend_aucs(fh, fr, idxs, y_val, w):
    return np.asarray([
        ts_auc_from(pd.Series(w * a + (1 - w) * b, index=ix).to_frame("prediction"), y_val)
        for a, b, ix in zip(fh, fr, idxs)])


def run_phase(grid, seeds, phase, keep=None):
    """Score COLSETS x grid x WSWEEP; `keep` (set of labels) restricts which cells are evaluated.
    Ranker fit once per colset, each HGB fit once, w-sweep free. Returns summarize() dict."""
    def label(cs, cname, w):
        return f"{cs}|{cname}|w{w:.2f}"

    store = {}
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_calibz")
        fr61, idx61 = ranker_cache(BASE61, S)
        fh61 = hgb_cache(BASE61, S, seed, dict(name="tuned", **TUNED))
        base = blend_aucs(fh61, fr61, idx61, S["y_val"], 0.60)
        log(f"[{phase} seed {seed}] shipped champion base={base.mean():.4f} ({time.time()-t0:.0f}s)")
        for cs, cols in COLSETS.items():
            fr, idxs = (fr61, idx61) if cs == "base61" else ranker_cache(cols, S)
            for c in grid:
                if keep is not None and not any(label(cs, c["name"], w) in keep for w in WSWEEP):
                    continue
                try:
                    fh = fh61 if (cs == "base61" and c["name"] == "tuned") else hgb_cache(cols, S, seed, c)
                except Exception:
                    log(f"  [{phase} s{seed}] {cs}|{c['name']} HGB-ERROR\n{traceback.format_exc()}")
                    continue
                for w in WSWEEP:
                    lab = label(cs, c["name"], w)
                    if keep is not None and lab not in keep:
                        continue
                    aug = blend_aucs(fh, fr, idxs, S["y_val"], w)
                    md, _s, t, _ = paired(aug, base)
                    store.setdefault(lab, []).append((md, t))
                    log(f"  [{phase} s{seed}] {cs:<6} {c['name']:<8} w{w:.2f} "
                        f"aug={aug.mean():.4f} d={md:+.4f} t={t:+.2f}")
        del S
    return summarize(store)


def main():
    log(f"\n===== NIGHT (hp) champion re-tune + blend sweep {datetime.now():%F %T} "
        f"smoke={os.environ.get('SMOKE') == '1'} =====")
    log(f"AUG_CALIB_COLS={AUG_CALIB_COLS[:3]}...({len(AUG_CALIB_COLS)}) colsets={list(COLSETS)}")
    grid = hgb_grid()
    log(f"screening {len(COLSETS)}x{len(grid)}x{len(WSWEEP)}="
        f"{len(COLSETS)*len(grid)*len(WSWEEP)} (colset,hgb,w) cells on {SEARCH_SEEDS}")
    screen = run_phase(grid, SEARCH_SEEDS, "SCREEN")
    log("\n----- (hp) SCREEN SUMMARY (top 20 by mean_d) -----")
    ranked = sorted(screen, key=lambda n: -screen[n][0])
    strong = []
    for name in ranked[:20]:
        md, mind, npos, mint, n = screen[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<28} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
            f"min_t {mint:+.2f} -> {v}")
    for name in screen:
        md, mind, npos, mint, n = screen[name]
        if npos == n and md >= 0.0015:
            strong.append(name)
    if strong:
        log(f"\nfresh-confirming {len(strong)} cells: {sorted(strong)[:12]}...")
        fresh = run_phase(grid, FRESH_SEEDS, "FRESH", keep=set(strong))
        log("\n----- (hp) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<28} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\n(hp) no cell screened promising (>=0.0015 & all-pos).")
    log(f"===== NIGHT (hp) DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
