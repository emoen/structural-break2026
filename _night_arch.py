"""E71: architecture exploration -- calib-specialist member, model diversity, meta-learners.

Open-ended search for an architecture that beats the shipped 2-member fixed-CDF champion
(0.6*F_hgb + 0.4*F_rank on the 61 cols). Uses the leak-free `eval_stack` (series-CV OOF ->
fixed-CDF -> meta). Given the strong NEGATIVE precedent (E63 5-member stack negative; E64
8-model+meta flipped on fresh), this run is deliberately SMALL and hypothesis-driven rather than a
broad sweep. Every stack is paired vs the SHIPPED champion (`_eval` on base61, w=0.6) per fold/seed.

Stacks under test (champ_calibz mode; calib block = E69 robust-z, cols 116..166):
  A calib_spec_mean  : {hgb@61, rank@61, hgb@calibz} + mean   -> add a calibration SPECIALIST member
  B calib_spec_logit : {hgb@61, rank@61, hgb@calibz} + logit  -> let a meta-learner weight the 3
  C fuse_aug_mean    : {hgb@aug, rank@aug} + mean             -> feature-fusion instead of specialist
  D blend_logit      : {hgb@61, rank@61} + logit             -> "learn the blend weight" meta control
  E diverse_logit    : {hgb@61, rank@61, extratrees@aug} + logit -> model diversity + meta
Screen {42,7,123} -> fresh {31337,2025,555,99}. Results append to _night_arch_results.log.

NOTE: calib block defaults to E69 robust-z (116..166). Update CALIB_COLS from the E69 fresh winner.
"""
import os
import time
import traceback
from datetime import datetime

import numpy as np

from _night_stack import (
    FRESH_SEEDS, SEARCH_SEEDS, NCHAMP_BASE,
    _eval, eval_stack, make_logger, paired, prepare_seed, summarize, verdict,
)
from _night_banks import NDEP

C = NCHAMP_BASE
CDEP = C
CMS = C + NDEP
CRB = CMS + NCHAMP_BASE

KEPT_DEP = [CDEP + i for i in (0, 1, 2, 3, 4, 5, 10, 11, 12, 13)]
BASE61 = np.concatenate([np.arange(C), np.asarray(KEPT_DEP, dtype=int)])
CALIB_COLS = np.asarray(list(range(CMS, CMS + NCHAMP_BASE)), dtype=int)   # E69 survivor z_meansd (robust-z was dead)
AUG = np.concatenate([BASE61, CALIB_COLS])


def stacks():
    hgb61 = dict(name="hgb61", kind="hgb", cols=BASE61)
    rank61 = dict(name="rank61", kind="ranker", cols=BASE61, obj="rank_xendcg")
    hgbcal = dict(name="hgbcal", kind="hgb", cols=CALIB_COLS)
    hgbaug = dict(name="hgbaug", kind="hgb", cols=AUG)
    rankaug = dict(name="rankaug", kind="ranker", cols=AUG, obj="rank_xendcg")
    etaug = dict(name="etaug", kind="extratrees", cols=AUG, min_samples_leaf=20)
    return [
        dict(name="A_calib_spec_mean", members=[hgb61, rank61, hgbcal], meta="mean"),
        dict(name="B_calib_spec_logit", members=[hgb61, rank61, hgbcal], meta="logit"),
        dict(name="C_fuse_aug_mean", members=[hgbaug, rankaug], meta="mean"),
        dict(name="D_blend_logit", members=[hgb61, rank61], meta="logit"),
        dict(name="E_diverse_logit", members=[hgb61, rank61, etaug], meta="logit"),
    ]


def run_phase(cfgs, seeds, phase):
    store = {c["name"]: [] for c in cfgs}
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_calibz")
        base = _eval(BASE61, S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"],
                     S["caches"], S["y_val"], seed)
        log(f"[{phase} seed {seed}] shipped champion base={base.mean():.4f} ({time.time()-t0:.0f}s)")
        for c in cfgs:
            try:
                aug = eval_stack(c["members"], c["meta"], S["Xtr"], S["ytr"], S["Ttr"], S["Str"],
                                 S["caches"], S["y_val"], seed)
                md, _s, t, _ = paired(aug, base)
                store[c["name"]].append((md, t))
                log(f"  [{phase} s{seed}] {c['name']:<20} stack={aug.mean():.4f} "
                    f"d={md:+.4f} t={t:+.2f}")
            except Exception:
                log(f"  [{phase} s{seed}] {c['name']:<20} ERROR\n{traceback.format_exc()}")
        del S
    return summarize(store)


log = make_logger("_night_arch_results.log")


def main():
    log(f"\n===== NIGHT (arch) architecture exploration {datetime.now():%F %T} "
        f"smoke={os.environ.get('SMOKE') == '1'} =====")
    cfgs = stacks()
    log(f"baseline = shipped 2-member fixed-CDF champion; screening {len(cfgs)} stacks on {SEARCH_SEEDS}")
    screen = run_phase(cfgs, SEARCH_SEEDS, "SCREEN")
    log("\n----- (arch) SCREEN SUMMARY -----")
    strong = []
    for name in sorted(screen, key=lambda n: -screen[n][0]):
        md, mind, npos, mint, n = screen[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<20} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
            f"min_t {mint:+.2f} -> {v}")
        if npos == n and md >= 0.0015:
            strong.append(name)
    if strong:
        conf = [c for c in cfgs if c["name"] in strong]
        log(f"\nfresh-confirming {strong} on {FRESH_SEEDS}")
        fresh = run_phase(conf, FRESH_SEEDS, "FRESH")
        log("\n----- (arch) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<20} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\n(arch) no stack screened promising.")
    log(f"===== NIGHT (arch) DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
