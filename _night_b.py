"""Deliverable (b): dependence-structure / early-onset lens.

Targets the EDA's documented blind spots -> latency on `ma`/`ar_coef` dependence breaks and the
strong-AR specificity leak. Adds 14 causal dependence features (studentized AR residual, early-onset
residual CUSUM, ACF-lag contrasts, arch contrast, sup-F/Chow scan) on top of the 51-col trail120
champion, and screens which sub-block lifts the shipped 0.6/0.4 HGB+ranker blend.

Baseline = champion (first 51 cols). Candidate = champion + dep sub-block, both scored by `_eval`.
Screen on 3 seeds, fresh-confirm STRONG on 4. Results append live to _night_b_results.log.
"""
import os
import time
import traceback
from datetime import datetime

import numpy as np

from _night_stack import (
    FRESH_SEEDS, SEARCH_SEEDS, NCHAMP_BASE,
    _eval, champion_base, make_logger, paired, prepare_seed, summarize, verdict,
)

log = make_logger("_night_b_results.log")

# dep columns live at 51..64 inside the 65-col champ_dep matrix
C = NCHAMP_BASE
STUDENT = [C + 0, C + 1]          # logvr, zstudent
CUSUM = [C + 2, C + 3, C + 4, C + 5]
ACCF = [C + 6, C + 7, C + 8, C + 9]
ARCH = [C + 10]
SUPF = [C + 11, C + 12, C + 13]


def _cfg(name, extra):
    return dict(name=name, cols=np.concatenate([np.arange(C), np.asarray(extra, dtype=int)]))


def configs():
    return [
        _cfg("b_all", STUDENT + CUSUM + ACCF + ARCH + SUPF),
        _cfg("b_student", STUDENT),
        _cfg("b_cusum", CUSUM),
        _cfg("b_student_cusum", STUDENT + CUSUM),
        _cfg("b_accf", ACCF),
        _cfg("b_supf", SUPF),
        _cfg("b_student_supf", STUDENT + SUPF),
        _cfg("b_no_accf", STUDENT + CUSUM + ARCH + SUPF),
    ]


def run_phase(cfgs, seeds, phase):
    store = {c["name"]: [] for c in cfgs}
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_dep")
        base = champion_base(S, seed)
        log(f"[{phase} seed {seed}] champion base={base.mean():.4f} ({time.time()-t0:.0f}s)")
        for c in cfgs:
            try:
                aug = _eval(c["cols"], S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"],
                            S["caches"], S["y_val"], seed)
                md, _s, t, _ = paired(aug, base)
                store[c["name"]].append((md, t))
                log(f"  [{phase} s{seed}] {c['name']:<18} aug={aug.mean():.4f} "
                    f"d={md:+.4f} t={t:+.2f}")
            except Exception:
                log(f"  [{phase} s{seed}] {c['name']:<18} ERROR\n{traceback.format_exc()}")
        del S
    return summarize(store)


def main():
    log(f"\n===== NIGHT (b) dependence/early-onset lens {datetime.now():%F %T} "
        f"smoke={os.environ.get('SMOKE') == '1'} =====")
    cfgs = configs()
    log(f"screening {len(cfgs)} dep variants on {SEARCH_SEEDS}")
    screen = run_phase(cfgs, SEARCH_SEEDS, "SCREEN")
    log("\n----- (b) SCREEN SUMMARY -----")
    strong = []
    for name in sorted(screen, key=lambda n: -screen[n][0]):
        md, mind, npos, mint, n = screen[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<18} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
            f"min_t {mint:+.2f} -> {v}")
        if npos == n and md >= 0.0015:
            strong.append(name)
    if strong:
        conf = [c for c in cfgs if c["name"] in strong]
        log(f"\nfresh-confirming {strong} on {FRESH_SEEDS}")
        fresh = run_phase(conf, FRESH_SEEDS, "FRESH")
        log("\n----- (b) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<18} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\n(b) no variant screened promising.")
    log(f"===== NIGHT (b) DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
