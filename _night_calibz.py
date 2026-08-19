"""Lever #2 refinement (E69): robust, z-ONLY cross-sectional calibration.

E68 found the studentized z on all 51 champion features positive on 7/7 seeds but sub-threshold
(FRESH +0.0019, min_t +0.03); the rank readout HURT and the whole beat every subset. Two changes
here try to lift t over the bar by cutting per-seed noise, NOT by adding features:
  1. z-ONLY (drop rank entirely -- confirmed harmful in E68).
  2. a better null estimate: 32 windows (up from 20) and a ROBUST center/scale (median / MAD) that
     the heavy-tailed champion features do not drag.
The calibz bank emits BOTH readouts (mean/sd z and robust median/MAD z) so one run isolates the
estimator effect. Baseline = shipped 61-col champion; candidate = baseline + a calibrated block, both
scored by the shipped 0.6/0.4 blend. Screen {42,7,123} -> fresh {31337,2025,555,99}. Results append
live to _night_calibz_results.log.
"""
import os
import time
import traceback
from datetime import datetime

import numpy as np

from _night_stack import (
    FRESH_SEEDS, SEARCH_SEEDS, NCHAMP_BASE,
    _eval, make_logger, paired, prepare_seed, summarize, verdict,
)
from _night_banks import NDEP

log = make_logger("_night_calibz_results.log")

C = NCHAMP_BASE                    # 51 champion cols
CDEP = C                           # dep bank at 51..64
CMS = C + NDEP                     # z_meansd block starts at 65
CRB = CMS + NCHAMP_BASE           # z_robust block starts at 116

KEPT_DEP = [CDEP + 0, CDEP + 1, CDEP + 2, CDEP + 3, CDEP + 4, CDEP + 5,
            CDEP + 10, CDEP + 11, CDEP + 12, CDEP + 13]
BASE61 = np.concatenate([np.arange(C), np.asarray(KEPT_DEP, dtype=int)])

CORE = [s * 15 + f for f in (0, 1, 2, 7, 13) for s in range(3)]   # 15 location/scale-shift contrasts
SCALESHIFT = [s * 15 + f for f in (0, 1) for s in range(3)]       # 6  (mean_z, log_var_ratio)
ALLF = list(range(NCHAMP_BASE))                                   # 51


def ms(idxs):
    return [CMS + i for i in idxs]


def rb(idxs):
    return [CRB + i for i in idxs]


def _cfg(name, extra):
    return dict(name=name, cols=np.concatenate([BASE61, np.asarray(extra, dtype=int)]))


def configs():
    return [
        _cfg("g_zms_all", ms(ALLF)),              # 51 mean/sd z @ 32 wins (E68 control, re-est.)
        _cfg("g_zr_all", rb(ALLF)),               # 51 robust median/MAD z  <- the hypothesis
        _cfg("g_zr_core", rb(CORE)),              # 15 robust core contrasts
        _cfg("g_zr_scaleshift", rb(SCALESHIFT)),  # 6  robust location/scale
        _cfg("g_zms_zr", ms(ALLF) + rb(ALLF)),    # 102 both z readouts stacked
    ]


def base61_eval(S, seed):
    return _eval(BASE61, S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"],
                 S["caches"], S["y_val"], seed)


def run_phase(cfgs, seeds, phase):
    store = {c["name"]: [] for c in cfgs}
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_calibz")
        base = base61_eval(S, seed)
        log(f"[{phase} seed {seed}] champion(61) base={base.mean():.4f} ({time.time()-t0:.0f}s)")
        for c in cfgs:
            try:
                aug = _eval(c["cols"], S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"],
                            S["caches"], S["y_val"], seed)
                md, _s, t, _ = paired(aug, base)
                store[c["name"]].append((md, t))
                log(f"  [{phase} s{seed}] {c['name']:<16} aug={aug.mean():.4f} "
                    f"d={md:+.4f} t={t:+.2f}")
            except Exception:
                log(f"  [{phase} s{seed}] {c['name']:<16} ERROR\n{traceback.format_exc()}")
        del S
    return summarize(store)


def main():
    log(f"\n===== NIGHT (gz) robust z-only calibration {datetime.now():%F %T} "
        f"smoke={os.environ.get('SMOKE') == '1'} =====")
    cfgs = configs()
    log(f"baseline = shipped 61-col champion; screening {len(cfgs)} robust-calib variants on {SEARCH_SEEDS}")
    screen = run_phase(cfgs, SEARCH_SEEDS, "SCREEN")
    log("\n----- (gz) SCREEN SUMMARY -----")
    strong = []
    for name in sorted(screen, key=lambda n: -screen[n][0]):
        md, mind, npos, mint, n = screen[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
            f"min_t {mint:+.2f} -> {v}")
        if npos == n and md >= 0.0015:
            strong.append(name)
    if strong:
        conf = [c for c in cfgs if c["name"] in strong]
        log(f"\nfresh-confirming {strong} on {FRESH_SEEDS}")
        fresh = run_phase(conf, FRESH_SEEDS, "FRESH")
        log("\n----- (gz) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\n(gz) no variant screened promising.")
    log(f"===== NIGHT (gz) DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
