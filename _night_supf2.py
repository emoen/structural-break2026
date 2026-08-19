"""Deliverable (b) v3: sup-F / Chow SCAN refinement on the shipped E62 champion.

sup-F is the one dependence-structure feature that PAID (it rode into the shipped b_no_accf), but
the shipped version is a coarse (~40-point) scan on the RAW window only. This sharpens that same
detector three ways, each isolable, on top of the NOW-SHIPPED 61-col champion:

  RAW   : a FINE full-grid scan + the variance-split position the coarse one omits
  RESID : the same Chow scan on the AR(8) RESIDUAL (productive residual formulation -> residual break)
  AGG   : Andrews-Ploberger ave-F / exp-F aggregations (stronger vs diffuse/gradual breaks)

Baseline = shipped 61-col champion (`_eval` over champ+kept-dep cols). Candidate = baseline + supf2
block, both scored by the shipped 0.6/0.4 blend. Screen {42,7,123} -> fresh {31337,2025,555,99}.
Results append live to _night_supf2_results.log.
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

log = make_logger("_night_supf2_results.log")

C = NCHAMP_BASE                    # 51 champion cols
CDEP = C                           # dep bank at 51..64
C2 = C + NDEP                      # supf2 bank starts at 65

# shipped 61-col champion = champion(51) + kept dep (STUDENT+CUSUM+ARCH+SUPF, drop ACCF 57-60)
KEPT_DEP = [CDEP + 0, CDEP + 1, CDEP + 2, CDEP + 3, CDEP + 4, CDEP + 5,
            CDEP + 10, CDEP + 11, CDEP + 12, CDEP + 13]
BASE61 = np.concatenate([np.arange(C), np.asarray(KEPT_DEP, dtype=int)])

# supf2 blocks (order matches supf2_feats output)
RAW = [C2 + 0, C2 + 1, C2 + 2]      # raw_t, raw_meanpos, raw_varpos
RESID = [C2 + 3, C2 + 4, C2 + 5]    # resid_t, resid_var, resid_pos
AGG = [C2 + 6, C2 + 7]              # avef, expf


def _cfg(name, extra):
    return dict(name=name, cols=np.concatenate([BASE61, np.asarray(extra, dtype=int)]))


def configs():
    return [
        _cfg("s2_raw", RAW),
        _cfg("s2_resid", RESID),
        _cfg("s2_agg", AGG),
        _cfg("s2_resid_raw", RESID + RAW),
        _cfg("s2_resid_agg", RESID + AGG),
        _cfg("s2_all", RAW + RESID + AGG),
    ]


def base61_eval(S, seed):
    return _eval(BASE61, S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"],
                 S["caches"], S["y_val"], seed)


def run_phase(cfgs, seeds, phase):
    store = {c["name"]: [] for c in cfgs}
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_supf2")
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
    log(f"\n===== NIGHT (b) v3 sup-F refinement {datetime.now():%F %T} "
        f"smoke={os.environ.get('SMOKE') == '1'} =====")
    cfgs = configs()
    log(f"baseline = shipped 61-col champion; screening {len(cfgs)} supf2 variants on {SEARCH_SEEDS}")
    screen = run_phase(cfgs, SEARCH_SEEDS, "SCREEN")
    log("\n----- (b v3) SCREEN SUMMARY -----")
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
        log("\n----- (b v3) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\n(b v3) no variant screened promising.")
    log(f"===== NIGHT (b) v3 DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
