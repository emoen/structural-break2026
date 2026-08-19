"""Deliverable (b) v2: dependence-STRUCTURE lens, mined after b_no_accf shipped (LB 62.48%).

The shipped champion (b_no_accf) owns residual-*variance* change + accumulation + a mean/var
split scan, but not a change in the dependence STRUCTURE itself when residual variance is ~fixed
(the ma/ar_coef axis). This screens five orthogonal dependence-structure blocks on top of the
NOW-SHIPPED 61-col champion (champion 51 + kept dep 10), each isolable so we learn which pay:

  ARCOEF : window-vs-history AR(8) coefficient-vector distance  (targets ar_coef breaks)
  PACF   : partial-autocorrelation direct-lag contrasts         (cleaner than the noise ACF)
  LJUNG  : Ljung-Box portmanteau of the historical-AR residual  (whiteness / mis-specification)
  VR     : Lo-MacKinlay multi-horizon variance-ratio profile    (MA-vs-AR signature -> ma breaks)
  HURST  : DFA long-memory exponent shift

Baseline = the shipped 61-col champion (`_eval` over champ+kept-dep cols). Candidate = baseline +
dep2 block, both scored by the shipped 0.6/0.4 HGB+ranker blend. Screen {42,7,123} ->
fresh-confirm STRONG on {31337,2025,555,99}. Results append live to _night_bv2_results.log.
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

log = make_logger("_night_bv2_results.log")

C = NCHAMP_BASE                    # 51 champion cols
CDEP = C                           # dep bank starts at 51
C2 = C + NDEP                      # dep2 bank starts at 65

# shipped 61-col champion = champion(51) + kept dep (STUDENT+CUSUM+ARCH+SUPF, drop ACCF 57-60)
KEPT_DEP = [CDEP + 0, CDEP + 1, CDEP + 2, CDEP + 3, CDEP + 4, CDEP + 5,
            CDEP + 10, CDEP + 11, CDEP + 12, CDEP + 13]
BASE61 = np.concatenate([np.arange(C), np.asarray(KEPT_DEP, dtype=int)])

# dep2 blocks (order matches dep2_feats output)
ARCOEF = [C2 + 0, C2 + 1, C2 + 2]
PACF = [C2 + 3, C2 + 4, C2 + 5]
LJUNG = [C2 + 6, C2 + 7, C2 + 8]
VR = [C2 + 9, C2 + 10, C2 + 11, C2 + 12]
HURST = [C2 + 13]
ALL2 = ARCOEF + PACF + LJUNG + VR + HURST


def _cfg(name, extra):
    return dict(name=name, cols=np.concatenate([BASE61, np.asarray(extra, dtype=int)]))


def configs():
    return [
        _cfg("v2_arcoef", ARCOEF),
        _cfg("v2_pacf", PACF),
        _cfg("v2_ljung", LJUNG),
        _cfg("v2_vr", VR),
        _cfg("v2_hurst", HURST),
        _cfg("v2_all", ALL2),
        _cfg("v2_arcoef_vr", ARCOEF + VR),
        _cfg("v2_arcoef_ljung", ARCOEF + LJUNG),
    ]


def base61_eval(S, seed):
    return _eval(BASE61, S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"],
                 S["caches"], S["y_val"], seed)


def run_phase(cfgs, seeds, phase):
    store = {c["name"]: [] for c in cfgs}
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_dep2")
        base = base61_eval(S, seed)
        log(f"[{phase} seed {seed}] champion(61) base={base.mean():.4f} ({time.time()-t0:.0f}s)")
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
    log(f"\n===== NIGHT (b) v2 dependence-structure lens {datetime.now():%F %T} "
        f"smoke={os.environ.get('SMOKE') == '1'} =====")
    cfgs = configs()
    log(f"baseline = shipped 61-col champion; screening {len(cfgs)} dep2 variants on {SEARCH_SEEDS}")
    screen = run_phase(cfgs, SEARCH_SEEDS, "SCREEN")
    log("\n----- (b v2) SCREEN SUMMARY -----")
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
        log("\n----- (b v2) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<18} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\n(b v2) no variant screened promising.")
    log(f"===== NIGHT (b) v2 DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
