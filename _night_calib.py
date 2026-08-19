"""Lever #2: cross-sectional calibration bank -- studentize / rank every champion base feature
against the SERIES' OWN pre-break null band, so a fixed GBT threshold means the same thing across
series (comparable under the cross-sectional TS-AUC metric that ranks series against each other at
every online step). Generalizes the E62 studentized AR8-residual win (+0.90 LB) to the full 51-col
champion base row.

For each series we build a null band of each of the 51 champion base features from sliding pre-break
windows, then emit two calibrated readouts of the online row:
  z    : studentized  (value - null_mean) / null_sd     -- 'k sigma above THIS series own null'
  rank : empirical null-CDF percentile in [0,1]          -- rank-normalized, null-shape robust

Baseline = shipped 61-col champion. Candidate = baseline + a calibrated block, both scored by the
shipped 0.6/0.4 blend. Screen {42,7,123} -> fresh {31337,2025,555,99}. Results append live to
_night_calib_results.log.
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

log = make_logger("_night_calib_results.log")

C = NCHAMP_BASE                    # 51 champion cols
CDEP = C                           # dep bank at 51..64
CZ = C + NDEP                      # calib z-block starts at 65
CR = CZ + NCHAMP_BASE             # calib rank-block starts at 116

# shipped 61-col champion = champion(51) + kept dep (STUDENT+CUSUM+ARCH+SUPF, drop ACCF 57-60)
KEPT_DEP = [CDEP + 0, CDEP + 1, CDEP + 2, CDEP + 3, CDEP + 4, CDEP + 5,
            CDEP + 10, CDEP + 11, CDEP + 12, CDEP + 13]
BASE61 = np.concatenate([np.arange(C), np.asarray(KEPT_DEP, dtype=int)])

# champion_feat layout: 3 scales x 15 feats (feat f at scale s -> s*15+f); AR residual at 45..50.
# core = location/scale-shift contrasts most helped by per-series calibration:
#   0 mean_z, 1 log_var_ratio, 2 ks, 7 iqr_ratio, 13 rng_ratio
CORE = [s * 15 + f for f in (0, 1, 2, 7, 13) for s in range(3)]   # 15
SCALESHIFT = [s * 15 + f for f in (0, 1) for s in range(3)]       # 6  (mean_z, log_var_ratio)
RESID = [45, 46, 47, 48, 49, 50]                                 # 6  AR8 residual var-ratios
ALLF = list(range(NCHAMP_BASE))                                   # 51


def zc(idxs):
    return [CZ + i for i in idxs]


def rc(idxs):
    return [CR + i for i in idxs]


def _cfg(name, extra):
    return dict(name=name, cols=np.concatenate([BASE61, np.asarray(extra, dtype=int)]))


def configs():
    return [
        _cfg("g_z_scaleshift", zc(SCALESHIFT)),           # 6  purest location/scale z
        _cfg("g_z_core", zc(CORE)),                       # 15 core contrasts z
        _cfg("g_z_resid", zc(RESID)),                     # 6  residual var-ratio z
        _cfg("g_z_all", zc(ALLF)),                        # 51 full z
        _cfg("g_rank_core", rc(CORE)),                    # 15 core rank
        _cfg("g_zr_core", zc(CORE) + rc(CORE)),           # 30 core z + rank
        _cfg("g_all", zc(ALLF) + rc(ALLF)),               # 102 full z + rank
    ]


def base61_eval(S, seed):
    return _eval(BASE61, S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"],
                 S["caches"], S["y_val"], seed)


def run_phase(cfgs, seeds, phase):
    store = {c["name"]: [] for c in cfgs}
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_calib")
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
    log(f"\n===== NIGHT (g) cross-sectional calibration bank {datetime.now():%F %T} "
        f"smoke={os.environ.get('SMOKE') == '1'} =====")
    cfgs = configs()
    log(f"baseline = shipped 61-col champion; screening {len(cfgs)} calib variants on {SEARCH_SEEDS}")
    screen = run_phase(cfgs, SEARCH_SEEDS, "SCREEN")
    log("\n----- (g) SCREEN SUMMARY -----")
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
        log("\n----- (g) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\n(g) no variant screened promising.")
    log(f"===== NIGHT (g) DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
