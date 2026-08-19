"""Deliverable (e): causal SEQUENTIAL change-point detector bank -- attack the early-detection cliff.

The extended EDA (_eda_early.py) showed the champion loses ~27 AUC points from a clean window
(0.97) to the just-after-break regime (0.71 at 10 post-break points), and the per-time-online
metric averages over MANY of those low-post steps -> that boundary regime is where the head-room
is. Adds online change-point statistics on the STUDENTIZED AR8 residual, read out with recency /
endpoint emphasis so they fire on FEW post-break points, on top of the NOW-SHIPPED 61-col champion:

  BB     : Brownian-bridge CUSUM of the residual (mean-shift)         -- bb_max, bb_min, bb_pos
  PH     : Page-Hinkley on the variance surprise (rs^2-1)             -- ph_max, ph_end
  EWMA   : EWMA of variance surprise, fast/slow decay (recent-weight) -- ewma_fend/fmax/send
  RECENT : recency short-window residual z (last 30/60 pts)           -- rv30_z, rv60_z, rmean30_z

Baseline = shipped 61-col champion (`_eval` over champ+kept-dep cols). Candidate = baseline + seq
block, both scored by the shipped 0.6/0.4 blend. Screen {42,7,123} -> fresh {31337,2025,555,99}.
Results append live to _night_e_results.log.
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

log = make_logger("_night_e_results.log")

C = NCHAMP_BASE                    # 51 champion cols
CDEP = C                           # dep bank at 51..64
C2 = C + NDEP                      # seq bank starts at 65

# shipped 61-col champion = champion(51) + kept dep (STUDENT+CUSUM+ARCH+SUPF, drop ACCF 57-60)
KEPT_DEP = [CDEP + 0, CDEP + 1, CDEP + 2, CDEP + 3, CDEP + 4, CDEP + 5,
            CDEP + 10, CDEP + 11, CDEP + 12, CDEP + 13]
BASE61 = np.concatenate([np.arange(C), np.asarray(KEPT_DEP, dtype=int)])

# seq blocks (order matches seq_feats output)
BB = [C2 + 0, C2 + 1, C2 + 2]       # bb_max, bb_min, bb_pos
PH = [C2 + 3, C2 + 4]               # ph_max, ph_end
EWMA = [C2 + 5, C2 + 6, C2 + 7]     # ewma_fend, ewma_fmax, ewma_send
RECENT = [C2 + 8, C2 + 9, C2 + 10]  # rv30_z, rv60_z, rmean30_z


def _cfg(name, extra):
    return dict(name=name, cols=np.concatenate([BASE61, np.asarray(extra, dtype=int)]))


def configs():
    return [
        _cfg("e_bb", BB),
        _cfg("e_ph", PH),
        _cfg("e_ewma", EWMA),
        _cfg("e_recent", RECENT),
        _cfg("e_ph_recent", PH + RECENT),
        _cfg("e_ewma_recent", EWMA + RECENT),
        _cfg("e_all", BB + PH + EWMA + RECENT),
    ]


def base61_eval(S, seed):
    return _eval(BASE61, S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"],
                 S["caches"], S["y_val"], seed)


def run_phase(cfgs, seeds, phase):
    store = {c["name"]: [] for c in cfgs}
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_seq")
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
    log(f"\n===== NIGHT (e) sequential-detector bank {datetime.now():%F %T} "
        f"smoke={os.environ.get('SMOKE') == '1'} =====")
    cfgs = configs()
    log(f"baseline = shipped 61-col champion; screening {len(cfgs)} seq variants on {SEARCH_SEEDS}")
    screen = run_phase(cfgs, SEARCH_SEEDS, "SCREEN")
    log("\n----- (e) SCREEN SUMMARY -----")
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
        log("\n----- (e) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\n(e) no variant screened promising.")
    log(f"===== NIGHT (e) DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
