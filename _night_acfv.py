"""Task 1 ship gate: does the pooled/late-offset ACF-VECTOR block lift the SHIPPED b_no_accf
champion (61 cols) under the real 0.6/0.4 HGB+ranker blend + TS-AUC, across seeds?

Base    = b_no_accf  = champ_base(51) + dep KEEP [0,1,2,3,4,5,10,11,12,13]  (61 cols)
Cand    = base + acfv (acfv_d1..d10, acfv_l2, acfv_l2_loc)                   (73 cols)
Screen on SEARCH_SEEDS, fresh-confirm STRONG on FRESH_SEEDS. Same gate as _night_b.
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
from _night_banks import NDEP, NACFV

log = make_logger("_night_acfv_results.log")

C = NCHAMP_BASE                       # 51
DEP_KEEP = [C + i for i in (0, 1, 2, 3, 4, 5, 10, 11, 12, 13)]   # b_no_accf dep cols
BASE_COLS = np.concatenate([np.arange(C), np.asarray(DEP_KEEP, dtype=int)])   # 61
ACFV_COLS = C + NDEP + np.arange(NACFV)                          # acfv block (cols 65..)
CAND_COLS = np.concatenate([BASE_COLS, ACFV_COLS.astype(int)])   # 61 + NACFV


def _score(cols, S, seed):
    return _eval(cols, S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"],
                 S["caches"], S["y_val"], seed)


def run_phase(seeds, phase):
    store = {"acfv": []}
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ_acfv")
        base = _score(BASE_COLS, S, seed)
        log(f"[{phase} seed {seed}] b_no_accf base={base.mean():.4f} ({time.time()-t0:.0f}s)")
        try:
            aug = _score(CAND_COLS, S, seed)
            md, _s, t, _ = paired(aug, base)
            store["acfv"].append((md, t))
            log(f"  [{phase} s{seed}] acfv  aug={aug.mean():.4f}  d={md:+.4f}  t={t:+.2f}")
        except Exception:
            log(f"  [{phase} s{seed}] acfv ERROR\n{traceback.format_exc()}")
        del S
    return summarize(store)


def main():
    log(f"\n===== TASK1 ACF-VECTOR gate {datetime.now():%F %T} "
        f"smoke={os.environ.get('SMOKE') == '1'} =====")
    screen = run_phase(SEARCH_SEEDS, "SCREEN")
    md, mind, npos, mint, n = screen["acfv"]
    v = verdict(md, mind, npos, mint, n)
    log(f"\n----- SCREEN: acfv mean_d={md:+.4f} min_d={mind:+.4f} "
        f"npos={npos}/{n} min_t={mint:+.2f} -> {v} -----")
    if v == "STRONG":
        log("SCREEN STRONG -> fresh confirm")
        conf = run_phase(FRESH_SEEDS, "CONFIRM")
        md, mind, npos, mint, n = conf["acfv"]
        v = verdict(md, mind, npos, mint, n)
        log(f"\n----- CONFIRM: acfv mean_d={md:+.4f} min_d={mind:+.4f} "
            f"npos={npos}/{n} min_t={mint:+.2f} -> {v} -----")
    log(f"\n===== TASK1 VERDICT: {v} =====")


if __name__ == "__main__":
    main()
