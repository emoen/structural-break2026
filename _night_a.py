"""Deliverable (a): model-diversity stacking upgrade.

We currently ship a fixed 2-model (HGB + LGBMRanker) 0.6/0.4 blend. The 2025 first place ran 8
diverse first-level models + a meta-learner. This experiment keeps the SAME 51-col champion feature
set but replaces the fixed blend with a leak-free stack: 5 diverse members (HGB, LGBM-binary,
ExtraTrees, rank_xendcg ranker, lambdarank ranker) -> series-level OOF -> fixed-CDF -> meta-learner.

Baseline = shipped champion blend. Candidate = eval_stack(...). Screen 3 seeds, fresh-confirm 4.
Results append live to _night_a_results.log.
"""
import os
import time
import traceback
from datetime import datetime

import numpy as np

from _night_stack import (
    FRESH_SEEDS, SEARCH_SEEDS, NCHAMP_BASE,
    champion_base, eval_stack, make_logger, paired, prepare_seed, summarize, verdict,
)

log = make_logger("_night_a_results.log")
CH = np.arange(NCHAMP_BASE)   # all 51 champion cols for every member


def members5():
    return [
        dict(name="hgb", kind="hgb", cols=CH),
        dict(name="lgbm", kind="lgbm", cols=CH, num_leaves=31, colsample=0.7),
        dict(name="extratrees", kind="extratrees", cols=CH),
        dict(name="rank_xendcg", kind="ranker", cols=CH, obj="rank_xendcg"),
        dict(name="lambdarank", kind="ranker", cols=CH, obj="lambdarank"),
    ]


def members3():
    return [
        dict(name="hgb", kind="hgb", cols=CH),
        dict(name="lgbm", kind="lgbm", cols=CH, num_leaves=31, colsample=0.7),
        dict(name="rank_xendcg", kind="ranker", cols=CH, obj="rank_xendcg"),
    ]


def members_clf3():
    return [
        dict(name="hgb", kind="hgb", cols=CH),
        dict(name="lgbm", kind="lgbm", cols=CH, num_leaves=31, colsample=0.7),
        dict(name="extratrees", kind="extratrees", cols=CH),
    ]


def configs():
    return [
        dict(name="a_stack5_logit", members=members5(), meta="logit"),
        dict(name="a_stack5_mean", members=members5(), meta="mean"),
        dict(name="a_stack5_gbm", members=members5(), meta="gbm"),
        dict(name="a_stack3_logit", members=members3(), meta="logit"),
        dict(name="a_clf3_logit", members=members_clf3(), meta="logit"),
    ]


def run_phase(cfgs, seeds, phase):
    store = {c["name"]: [] for c in cfgs}
    for seed in seeds:
        t0 = time.time()
        S = prepare_seed(seed, "champ")
        base = champion_base(S, seed)
        log(f"[{phase} seed {seed}] champion base={base.mean():.4f} ({time.time()-t0:.0f}s)")
        for c in cfgs:
            try:
                ts = time.time()
                aug = eval_stack(c["members"], c["meta"], S["Xtr"], S["ytr"], S["Ttr"],
                                 S["Str"], S["caches"], S["y_val"], seed)
                md, _s, t, _ = paired(aug, base)
                store[c["name"]].append((md, t))
                log(f"  [{phase} s{seed}] {c['name']:<18} stack={aug.mean():.4f} "
                    f"d={md:+.4f} t={t:+.2f} ({time.time()-ts:.0f}s)")
            except Exception:
                log(f"  [{phase} s{seed}] {c['name']:<18} ERROR\n{traceback.format_exc()}")
        del S
    return summarize(store)


def main():
    log(f"\n===== NIGHT (a) model-diversity stack {datetime.now():%F %T} "
        f"smoke={os.environ.get('SMOKE') == '1'} =====")
    cfgs = configs()
    log(f"screening {len(cfgs)} stack variants on {SEARCH_SEEDS}")
    screen = run_phase(cfgs, SEARCH_SEEDS, "SCREEN")
    log("\n----- (a) SCREEN SUMMARY -----")
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
        log("\n----- (a) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<18} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\n(a) no variant screened promising.")
    log(f"===== NIGHT (a) DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
