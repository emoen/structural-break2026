"""Deliverable (c): Alphabot replication (2025 first place, adapted to our causal streaming metric).

Builds ~1000s of features across independent banks (51-col champion + 14 dependence + 137 Alphabot
primitives + 900 deterministic composites = 1102 wide) and stacks 8 diverse first-level models
across banks/objectives -> series-level OOF -> fixed-CDF -> meta-learner. Mirrors their
"diversity + leak-free stacking" edge rather than any single feature.

CAVEAT vs 2025: their competition was an offline pre/post batch; ours is causal (window-vs-history,
growing). So every feature here is computed on the causal trailing window only.

Cost control: the 1102-col feature build over ~1M val rows/seed is the bottleneck, so (c) runs a
REDUCED 2 search + 2 fresh seeds and a wall-clock BUDGET guard -- it stops gracefully and logs a
partial verdict if it would overrun the overnight window. Results append live to _night_c_results.log.
"""
import os
import time
import traceback
from datetime import datetime

from _night_banks import wide_blocks
from _night_stack import (
    champion_base, eval_stack, make_logger, paired, prepare_seed, summarize, verdict,
)

log = make_logger("_night_c_results.log")

SMOKE = os.environ.get("SMOKE") == "1"
C_SEARCH = [42, 7]
C_FRESH = [123, 31337] if not SMOKE else [2025]
# wall-clock budget (seconds) for the whole (c) run; guard stops before overrun.
BUDGET = float(os.environ.get("C_BUDGET", "1800" if SMOKE else "18000"))
_START = time.time()

B = wide_blocks()


def members8():
    """8 first-level models across banks and objectives (Alphabot-style diversity)."""
    return [
        dict(name="hgb_champdep", kind="hgb", cols=B["champ_dep"]),
        dict(name="lgbm_all", kind="lgbm", cols=B["all"], num_leaves=63, colsample=0.4),
        dict(name="lgbm_alpha", kind="lgbm", cols=B["alpha_all"], num_leaves=63, colsample=0.5),
        dict(name="et_champdep", kind="extratrees", cols=B["champ_dep"]),
        dict(name="et_prim", kind="extratrees", cols=B["alpha_prim"]),
        dict(name="rank_all", kind="ranker", cols=B["all"], obj="rank_xendcg"),
        dict(name="lambda_champdep", kind="ranker", cols=B["champ_dep"], obj="lambdarank"),
        dict(name="lgbm_comp", kind="lgbm", cols=B["alpha_comp"], num_leaves=63, colsample=0.3),
    ]


def configs():
    m = members8()
    return [
        dict(name="c_stack8_logit", members=m, meta="logit"),
        dict(name="c_stack8_gbm", members=m, meta="gbm"),
        dict(name="c_stack8_mean", members=m, meta="mean"),
    ]


def _time_left():
    return BUDGET - (time.time() - _START)


def run_phase(cfgs, seeds, phase):
    store = {c["name"]: [] for c in cfgs}
    for seed in seeds:
        if _time_left() < 0.15 * BUDGET:
            log(f"[{phase}] BUDGET guard: {_time_left():.0f}s left, skipping seed {seed}+")
            break
        t0 = time.time()
        S = prepare_seed(seed, "wide", rr_mode="champ")
        base = champion_base(S, seed)
        log(f"[{phase} seed {seed}] champion base={base.mean():.4f} "
            f"build+base {time.time()-t0:.0f}s, {_time_left():.0f}s left")
        for c in cfgs:
            if _time_left() < 0.05 * BUDGET:
                log(f"  [{phase} s{seed}] BUDGET guard: skipping {c['name']}+")
                break
            try:
                ts = time.time()
                aug = eval_stack(c["members"], c["meta"], S["Xtr"], S["ytr"], S["Ttr"],
                                 S["Str"], S["caches"], S["y_val"], seed)
                md, _s, t, _ = paired(aug, base)
                store[c["name"]].append((md, t))
                log(f"  [{phase} s{seed}] {c['name']:<16} stack={aug.mean():.4f} "
                    f"d={md:+.4f} t={t:+.2f} ({time.time()-ts:.0f}s, {_time_left():.0f}s left)")
            except Exception:
                log(f"  [{phase} s{seed}] {c['name']:<16} ERROR\n{traceback.format_exc()}")
        del S
    return summarize(store)


def main():
    log(f"\n===== NIGHT (c) Alphabot wide-stack {datetime.now():%F %T} "
        f"smoke={SMOKE} budget={BUDGET:.0f}s =====")
    cfgs = configs()
    log(f"screening {len(cfgs)} wide-stack variants on {C_SEARCH} (1102 feats, 8 members)")
    screen = run_phase(cfgs, C_SEARCH, "SCREEN")
    log("\n----- (c) SCREEN SUMMARY -----")
    strong = []
    for name in sorted(screen, key=lambda n: -screen[n][0]):
        md, mind, npos, mint, n = screen[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
            f"min_t {mint:+.2f} -> {v}")
        if npos == n and md >= 0.0015:
            strong.append(name)
    if strong and _time_left() > 0.2 * BUDGET:
        conf = [c for c in cfgs if c["name"] in strong]
        log(f"\nfresh-confirming {strong} on {C_FRESH} ({_time_left():.0f}s left)")
        fresh = run_phase(conf, C_FRESH, "FRESH")
        log("\n----- (c) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    elif not strong:
        log("\n(c) no variant screened promising.")
    else:
        log(f"\n(c) BUDGET guard: skipping fresh-confirm ({_time_left():.0f}s left).")
    log(f"===== NIGHT (c) DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
