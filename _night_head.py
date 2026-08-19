"""E72 (lever #3): target / aggregation head on the champion.

Every recent axis changed the FEATURES (E52-E67), the CALIBRATION (E68/E69), the HGB
HYPERPARAMS (E70) or the ARCHITECTURE (E71). Lever #3 is the one axis left in the strategy
checkpoint: the TRAINING TARGET / SAMPLE-WEIGHTING of the temporal head. The champion trains
its 0.6 HGB member as a plain binary classifier (label = 1 iff online step t >= tau) with unit
weights, blended 0.6/0.4 with a rank_xendcg ranker over fixed train-CDF percentiles.

The head-room (per the checkpoint's `_eda_early.py`) is the early-detection cliff: pooled AUC is
~0.71 at 10 post-break points, ~0.79 at 20, ~0.88 at 40, vs ~0.97 on the full window. The metric
averages over the many low-post steps just after tau. This run attacks that regime from the LABEL
side (not the feature side, which E67 closed): give the HGB more incentive/capacity on the hard
just-after-tau rows, without adding any information.

Two mechanism families, all on the SHIPPED 61-col champion features (only the head changes):
  (A) sample-weighting the classifier  -- up-weight rows by post-break offset o = t - tau
        we_*   : w = 1 + A*exp(-o/S) for positives (o>=0), 1 else  -> emphasise just-after-tau
        wb_*   : w = 1 + A*exp(-|o|/S)                             -> emphasise the tau boundary
  (B) soft / hazard target via a regressor (raw output feeds the same fixed-CDF rank blend)
        haz_S  : y = 1 - exp(-max(o,0)/S)   monotone hazard ramp (0 pre-break)
        ramp_H : y = clip(o/H, 0, 1)        E26's ramp, re-baselined on the CURRENT champion

PRECEDENT / GUARD (E26): a ramp target `clip((t-tau)/H,0,1)` and a *deweight*-near-tau scheme were
already NOISE on the OLD 45-col champion. So `ramp_H` is a deliberate re-baseline, and the novel
bet is the OPPOSITE-SIGN `we_*` (UP-weight, not deweight, the early positives) plus the hazard
regressor -- the specifically-untested variants for the cliff.

Ranker is left EXACTLY as shipped (it is already a ranking/AUC-shaped objective); only the HGB head
varies. Blend weight fixed at 0.6 (E70 confirmed near-optimal). Every variant is paired vs the
shipped champion (ctrl = tuned HGB, unit weights). Screen {42,7,123} -> fresh {31337,2025,555,99}.
Results append to _night_head_results.log.
"""
import os
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingRegressor

from _night_stack import (
    FRESH_SEEDS, SEARCH_SEEDS, NCHAMP_BASE,
    make_logger, paired, prepare_seed, summarize, verdict,
)
from _ensemble_tune import load_pool_strat, make_hgb
from _residual_overnight import _score, pct, TUNED
from _lambdarank import fit_ranker, sort_into_groups
from _exp_fast import ts_auc_from

log = make_logger("_night_head_results.log")

C = NCHAMP_BASE
CDEP = C
# Shipped 61-col champion = champ(51) + dep-kept(10) in the champ_calibz layout (same as E70).
KEPT_DEP = [CDEP + i for i in (0, 1, 2, 3, 4, 5, 10, 11, 12, 13)]
BASE61 = np.concatenate([np.arange(C), np.asarray(KEPT_DEP, dtype=int)])

BLEND_W = 0.60


# ---------------------------------------------------------------------------
# offset reconstruction: o = t - tau per training row (from Ttr / Str + yi)
# ---------------------------------------------------------------------------
def prep(seed):
    """prepare_seed(champ_calibz) + the per-row post-break offset o = t - tau.

    prepare_seed sets ET.SEED=seed and calls load_pool_strat(); we re-call it (deterministic on
    the same SEED) purely to recover tau_index per series, then map through Str.
    """
    S = prepare_seed(seed, "champ_calibz")
    _X, yi, _yv, train_ids, _folds = load_pool_strat()
    tau = {int(sid): int(yi.loc[sid, "tau_index"]) for sid in train_ids}
    NEG = np.int64(-10**9)  # no-break series (tau < 0): treat every row as far pre-break
    off = np.asarray(
        [int(t) - tau[int(s)] if tau[int(s)] >= 0 else NEG for t, s in zip(S["Ttr"], S["Str"])],
        dtype=np.int64)
    # consistency: label==1 exactly where the row is at/after a real tau
    assert np.array_equal((off >= 0).astype(np.int64), S["ytr"].astype(np.int64)), \
        "offset sign disagrees with ytr -- tau/Str misalignment"
    return S, off


# ---------------------------------------------------------------------------
# head configs: (A) classifier sample-weight  (B) regressor soft target
# ---------------------------------------------------------------------------
def w_early(A, s):
    def f(off, y):
        w = np.ones(len(y), dtype=np.float64)
        pos = off >= 0
        w[pos] = 1.0 + A * np.exp(-off[pos] / s)
        return w
    return f


def w_boundary(A, s):
    def f(off, y):
        return 1.0 + A * np.exp(-np.abs(off) / s)
    return f


def haz(s):
    def f(off):
        return 1.0 - np.exp(-np.clip(off, 0, None) / s)
    return f


def ramp(H):
    def f(off):
        return np.clip(off / H, 0.0, 1.0)
    return f


def head_grid():
    return [
        dict(name="ctrl",     kind="clf", fn=None),                 # unit weights == shipped
        dict(name="we_a2s20", kind="clf", fn=w_early(2.0, 20.0)),
        dict(name="we_a4s12", kind="clf", fn=w_early(4.0, 12.0)),
        dict(name="we_a1s40", kind="clf", fn=w_early(1.0, 40.0)),
        dict(name="wb_a2s15", kind="clf", fn=w_boundary(2.0, 15.0)),
        dict(name="haz_s20",  kind="reg", fn=haz(20.0)),
        dict(name="ramp_h40", kind="reg", fn=ramp(40.0)),
    ]


def make_reg(seed):
    p = dict(early_stopping=True)
    p.update(TUNED)
    return HistGradientBoostingRegressor(random_state=seed, **p)


def _score_reg(model, cache, cols):
    full = np.zeros(len(cache["index"]))
    if len(cache["feat"]):
        full[cache["active"]] = model.predict(cache["feat"][:, cols])
    return pd.Series(full, index=cache["index"])


def ranker_cache(cols, S):
    rrows, rlab, rgrp = sort_into_groups(S["Rr"][:, cols], S["Rl"], S["Rt"])
    ranker = fit_ranker("rank_xendcg", rrows, rlab, rgrp)
    ref_r = np.sort(ranker.r.predict(S["Rr"][:, cols]))
    rs = [_score(ranker, c, cols) for c in S["caches"]]
    idxs = [s.index for s in rs]
    fr = [pct(s.to_numpy(), ref_r) for s in rs]
    return fr, idxs


def hgb_head_cache(cols, S, off, seed, cfg):
    Xc = S["Xtr"][:, cols]
    if cfg["kind"] == "clf":
        w = None if cfg["fn"] is None else cfg["fn"](off, S["ytr"])
        hgb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            hgb.fit(Xc, S["ytr"], sample_weight=w)
        ref = np.sort(hgb.predict_proba(Xc)[:, 1])
        hs = [_score(hgb, c, cols) for c in S["caches"]]
    else:
        ysoft = cfg["fn"](off)
        reg = make_reg(seed)
        with threadpool_limits(limits=1):
            reg.fit(Xc, ysoft)
        ref = np.sort(reg.predict(Xc))
        hs = [_score_reg(reg, c, cols) for c in S["caches"]]
    return [pct(s.to_numpy(), ref) for s in hs]


def blend_aucs(fh, fr, idxs, y_val, w=BLEND_W):
    return np.asarray([
        ts_auc_from(pd.Series(w * a + (1 - w) * b, index=ix).to_frame("prediction"), y_val)
        for a, b, ix in zip(fh, fr, idxs)])


def run_phase(grid, seeds, phase, keep=None):
    store = {}
    for seed in seeds:
        t0 = time.time()
        S, off = prep(seed)
        fr, idxs = ranker_cache(BASE61, S)
        fh_ctrl = hgb_head_cache(BASE61, S, off, seed, dict(kind="clf", fn=None))
        base = blend_aucs(fh_ctrl, fr, idxs, S["y_val"])
        log(f"[{phase} seed {seed}] shipped champion base={base.mean():.4f} "
            f"npos={int(S['ytr'].sum())}/{len(S['ytr'])} ({time.time()-t0:.0f}s)")
        for cfg in grid:
            name = cfg["name"]
            if keep is not None and name not in keep:
                continue
            if name == "ctrl":
                continue
            try:
                fh = hgb_head_cache(BASE61, S, off, seed, cfg)
            except Exception:
                log(f"  [{phase} s{seed}] {name} HEAD-ERROR\n{traceback.format_exc()}")
                continue
            aug = blend_aucs(fh, fr, idxs, S["y_val"])
            md, _s, t, _ = paired(aug, base)
            store.setdefault(name, []).append((md, t))
            log(f"  [{phase} s{seed}] {name:<9} aug={aug.mean():.4f} d={md:+.4f} t={t:+.2f}")
        del S
    return summarize(store)


def main():
    smoke = os.environ.get("SMOKE") == "1"
    log(f"\n===== NIGHT (head/lever3) target-head search {datetime.now():%F %T} smoke={smoke} =====")
    grid = head_grid()
    log(f"heads: {[c['name'] for c in grid]}  base61={len(BASE61)} cols  w={BLEND_W}")
    seeds = [SEARCH_SEEDS[0]] if smoke else SEARCH_SEEDS
    screen = run_phase(grid, seeds, "SCREEN")
    log("\n----- (head) SCREEN SUMMARY (by mean_d) -----")
    strong = []
    for name in sorted(screen, key=lambda n: -screen[n][0]):
        md, mind, npos, mint, n = screen[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<10} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
            f"min_t {mint:+.2f} -> {v}")
        if npos == n and md >= 0.0015:
            strong.append(name)
    if smoke:
        log("(smoke) done.")
        return
    if strong:
        log(f"\nfresh-confirming {len(strong)} heads: {sorted(strong)}")
        fresh = run_phase(grid, FRESH_SEEDS, "FRESH", keep=set(strong))
        log("\n----- (head) FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<10} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
                f"min_t {mint:+.2f} -> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\n(head) no head screened promising (>=0.0015 & all-pos).")
    log(f"===== NIGHT (head/lever3) DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
