"""Orthogonal lane: explicit FEATURE INTERACTIONS on top of the E42 residual champion.

The GPU box owns the residual-AR axis (order sets / multi-scale / robust / GARCH). This is the
laptop's non-overlapping lane: does adding explicit pairwise INTERACTION features (products of
existing champion features) lift the shipped blend? Gradient boosting captures interactions
implicitly via tree splits, so the prior is LOW — but the BoNing-Gu 5th-place solution used
"top-10 feature interactions" as part of its win, and we now have a strong NEW feature (the AR
residual-variance-ratio) whose *joint* signal with the base contrasts (residual blowup AND a
distribution shift) a fixed-CDF blend may underuse. This screens that directly.

Re-baselined vs the CURRENT champion (45 champion feats + AR{1,2,3,4} residual block, STEPS=30) —
the residual block is FROZEN at the shipped config (this is NOT a residual search). Each block is
a set of product columns appended to BOTH members' feature matrix (mirrors how the residual block
was added to both). Real fixed-CDF TS-AUC, paired-t vs the champion, 2-stage seed gate.

Base matrix (49 cols): scale60 = cols 0..14, scale130 = 15..29, scale250 = 30..44
  (per scale: mean_z, log_var_ratio, ks, skew_diff, kurt_diff, ac1_diff, ac2_diff, iqr_ratio,
   tail2, tail3, zero_cross, slope_norm, spec_centroid_diff, rng_ratio, spec_entropy)
  residual AR{1,2,3,4} log-var-ratio = cols 45,46,47,48.

Gate: STRONG = pos on all seeds AND mean_d>=0.0015 AND min_t>=2.0 -> fresh-confirm {31337,2025,555,99}.

Usage:
  SMOKE=1 .venv/bin/python -u _interactions.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _interactions.py > _interactions.log 2>&1 & disown
"""
import os
import time
import traceback
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, load_pool_strat, select_picks
from _exp_fast import split_series
from _overnight import MIN_N, paired
from _residual_overnight import NCHAMP, precompute_fold, build_rows, _eval, _nextra, DENSE_K

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]
FRESH_SEEDS = [31337, 2025] if SMOKE else [31337, 2025, 555, 99]
RESULTS = "_interactions_results.log"

# Frozen shipped champion residual block (NOT searched here).
SPEC = dict(name="ctrl_ar1234", model="ar", orders=[1, 2, 3, 4], scales=[WINDOW])

# column indices in the 49-col base matrix
MZ, LVR, KS = 0, 1, 2                      # scale-60 core
MZ1, LVR1, KS1 = 15, 16, 17                # scale-130 core
MZ2, LVR2, KS2 = 30, 31, 32                # scale-250 core
AR = [45, 46, 47, 48]                      # residual AR{1,2,3,4}

# Each block = list of (i, j) column pairs -> product columns appended to the base matrix.
BLOCKS = {
    "resid_x_lvr250":   [(p, LVR2) for p in AR],                 # residual blowup AND raw variance
    "resid_x_ks250":    [(p, KS2) for p in AR],                  # residual blowup AND distribution shift
    "resid_x_mz250":    [(p, MZ2) for p in AR],                  # residual blowup AND mean shift
    "resid_x_core250":  [(p, c) for p in AR for c in (MZ2, LVR2, KS2)],  # broad residual x core (12)
    "resid_sq":         [(p, p) for p in AR],                    # curvature of the new feature
    "mz_x_lvr":         [(MZ, LVR), (MZ1, LVR1), (MZ2, LVR2)],   # level AND scale, per scale
    "lvr_x_ks":         [(LVR, KS), (LVR1, KS1), (LVR2, KS2)],   # variance AND distribution, per scale
    "lvr_cross_scale":  [(LVR, LVR1), (LVR1, LVR2), (LVR, LVR2)],  # variance persistence across scales
    "ks_cross_scale":   [(KS, KS1), (KS1, KS2), (KS, KS2)],      # KS persistence across scales
}


def log(msg):
    with open(RESULTS, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def _augment(M, pairs):
    if M.shape[0] == 0:
        return np.zeros((0, M.shape[1] + len(pairs)))
    extra = np.column_stack([M[:, i] * M[:, j] for (i, j) in pairs])
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([M, extra])


def _aug_caches(caches, pairs):
    return [{"index": c["index"], "active": c["active"], "feat": _augment(c["feat"], pairs)}
            for c in caches]


def run_phase(blocks, seeds, phase):
    store = {name: [] for name in blocks}
    nx = _nextra(SPEC)
    base_ncol = NCHAMP + nx
    for seed in seeds:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        picks = select_picks(X, train_ids)
        rng = np.random.default_rng(seed + 202)
        rank_steps = {}
        for sid in train_ids:
            L = len(split_series(X.loc[sid])[1].to_numpy())
            if L <= MIN_N:
                continue
            cand = np.arange(MIN_N, L)
            rank_steps[sid] = rng.choice(cand, size=min(DENSE_K, len(cand)), replace=False)

        caches = [precompute_fold(X, f, SPEC, nx) for f in folds]
        Xtr, ytr, _ = build_rows(X, yi, picks, SPEC, nx)
        Rr, Rl, Rt = build_rows(X, yi, rank_steps, SPEC, nx)
        base = _eval(np.arange(base_ncol), Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
        log(f"[{phase} seed {seed}] champion base={base.mean():.4f} ({time.time()-t0:.0f}s, "
            f"{base_ncol} cols)")

        for name, pairs in blocks.items():
            try:
                cols = np.arange(base_ncol + len(pairs))
                aug = _eval(cols, _augment(Xtr, pairs), ytr, _augment(Rr, pairs), Rl, Rt,
                            _aug_caches(caches, pairs), y_val, seed)
                md, _s, t, _ = paired(aug, base)
                store[name].append((md, t))
                log(f"  [{phase} s{seed}] {name:<16} +{len(pairs):>2}f aug={aug.mean():.4f} "
                    f"d={md:+.4f} t={t:+.2f}")
            except Exception:
                log(f"  [{phase} s{seed}] {name:<16} ERROR\n{traceback.format_exc()}")
        del X, caches
    out = {}
    for name, dt in store.items():
        if not dt:
            continue
        ds = [d for d, _t in dt]
        ts = [t for _d, t in dt]
        out[name] = (float(np.mean(ds)), float(min(ds)), sum(1 for d in ds if d > 0),
                     float(min(ts)), len(ds))
    return out


def verdict(mean_d, min_d, npos, min_t, n):
    if npos == n and mean_d >= 0.0015 and min_t >= 2.0:
        return "STRONG"
    return "pos" if npos == n and mean_d > 0 else "noise"


def main():
    log(f"\n===== FEATURE-INTERACTION campaign {datetime.now():%F %T} (smoke={SMOKE}) =====")
    log(f"re-baselined vs residual champion (45 + AR1234 = {NCHAMP + _nextra(SPEC)} cols); "
        f"screening {len(BLOCKS)} blocks on {SEARCH_SEEDS}")
    screen = run_phase(BLOCKS, SEARCH_SEEDS, "SCREEN")

    log("\n----- SCREEN SUMMARY (search seeds) -----")
    strong = []
    for name in sorted(screen, key=lambda n: -screen[n][0]):
        md, mind, npos, mint, n = screen[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} min_t {mint:+.2f} -> {v}")
        if npos == n and md >= 0.0015:
            strong.append(name)

    if strong:
        log(f"\nfresh-confirming {len(strong)} promising blocks on {FRESH_SEEDS}: {strong}")
        fresh = run_phase({k: BLOCKS[k] for k in strong}, FRESH_SEEDS, "FRESH")
        log("\n----- FRESH CONFIRM SUMMARY -----")
        for name in sorted(fresh, key=lambda n: -fresh[n][0]):
            md, mind, npos, mint, n = fresh[name]
            log(f"{name:<16} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} min_t {mint:+.2f} "
                f"-> {verdict(md, mind, npos, mint, n)}")
    else:
        log("\nno block screened promising; nothing to fresh-confirm.")
    log(f"===== INTERACTION DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()
