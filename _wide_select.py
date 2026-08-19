"""Laptop lane: WIDE feature bank + SHAP/gain SELECTION (the missing "select" step).

The 2025 2nd-place pipeline's edge was transform -> contrast -> SELECT: generate a large feature
bank, then keep the SHAP-top-K and LightGBM-gain-top-K, and train the final model on the selected
union. We have the contrast step (our whole feature set) but NO selection step. This tests whether a
wide bank + real importance selection beats the hand-curated 49-col residual champion.

Bank (69 cols): champion(45) + AR{1,2,3,4} residual(4) + transform-lens(11, from _transform_lens)
+ local-reference(9, from _local_ref). Individually those extra blocks were noise (E51/E52 + the
local-ref screen), so the only way they help is if SELECTION rescues a useful subset the block-level
gate missed. Importances come from ONE LightGBM fit on the training rows (honest: train-only);
SHAP via LightGBM's native pred_contrib (no shap package needed).

Configs vs the champion (49 cols): all69 (no selection = dilution baseline), gain/shap top-K,
union of gain+shap top-K, and champion+top-K-extras. Same 2-stage seed gate + fixed-CDF TS-AUC.

Usage:
  SMOKE=1 .venv/bin/python -u _wide_select.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _wide_select.py > _wide_select.log 2>&1 & disown
"""
import os
import time
import traceback
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import lightgbm as lgb
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, champion_feat, load_pool_strat, select_picks
from _exp_fast import HIST_CAP, hist_context, split_series
from _overnight import MIN_N, N_JOBS, paired
from _residual_overnight import (NCHAMP, _active, _extra_feats, _eval, _fit_refs, _nextra)
from _transform_lens import lens_feats
from _local_ref import local_ref_feats

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]
FRESH_SEEDS = [31337, 2025] if SMOKE else [31337, 2025, 555, 99]
DENSE_K = 30 if SMOKE else 60
RESULTS = "_wide_select_results.log"

# Re-baselined vs the CURRENT LB champion (E54): multi-scale AR8 residual var-ratio on
# trailing sub-windows 60/130/250 (not the superseded single-scale AR{1,2,3,4}).
RESID_SCALES = [60, 130, 250]
SPEC = dict(name="ar8_ms", model="ar", orders=[8], scales=RESID_SCALES)
NX = _nextra(SPEC)                 # 3
BASE_NCOL = NCHAMP + NX            # 48  (champion + multi-scale AR8 residual)
NLENS = 11                         # transform-lens
NLOC = 9                           # local-reference
NEXTRA = NLENS + NLOC              # 20
NCOMB = BASE_NCOL + NEXTRA         # 68

_EPS = 1e-12

# --- human-readable names for every bank column (for selection reporting) ---
_BASE15 = ["mean_z", "log_var_ratio", "ks", "skew_diff", "kurt_diff",
           "ac1_diff", "ac2_diff", "iqr_ratio", "tail2", "tail3",
           "zero_cross", "slope_norm", "spec_centroid_diff", "rng_ratio", "spec_entropy"]
_LENS11 = ["cusum_maxabs", "cusum_end", "cusum_range", "pit_mean_dev", "pit_std", "pit_ks",
           "abs_mean_ratio", "abs_var_ratio", "rollstd_level", "rollstd_cv", "levene_f"]
_LOC9 = [f"loc{L}_{s}" for L in (30, 60, 120) for s in ("mz", "lvr", "ks")]


def _build_names():
    names = []
    for s in (60, 130, 250):
        names += [f"ch{s}_{k}" for k in _BASE15]      # 0-44 champion
    names += [f"ar8res_s{s}" for s in RESID_SCALES]   # 45-47 multi-scale AR8 residual
    names += _LENS11                                   # 48-58 transform-lens
    names += _LOC9                                     # 59-67 local-reference
    return names


FEATURE_NAMES = _build_names()
assert len(FEATURE_NAMES) == NCOMB, (len(FEATURE_NAMES), NCOMB)


def _names(cols):
    return [FEATURE_NAMES[c] for c in cols]


def log(msg):
    with open(RESULTS, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def _row(win, ctx, refs):
    return np.concatenate([champion_feat(win, ctx), _extra_feats(win, refs, SPEC),
                           np.asarray(lens_feats(win, ctx), dtype=np.float64),
                           np.asarray(local_ref_feats(win), dtype=np.float64)])


def _series_feat(x_on, ctx, refs, active):
    rows = [_row(x_on[max(0, t - WINDOW + 1):t + 1], ctx, refs) for t in active]
    if not rows:
        return np.zeros((0, NCOMB))
    return np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _val_series(sid, x_h, x_on, oi):
    import pandas as pd
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    feat = _series_feat(x_on, ctx, _fit_refs(x_h, SPEC), active)
    midx = pd.MultiIndex.from_product([[sid], oi], names=["id", "time"])
    return midx, active, feat


def _train_series(sid, x_h, x_on, tau, steps):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return np.zeros((0, NCOMB)), [], []
    refs = _fit_refs(x_h, SPEC)
    base = active[0]
    rows, labels, tvals = [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        win = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(_row(win, ctx, refs))
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    feat = np.nan_to_num(np.asarray(rows), nan=0.0, posinf=0.0, neginf=0.0) if rows \
        else np.zeros((0, NCOMB))
    return feat, labels, tvals


def precompute_fold(X, fold_ids):
    jobs = [(sid, np.asarray(split_series(X.loc[sid])[0], dtype=np.float64),
             split_series(X.loc[sid])[1].to_numpy(dtype=np.float64),
             split_series(X.loc[sid])[1].index) for sid in fold_ids]
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series)(sid, x_h, x_on, oi) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, active_all, offset = [], [], [], 0
    for midx, active, feat in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, NCOMB)))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCOMB))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


def build_rows(X, yi, sid_to_steps):
    jobs = []
    for sid, steps in sid_to_steps.items():
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64),
                     int(yi.loc[sid, "tau_index"]), np.asarray(steps)))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_train_series)(sid, x_h, x_on, tau, steps)
            for (sid, x_h, x_on, tau, steps) in jobs)
    rows = [r for r, _l, _t in parts if len(r)]
    labels = np.concatenate([l for _r, l, _t in parts if len(l)]) if rows else np.zeros(0, int)
    tvals = np.concatenate([t for _r, _l, t in parts if len(t)]) if rows else np.zeros(0, int)
    feat = np.vstack(rows) if rows else np.zeros((0, NCOMB))
    return feat, labels.astype(np.int64), tvals.astype(np.int64)


def _importances(Xtr, ytr, seed):
    """Return (gain, shap) importance arrays over NCOMB features from one LightGBM fit."""
    clf = lgb.LGBMClassifier(n_estimators=750, learning_rate=0.01, colsample_bytree=0.3,
                             max_depth=8, random_state=seed, verbosity=-1, n_jobs=N_JOBS)
    with threadpool_limits(limits=1):
        clf.fit(Xtr, ytr)
    gain = np.asarray(clf.booster_.feature_importance(importance_type="gain"), dtype=np.float64)
    contrib = clf.booster_.predict(Xtr, pred_contrib=True)   # (n, NCOMB+1), last = base value
    shap = np.abs(contrib[:, :-1]).mean(axis=0)
    return gain, shap


def _configs(gain, shap):
    order_g = np.argsort(-gain)
    order_s = np.argsort(-shap)
    extras = np.arange(BASE_NCOL, NCOMB)
    eg = extras[np.argsort(-gain[extras])]           # extras ranked by gain
    champ = np.arange(BASE_NCOL)

    def uni(a, b):
        return np.array(sorted(set(a.tolist()) | set(b.tolist())))

    return {
        "all68":      np.arange(NCOMB),
        "gain40":     np.sort(order_g[:40]),
        "gain55":     np.sort(order_g[:55]),
        "shap40":     np.sort(order_s[:40]),
        "shap55":     np.sort(order_s[:55]),
        "union_40":   uni(order_g[:40], order_s[:40]),
        "champ_ex4":  uni(champ, eg[:4]),
        "champ_ex8":  uni(champ, eg[:8]),
        "champ_ex15": uni(champ, eg[:15]),
    }


def _extras_by_gain(gain):
    """Indices of the 20 extra (lens+local) columns ranked by descending gain."""
    extras = np.arange(BASE_NCOL, NCOMB)
    return extras[np.argsort(-gain[extras])]


def run_phase(seeds, phase):
    from collections import Counter
    store = {}
    base_cols = np.arange(BASE_NCOL)
    sel_rows = []                 # (seed, top4_names, top8_names)
    cnt8 = Counter()              # how often each extra lands in the gain-top-8
    cnt4 = Counter()              # ... and in the gain-top-4
    gain_accum = {}               # extra name -> list of gain values across seeds
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

        caches = [precompute_fold(X, f) for f in folds]
        Xtr, ytr, _ = build_rows(X, yi, picks)
        Rr, Rl, Rt = build_rows(X, yi, rank_steps)
        base = _eval(base_cols, Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
        gain, shap = _importances(Xtr, ytr, seed)
        cfgs = _configs(gain, shap)

        eg = _extras_by_gain(gain)
        top8, top4 = _names(eg[:8]), _names(eg[:4])
        sel_rows.append((seed, top4, top8))
        for nm in top8:
            cnt8[nm] += 1
        for nm in top4:
            cnt4[nm] += 1
        for c in np.arange(BASE_NCOL, NCOMB):
            gain_accum.setdefault(FEATURE_NAMES[c], []).append(float(gain[c]))
        log(f"[{phase} seed {seed}] champion base={base.mean():.4f} ({time.time()-t0:.0f}s, "
            f"{BASE_NCOL} cols)")
        log(f"    gain-top4 extras = {top4}")
        log(f"    gain-top8 extras = {top8}")

        for name, cols in cfgs.items():
            store.setdefault(name, [])
            try:
                aug = _eval(np.asarray(cols), Xtr, ytr, Rr, Rl, Rt, caches, y_val, seed)
                md, _s, t, d = paired(aug, base)
                store[name].append((md, t, np.asarray(d)))
                log(f"  [{phase} s{seed}] {name:<11} {len(cols):>2}c aug={aug.mean():.4f} "
                    f"d={md:+.4f} t={t:+.2f}")
            except Exception:
                log(f"  [{phase} s{seed}] {name:<11} ERROR\n{traceback.format_exc()}")
        del X, caches
    out = {}
    for name, dt in store.items():
        if not dt:
            continue
        ds = [d for d, _t, _a in dt]
        ts = [t for _d, t, _a in dt]
        dd = np.concatenate([a for _d, _t, a in dt])          # pooled per-fold diffs
        sd = float(dd.std(ddof=1))
        t_agg = float(dd.mean() / (sd / np.sqrt(len(dd)))) if sd > 0 else float("inf")
        out[name] = (float(np.mean(ds)), float(min(ds)), sum(1 for d in ds if d > 0),
                     float(min(ts)), len(ds), t_agg, [float(t) for t in ts])
    sel = {"rows": sel_rows, "cnt8": cnt8, "cnt4": cnt4, "gain": gain_accum, "nseeds": len(seeds)}
    return out, sel


def _log_selection(phase, sel):
    ns = sel["nseeds"]
    log(f"\n----- {phase}: extras selected per seed -----")
    for seed, e4, e8 in sel["rows"]:
        log(f"  s{seed:<6} top4={e4}")
        log(f"  s{seed:<6} top8={e8}")
    mean_gain = {nm: float(np.mean(v)) for nm, v in sel["gain"].items()}
    log(f"\n----- {phase}: extra representation across {ns} seeds (times in gain-top-K) -----")
    log(f"  {'feature':<16} {'top4':>5} {'top8':>5}  {'mean_gain':>10}")
    for nm in sorted(mean_gain, key=lambda n: -mean_gain[n]):
        log(f"  {nm:<16} {sel['cnt4'][nm]:>5} {sel['cnt8'][nm]:>5}  {mean_gain[nm]:>10.1f}")



def verdict(mean_d, min_d, npos, min_t, n):
    if npos == n and mean_d >= 0.0015 and min_t >= 2.0:
        return "STRONG"
    return "pos" if npos == n and mean_d > 0 else "noise"


def _fmt_ts(tlist):
    return "[" + ", ".join(f"{t:+.2f}" for t in tlist) + "]"


def _summary(title, res, only=None):
    log(f"\n----- {title} -----")
    for name in sorted(res, key=lambda n: -res[n][0]):
        if only is not None and name not in only:
            continue
        md, mind, npos, mint, n, tagg, tlist = res[name]
        v = verdict(md, mind, npos, mint, n)
        log(f"{name:<11} mean_d {md:+.4f} min_d {mind:+.4f} pos {npos}/{n} "
            f"t_agg {tagg:+.2f} min_t {mint:+.2f} t_seeds {_fmt_ts(tlist)} -> {v}")


def main():
    log(f"\n===== WIDE-BANK+SELECT campaign {datetime.now():%F %T} (smoke={SMOKE}) =====")
    log(f"bank={NCOMB} cols (champ{BASE_NCOL} [45 champion + AR8-ms residual {NX}] + lens{NLENS} "
        f"+ local{NLOC}); re-baselined vs NEW multi-scale AR8 champion ({BASE_NCOL}); "
        f"screen {SEARCH_SEEDS}")
    screen, sel_s = run_phase(SEARCH_SEEDS, "SCREEN")
    _log_selection("SCREEN", sel_s)

    _summary("SCREEN SUMMARY (search seeds)", screen)
    promising = [name for name in screen
                 if screen[name][2] == screen[name][4] and screen[name][0] >= 0.0015]

    if promising:
        log(f"\nfresh-confirming {len(promising)} configs on {FRESH_SEEDS}: {sorted(promising)}")
        fresh, sel_f = run_phase(FRESH_SEEDS, "FRESH")
        _log_selection("FRESH", sel_f)
        _summary("FRESH CONFIRM SUMMARY", fresh, only=set(promising))
    else:
        log("\nno config screened promising; nothing to fresh-confirm.")
    log(f"===== WIDE-BANK+SELECT DONE {datetime.now():%F %T} =====")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
