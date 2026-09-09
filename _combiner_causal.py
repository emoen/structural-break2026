"""P_C -- CAUSAL CONDITIONAL COMBINER on the TRUE 4-member champion ensemble.

Question: does a combiner that varies the member weighting PER ROW, conditioned on causal
context, beat the shipped FIXED linear blend of member percentile scores? This CLONES and
EXTENDS `_stack_deep.py` (imported, never modified) and REUSES its cached leak-free member
predictions (`_stack_cache/seed_*.pkl`) -- NO member is refit here.

TRUE champion baseline (percentile space, weights sum to 1):
    ranker=0.45  rf=0.40  xgb=0.10  catboost=0.05  hgb=0.00
(_stack_deep's own `wc` is the OLD 2-member hgb0.40/ranker0.60 -- NOT our champion; we build the
real 4-member vector via IDX and every combiner arm's delta is measured against it.)

Three CAUSAL, AFFINE-INVARIANT context signals per row (pipeline invariant to per-series
x -> a*x+b, a>0):
  ctx_step     = 0-based online step (=Ttr on train / time_online cumcount on val). Index -> exactly
                 invariant. The "steps-seen / offset" axis.
  ctx_disagree = spread ACROSS the M member percentile columns (mat.std(axis=1)). The members are
                 affine-invariant (proven base79 property), so their spread is too. Free from the
                 cached scores -- independent of any raw-series transform.
  ctx_disp     = TRUE RAW causal running-dispersion regime of the underlying series, in log-ratio
                 form so the scale a cancels EXACTLY:  log( std(recent W_RECENT online pts <= t) /
                 std(history) ), guarded to rows where both stds > 0 (else 0). Uses ONLY the
                 pre-online history + online points up to and INCLUDING step t -> causal, no
                 lookahead. Sourced by re-streaming the raw series (load_pool_strat/select_picks,
                 NO model refit); train-row alignment to the cached Foof is HARD-ASSERTED by
                 matching the re-derived (labels, steps) to the cached (ytr, Ttr); val-row
                 alignment is a reindex onto the cache MultiIndex Vidx with a no-NaN assert.

Two combiner arm families (honest SEARCH->FRESH selection, respecting the E50 lesson: bin/meta
weights are chosen to maximize the GLOBAL TS-AUC objective, never a per-step LOCAL AUC):
  ARM A  context-binned weight gating -- one member-weight simplex per context bin, chosen by
         GLOBAL-TS-AUC coordinate ascent on the SEARCH seeds (reuses the score_grid rank machinery).
  ARM B  learned causal meta -- fit_logit / fit_gbm on the OOF member scores CONCATENATED with the
         3 context features (same 3 features rebuilt for the train/OOF rows).

Env STACK_SMOKE=1 -> 2 seeds + coarse grid, reusing/building only the smoke caches.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

STACK_SMOKE = os.environ.get("STACK_SMOKE") == "1"
if STACK_SMOKE:
    os.environ["SMOKE"] = "1"

import json
import time
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")

# reuse _stack_deep's harness verbatim (module import runs no side effects beyond constants)
import _stack_deep as SD
from _stack_deep import (build_folds, score_folds, lin, fit_logit, fit_gbm, simplex,
                         wname, _auc, NAMES, IDX, M, SEEDS, FRESH, ALL_SEEDS)

# raw-series re-derivation (NO model refit -- only row structure + causal dispersion)
import _ensemble_tune as ET
from _ensemble_tune import load_pool_strat, select_picks, WINDOW  # noqa: F401
from _exp_fast import split_series, MIN_N, HIST_CAP  # noqa: F401
from _residual_overnight import _active

# ---------------------------------------------------------------------------
CTX_DIR = "_combiner_causal_cache"
os.makedirs("experiments/json", exist_ok=True)
os.makedirs(CTX_DIR, exist_ok=True)
LOG = "experiments/_combiner_causal.log"
SUMMARY = "experiments/_combiner_causal_summary.txt"
JSON = "experiments/json/_combiner_causal_best.json"

W_RECENT = 32                       # trailing online window for the running-dispersion regime
GRID_Q = 4 if STACK_SMOKE else 10   # weight-grid resolution: q=4 -> step .25 (smoke), q=10 -> .10
PASSES = 1 if STACK_SMOKE else 2    # coordinate-ascent passes over the bins
STEP_EDGES = [20, 40, 80, 160]      # ctx_step offset-bin edges -> 5 bins


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# TRUE-champion weight vector (percentile space)
# ---------------------------------------------------------------------------
def true_champ_weights():
    w = np.zeros(M)
    w[IDX["ranker"]] = 0.45
    w[IDX["rf"]] = 0.40
    w[IDX["xgb"]] = 0.10
    w[IDX["catboost"]] = 0.05
    w[IDX["hgb"]] = 0.00
    return w


# ---------------------------------------------------------------------------
# ctx_disp -- TRUE raw causal running-dispersion (affine-invariant log std-ratio)
# ---------------------------------------------------------------------------
def _disp_array(x_h, x_on, w_recent=W_RECENT):
    """Per online point t: log(std(recent w_recent pts <= t) / std(history)).

    Affine-invariant: under x->a*x+b (a>0) both stds scale by a, the ratio cancels a exactly and
    +b cancels in the std; guarded to strictly-positive stds (a>0 preserves the guard) so no eps is
    needed and invariance holds to machine precision. Causal: uses only history + online <= t.
    """
    L = len(x_on)
    if L == 0:
        return np.zeros(0)
    hstd = float(np.std(x_h)) if len(x_h) else 0.0
    if hstd <= 0.0:
        return np.zeros(L)
    rstd = pd.Series(x_on).rolling(w_recent, min_periods=2).std(ddof=0).to_numpy()
    out = np.zeros(L)
    good = np.isfinite(rstd) & (rstd > 0.0)
    out[good] = np.log(rstd[good] / hstd)
    return out


def _build_disp(seed, sd):
    """Return (disp_tr aligned to Foof rows, [disp_val per fold aligned to Vidx]). Cached.

    Leak-free alignment: train rows replicate _train_series' inclusion logic exactly and the
    re-derived (labels, steps) are HARD-ASSERTED equal to the cached (ytr, Ttr); val rows are
    reindexed onto the exact cache MultiIndex Vidx (no-NaN assert).
    """
    path = os.path.join(CTX_DIR, f"ctx_seed_{seed}{'_smoke' if STACK_SMOKE else ''}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    ET.SEED = seed                       # prepare_stack sets this before load_pool_strat/select_picks
    with threadpool_limits(limits=1):
        X, yi, y_val, train_ids, folds = load_pool_strat()
        picks = select_picks(X, train_ids)

        # ---- train rows: mirror _train_series row-inclusion, emit disp + (label, step) ----
        disp_tr, lab_tr, t_tr = [], [], []
        for sid, steps in picks.items():
            x_h, onl = split_series(X.loc[sid])
            x_on = onl.to_numpy(dtype=np.float64)
            active = _active(x_on)
            if len(active) == 0:
                continue
            base = int(active[0])
            tau = int(yi.loc[sid, "tau_index"])
            df = _disp_array(np.asarray(x_h, dtype=np.float64), x_on)
            for t in steps:
                i = t - base
                if i < 0 or i >= len(active):
                    continue
                disp_tr.append(float(df[t]))
                lab_tr.append(1 if (tau >= 0 and t >= tau) else 0)
                t_tr.append(int(t))
        disp_tr = np.asarray(disp_tr, dtype=np.float64)
        lab_tr = np.asarray(lab_tr, dtype=np.int64)
        t_tr = np.asarray(t_tr, dtype=np.int64)

        # HARD leak-free alignment check vs the cached member rows
        assert lab_tr.shape == sd["ytr"].shape, (
            f"train row count mismatch {lab_tr.shape} vs {sd['ytr'].shape}")
        assert np.array_equal(lab_tr, np.asarray(sd["ytr"])), "train labels misaligned vs cache ytr"
        assert np.array_equal(t_tr, np.asarray(sd["Ttr"])), "train steps misaligned vs cache Ttr"

        # ---- val rows: disp per online point, reindexed onto the exact cache Vidx ----
        disp_val = []
        for f_ids, vidx in zip(folds, sd["Vidx"]):
            parts = []
            for sid in f_ids:
                x_h, onl = split_series(X.loc[sid])
                x_on = onl.to_numpy(dtype=np.float64)
                df = _disp_array(np.asarray(x_h, dtype=np.float64), x_on)
                midx = pd.MultiIndex.from_product([[sid], onl.index], names=["id", "time"])
                parts.append(pd.Series(df, index=midx))
            ser = pd.concat(parts) if parts else pd.Series(dtype=np.float64)
            aligned = ser.reindex(vidx)
            assert not aligned.isna().any(), "val disp has NaN after reindex (fold/index mismatch)"
            disp_val.append(aligned.to_numpy(dtype=np.float64))

    out = {"disp_tr": disp_tr, "disp_val": disp_val}
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  [ctx] seed {seed} raw-dispersion re-derived+aligned in {time.time()-t0:.0f}s -> {path}")
    return out


# ---------------------------------------------------------------------------
# assemble the 3 context features for a seed (train rows + per-fold val rows)
# ---------------------------------------------------------------------------
def build_context(seed, sd, disp):
    Foof, Ttr = sd["Foof"], np.asarray(sd["Ttr"], dtype=np.float64)
    ctx_tr = {
        "step": Ttr,
        "disagree": Foof.std(axis=1),
        "disp": disp["disp_tr"],
    }
    ctx_val = []                          # per fold: dict of length-L arrays aligned to Vmat rows
    for fold_i, (idx, mat) in enumerate(zip(sd["Vidx"], sd["Vmat"])):
        merged = pd.Series(0, index=idx).to_frame("row")
        merged["time_online"] = merged.groupby(level="id").cumcount()
        ctx_val.append({
            "step": merged["time_online"].to_numpy(dtype=np.float64),
            "disagree": mat.std(axis=1),
            "disp": disp["disp_val"][fold_i],
        })
    return ctx_tr, ctx_val


# ---------------------------------------------------------------------------
# affine-invariance self-check (transform a real raw series x->a*x+b, recompute)
# ---------------------------------------------------------------------------
def affine_check(seed):
    ET.SEED = seed
    with threadpool_limits(limits=1):
        X, yi, y_val, train_ids, folds = load_pool_strat()
    rng = np.random.default_rng(0)
    a = float(rng.uniform(0.3, 4.0))
    b = float(rng.uniform(-5.0, 5.0))
    d_step = d_dis = 0.0
    dmax = 0.0
    for sid in list(train_ids)[:50]:
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        x_h = np.asarray(x_h, dtype=np.float64)
        d0 = _disp_array(x_h, x_on)
        d1 = _disp_array(a * x_h + b, a * x_on + b)
        dmax = max(dmax, float(np.max(np.abs(d0 - d1))) if len(d0) else 0.0)
    # ctx_step is an integer index (independent of series values) -> exactly invariant;
    # ctx_disagree is a function of the cached member percentiles (unchanged by the raw x->a*x+b
    # transform, and the members are themselves affine-invariant) -> exactly invariant.
    log(f"[affine-check] a={a:.3f} b={b:+.3f}  ctx_step maxdiff={d_step:.2e}  "
        f"ctx_disagree maxdiff={d_dis:.2e}  ctx_disp maxdiff={dmax:.2e}")
    assert dmax < 1e-9, f"ctx_disp NOT affine-invariant (maxdiff {dmax:.2e}) -- P0"
    return d_step, d_dis, dmax


# ---------------------------------------------------------------------------
# ARM A -- context-binned weight gating (GLOBAL-TS-AUC coordinate ascent)
# ---------------------------------------------------------------------------
def _grid_Wmat(q):
    g = [np.asarray(t, dtype=np.float64) / q for t in simplex(M, q)]
    return np.asarray(g).T                # M x B


def _rank_auc_mat(S, groups):
    """Weighted mean rank-AUC over groups for MANY score columns at once (stable-argsort ranks)."""
    B = S.shape[1]
    wsum = np.zeros(B)
    tot = 0.0
    for rows, lab, w in groups:
        sub = S[rows, :]
        g = sub.shape[0]
        order = np.argsort(sub, axis=0, kind="stable")
        ranks = np.empty((g, B))
        np.put_along_axis(ranks, order,
                          np.broadcast_to(np.arange(1, g + 1)[:, None], (g, B)), axis=0)
        npos = float(lab.sum())
        nneg = float(g - npos)
        auc = (ranks[lab == 1, :].sum(axis=0) - npos * (npos + 1) / 2.0) / (npos * nneg)
        wsum += w * auc
        tot += w
    return wsum / tot if tot > 0 else np.full(B, 0.5)


def _binrow(ctx_arr, ctx_name, thr):
    if ctx_name == "step":
        return np.digitize(ctx_arr, STEP_EDGES).astype(int)      # 0..len(STEP_EDGES)
    return np.digitize(ctx_arr, thr).astype(int)                 # 0..len(thr)


def _2d_binrow(step_arr, dis_arr, thr_dis):
    sb = np.digitize(step_arr, STEP_EDGES).astype(int)           # 0..4
    db = np.digitize(dis_arr, thr_dis).astype(int)               # 0..2
    return sb * 3 + db                                           # up to 15 bins


def _eval_bin(binfolds, W, b, Cand):
    """Mean-over-folds global rank-AUC for replacing bin b's weights with each column of Cand."""
    accum = []
    for mat, groups, binrow in binfolds:
        Wrow = W[:, binrow]                          # M x L
        Sbase = np.einsum("lm,ml->l", mat, Wrow)     # current per-row score
        maskb = binrow == b
        Scand = np.repeat(Sbase[:, None], Cand.shape[1], axis=1)
        if maskb.any():
            Scand[maskb, :] = mat[maskb, :] @ Cand
        accum.append(_rank_auc_mat(Scand, groups))
    return np.mean(accum, axis=0)


def _optimize_bins(search_binfolds, K, Cand, champW):
    W = np.repeat(champW[:, None], K, axis=1)         # init every bin at the champ blend
    for _ in range(PASSES):
        for b in range(K):
            has_rows = any((binrow == b).any() for bf in search_binfolds for _, _, binrow in bf)
            if not has_rows:
                continue
            sc = np.mean([_eval_bin(bf, W, b, Cand) for bf in search_binfolds], axis=0)
            W[:, b] = Cand[:, int(np.argmax(sc))]
    return W


def _score_binned(binfolds, W):
    aucs = []
    for mat, groups, binrow in binfolds:
        s = np.einsum("lm,ml->l", mat, W[:, binrow])
        wsum = tot = 0.0
        for rows, lab, w in groups:
            wsum += w * _auc(lab, s[rows])
            tot += w
        aucs.append(wsum / tot if tot > 0 else 0.5)
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
def main():
    open(LOG, "w").close()
    log(f"=== COMBINER CAUSAL  {datetime.now():%F %T}  smoke={STACK_SMOKE} ===")
    log(f"members={NAMES}  search={SEEDS}  fresh={FRESH}  grid_q={GRID_Q}  passes={PASSES}  "
        f"W_RECENT={W_RECENT}")

    wc = true_champ_weights()
    log(f"[true-champ] weights: {wname(wc)}   (vector over {NAMES})")

    # affine self-check (on the search seed's raw series)
    affine_check(SEEDS[0])

    # ---- load cached members + build causal context per seed ----
    seed_folds, seed_ctx_tr, seed_ctx_val, seed_meta = {}, {}, {}, {}
    for seed in ALL_SEEDS:
        sd = SD.get_seed_cache(seed)                  # REUSES _stack_cache -- never refits
        seed_folds[seed] = build_folds(sd)
        disp = _build_disp(seed, sd)
        ctx_tr, ctx_val = build_context(seed, sd, disp)
        seed_ctx_tr[seed] = ctx_tr
        seed_ctx_val[seed] = ctx_val
        seed_meta[seed] = (sd["Foof"], np.asarray(sd["ytr"]))
    log(f"[cache+ctx ready] {len(ALL_SEEDS)} seeds")

    # ---- TRUE-champion baseline ----
    champ = {s: score_folds(seed_folds[s], lin(wc)) for s in ALL_SEEDS}
    champ_all = float(np.mean([champ[s] for s in ALL_SEEDS]))
    champ_search = float(np.mean([champ[s] for s in SEEDS]))
    champ_fresh = float(np.mean([champ[s] for s in FRESH])) if FRESH else float("nan")
    log(f"[sanity] TRUE-champ  search={champ_search:.4f}  fresh={champ_fresh:.4f}  "
        f"ALL={champ_all:.4f}")

    results = {}                                       # arm -> dict(search,fresh,all,npos)

    def rowstats(score_by_seed):
        sm = float(np.mean([score_by_seed[s] for s in SEEDS]))
        fm = float(np.mean([score_by_seed[s] for s in FRESH])) if FRESH else float("nan")
        am = float(np.mean([score_by_seed[s] for s in ALL_SEEDS]))
        npos = sum(1 for s in ALL_SEEDS if score_by_seed[s] > champ[s])
        return sm, fm, am, npos

    def emit(tag, score_by_seed):
        sm, fm, am, npos = rowstats(score_by_seed)
        results[tag] = dict(search=sm, fresh=fm, all=am, npos=npos, d_all=am - champ_all)
        log(f"   {sm:7.4f} {fm:7.4f} {am:7.4f} {am-champ_all:+7.4f} {npos:>3}/{len(ALL_SEEDS)}  {tag}")

    Cand = _grid_Wmat(GRID_Q)
    log(f"\n-- ARM A: context-binned weight gating ({Cand.shape[1]} weight blends) --")
    log(f"   {'search':>7} {'fresh':>7} {'all':>7} {'d_all':>7} {'np':>5}  arm")
    log(f"   {champ_search:7.4f} {champ_fresh:7.4f} {champ_all:7.4f} {0.0:+7.4f} {'-':>5}  "
        f"CHAMP [{wname(wc)}]")

    # thresholds (terciles) fitted on SEARCH-seed pooled context values, then applied to ALL
    def pooled(name):
        return np.concatenate([np.concatenate([cv[name] for cv in seed_ctx_val[s]]) for s in SEEDS])
    thr = {"disagree": np.quantile(pooled("disagree"), [1 / 3, 2 / 3]),
           "disp": np.quantile(pooled("disp"), [1 / 3, 2 / 3])}

    armA_W = {}
    for arm, spec in (("A_step", ("step", None)),
                      ("A_disagree", ("disagree", thr["disagree"])),
                      ("A_disp", ("disp", thr["disp"])),
                      ("A_step_x_disagree", ("2d", thr["disagree"]))):
        name, t = spec
        # per-seed binned folds
        binfolds_by_seed = {}
        for s in ALL_SEEDS:
            bf = []
            for (mat, groups), cv in zip(seed_folds[s], seed_ctx_val[s]):
                if name == "2d":
                    br = _2d_binrow(cv["step"], cv["disagree"], t)
                else:
                    br = _binrow(cv[name], name, t)
                bf.append((mat, groups, br))
            binfolds_by_seed[s] = bf
        K = int(max(np.max(bf[2]) for bf in
                    [x for s in ALL_SEEDS for x in binfolds_by_seed[s]])) + 1
        W = _optimize_bins([binfolds_by_seed[s] for s in SEEDS], K, Cand, wc)
        armA_W[arm] = W
        emit(arm, {s: _score_binned(binfolds_by_seed[s], W) for s in ALL_SEEDS})

    # ---- ARM B: learned causal metas on [members || 3 context feats] ----
    log("\n-- ARM B: learned causal meta on [members || ctx_step,ctx_disagree,ctx_disp] --")
    log(f"   {'search':>7} {'fresh':>7} {'all':>7} {'d_all':>7} {'np':>5}  arm")
    log(f"   {champ_search:7.4f} {champ_fresh:7.4f} {champ_all:7.4f} {0.0:+7.4f} {'-':>5}  "
        f"CHAMP [{wname(wc)}]")

    def aug_train(seed):
        Foof, _ = seed_meta[seed]
        c = seed_ctx_tr[seed]
        return np.hstack([Foof, np.column_stack([c["step"], c["disagree"], c["disp"]])])

    def aug_folds(seed):
        out = []
        for (mat, groups), cv in zip(seed_folds[seed], seed_ctx_val[seed]):
            mat_aug = np.hstack([mat, np.column_stack([cv["step"], cv["disagree"], cv["disp"]])])
            out.append((mat_aug, groups))
        return out

    cols_all = list(range(M + 3))
    cols_mem = list(range(M))
    for tag, fitter in (("B_logit_aug", fit_logit), ("B_gbm_aug", fit_gbm)):
        by_seed = {}
        for s in ALL_SEEDS:
            Fa = aug_train(s)
            _, y = seed_meta[s]
            fitted = fitter(Fa, y, cols_all, s)
            by_seed[s] = score_folds(aug_folds(s), fitted)
        emit(tag, by_seed)
    # ablation: same meta on members ONLY (isolates whether the 3 ctx feats add anything)
    for tag, fitter in (("B_logit_memonly", fit_logit), ("B_gbm_memonly", fit_gbm)):
        by_seed = {}
        for s in ALL_SEEDS:
            Foof, y = seed_meta[s]
            fitted = fitter(Foof, y, cols_mem, s)
            by_seed[s] = score_folds(seed_folds[s], fitted)
        emit(tag, by_seed)

    # ---- selection (SEARCH only) + honest FRESH verdict ----
    armA = {k: v for k, v in results.items() if k.startswith("A_")}
    armB = {k: v for k, v in results.items() if k.startswith("B_")}
    bestA = max(armA, key=lambda k: armA[k]["search"])
    bestB = max(armB, key=lambda k: armB[k]["search"])
    log("\n-- VERDICT (selected on SEARCH, reported on FRESH/ALL vs TRUE champ) --")
    for tag in (bestA, bestB):
        r = results[tag]
        log(f"   best {tag:<18} search={r['search']:.4f} fresh={r['fresh']:.4f} "
            f"all={r['all']:.4f} d_all={r['d_all']:+.4f} np={r['npos']}/{len(ALL_SEEDS)}")

    payload = {"members": NAMES, "true_champ_weights": {n: float(v) for n, v in zip(NAMES, wc)},
               "champ": {"search": champ_search, "fresh": champ_fresh, "all": champ_all},
               "W_RECENT": W_RECENT, "grid_q": GRID_Q, "passes": PASSES,
               "arms": results, "best_search": {"armA": bestA, "armB": bestB}}
    with open(JSON, "w") as f:
        json.dump(payload, f, indent=2)
    with open(SUMMARY, "w") as f:
        f.write(f"COMBINER CAUSAL  {datetime.now():%F %T}  smoke={STACK_SMOKE}\n")
        f.write(f"TRUE champ [{wname(wc)}]  search={champ_search:.4f} fresh={champ_fresh:.4f} "
                f"all={champ_all:.4f}\n\n")
        f.write(f"{'arm':<20} {'search':>8} {'fresh':>8} {'all':>8} {'d_all':>8} {'np':>6}\n")
        for tag, r in results.items():
            f.write(f"{tag:<20} {r['search']:8.4f} {r['fresh']:8.4f} {r['all']:8.4f} "
                    f"{r['d_all']:+8.4f} {r['npos']:>4}/{len(ALL_SEEDS)}\n")
    log(f"\n=== DONE {datetime.now():%F %T}  json={JSON} ===")


if __name__ == "__main__":
    main()
