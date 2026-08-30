"""Stack the 5-member pool (champion HGB + LGBMRanker + zoo winners rf/xgb/catboost) and test
whether any combination beats the deployed 2-member champion blend (w=0.40 percentile blend).

Leak-free protocol (mirrors the champion + eval_stack machinery):
  * one prepare_stack(seed) builds: pointwise train rows Xtr (10 steps/series), the DENSE ranker
    rows Rr (60 steps/series, the champion ranker recipe), 10 stratified val folds, y_val.
  * OOF member predictions on the Xtr rows via series-level K-fold (no series in both train &
    OOF) -> fixed train-CDF percentile calibration -> aligned meta-feature matrix Foof.
      - pointwise members (hgb/rf/xgb/catboost) OOF-fit on Xtr[tr], predict Xtr[held].
      - ranker OOF-fit on the DENSE rows of non-held series, predict the held Xtr rows (keeps the
        member at champion strength while staying leak-free & row-aligned to the others).
  * full-fit each member -> per-val-fold percentile predictions Vmat (aligned to cache index).

Combination methods evaluated per seed (all off the SAME member preds, so apples-to-apples):
  champ     0.40*pct_hgb + 0.60*pct_rank                      (deployed baseline)
  mean5     equal mean of all 5 members
  trees4    equal mean of hgb+rf+xgb+catboost (drop ranker)   (control: ranker's value)
  logit_HR  learned logistic on [hgb,ranker] only             (control: is base weight improvable)
  logit5    learned logistic meta-stacker on all 5 (OOF-fit)
  gbm5      learned LightGBM meta-stacker on all 5 (OOF-fit)
  greedy5   Caruana greedy ensemble selection on OOF (train TS-AUC objective) -> integer weights

Reports per-seed val TS-AUC (search seeds [42,7,123] + fresh [31337,2025,555,99]) and the paired
delta vs champ. Env STACK_SMOKE=1 -> tiny pool / 2 seeds for a quick end-to-end check.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

STACK_SMOKE = os.environ.get("STACK_SMOKE") == "1"
if STACK_SMOKE:
    os.environ["SMOKE"] = "1"

import time
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import _ensemble_tune as ET
from _ensemble_tune import load_pool_strat, select_picks, make_hgb
from _night_stack import (SEARCH_SEEDS, FRESH_SEEDS, DENSE_K, NCHAMP_BASE,
                          precompute_fold_wide, build_rows_wide)
from _exp_fast import split_series, ts_auc_from
from _residual_overnight import pct, TUNED
from _lambdarank import sort_into_groups, fit_ranker
from _overnight import MIN_N, N_JOBS, paired

# ---------------------------------------------------------------------------
MODE = "champ_dep"
C = NCHAMP_BASE
DEP_KEEP = [C + i for i in (0, 1, 2, 3, 4, 5, 10, 11, 12, 13)]
COLS = np.concatenate([np.arange(C), np.asarray(DEP_KEEP, dtype=int)])     # 61-col champion set

SEEDS = SEARCH_SEEDS
FRESH = FRESH_SEEDS
MFOLD = 3
W_CHAMP = 0.40                                # deployed blend: W*pct_hgb + (1-W)*pct_rank
os.makedirs("experiments/json", exist_ok=True)
LOG = "experiments/_stack_zoo.log"

MEMBERS = [
    dict(name="hgb",      kind="hgb",      params=dict(TUNED)),
    dict(name="ranker",   kind="ranker",   params={}),
    dict(name="rf",       kind="rf",       params=dict(n_estimators=800, max_features=0.5,
                                                       min_samples_leaf=20)),
    dict(name="xgb",      kind="xgb",      params=dict(n_estimators=400, learning_rate=0.03,
                                                       max_depth=6, subsample=0.8,
                                                       colsample_bytree=0.7, reg_lambda=1.0,
                                                       min_child_weight=5)),
    dict(name="catboost", kind="catboost", params=dict(iterations=600, learning_rate=0.03,
                                                       depth=6, l2_leaf_reg=10.0)),
]
NAMES = [m["name"] for m in MEMBERS]
IDX = {n: i for i, n in enumerate(NAMES)}


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# per-seed build: pointwise rows Xtr + dense ranker rows Rr(+sids) + val caches
# ---------------------------------------------------------------------------
def prepare_stack(seed):
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
    caches = [precompute_fold_wide(X, f, MODE) for f in folds]
    Xtr, ytr, Ttr, Str = build_rows_wide(X, yi, picks, MODE)
    Rr, Rl, Rt, Rs = build_rows_wide(X, yi, rank_steps, MODE)   # keep ranker-row sids for OOF
    return dict(Xtr=np.asarray(Xtr), ytr=np.asarray(ytr), Ttr=np.asarray(Ttr),
                Str=np.asarray(Str), Rr=np.asarray(Rr), Rl=np.asarray(Rl),
                Rt=np.asarray(Rt), Rs=np.asarray(Rs), caches=caches, y_val=y_val)


def fit_pointwise(spec, X, y, seed):
    k, p = spec["kind"], spec["params"]
    if k == "hgb":
        m = make_hgb(random_state=seed, **p)
        with threadpool_limits(limits=1):
            m.fit(X, y)
        return m
    if k == "rf":
        m = RandomForestClassifier(n_jobs=N_JOBS, random_state=seed, **p)
    elif k == "xgb":
        m = xgb.XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=N_JOBS,
                              random_state=seed, verbosity=0, **p)
    elif k == "catboost":
        m = CatBoostClassifier(random_seed=seed, thread_count=N_JOBS, verbose=0,
                               allow_writing_files=False, **p)
    else:
        raise ValueError(k)
    m.fit(X, y)
    return m


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


def member_preds(seed):
    """Return OOF meta-feature matrix (Foof) + per-fold val percentile matrices (Vmat)."""
    S = prepare_stack(seed)
    Xtr, ytr, Ttr, Str = S["Xtr"], S["ytr"], S["Ttr"], S["Str"]
    Rr, Rl, Rt, Rs = S["Rr"], S["Rl"], S["Rt"], S["Rs"]
    caches, y_val = S["caches"], S["y_val"]
    cols = COLS
    M, n = len(MEMBERS), len(ytr)

    # ---- OOF meta-features on the Xtr rows (series-level K-fold) ----
    Foof = np.zeros((n, M))
    sids = np.unique(Str)
    rng = np.random.default_rng(seed + 7)
    rng.shuffle(sids)
    kfolds = np.array_split(sids, MFOLD)
    for f in range(MFOLD):
        held = np.isin(Str, kfolds[f])
        tr = ~held
        Xh = Xtr[held][:, cols]
        for mi, spec in enumerate(MEMBERS):
            if spec["kind"] == "ranker":
                rtr = ~np.isin(Rs, kfolds[f])
                rows, lab, grp = sort_into_groups(Rr[rtr][:, cols], Rl[rtr], Rt[rtr])
                rk = fit_ranker("rank_xendcg", rows, lab, grp)
                Foof[held, mi] = _proba(rk, Xh)
            else:
                m = fit_pointwise(spec, Xtr[tr][:, cols], ytr[tr], seed)
                Foof[held, mi] = _proba(m, Xh)

    # ---- full fits + train-CDF refs + val percentile predictions ----
    refs, full = [], []
    for mi, spec in enumerate(MEMBERS):
        if spec["kind"] == "ranker":
            rows, lab, grp = sort_into_groups(Rr[:, cols], Rl, Rt)
            mdl = fit_ranker("rank_xendcg", rows, lab, grp)
            ref = np.sort(_proba(mdl, Rr[:, cols]))
        else:
            mdl = fit_pointwise(spec, Xtr[:, cols], ytr, seed)
            ref = np.sort(_proba(mdl, Xtr[:, cols]))
        refs.append(ref)
        full.append(mdl)
        Foof[:, mi] = pct(Foof[:, mi], ref)

    Vidx, Vmat = [], []
    for c in caches:
        L = len(c["index"])
        mat = np.zeros((L, M))
        for mi, mdl in enumerate(full):
            s = np.zeros(L)
            if len(c["feat"]):
                s[c["active"]] = _proba(mdl, c["feat"][:, cols])
            mat[:, mi] = pct(s, refs[mi])
        Vidx.append(c["index"])
        Vmat.append(mat)

    return dict(Foof=Foof, ytr=ytr, Ttr=Ttr, Vidx=Vidx, Vmat=Vmat, y_val=y_val)


# ---------------------------------------------------------------------------
# combination methods
# ---------------------------------------------------------------------------
def _fast_auc(lab, sc):
    order = np.argsort(sc, kind="stable")
    ranks = np.empty(len(sc), dtype=np.float64)
    ranks[order] = np.arange(1, len(sc) + 1)
    npos = float(lab.sum())
    nneg = float(len(lab) - npos)
    if npos == 0 or nneg == 0:
        return None
    return (ranks[lab == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def train_ts_auc(score, y, groups):
    wsum = tot = 0.0
    for g in groups:
        a = _fast_auc(y[g], score[g])
        if a is None:
            continue
        lab = y[g]
        npos = float(lab.sum())
        nneg = float(len(lab) - npos)
        w = npos * nneg
        wsum += w * a
        tot += w
    return wsum / tot if tot > 0 else 0.5


def greedy_select(Foof, y, groups, max_iters=30):
    M = Foof.shape[1]
    singles = [train_ts_auc(Foof[:, m], y, groups) for m in range(M)]
    best_m = int(np.argmax(singles))
    counts = np.zeros(M)
    counts[best_m] = 1.0
    cur_sum = Foof[:, best_m].copy()
    best_score = singles[best_m]
    for _ in range(max_iters - 1):
        scores = [train_ts_auc((cur_sum + Foof[:, m]) / (counts.sum() + 1), y, groups)
                  for m in range(M)]
        m = int(np.argmax(scores))
        if scores[m] <= best_score + 1e-6:
            break
        counts[m] += 1.0
        cur_sum += Foof[:, m]
        best_score = scores[m]
    return counts / counts.sum(), best_score


def combine(seed_data, method, meta_fit=None):
    """Return per-fold val TS-AUC array for `method` on one seed's member preds."""
    Foof, ytr, Vmat, Vidx, y_val = (seed_data["Foof"], seed_data["ytr"], seed_data["Vmat"],
                                    seed_data["Vidx"], seed_data["y_val"])
    aucs = []
    for mat, idx in zip(Vmat, Vidx):
        if method == "champ":
            s = W_CHAMP * mat[:, IDX["hgb"]] + (1 - W_CHAMP) * mat[:, IDX["ranker"]]
        elif method == "mean5":
            s = mat.mean(axis=1)
        elif method == "trees4":
            s = mat[:, [IDX["hgb"], IDX["rf"], IDX["xgb"], IDX["catboost"]]].mean(axis=1)
        elif method in ("logit_HR", "logit5", "gbm5"):
            cols = [IDX["hgb"], IDX["ranker"]] if method == "logit_HR" else list(range(mat.shape[1]))
            s = meta_fit.predict_proba(mat[:, cols])[:, 1]
        elif method == "greedy5":
            s = mat @ meta_fit                     # meta_fit = weight vector
        else:
            raise ValueError(method)
        aucs.append(ts_auc_from(pd.Series(s, index=idx).to_frame("prediction"), y_val))
    return np.asarray(aucs)


def fit_meta(seed_data, method, seed):
    Foof, ytr = seed_data["Foof"], seed_data["ytr"]
    if method == "logit_HR":
        cols = [IDX["hgb"], IDX["ranker"]]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000,
                                                               random_state=seed))
        m.fit(Foof[:, cols], ytr)
        return m
    if method == "logit5":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000,
                                                               random_state=seed))
        m.fit(Foof, ytr)
        return m
    if method == "gbm5":
        m = lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=0.05,
                               num_leaves=15, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                               n_jobs=N_JOBS, verbose=-1)
        m.fit(Foof, ytr)
        return m
    return None


METHODS = ["champ", "mean5", "trees4", "logit_HR", "logit5", "gbm5", "greedy5"]


def main():
    open(LOG, "w").close()
    log(f"=== STACK ZOO  {datetime.now():%F %T}  smoke={STACK_SMOKE} ===")
    log(f"members={NAMES}  search={SEEDS}  fresh={FRESH}  cols={len(COLS)}  mfold={MFOLD}")

    results = {m: {} for m in METHODS}     # method -> {seed: per-fold auc array}
    gweights = {}                          # seed -> greedy weight vector
    all_seeds = SEEDS + FRESH
    for seed in all_seeds:
        t0 = time.time()
        sd = member_preds(seed)
        groups = [np.where(sd["Ttr"] == t)[0] for t in np.unique(sd["Ttr"])]
        gw, gscore = greedy_select(sd["Foof"], sd["ytr"], groups)
        gweights[seed] = gw
        metas = {m: fit_meta(sd, m, seed) for m in ("logit_HR", "logit5", "gbm5")}
        for m in METHODS:
            mf = gw if m == "greedy5" else metas.get(m)
            results[m][seed] = combine(sd, m, mf)
        gwline = " ".join(f"{n}={w:.2f}" for n, w in zip(NAMES, gw) if w > 0)
        log(f"  [seed {seed}] champ={results['champ'][seed].mean():.4f} "
            f"mean5={results['mean5'][seed].mean():.4f} logit5={results['logit5'][seed].mean():.4f} "
            f"gbm5={results['gbm5'][seed].mean():.4f} greedy5={results['greedy5'][seed].mean():.4f} "
            f"| greedy[{gwline}] ({time.time()-t0:.0f}s)")
        del sd

    # ---- tables ----
    def block(name, seeds):
        log(f"\n-- {name} ({len(seeds)} seeds) --")
        log(f"   {'method':<9} {'mean':>8} {'d_vs_champ':>11} {'npos':>6} {'min_t':>7}")
        base = np.mean([results['champ'][s].mean() for s in seeds])
        rows = []
        for m in METHODS:
            mean = np.mean([results[m][s].mean() for s in seeds])
            mds, ts = [], []
            for s in seeds:
                md, _sd, t, _d = paired(results[m][s], results['champ'][s])
                mds.append(md)
                ts.append(t)
            npos = sum(1 for d in mds if d > 0)
            log(f"   {m:<9} {mean:8.4f} {mean-base:+11.4f} {npos:>4}/{len(seeds)} "
                f"{min(ts):+7.2f}")
            rows.append((m, mean, mean - base, npos, len(seeds), min(ts)))
        return rows

    srows = block("SEARCH", SEEDS)
    if FRESH:
        block("FRESH", FRESH)
        block("ALL", all_seeds)

    # mean greedy weights
    gw_mean = np.mean([gweights[s] for s in all_seeds], axis=0)
    log("\n-- mean greedy weights --")
    log("   " + "  ".join(f"{n}={w:.3f}" for n, w in zip(NAMES, gw_mean)))

    # verdict vs champion on ALL seeds
    log("\n-- verdict vs champ (ALL seeds: mean_d>0 on all + best mean) --")
    best = None
    for m in METHODS:
        if m == "champ":
            continue
        mds = [paired(results[m][s], results['champ'][s])[0] for s in all_seeds]
        npos = sum(1 for d in mds if d > 0)
        mean = np.mean([results[m][s].mean() for s in all_seeds])
        tag = "CONSISTENT" if npos == len(all_seeds) else f"{npos}/{len(all_seeds)}"
        log(f"   {m:<9} mean={mean:.4f} mean_d={np.mean(mds):+.4f} {tag}")
        if best is None or mean > best[1]:
            best = (m, mean, np.mean(mds), npos)
    log(f"\n>> best method: {best[0]} mean={best[1]:.4f} mean_d={best[2]:+.4f} "
        f"pos {best[3]}/{len(all_seeds)}")

    with open("experiments/json/_stack_zoo_best.json", "w") as f:
        json.dump({"members": NAMES, "methods": METHODS,
                   "mean_greedy_weights": {n: float(w) for n, w in zip(NAMES, gw_mean)},
                   "search_rows": [list(r) for r in srows]}, f, indent=2)
    log(f"\n=== DONE {datetime.now():%F %T} ===")


if __name__ == "__main__":
    main()
