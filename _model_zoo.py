"""Overnight MODEL ZOO — feed many model families the SAME champion features and score each.

Goal (runs unattended, one process, no approvals):
  1. DEFAULT table  : fit every model family with library-default-ish params on the champion
     61-col feature matrix (b_no_accf, = champ51 + 10 dep cols), score per-val-series TS-AUC
     (mean over search seeds). Same data, same features, same metric for every model.
  2. ROUGH grid     : a small parameter grid per family; keep each family's best rough config.
  3. TOP-3          : rank families by their best (default ∪ rough) mean TS-AUC, take the top 3
     (score-agnostic selection — we just take whoever wins).
  4. FINE grid      : a larger parameter grid for those 3 families only; keep the best fine config.
  5. FINAL table    : re-score the 3 winners' best configs on search + fresh seeds and print the
     table. Also dumps per-seed/per-fold predictions of the winners (+ the champion HGB member)
     to a pickle so the morning stacking work can reuse them without refitting.

Every model is fit as a single CLASSIFIER producing a continuous score per row; scores are
percentile-calibrated against the train CDF (monotone, mirrors the champion HGB member exactly),
then TS-AUC is computed per val series. This isolates each model as a stackable first-level member.

Scoring harness (prepare_light / eval_config) mirrors _bayescp_ship / _stack_champ member fits.

Env:
  ZOO_SMOKE=1   fast end-to-end check (tiny pool, 1 seed, truncated grids, no fresh).

Outputs (all in repo root):
  _model_zoo.log            human-readable progress + final tables (also stdout)
  _model_zoo_default.tsv    default-config scores, every family
  _model_zoo_rough.tsv      every rough-grid config score
  _model_zoo_fine.tsv       every fine-grid config score (top-3 families)
  _model_zoo_final.tsv      winners' best config, search+fresh scores
  _model_zoo_best.json      best config per family + chosen top-3
  _model_zoo_preds.pkl      winners' (+ champ HGB) per-seed/per-fold val predictions for stacking
"""
import os

# thread budget: allow model fits some parallelism; feature-build workers pin BLAS to 1 themselves.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

ZOO_SMOKE = os.environ.get("ZOO_SMOKE") == "1"
if ZOO_SMOKE:
    os.environ["SMOKE"] = "1"          # shrink the underlying pool/folds for a quick pass

import json
import pickle
import time
import warnings
import traceback
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _ensemble_tune import load_pool_strat, select_picks, make_hgb
import _ensemble_tune as ET
from _night_stack import (SEARCH_SEEDS, FRESH_SEEDS, NCHAMP_BASE,
                          precompute_fold_wide, build_rows_wide)
from _residual_overnight import pct, TUNED
from _exp_fast import ts_auc_from
from _overnight import N_JOBS

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
MODE = "champ_dep"                                    # builds champ51 + 14 dep cols (65)
C = NCHAMP_BASE
DEP_KEEP = [C + i for i in (0, 1, 2, 3, 4, 5, 10, 11, 12, 13)]
COLS = np.concatenate([np.arange(C), np.asarray(DEP_KEEP, dtype=int)])   # 61-col b_no_accf champion set

SEEDS = [42] if ZOO_SMOKE else SEARCH_SEEDS          # search seeds (feature build held in RAM)
FRESH = [] if ZOO_SMOKE else FRESH_SEEDS             # extra confirm seeds for the final table
N_TOP = 3
BUDGET_SEC = 7.5 * 3600                              # hard stop; write whatever we have
T0 = time.time()

os.makedirs("experiments/json", exist_ok=True)
LOG = "experiments/_model_zoo.log"


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def budget_left():
    return BUDGET_SEC - (time.time() - T0)


# ---------------------------------------------------------------------------
# feature build (light: caches + train rows only, no ranker rows) — held per seed
# ---------------------------------------------------------------------------
def prepare_light(seed):
    ET.SEED = seed
    X, yi, y_val, train_ids, folds = load_pool_strat()
    picks = select_picks(X, train_ids)
    caches = [precompute_fold_wide(X, f, MODE) for f in folds]
    Xtr, ytr, _Ttr, _Str = build_rows_wide(X, yi, picks, MODE)
    return dict(caches=caches, Xtr=np.asarray(Xtr), ytr=np.asarray(ytr), y_val=y_val)


def raw_scores(model, X):
    """Continuous per-row score: predict_proba[:,1] -> decision_function -> predict."""
    try:
        p = np.asarray(model.predict_proba(X))
        return p[:, 1] if (p.ndim == 2 and p.shape[1] > 1) else p.ravel()
    except Exception:
        pass
    if hasattr(model, "decision_function"):
        d = np.asarray(model.decision_function(X))
        return d[:, 0] if d.ndim > 1 else d
    return np.asarray(model.predict(X), dtype=float).ravel()


def eval_config(ctor, params, S, seed, scale=False, sub=None, want_preds=False):
    """Fit one model on champion cols; return (per-fold TS-AUC array, preds|None).

    preds (when requested) = list of pd.Series (percentile scores) aligned to each fold cache.
    """
    Xtr, ytr, caches, y_val = S["Xtr"], S["ytr"], S["caches"], S["y_val"]
    Xt = Xtr[:, COLS].astype(np.float64)

    if sub is not None and len(ytr) > sub:
        rng = np.random.default_rng(seed + 99)
        idx = rng.choice(len(ytr), sub, replace=False)
        Xfit, yfit = Xt[idx], ytr[idx]
    else:
        Xfit, yfit = Xt, ytr

    scaler = StandardScaler().fit(Xfit) if scale else None
    if scaler is not None:
        Xfit = scaler.transform(Xfit)

    model = ctor(params, seed)
    model.fit(Xfit, yfit)
    ref = np.sort(raw_scores(model, Xfit))

    aucs, preds = [], []
    for c in caches:
        L = len(c["index"])
        s = np.zeros(L)
        if len(c["feat"]):
            fv = c["feat"][:, COLS].astype(np.float64)
            if scaler is not None:
                fv = scaler.transform(fv)
            s[c["active"]] = raw_scores(model, fv)
        ser = pd.Series(pct(s, ref), index=c["index"])
        aucs.append(ts_auc_from(ser.to_frame("prediction"), y_val))
        if want_preds:
            preds.append(ser)
    return np.asarray(aucs), (preds if want_preds else None)


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------
def _hgb(p, s):
    return make_hgb(random_state=s, **p)


def _lgbm(p, s):
    return lgb.LGBMClassifier(objective="binary", subsample_freq=1, random_state=s,
                              n_jobs=N_JOBS, verbose=-1, **p)


def _xgb(p, s):
    return xgb.XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=N_JOBS,
                             random_state=s, verbosity=0, **p)


def _cat(p, s):
    return CatBoostClassifier(random_seed=s, thread_count=N_JOBS, verbose=0,
                              allow_writing_files=False, **p)


def _gbdt(p, s):
    return GradientBoostingClassifier(random_state=s, **p)


def _rf(p, s):
    return RandomForestClassifier(n_jobs=N_JOBS, random_state=s, **p)


def _et(p, s):
    return ExtraTreesClassifier(n_jobs=N_JOBS, random_state=s, **p)


def _logreg(p, s):
    return LogisticRegression(max_iter=2000, random_state=s, **p)


def _ridge(p, s):
    return RidgeClassifier(random_state=s, **p)


def _sgd(p, s):
    return SGDClassifier(loss="log_loss", max_iter=2000, random_state=s, **p)


def _lda(p, s):
    return LinearDiscriminantAnalysis(**p)


def _qda(p, s):
    return QuadraticDiscriminantAnalysis(**p)


def _gnb(p, s):
    return GaussianNB(**p)


def _knn(p, s):
    return KNeighborsClassifier(n_jobs=N_JOBS, **p)


def _svc(p, s):
    return SVC(random_state=s, **p)


def _mlp(p, s):
    return MLPClassifier(random_state=s, **p)


# family -> dict(ctor, scale, sub, default, rough)
FAMILIES = {}

FAMILIES["hgb"] = dict(
    ctor=_hgb, scale=False, sub=None,
    default=dict(TUNED),                              # the shipped champion HGB member
    rough=[
        dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20,
             l2_regularization=1.0, max_features=0.7),
        dict(learning_rate=0.03, max_iter=400, max_leaf_nodes=63, min_samples_leaf=20,
             l2_regularization=1.0, max_features=0.5),
        dict(learning_rate=0.02, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40,
             l2_regularization=2.0, max_features=0.5),
        dict(learning_rate=0.08, max_iter=300, max_leaf_nodes=15, min_samples_leaf=50,
             l2_regularization=1.0, max_features=0.8),
    ],
)

FAMILIES["lgbm"] = dict(
    ctor=_lgbm, scale=False, sub=None,
    default=dict(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                 colsample_bytree=0.7, reg_lambda=1.0),
    rough=[
        dict(n_estimators=600, learning_rate=0.02, num_leaves=31, subsample=0.8,
             colsample_bytree=0.6, reg_lambda=1.0),
        dict(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8,
             colsample_bytree=0.7, reg_lambda=5.0),
        dict(n_estimators=800, learning_rate=0.02, num_leaves=15, subsample=0.9,
             colsample_bytree=0.8, reg_lambda=1.0),
        dict(n_estimators=400, learning_rate=0.03, num_leaves=63, subsample=0.7,
             colsample_bytree=0.5, reg_lambda=10.0),
    ],
)

if HAS_XGB:
    FAMILIES["xgb"] = dict(
        ctor=_xgb, scale=False, sub=None,
        default=dict(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8,
                     colsample_bytree=0.7, reg_lambda=1.0, min_child_weight=5),
        rough=[
            dict(n_estimators=600, learning_rate=0.02, max_depth=4, subsample=0.8,
                 colsample_bytree=0.7, reg_lambda=1.0, min_child_weight=5),
            dict(n_estimators=300, learning_rate=0.05, max_depth=8, subsample=0.8,
                 colsample_bytree=0.6, reg_lambda=5.0, min_child_weight=10),
            dict(n_estimators=800, learning_rate=0.02, max_depth=6, subsample=0.9,
                 colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=20),
        ],
    )

if HAS_CAT:
    FAMILIES["catboost"] = dict(
        ctor=_cat, scale=False, sub=None,
        default=dict(iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0),
        rough=[
            dict(iterations=800, learning_rate=0.02, depth=4, l2_leaf_reg=3.0),
            dict(iterations=400, learning_rate=0.05, depth=8, l2_leaf_reg=6.0),
            dict(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=10.0),
        ],
    )

FAMILIES["gbdt"] = dict(
    ctor=_gbdt, scale=False, sub=None,
    default=dict(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8),
    rough=[
        dict(n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.8),
        dict(n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.7),
    ],
)

FAMILIES["rf"] = dict(
    ctor=_rf, scale=False, sub=None,
    default=dict(n_estimators=400, max_features="sqrt", min_samples_leaf=20),
    rough=[
        dict(n_estimators=600, max_features="sqrt", min_samples_leaf=10),
        dict(n_estimators=400, max_features=0.5, min_samples_leaf=20, max_depth=12),
        dict(n_estimators=600, max_features="log2", min_samples_leaf=40),
    ],
)

FAMILIES["extratrees"] = dict(
    ctor=_et, scale=False, sub=None,
    default=dict(n_estimators=400, max_features="sqrt", min_samples_leaf=20),
    rough=[
        dict(n_estimators=600, max_features="sqrt", min_samples_leaf=10),
        dict(n_estimators=400, max_features=0.5, min_samples_leaf=20, max_depth=16),
        dict(n_estimators=800, max_features="log2", min_samples_leaf=40),
    ],
)

FAMILIES["logreg"] = dict(
    ctor=_logreg, scale=True, sub=None,
    default=dict(C=1.0),
    rough=[dict(C=0.1), dict(C=10.0), dict(C=1.0, class_weight="balanced")],
)

FAMILIES["ridge"] = dict(
    ctor=_ridge, scale=True, sub=None,
    default=dict(alpha=1.0),
    rough=[dict(alpha=0.1), dict(alpha=10.0), dict(alpha=100.0)],
)

FAMILIES["sgd"] = dict(
    ctor=_sgd, scale=True, sub=None,
    default=dict(alpha=1e-4, penalty="l2"),
    rough=[dict(alpha=1e-3, penalty="l2"), dict(alpha=1e-4, penalty="elasticnet", l1_ratio=0.15)],
)

FAMILIES["lda"] = dict(
    ctor=_lda, scale=True, sub=None,
    default=dict(solver="lsqr", shrinkage="auto"),
    rough=[dict(solver="lsqr", shrinkage=0.1), dict(solver="lsqr", shrinkage=0.5)],
)

FAMILIES["qda"] = dict(
    ctor=_qda, scale=True, sub=None,
    default=dict(reg_param=0.1),
    rough=[dict(reg_param=0.0), dict(reg_param=0.5)],
)

FAMILIES["gnb"] = dict(
    ctor=_gnb, scale=False, sub=None,
    default=dict(var_smoothing=1e-9),
    rough=[dict(var_smoothing=1e-7), dict(var_smoothing=1e-5)],
)

FAMILIES["knn"] = dict(
    ctor=_knn, scale=True, sub=6000,
    default=dict(n_neighbors=25, weights="distance"),
    rough=[dict(n_neighbors=50, weights="distance"), dict(n_neighbors=15, weights="uniform")],
)

FAMILIES["svc"] = dict(
    ctor=_svc, scale=True, sub=4000,
    default=dict(kernel="rbf", C=1.0, gamma="scale"),
    rough=[dict(kernel="rbf", C=3.0, gamma="scale"), dict(kernel="rbf", C=1.0, gamma=0.01)],
)

FAMILIES["mlp"] = dict(
    ctor=_mlp, scale=True, sub=None,
    default=dict(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=120, early_stopping=True),
    rough=[
        dict(hidden_layer_sizes=(128, 32), alpha=1e-3, max_iter=150, early_stopping=True),
        dict(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=150, early_stopping=True),
    ],
)


# ---------------------------------------------------------------------------
# fine grids (only built for the top-3 families) — capped ~24 configs each
# ---------------------------------------------------------------------------
def _prod(**axes):
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[axes[k] for k in keys])]


def fine_grid(name):
    if name == "hgb":
        g = []
        for lr, it in [(0.015, 800), (0.02, 600), (0.03, 400), (0.05, 250)]:
            for mln in [15, 31, 63]:
                for msl in [20, 40]:
                    g.append(dict(learning_rate=lr, max_iter=it, max_leaf_nodes=mln,
                                  min_samples_leaf=msl, l2_regularization=1.0, max_features=0.5))
        return g
    if name == "lgbm":
        g = []
        for lr, ne in [(0.02, 700), (0.03, 500), (0.05, 300)]:
            for nl in [15, 31, 63]:
                for cs in [0.6, 0.8]:
                    g.append(dict(n_estimators=ne, learning_rate=lr, num_leaves=nl,
                                  subsample=0.8, colsample_bytree=cs, reg_lambda=2.0))
        return g
    if name == "xgb":
        g = []
        for lr, ne in [(0.02, 700), (0.03, 500), (0.05, 300)]:
            for md in [4, 6, 8]:
                for cs in [0.6, 0.8]:
                    g.append(dict(n_estimators=ne, learning_rate=lr, max_depth=md, subsample=0.8,
                                  colsample_bytree=cs, reg_lambda=2.0, min_child_weight=10))
        return g
    if name == "catboost":
        g = []
        for lr, it in [(0.02, 800), (0.03, 600), (0.05, 400)]:
            for dp in [4, 6, 8]:
                for l2 in [3.0, 6.0]:
                    g.append(dict(iterations=it, learning_rate=lr, depth=dp, l2_leaf_reg=l2))
        return g
    if name == "gbdt":
        return _prod(n_estimators=[150, 250], learning_rate=[0.03, 0.05],
                     max_depth=[3, 4], subsample=[0.7, 0.9])
    if name in ("rf", "extratrees"):
        return _prod(n_estimators=[400, 800], max_features=["sqrt", 0.5, "log2"],
                     min_samples_leaf=[5, 20, 40])
    if name == "logreg":
        return _prod(C=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0], class_weight=[None, "balanced"])
    if name == "ridge":
        return [dict(alpha=a) for a in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)]
    if name == "sgd":
        return _prod(alpha=[1e-5, 1e-4, 1e-3, 1e-2], penalty=["l2", "elasticnet"])
    if name == "lda":
        return [dict(solver="lsqr", shrinkage=v) for v in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, "auto")]
    if name == "qda":
        return [dict(reg_param=v) for v in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)]
    if name == "gnb":
        return [dict(var_smoothing=v) for v in (1e-11, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)]
    if name == "knn":
        return _prod(n_neighbors=[10, 25, 50, 100], weights=["uniform", "distance"])
    if name == "svc":
        return _prod(kernel=["rbf"], C=[0.5, 1.0, 3.0, 10.0], gamma=["scale", 0.01, 0.03])
    if name == "mlp":
        return [dict(hidden_layer_sizes=h, alpha=a, max_iter=180, early_stopping=True)
                for h in [(64,), (128,), (128, 32), (256, 64)] for a in (1e-3, 1e-2)]
    return []


# ---------------------------------------------------------------------------
# aggregation / tables
# ---------------------------------------------------------------------------
def agg(aucs_by_seed):
    seed_means = [float(a.mean()) for a in aucs_by_seed]
    return float(np.mean(seed_means)), float(np.std(seed_means)), seed_means


def tsv_write(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def run_config(fam, params, S_by_seed, seeds, scale, sub, want_preds=False):
    """Evaluate one config across seeds. Returns dict(mean,std,seed_means,aucs,preds,secs) or None."""
    t = time.time()
    aucs_by_seed, preds_by_seed = [], {}
    for seed in seeds:
        a, pr = eval_config(fam["ctor"], params, S_by_seed[seed], seed,
                            scale=scale, sub=sub, want_preds=want_preds)
        aucs_by_seed.append(a)
        if want_preds:
            preds_by_seed[seed] = pr
    m, sd, sm = agg(aucs_by_seed)
    return dict(mean=m, std=sd, seed_means=sm, aucs=aucs_by_seed,
                preds=preds_by_seed, secs=time.time() - t)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    open(LOG, "w").close()
    log(f"=== MODEL ZOO  {datetime.now():%F %T}  smoke={ZOO_SMOKE} ===")
    log(f"features: mode={MODE} cols={len(COLS)} (b_no_accf champion set)")
    log(f"search seeds={SEEDS}  fresh seeds={FRESH}  families={list(FAMILIES)}")

    # build + hold search-seed features in RAM (reused across all stages)
    log("\n-- building features (search seeds) --")
    S_by_seed = {}
    for seed in SEEDS:
        t = time.time()
        S_by_seed[seed] = prepare_light(seed)
        n = len(S_by_seed[seed]["ytr"])
        log(f"   seed {seed}: {n} train rows, {len(S_by_seed[seed]['caches'])} folds "
            f"({time.time()-t:.0f}s)")

    tag_scores = {}                       # "fam/tag" -> result dict (search seeds)
    fam_configs = {}                      # fam -> list of (tag, params)

    # ---- STAGE 1+2: defaults + rough grid ----------------------------------
    log("\n== STAGE 1+2: default + rough grid (search seeds) ==")
    default_rows, rough_rows = [], []
    for fam_name, fam in FAMILIES.items():
        configs = [("default", fam["default"])]
        configs += [(f"rough{i}", p) for i, p in enumerate(fam["rough"])]
        fam_configs[fam_name] = configs
        for tag, params in configs:
            if budget_left() < 600:
                log("!! budget nearly exhausted — stopping grid early")
                break
            key = f"{fam_name}/{tag}"
            try:
                r = run_config(fam, params, S_by_seed, SEEDS, fam["scale"], fam["sub"])
            except Exception as e:
                log(f"   {key:<22} FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            tag_scores[key] = r
            row = [fam_name, tag, f"{r['mean']:.4f}", f"{r['std']:.4f}",
                   ",".join(f"{v:.4f}" for v in r["seed_means"]), json.dumps(params), f"{r['secs']:.0f}"]
            (default_rows if tag == "default" else rough_rows).append(row)
            log(f"   {key:<22} mean={r['mean']:.4f} sd={r['std']:.4f} "
                f"seeds={[round(v,4) for v in r['seed_means']]} ({r['secs']:.0f}s)")

    hdr = ["family", "tag", "mean", "std", "seed_means", "params", "secs"]
    tsv_write("_model_zoo_default.tsv", hdr, default_rows)
    tsv_write("_model_zoo_rough.tsv", hdr, rough_rows)

    # default table
    log("\n-- DEFAULT scores (search-seed mean TS-AUC) --")
    for fam_name in FAMILIES:
        k = f"{fam_name}/default"
        if k in tag_scores:
            log(f"   {fam_name:<12} {tag_scores[k]['mean']:.4f}")

    # ---- STAGE 3: best-per-family -> top-3 ---------------------------------
    best_per_fam = {}                     # fam -> (tag, params, mean)
    for fam_name, configs in fam_configs.items():
        cands = [(tag, params, tag_scores[f"{fam_name}/{tag}"]["mean"])
                 for tag, params in configs if f"{fam_name}/{tag}" in tag_scores]
        if cands:
            best_per_fam[fam_name] = max(cands, key=lambda z: z[2])

    ranked = sorted(best_per_fam.items(), key=lambda kv: kv[1][2], reverse=True)
    log("\n-- best config per family (default ∪ rough) --")
    for fam_name, (tag, _p, m) in ranked:
        log(f"   {fam_name:<12} {m:.4f}  ({tag})")

    top3 = [fam_name for fam_name, _ in ranked[:N_TOP]]
    log(f"\n== TOP-{N_TOP} families selected: {top3} ==")

    # ---- STAGE 4: fine grid on top-3 ---------------------------------------
    log("\n== STAGE 4: fine grid (top-3, search seeds) ==")
    fine_rows = []
    best_fine = {}                        # fam -> (tag, params, mean, result)
    for fam_name in top3:
        fam = FAMILIES[fam_name]
        grid = fine_grid(fam_name)
        # seed the fine search with the family's current best config too
        best_prev = best_per_fam[fam_name]
        grid = [best_prev[1]] + grid
        best = best_per_fam[fam_name]      # (tag, params, mean)
        best_res = tag_scores.get(f"{fam_name}/{best[0]}")
        log(f"\n   -- {fam_name}: {len(grid)} fine configs --")
        for i, params in enumerate(grid):
            if budget_left() < 400:
                log("!! budget nearly exhausted — stopping fine grid")
                break
            try:
                r = run_config(fam, params, S_by_seed, SEEDS, fam["scale"], fam["sub"])
            except Exception as e:
                log(f"      fine{i:<3} FAILED: {type(e).__name__}: {e}")
                continue
            fine_rows.append([fam_name, f"fine{i}", f"{r['mean']:.4f}", f"{r['std']:.4f}",
                              ",".join(f"{v:.4f}" for v in r["seed_means"]),
                              json.dumps(params), f"{r['secs']:.0f}"])
            marker = ""
            if r["mean"] > best[2]:
                best = (f"fine{i}", params, r["mean"])
                best_res = r
                marker = "  <== new best"
            log(f"      fine{i:<3} mean={r['mean']:.4f} sd={r['std']:.4f}{marker} ({r['secs']:.0f}s)")
        best_fine[fam_name] = (best[0], best[1], best[2], best_res)
        log(f"   >> {fam_name} best: {best[2]:.4f}  {json.dumps(best[1])}")

    tsv_write("_model_zoo_fine.tsv", hdr, fine_rows)

    # ---- STAGE 5: final table (search + fresh) + prediction dump -----------
    log("\n== STAGE 5: FINAL — winners on search + fresh seeds ==")
    # winners = top-3 best fine configs; also carry the champion HGB member for stacking
    winners = {fam_name: best_fine[fam_name][1] for fam_name in top3}
    if "hgb" not in winners:
        winners["hgb_champ"] = dict(TUNED)

    def fam_of(tag):
        return "hgb" if tag == "hgb_champ" else tag

    # evaluate on search seeds (reuse held S) collecting preds
    final = {tag: {"search": [], "fresh": [], "preds": {}} for tag in winners}
    for tag, params in winners.items():
        fam = FAMILIES[fam_of(tag)]
        for seed in SEEDS:
            a, pr = eval_config(fam["ctor"], params, S_by_seed[seed], seed,
                                scale=fam["scale"], sub=fam["sub"], want_preds=True)
            final[tag]["search"].append(a)
            final[tag]["preds"][seed] = pr

    y_val_by_seed = {seed: S_by_seed[seed]["y_val"] for seed in SEEDS}

    # free search features, then stream fresh seeds one at a time
    S_by_seed.clear()
    for seed in FRESH:
        if budget_left() < 300:
            log("!! budget nearly exhausted — skipping remaining fresh seeds")
            break
        t = time.time()
        Sf = prepare_light(seed)
        log(f"   fresh seed {seed} built ({time.time()-t:.0f}s)")
        y_val_by_seed[seed] = Sf["y_val"]
        for tag, params in winners.items():
            fam = FAMILIES[fam_of(tag)]
            a, pr = eval_config(fam["ctor"], params, Sf, seed,
                                scale=fam["scale"], sub=fam["sub"], want_preds=True)
            final[tag]["fresh"].append(a)
            final[tag]["preds"][seed] = pr
        del Sf

    # final table
    log("\n-- FINAL TABLE (winners) --")
    log(f"   {'model':<14} {'search':>8} {'fresh':>8} {'all':>8}   config")
    final_rows = []
    for tag, params in winners.items():
        sm = np.mean([a.mean() for a in final[tag]["search"]]) if final[tag]["search"] else float("nan")
        fm = np.mean([a.mean() for a in final[tag]["fresh"]]) if final[tag]["fresh"] else float("nan")
        allv = final[tag]["search"] + final[tag]["fresh"]
        am = np.mean([a.mean() for a in allv]) if allv else float("nan")
        log(f"   {tag:<14} {sm:8.4f} {fm:8.4f} {am:8.4f}   {json.dumps(params)}")
        final_rows.append([tag, fam_of(tag), f"{sm:.4f}", f"{fm:.4f}", f"{am:.4f}", json.dumps(params)])
    tsv_write("_model_zoo_final.tsv",
              ["model", "family", "search_mean", "fresh_mean", "all_mean", "params"], final_rows)

    # best-config json
    with open("experiments/json/_model_zoo_best.json", "w") as f:
        json.dump({
            "features": {"mode": MODE, "cols": [int(x) for x in COLS]},
            "seeds": {"search": SEEDS, "fresh": FRESH},
            "top3": top3,
            "best_per_family": {k: {"tag": v[0], "params": v[1], "search_mean": v[2]}
                                for k, v in best_per_fam.items()},
            "winners": {tag: params for tag, params in winners.items()},
        }, f, indent=2, default=str)

    # prediction dump for morning stacking (percentile scores, aligned per fold)
    dump = {"mode": MODE, "cols": [int(x) for x in COLS],
            "y_val_by_seed": y_val_by_seed, "winners": {}}
    for tag in winners:
        dump["winners"][tag] = {
            "family": fam_of(tag), "params": winners[tag],
            "preds": final[tag]["preds"],           # {seed: [pd.Series per fold]}
        }
    with open("_model_zoo_preds.pkl", "wb") as f:
        pickle.dump(dump, f)

    log(f"\n=== DONE {datetime.now():%F %T}  elapsed {(time.time()-T0)/60:.1f} min ===")
    log("outputs: _model_zoo_default.tsv _model_zoo_rough.tsv _model_zoo_fine.tsv "
        "_model_zoo_final.tsv _model_zoo_best.json _model_zoo_preds.pkl")


if __name__ == "__main__":
    main()
