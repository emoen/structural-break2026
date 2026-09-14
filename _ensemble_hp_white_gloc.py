"""Paired four-member hyperparameter screen on calib + WBLOCK + white GLOC.

The feature cache, sampled rows, folds, and model seed are shared by the baseline and every
candidate. Each candidate replaces exactly one member in the deployed fixed-CDF blend. The
random forest is screened using the first 250 trees of the exact 800-tree baseline, then the
top two parameter settings are refit and compared at 800 trees.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("GLOC6", "1")
os.environ.setdefault("WBLOCK", "1")

import json
import math
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

import _calib_rankerrf as CR


HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "_ensemble_hp_white_gloc.log")
RESULTS = os.path.join(HERE, "_ensemble_hp_white_gloc.json")
DATA_SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7,123").split(",") if s]
MODEL_SEED = int(os.environ.get("MODEL_SEED", "42"))
N_JOBS = int(os.environ.get("N_JOBS", "15"))
RF_SCREEN_TREES = int(os.environ.get("RF_SCREEN_TREES", "250"))
SMOKE = os.environ.get("SMOKE") == "1"

WEIGHTS = {"ranker": 0.45, "rf": 0.40, "xgb": 0.10, "catboost": 0.05}
FEATURE_ARMS = ["czrb_all79", "wblock", "white_gloc"]

RANK_BASE = dict(
    objective="rank_xendcg", n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, reg_lambda=1.0, subsample=1.0, subsample_freq=0,
    colsample_bytree=0.7,
)
RF_BASE = dict(n_estimators=800, max_features=0.5, min_samples_leaf=20)
XGB_BASE = dict(
    n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8,
    colsample_bytree=0.7, reg_lambda=1.0, min_child_weight=5,
)
CAT_BASE = dict(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=10.0)


def one_factor_grid(base, changes):
    out = [("baseline", dict(base))]
    for name, override in changes:
        params = dict(base)
        params.update(override)
        out.append((name, params))
    return out


RANK_GRID = one_factor_grid(RANK_BASE, [
    ("leaves15", {"num_leaves": 15}),
    ("leaves63", {"num_leaves": 63}),
    ("child10", {"min_child_samples": 10}),
    ("child40", {"min_child_samples": 40}),
    ("cols055", {"colsample_bytree": 0.55}),
    ("cols085", {"colsample_bytree": 0.85}),
    ("cols100", {"colsample_bytree": 1.0}),
    ("l2_3", {"reg_lambda": 3.0}),
    ("l2_5", {"reg_lambda": 5.0}),
    ("slow500", {"n_estimators": 500, "learning_rate": 0.03}),
])
RF_GRID = one_factor_grid({**RF_BASE, "n_estimators": RF_SCREEN_TREES}, [
    ("mf030", {"max_features": 0.3}),
    ("mf070", {"max_features": 0.7}),
    ("leaf10", {"min_samples_leaf": 10}),
    ("leaf40", {"min_samples_leaf": 40}),
    ("mf030_leaf10", {"max_features": 0.3, "min_samples_leaf": 10}),
    ("mf030_leaf40", {"max_features": 0.3, "min_samples_leaf": 40}),
    ("mf070_leaf10", {"max_features": 0.7, "min_samples_leaf": 10}),
    ("mf070_leaf40", {"max_features": 0.7, "min_samples_leaf": 40}),
])
XGB_GRID = one_factor_grid(XGB_BASE, [
    ("depth4", {"max_depth": 4}),
    ("depth5", {"max_depth": 5}),
    ("depth8", {"max_depth": 8}),
    ("child2", {"min_child_weight": 2}),
    ("child10", {"min_child_weight": 10}),
    ("rows065", {"subsample": 0.65}),
    ("rows100", {"subsample": 1.0}),
    ("cols050", {"colsample_bytree": 0.5}),
    ("cols085", {"colsample_bytree": 0.85}),
    ("cols100", {"colsample_bytree": 1.0}),
    ("l2_3", {"reg_lambda": 3.0}),
    ("l2_10", {"reg_lambda": 10.0}),
])
CAT_GRID = one_factor_grid(CAT_BASE, [
    ("depth5", {"depth": 5}),
    ("depth7", {"depth": 7}),
    ("depth8", {"depth": 8}),
    ("l2_3", {"l2_leaf_reg": 3.0}),
    ("l2_30", {"l2_leaf_reg": 30.0}),
    ("rs0", {"random_strength": 0.0}),
    ("rs05", {"random_strength": 0.5}),
    ("rs2", {"random_strength": 2.0}),
    ("bern08", {"bootstrap_type": "Bernoulli", "subsample": 0.8}),
    ("bayes05", {"bootstrap_type": "Bayesian", "bagging_temperature": 0.5}),
])
GRIDS = {"ranker": RANK_GRID, "rf": RF_GRID, "xgb": XGB_GRID, "catboost": CAT_GRID}


def log(message):
    line = str(message)
    with open(LOG, "a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def proba(model, matrix):
    if isinstance(model, lgb.LGBMRanker):
        return model.predict(matrix)
    return model.predict_proba(matrix)[:, 1]


def pct(values, reference):
    return np.searchsorted(reference, values, side="right") / len(reference)


def fit_member(name, params, sd, cols):
    if name == "ranker":
        rows, labels, groups = CR.sort_into_groups(sd["Rr"][:, cols], sd["Rl"], sd["Rt"])
        model = lgb.LGBMRanker(
            random_state=MODEL_SEED, n_jobs=N_JOBS, verbose=-1, **params
        )
        with threadpool_limits(limits=1):
            model.fit(rows, labels, group=groups)
        reference_matrix = rows
    elif name == "rf":
        model = RandomForestClassifier(
            n_jobs=N_JOBS, random_state=MODEL_SEED, **params
        )
        model.fit(sd["Xtr"][:, cols], sd["ytr"])
        reference_matrix = sd["Xtr"][:, cols]
    elif name == "xgb":
        model = xgb.XGBClassifier(
            tree_method="hist", eval_metric="logloss", n_jobs=N_JOBS,
            random_state=MODEL_SEED, verbosity=0, **params
        )
        model.fit(sd["Xtr"][:, cols], sd["ytr"])
        reference_matrix = sd["Xtr"][:, cols]
    elif name == "catboost":
        model = CatBoostClassifier(
            random_seed=MODEL_SEED, thread_count=N_JOBS, verbose=0,
            allow_writing_files=False, **params
        )
        model.fit(sd["Xtr"][:, cols], sd["ytr"])
        reference_matrix = sd["Xtr"][:, cols]
    else:
        raise ValueError(name)

    reference = np.sort(proba(model, reference_matrix))
    predictions = []
    for cache in sd["caches"]:
        scores = np.zeros(len(cache["index"]), dtype=np.float64)
        if len(cache["feat"]):
            scores[cache["active"]] = proba(model, cache["feat"][:, cols])
        predictions.append(pct(scores, reference))
    return model, predictions


def forest_prefix_predictions(model, sd, cols, n_trees):
    def predict_prefix(matrix):
        total = np.zeros(len(matrix), dtype=np.float64)
        for tree in model.estimators_[:n_trees]:
            total += tree.predict_proba(matrix)[:, 1]
        return total / n_trees

    reference = np.sort(predict_prefix(sd["Xtr"][:, cols]))
    predictions = []
    for cache in sd["caches"]:
        scores = np.zeros(len(cache["index"]), dtype=np.float64)
        if len(cache["feat"]):
            scores[cache["active"]] = predict_prefix(cache["feat"][:, cols])
        predictions.append(pct(scores, reference))
    return predictions


def fold_aucs(predictions, metas):
    aucs = []
    for fold_predictions, (_length, _active, groups) in zip(predictions, metas):
        weighted = total = 0.0
        for rows, labels, weight in groups:
            auc = CR._auc(labels, fold_predictions[rows])
            if auc is not None:
                weighted += weight * auc
                total += weight
        aucs.append(weighted / total if total else 0.5)
    return np.asarray(aucs)


def blend(member_predictions):
    return [
        sum(WEIGHTS[name] * member_predictions[name][fold]
            for name in WEIGHTS)
        for fold in range(len(next(iter(member_predictions.values()))))
    ]


def paired_stats(candidate, baseline):
    delta = np.asarray(candidate) - np.asarray(baseline)
    if len(delta) > 1 and np.std(delta, ddof=1) > 0:
        t_score = float(ttest_1samp(delta, 0.0).statistic)
    else:
        t_score = math.inf if delta.mean() > 0 else -math.inf
    return float(np.mean(delta)), t_score


def add_result(store, member, config, seed, candidate, baseline, params, stage="screen"):
    mean_delta, t_score = paired_stats(candidate, baseline)
    key = f"{stage}|{member}|{config}"
    store.setdefault(key, {"member": member, "config": config, "stage": stage,
                           "params": params, "seeds": {}})
    store[key]["seeds"][str(seed)] = {
        "baseline_auc": float(np.mean(baseline)),
        "candidate_auc": float(np.mean(candidate)),
        "delta": mean_delta,
        "t": t_score,
    }
    log(f"  {member:<8} {config:<16} auc={np.mean(candidate):.6f} "
        f"d={mean_delta:+.6f} t={t_score:+.2f}")


def aggregate(entry):
    rows = list(entry["seeds"].values())
    deltas = np.asarray([row["delta"] for row in rows])
    return float(deltas.mean()), float(deltas.min()), int(np.sum(deltas > 0)), len(deltas)


def write_results(store):
    with open(RESULTS, "w") as stream:
        json.dump(store, stream, indent=2, sort_keys=True)


def main():
    open(LOG, "w").close()
    cols = CR.cand_cols(FEATURE_ARMS)
    log(f"=== WHITE-GLOC FOUR-MEMBER HP SCREEN {datetime.now():%F %T} ===")
    log(f"data_seeds={DATA_SEEDS} model_seed={MODEL_SEED} cols={len(cols)} "
        f"rf_screen_trees={RF_SCREEN_TREES} jobs={N_JOBS}")
    log("candidate_counts=" + str({name: len(grid) - 1 for name, grid in GRIDS.items()}))
    if SMOKE:
        log("SMOKE=1: restricting to first data seed and first non-baseline candidate/member")

    seeds = DATA_SEEDS[:1] if SMOKE else DATA_SEEDS
    store = {}
    for seed in seeds:
        started = time.time()
        sd = CR.prepare_seed(seed)
        metas = CR.make_folds_meta(sd)
        log(f"\n[seed {seed}] cache ready Xtr={sd['Xtr'].shape} Rr={sd['Rr'].shape}")

        base_models = {}
        base_predictions = {}
        for member, params in (("ranker", RANK_BASE), ("rf", RF_BASE),
                               ("xgb", XGB_BASE), ("catboost", CAT_BASE)):
            fit_started = time.time()
            base_models[member], base_predictions[member] = fit_member(member, params, sd, cols)
            log(f"  baseline {member:<8} fit+predict {time.time()-fit_started:.0f}s")
        base_aucs = fold_aucs(blend(base_predictions), metas)
        log(f"  four-member baseline auc={base_aucs.mean():.6f}")

        grids = {name: grid[:2] for name, grid in GRIDS.items()} if SMOKE else GRIDS
        for member, grid in grids.items():
            if member == "rf":
                rf_screen_base = dict(base_predictions)
                rf_screen_base["rf"] = forest_prefix_predictions(
                    base_models["rf"], sd, cols, RF_SCREEN_TREES
                )
                member_baseline_aucs = fold_aucs(blend(rf_screen_base), metas)
            else:
                member_baseline_aucs = base_aucs

            for config, params in grid[1:]:
                fit_started = time.time()
                _model, candidate_predictions = fit_member(member, params, sd, cols)
                mixed = dict(base_predictions)
                if member == "rf":
                    mixed["rf"] = candidate_predictions
                    for other in ("ranker", "xgb", "catboost"):
                        mixed[other] = rf_screen_base[other]
                else:
                    mixed[member] = candidate_predictions
                candidate_aucs = fold_aucs(blend(mixed), metas)
                add_result(store, member, config, seed, candidate_aucs,
                           member_baseline_aucs, params)
                log(f"    elapsed={time.time()-fit_started:.0f}s")
                write_results(store)
        log(f"[seed {seed}] done in {(time.time()-started)/60:.1f} min")

    if not SMOKE:
        rf_entries = [entry for entry in store.values()
                      if entry["member"] == "rf" and entry["stage"] == "screen"]
        rf_entries.sort(key=lambda entry: aggregate(entry)[0], reverse=True)
        finalists = rf_entries[:2]
        log("\n=== RANDOM FOREST 800-TREE FINALISTS ===")
        for entry in finalists:
            log(f"  {entry['config']}: screen mean_d={aggregate(entry)[0]:+.6f}")
        for seed in seeds:
            sd = CR.prepare_seed(seed)
            metas = CR.make_folds_meta(sd)
            _, baseline_rf = fit_member("rf", RF_BASE, sd, cols)
            other_predictions = {}
            for member, params in (("ranker", RANK_BASE), ("xgb", XGB_BASE),
                                   ("catboost", CAT_BASE)):
                _, other_predictions[member] = fit_member(member, params, sd, cols)
            baseline_predictions = {**other_predictions, "rf": baseline_rf}
            baseline_aucs = fold_aucs(blend(baseline_predictions), metas)
            for entry in finalists:
                params = dict(entry["params"])
                params["n_estimators"] = 800
                _, candidate_rf = fit_member("rf", params, sd, cols)
                candidate_aucs = fold_aucs(blend({**other_predictions, "rf": candidate_rf}), metas)
                add_result(store, "rf", entry["config"], seed, candidate_aucs,
                           baseline_aucs, params, stage="final800")
                write_results(store)

    log("\n=== SCREEN SUMMARY ===")
    ranked = sorted(store.values(), key=lambda entry: aggregate(entry)[0], reverse=True)
    log(f"{'stage/member/config':<38} {'mean_d':>10} {'min_d':>10} {'pos':>7}")
    for entry in ranked:
        mean_delta, min_delta, positive, total = aggregate(entry)
        log(f"{entry['stage']+'/'+entry['member']+'/'+entry['config']:<38} "
            f"{mean_delta:+10.6f} {min_delta:+10.6f} {positive:>3}/{total}")
    write_results(store)
    log(f"=== DONE {datetime.now():%F %T} ===")


if __name__ == "__main__":
    main()