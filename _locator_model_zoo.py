"""Model-zoo and out-of-fold meta screen for the 27-column locator bank.

Running without --full prints the plan and exits. The full run reuses frozen E116
predictions and feature caches, fits only locator specialists, and compares every
candidate directly with E116. It never edits notebooks, builds artifacts, or submits.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import ttest_1samp
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _locator_specialist as locator
import _post_white_pipeline as pipeline

HERE = Path(__file__).resolve().parent
CACHE_DIR = pipeline.CACHE_DIR
OUTPUT = HERE / "_locator_model_zoo.json"
REPORT = HERE / "_locator_model_zoo_summary.md"
LOG = HERE / "_locator_model_zoo.log"
N_JOBS = int(os.environ.get("N_JOBS", "15"))
WEIGHTS = [0.025, 0.05, 0.075, 0.10]
MODELS = ["extra_trees", "random_forest", "point_lgbm", "xgboost", "catboost", "logistic"]
METAS = ["meta_logistic", "meta_lgbm"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def log(message):
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(str(message) + "\n")
    print(message, flush=True)


def predict_score(model, matrix):
    return model.predict_proba(matrix)[:, 1]


def make_specialist(name, seed):
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400, max_features=0.7, min_samples_leaf=20,
            n_jobs=N_JOBS, random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, max_features=0.7, min_samples_leaf=20,
            n_jobs=N_JOBS, random_state=seed,
        )
    if name == "point_lgbm":
        return lgb.LGBMClassifier(
            objective="binary", n_estimators=300, learning_rate=0.03,
            num_leaves=15, min_child_samples=30, reg_lambda=2.0,
            colsample_bytree=0.8, random_state=seed, n_jobs=N_JOBS, verbose=-1,
        )
    if name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, tree_method="hist", eval_metric="logloss",
            n_jobs=N_JOBS, random_state=seed, verbosity=0,
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=400, learning_rate=0.03, depth=6, l2_leaf_reg=10.0,
            random_seed=seed, thread_count=N_JOBS, verbose=0,
            allow_writing_files=False,
        )
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=3000, random_state=seed),
        )
    raise ValueError(name)


def fit_specialist(name, data_seed, seed_data, columns):
    path = CACHE_DIR / f"specialist_e120_{name}_seed{data_seed}.pkl"
    if path.exists():
        saved = pipeline.load_pickle(path)
        if saved.get("columns") == columns.tolist():
            return saved["predictions"]

    model = make_specialist(name, 42)
    train = seed_data["Xtr"][:, columns]
    model.fit(train, seed_data["ytr"])
    reference = np.sort(predict_score(model, train))
    predictions = []
    for cache in seed_data["caches"]:
        scores = np.zeros(len(cache["index"]), dtype=np.float64)
        if len(cache["feat"]):
            scores[cache["active"]] = predict_score(model, cache["feat"][:, columns])
        predictions.append(
            np.asarray(pipeline.HP.pct(scores, reference), dtype=np.float32)
        )
    pipeline.dump_pickle(path, {
        "model": name,
        "columns": columns.tolist(),
        "predictions": predictions,
    })
    return predictions


def exact_e116(seed_data, data_seed, stage5):
    stage1 = pipeline.load_pickle(pipeline.stage1_path(data_seed))
    stage2 = pipeline.load_pickle(pipeline.stage2_path(data_seed, 42))
    configured = pipeline.configured_predictions(
        stage1, stage2, int(stage5["winner_mask"]), list(stage5["accepted_params"])
    )
    return pipeline.blend_predictions(configured, dict(stage5["weights"]))


def fold_labels(seed_data):
    labels = []
    for cache in seed_data["caches"]:
        merged = cache["index"].to_frame(index=False).merge(
            seed_data["y_val"].reset_index(), how="left", on=["id", "time"]
        )
        target = merged["target"].to_numpy(dtype=np.int64)
        if len(target) != len(cache["index"]) or np.isnan(target).any():
            raise RuntimeError("Could not align validation labels for meta-learning")
        labels.append(target)
    return labels


def make_meta(name, seed):
    if name == "meta_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=3000, random_state=seed),
        )
    if name == "meta_lgbm":
        return lgb.LGBMClassifier(
            objective="binary", n_estimators=120, learning_rate=0.03,
            num_leaves=7, min_child_samples=100, reg_lambda=5.0,
            colsample_bytree=1.0, random_state=seed, n_jobs=N_JOBS, verbose=-1,
        )
    raise ValueError(name)


def meta_predictions(name, feature_folds, label_folds, seed):
    predictions = []
    for held in range(len(feature_folds)):
        train_x = np.vstack([values for index, values in enumerate(feature_folds) if index != held])
        train_y = np.concatenate([values for index, values in enumerate(label_folds) if index != held])
        model = make_meta(name, seed + held)
        model.fit(train_x, train_y)
        predictions.append(predict_score(model, feature_folds[held]))
    return predictions


def aggregate(candidate_by_seed, baseline_by_seed, metas_by_seed):
    rows = []
    for seed in pipeline.DATA_SEEDS:
        candidate = pipeline.fold_aucs(candidate_by_seed[seed], metas_by_seed[seed])
        baseline = pipeline.fold_aucs(baseline_by_seed[seed], metas_by_seed[seed])
        rows.append(pipeline.paired_summary(candidate, baseline))
    return pipeline.aggregate_seed_rows(rows)


def best_blend(specialist_by_seed, baseline_by_seed, metas_by_seed):
    rows = {}
    for weight in WEIGHTS:
        candidate = {
            seed: [
                (1.0 - weight) * base + weight * extra
                for base, extra in zip(baseline_by_seed[seed], specialist_by_seed[seed])
            ]
            for seed in pipeline.DATA_SEEDS
        }
        rows[str(weight)] = aggregate(candidate, baseline_by_seed, metas_by_seed)
    best_weight, best = max(rows.items(), key=lambda item: item[1]["mean_delta"])
    return {"weights": rows, "best_weight": float(best_weight), "best": best}


def result_row(name, row):
    cells = [
        name, f"{row['candidate_auc']:.6f}", f"{row['mean_delta']:+.6f}",
        f"{row['min_delta']:+.6f}", f"{row['positive']}/{row['total']}",
        f"{row['aggregate_t']:+.2f}",
    ]
    for seed in pipeline.DATA_SEEDS:
        seed_row = row["seeds"][str(seed)]
        cells.append(f"{seed_row['delta']:+.6f} / {seed_row['t']:+.2f}")
    return "| " + " | ".join(cells) + " |"


def markdown(results):
    headers = ["config", "absolute TS-AUC", "mean_d", "min_d", "pos", "t_agg"]
    headers.extend(f"seed {seed} d / t" for seed in pipeline.DATA_SEEDS)
    lines = [
        "# Locator Model Zoo Against E116", "",
        "Every fixed blend and out-of-fold meta-learner is compared directly with exact E116.", "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name in MODELS:
        model = results["models"][name]
        lines.append(result_row(f"{name} @ {model['best_weight']:.3f}", model["best"]))
    for name in METAS:
        lines.append(result_row(name, results["metas"][name]))
    return "\n".join(lines) + "\n"


def run():
    LOG.write_text("", encoding="utf-8")
    stage5 = locator.load_manifest()
    columns = pipeline.specialist_columns()
    baseline_by_seed = {}
    metas_by_seed = {}
    specialists = {name: {} for name in MODELS}
    labels_by_seed = {}

    for data_seed in pipeline.DATA_SEEDS:
        started = time.time()
        seed_data = pipeline.CR.prepare_seed(data_seed)
        metas_by_seed[data_seed] = pipeline.CR.make_folds_meta(seed_data)
        baseline_by_seed[data_seed] = exact_e116(seed_data, data_seed, stage5)
        labels_by_seed[data_seed] = fold_labels(seed_data)
        for name in MODELS:
            fit_started = time.time()
            specialists[name][data_seed] = fit_specialist(name, data_seed, seed_data, columns)
            log(f"seed {data_seed} {name} complete in {time.time() - fit_started:.0f}s")
        log(f"seed {data_seed} complete in {(time.time() - started) / 60:.1f}m")

    model_results = {
        name: best_blend(specialists[name], baseline_by_seed, metas_by_seed)
        for name in MODELS
    }

    meta_results = {}
    for meta_name in METAS:
        candidate_by_seed = {}
        for data_seed in pipeline.DATA_SEEDS:
            feature_folds = [
                np.column_stack([
                    baseline_by_seed[data_seed][fold],
                    *[specialists[name][data_seed][fold] for name in MODELS],
                ])
                for fold in range(len(baseline_by_seed[data_seed]))
            ]
            candidate_by_seed[data_seed] = meta_predictions(
                meta_name, feature_folds, labels_by_seed[data_seed], data_seed
            )
        meta_results[meta_name] = aggregate(candidate_by_seed, baseline_by_seed, metas_by_seed)

    all_rows = [model_results[name]["best"] for name in MODELS]
    all_rows.extend(meta_results.values())
    best = max(all_rows, key=lambda row: row["mean_delta"])
    payload = {
        "baseline": "E116",
        "columns": columns.tolist(),
        "models": model_results,
        "metas": meta_results,
        "best": best,
        "passes_gate": bool(
            best["mean_delta"] >= pipeline.PROMOTION_DELTA
            and best["positive"] == len(pipeline.DATA_SEEDS)
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT.write_text(markdown(payload), encoding="utf-8")
    log(
        f"best mean_d={best['mean_delta']:+.6f} "
        f"positive={best['positive']}/{best['total']} gate={payload['passes_gate']}"
    )


def main():
    args = parse_args()
    if not args.full:
        print("Locator model zoo is prepared but not launched.")
        print("Run: env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u "
              "_locator_model_zoo.py --full > _locator_model_zoo.out 2>&1")
        return
    run()


if __name__ == "__main__":
    main()