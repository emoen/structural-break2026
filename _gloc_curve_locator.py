"""E122 full white-GLOC likelihood-curve specialists against exact E116.

Running without --full prints the plan and exits. The full run builds a dedicated
48-bin curve cache, verifies its established 364-column prefix against E116's
cache, fits one compact ranker and one pointwise classifier, and screens fixed
blend weights. It never edits notebooks, builds submission artifacts, or submits.
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("GLOC6", "1")
os.environ.setdefault("WBLOCK", "1")
os.environ.setdefault("GLOC_CURVE", "1")

import lightgbm as lgb
import numpy as np

import _locator_specialist as locator
import _post_white_pipeline as pipeline

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "_gloc_curve_locator.json"
REPORT = HERE / "_gloc_curve_locator_summary.md"
LOG = HERE / "_gloc_curve_locator.log"
PREDICTION_CACHE = HERE / "_gloc_curve_locator_cache"
FROZEN_FEATURE_CACHE = HERE / "_calib_cache_gloc6_wblock"
WEIGHTS = [0.025, 0.05, 0.075, 0.10]
MODELS = ["curve_ranker", "curve_point_lgbm"]
N_JOBS = int(os.environ.get("N_JOBS", "15"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def log(message):
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(str(message) + "\n")
    print(message, flush=True)


def frozen_seed_path(seed):
    return FROZEN_FEATURE_CACHE / f"seed_{seed}.pkl"


def validate_alignment(seed, curve_data):
    path = frozen_seed_path(seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen E116 feature cache: {path}")
    with path.open("rb") as stream:
        frozen = pickle.load(stream)
    for name in ("ytr", "Rl", "Rt"):
        if not np.array_equal(curve_data[name], frozen[name]):
            raise RuntimeError(f"seed {seed}: {name} changed")
    for name in ("Xtr", "Rr"):
        if curve_data[name].shape[0] != frozen[name].shape[0]:
            raise RuntimeError(f"seed {seed}: {name} row count changed")
        difference = float(np.max(np.abs(curve_data[name][:, :364] - frozen[name])))
        if difference != 0.0:
            raise RuntimeError(f"seed {seed}: {name} prefix maxdiff={difference}")
    if len(curve_data["caches"]) != len(frozen["caches"]):
        raise RuntimeError(f"seed {seed}: fold count changed")
    for fold, (candidate, baseline) in enumerate(zip(curve_data["caches"], frozen["caches"])):
        if not candidate["index"].equals(baseline["index"]):
            raise RuntimeError(f"seed {seed} fold {fold}: index changed")
        if not np.array_equal(candidate["active"], baseline["active"]):
            raise RuntimeError(f"seed {seed} fold {fold}: active rows changed")
        difference = float(np.max(np.abs(candidate["feat"][:, :364] - baseline["feat"])))
        if difference != 0.0:
            raise RuntimeError(f"seed {seed} fold {fold}: feature prefix maxdiff={difference}")
    del frozen
    log(f"seed {seed} cache alignment exact; established prefix maxdiff=0.0")


def point_predictions(seed_data, columns):
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        reg_lambda=2.0,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    train = seed_data["Xtr"][:, columns]
    model.fit(train, seed_data["ytr"])
    reference = np.sort(model.predict_proba(train)[:, 1])
    predictions = []
    for cache in seed_data["caches"]:
        scores = np.zeros(len(cache["index"]), dtype=np.float64)
        if len(cache["feat"]):
            scores[cache["active"]] = model.predict_proba(cache["feat"][:, columns])[:, 1]
        predictions.append(np.asarray(pipeline.HP.pct(scores, reference), dtype=np.float32))
    return predictions


def specialist_predictions(name, seed, seed_data, columns):
    PREDICTION_CACHE.mkdir(exist_ok=True)
    path = PREDICTION_CACHE / f"{name}_seed{seed}.pkl"
    if path.exists():
        saved = pipeline.load_pickle(path)
        if saved.get("columns") == columns.tolist():
            return saved["predictions"]
    if name == "curve_ranker":
        params = {
            **pipeline.HP.RANK_BASE,
            "n_estimators": 200,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "min_child_samples": 30,
        }
        predictions = pipeline.fit_predictions("ranker", params, 42, seed_data, columns)
    elif name == "curve_point_lgbm":
        predictions = point_predictions(seed_data, columns)
    else:
        raise ValueError(name)
    pipeline.dump_pickle(path, {
        "model": name,
        "columns": columns.tolist(),
        "predictions": predictions,
    })
    return predictions


def exact_e116(seed, stage5):
    stage1 = pipeline.load_pickle(pipeline.stage1_path(seed))
    stage2 = pipeline.load_pickle(pipeline.stage2_path(seed, 42))
    configured = pipeline.configured_predictions(
        stage1, stage2, int(stage5["winner_mask"]), list(stage5["accepted_params"])
    )
    return pipeline.blend_predictions(configured, dict(stage5["weights"]))


def screen_model(predictions, baselines, metas):
    results = {}
    for weight in WEIGHTS:
        rows = []
        for seed in pipeline.DATA_SEEDS:
            blended = [
                (1.0 - weight) * base + weight * specialist
                for base, specialist in zip(baselines[seed], predictions[seed])
            ]
            candidate_auc = pipeline.fold_aucs(blended, metas[seed])
            baseline_auc = pipeline.fold_aucs(baselines[seed], metas[seed])
            rows.append(pipeline.paired_summary(candidate_auc, baseline_auc))
        results[str(weight)] = pipeline.aggregate_seed_rows(rows)
    best_weight, best = max(results.items(), key=lambda item: item[1]["mean_delta"])
    return {"weights": results, "best_weight": float(best_weight), "best": best}


def markdown(payload):
    headers = ["config", "absolute TS-AUC", "mean_d", "min_d", "pos", "t_agg"]
    headers.extend(f"seed {seed} d / t" for seed in pipeline.DATA_SEEDS)
    lines = [
        "# E122 Full White-GLOC Likelihood-Curve Screen",
        "",
        "Specialists consume the 48-bin causal, affine-invariant Gaussian gain curve only.",
        "Baseline is exact E116 with its accepted allocation and fixed-CDF weights.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name in MODELS:
        result = payload["models"][name]
        row = result["best"]
        cells = [
            f"{name} @ {result['best_weight']:.3f}",
            f"{row['candidate_auc']:.6f}",
            f"{row['mean_delta']:+.6f}",
            f"{row['min_delta']:+.6f}",
            f"{row['positive']}/{row['total']}",
            f"{row['aggregate_t']:+.2f}",
        ]
        for seed in pipeline.DATA_SEEDS:
            seed_row = row["seeds"][str(seed)]
            cells.append(f"{seed_row['delta']:+.6f} / {seed_row['t']:+.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run():
    LOG.write_text("", encoding="utf-8")
    stage5 = locator.load_manifest()
    columns = np.asarray(pipeline.CR.CAND_SLICE["gloc_curve"], dtype=int)
    if not np.array_equal(columns, np.arange(364, 412)):
        raise RuntimeError(f"Unexpected curve columns: {columns.tolist()}")

    baselines = {}
    metas = {}
    predictions = {name: {} for name in MODELS}
    for seed in pipeline.DATA_SEEDS:
        started = time.time()
        seed_data = pipeline.CR.prepare_seed(seed)
        validate_alignment(seed, seed_data)
        baselines[seed] = exact_e116(seed, stage5)
        metas[seed] = pipeline.CR.make_folds_meta(seed_data)
        for name in MODELS:
            fit_started = time.time()
            predictions[name][seed] = specialist_predictions(name, seed, seed_data, columns)
            log(f"seed {seed} {name} complete in {time.time() - fit_started:.0f}s")
        log(f"seed {seed} complete in {(time.time() - started) / 60:.1f}m")

    results = {name: screen_model(predictions[name], baselines, metas) for name in MODELS}
    best_name, best_result = max(
        results.items(), key=lambda item: item[1]["best"]["mean_delta"]
    )
    best = best_result["best"]
    payload = {
        "experiment": "E122",
        "baseline": "E116",
        "representation": "white_gloc_gain_curve_log1p_per_n_48",
        "columns": columns.tolist(),
        "models": results,
        "best_model": best_name,
        "best_weight": best_result["best_weight"],
        "best": best,
        "passes_gate": bool(
            best["mean_delta"] >= pipeline.PROMOTION_DELTA
            and best["positive"] == len(pipeline.DATA_SEEDS)
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT.write_text(markdown(payload), encoding="utf-8")
    log(
        f"best={best_name} weight={best_result['best_weight']:.3f} "
        f"mean_d={best['mean_delta']:+.6f} positive={best['positive']}/{best['total']} "
        f"gate={payload['passes_gate']}"
    )


def main():
    args = parse_args()
    if not args.full:
        print("E122 curve-locator screen is prepared but not launched.")
        print("Run: env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u "
              "_gloc_curve_locator.py --full > _gloc_curve_locator.out 2>&1")
        return
    run()


if __name__ == "__main__":
    main()
