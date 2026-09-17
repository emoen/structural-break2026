"""E123 fixed-position multichannel white-GLOC curve against exact E116.

Running without --full prints the plan and exits. The full run builds a dedicated
192-column curve cache, verifies its established 364-column prefix against E116's
cache, fits one compact grouped ranker, and screens fixed blend weights. It never
edits notebooks, builds submission artifacts, or submits.
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
os.environ.setdefault("GLOC_CURVE2", "1")

import numpy as np

import _locator_specialist as locator
import _post_white_pipeline as pipeline

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "_gloc_curve2_locator.json"
REPORT = HERE / "_gloc_curve2_locator_summary.md"
LOG = HERE / "_gloc_curve2_locator.log"
PREDICTION_CACHE = HERE / "_gloc_curve2_locator_cache"
FROZEN_FEATURE_CACHE = HERE / "_calib_cache_gloc6_wblock"
WEIGHTS = [0.025, 0.05, 0.075, 0.10]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def log(message):
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(str(message) + "\n")
    print(message, flush=True)


def validate_alignment(seed, curve_data):
    path = FROZEN_FEATURE_CACHE / f"seed_{seed}.pkl"
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
    log(f"seed {seed} cache alignment exact; established prefix maxdiff=0.0")


def specialist_predictions(seed, seed_data, columns):
    PREDICTION_CACHE.mkdir(exist_ok=True)
    path = PREDICTION_CACHE / f"curve2_ranker_seed{seed}.pkl"
    if path.exists():
        saved = pipeline.load_pickle(path)
        if saved.get("columns") == columns.tolist():
            return saved["predictions"]
    params = {
        **pipeline.HP.RANK_BASE,
        "n_estimators": 200,
        "learning_rate": 0.04,
        "num_leaves": 15,
        "min_child_samples": 30,
    }
    predictions = pipeline.fit_predictions("ranker", params, 42, seed_data, columns)
    pipeline.dump_pickle(path, {
        "model": "curve2_ranker",
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


def screen(predictions, baselines, metas):
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
        "# E123 Fixed-Position Multichannel White-GLOC Curve",
        "",
        "The compact grouped ranker consumes 48-bin amplitude, normalized-shape,",
        "posterior-mass, and validity-mask channels on a fixed relative-position grid.",
        "Baseline is exact E116 with its accepted allocation and fixed-CDF weights.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for weight in WEIGHTS:
        row = payload["weights"][str(weight)]
        cells = [
            f"curve2_ranker @ {weight:.3f}",
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
    columns = np.asarray(pipeline.CR.CAND_SLICE["gloc_curve2"], dtype=int)
    if not np.array_equal(columns, np.arange(364, 556)):
        raise RuntimeError(f"Unexpected curve2 columns: {columns.tolist()}")

    baselines = {}
    metas = {}
    predictions = {}
    for seed in pipeline.DATA_SEEDS:
        started = time.time()
        seed_data = pipeline.CR.prepare_seed(seed)
        validate_alignment(seed, seed_data)
        baselines[seed] = exact_e116(seed, stage5)
        metas[seed] = pipeline.CR.make_folds_meta(seed_data)
        fit_started = time.time()
        predictions[seed] = specialist_predictions(seed, seed_data, columns)
        log(f"seed {seed} curve2_ranker complete in {time.time() - fit_started:.0f}s")
        log(f"seed {seed} complete in {(time.time() - started) / 60:.1f}m")

    result = screen(predictions, baselines, metas)
    best = result["best"]
    payload = {
        "experiment": "E123",
        "baseline": "E116",
        "representation": "fixed_grid_48_x_amplitude_shape_posterior_validity",
        "columns": columns.tolist(),
        **result,
        "passes_gate": bool(
            best["mean_delta"] >= pipeline.PROMOTION_DELTA
            and best["positive"] == len(pipeline.DATA_SEEDS)
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT.write_text(markdown(payload), encoding="utf-8")
    log(
        f"best weight={result['best_weight']:.3f} mean_d={best['mean_delta']:+.6f} "
        f"positive={best['positive']}/{best['total']} gate={payload['passes_gate']}"
    )


def main():
    args = parse_args()
    if not args.full:
        print("E123 curve-refinement screen is prepared but not launched.")
        print("Run: env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u "
              "_gloc_curve2_locator.py --full > _gloc_curve2_locator.out 2>&1")
        return
    run()


if __name__ == "__main__":
    main()
