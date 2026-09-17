"""Independent feature-bank ranker specialists against exact E116.

Running without --full prints the plan and exits. The full run reuses frozen E116
predictions and cached feature matrices, fits compact rankers on deliberately
separated banks, and performs no notebook, artifact, cloud, or submission work.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import _locator_specialist as locator
import _post_white_pipeline as pipeline

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "_split_bank_specialists.json"
REPORT = HERE / "_split_bank_specialists_summary.md"
LOG = HERE / "_split_bank_specialists.log"
WEIGHTS = [0.025, 0.05, 0.075, 0.10]
BANKS = ["raw79", "calibration79", "whitening15", "calibration_whitening_equal"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def log(message):
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(str(message) + "\n")
    print(message, flush=True)


def bank_columns():
    raw = np.asarray(pipeline.CR.BASE_COLS, dtype=int)
    calibration = np.asarray(pipeline.CR.CAND_SLICE["czrb_all79"], dtype=int)
    whitening = np.concatenate([
        pipeline.CR.CAND_SLICE["wblock"],
        pipeline.CR.CAND_SLICE["white_gloc"],
    ]).astype(int)
    banks = {
        "raw79": raw,
        "calibration79": calibration,
        "whitening15": whitening,
    }
    expected = {"raw79": 79, "calibration79": 79, "whitening15": 15}
    for name, columns in banks.items():
        if len(columns) != expected[name] or len(np.unique(columns)) != len(columns):
            raise RuntimeError(f"Invalid {name} columns: {columns.tolist()}")
    if any(np.intersect1d(left, right).size for index, left in enumerate(banks.values())
           for right in list(banks.values())[index + 1:]):
        raise RuntimeError("Split feature banks overlap")
    return banks


def compact_ranker_predictions(name, data_seed, seed_data, columns):
    path = pipeline.CACHE_DIR / f"specialist_e121_{name}_seed{data_seed}.pkl"
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
        "bank": name,
        "columns": columns.tolist(),
        "predictions": predictions,
    })
    return predictions


def exact_e116(data_seed, stage5):
    stage1 = pipeline.load_pickle(pipeline.stage1_path(data_seed))
    stage2 = pipeline.load_pickle(pipeline.stage2_path(data_seed, 42))
    configured = pipeline.configured_predictions(
        stage1, stage2, int(stage5["winner_mask"]), list(stage5["accepted_params"])
    )
    return pipeline.blend_predictions(configured, dict(stage5["weights"]))


def aggregate(candidate_by_seed, baseline_by_seed, metas_by_seed):
    rows = []
    for seed in pipeline.DATA_SEEDS:
        candidate = pipeline.fold_aucs(candidate_by_seed[seed], metas_by_seed[seed])
        baseline = pipeline.fold_aucs(baseline_by_seed[seed], metas_by_seed[seed])
        rows.append(pipeline.paired_summary(candidate, baseline))
    return pipeline.aggregate_seed_rows(rows)


def weight_screen(specialist_by_seed, baseline_by_seed, metas_by_seed):
    results = {}
    for weight in WEIGHTS:
        candidate = {
            seed: [
                (1.0 - weight) * base + weight * specialist
                for base, specialist in zip(
                    baseline_by_seed[seed], specialist_by_seed[seed]
                )
            ]
            for seed in pipeline.DATA_SEEDS
        }
        results[str(weight)] = aggregate(candidate, baseline_by_seed, metas_by_seed)
    best_weight, best = max(results.items(), key=lambda item: item[1]["mean_delta"])
    return {"weights": results, "best_weight": float(best_weight), "best": best}


def result_row(name, result):
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
    return "| " + " | ".join(cells) + " |"


def markdown(payload):
    headers = ["config", "absolute TS-AUC", "mean_d", "min_d", "pos", "t_agg"]
    headers.extend(f"seed {seed} d / t" for seed in pipeline.DATA_SEEDS)
    lines = [
        "# Split-Bank Specialists Against E116",
        "",
        "Compact rankers use mutually exclusive raw, calibration, or whitening banks.",
        "The equal arm averages calibration and whitening specialist percentiles before blending.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(result_row(name, payload["banks"][name]) for name in BANKS)
    return "\n".join(lines) + "\n"


def run():
    LOG.write_text("", encoding="utf-8")
    stage5 = locator.load_manifest()
    columns_by_bank = bank_columns()
    baseline_by_seed = {}
    metas_by_seed = {}
    predictions = {name: {} for name in columns_by_bank}

    for data_seed in pipeline.DATA_SEEDS:
        started = time.time()
        seed_data = pipeline.CR.prepare_seed(data_seed)
        metas_by_seed[data_seed] = pipeline.CR.make_folds_meta(seed_data)
        baseline_by_seed[data_seed] = exact_e116(data_seed, stage5)
        for name, columns in columns_by_bank.items():
            fit_started = time.time()
            predictions[name][data_seed] = compact_ranker_predictions(
                name, data_seed, seed_data, columns
            )
            log(f"seed {data_seed} {name} complete in {time.time() - fit_started:.0f}s")
        log(f"seed {data_seed} complete in {(time.time() - started) / 60:.1f}m")

    predictions["calibration_whitening_equal"] = {
        seed: [
            0.5 * calibration + 0.5 * whitening
            for calibration, whitening in zip(
                predictions["calibration79"][seed], predictions["whitening15"][seed]
            )
        ]
        for seed in pipeline.DATA_SEEDS
    }
    results = {
        name: weight_screen(predictions[name], baseline_by_seed, metas_by_seed)
        for name in BANKS
    }
    best_name, best_result = max(
        results.items(), key=lambda item: item[1]["best"]["mean_delta"]
    )
    best = best_result["best"]
    payload = {
        "baseline": "E116",
        "columns": {name: values.tolist() for name, values in columns_by_bank.items()},
        "banks": results,
        "best_bank": best_name,
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
        f"mean_d={best['mean_delta']:+.6f} "
        f"positive={best['positive']}/{best['total']} gate={payload['passes_gate']}"
    )


def main():
    args = parse_args()
    if not args.full:
        print("Split-bank specialist screen is prepared but not launched.")
        print("Run: env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u "
              "_split_bank_specialists.py --full > _split_bank_specialists.out 2>&1")
        return
    run()


if __name__ == "__main__":
    main()