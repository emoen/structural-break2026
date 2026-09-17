"""Corrected locator-specialist screen against the exact E116 ensemble.

Running without --full prints the plan and exits. This script reuses frozen E116
member-prediction checkpoints and fits only one compact ranker on the 27-column
locator bank. It does not edit notebooks, build submission artifacts, or submit.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import _post_white_pipeline as pipeline

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "_post_white_pipeline.json"
OUTPUT = HERE / "_locator_specialist.json"
REPORT = HERE / "_locator_specialist_summary.md"
LOG = HERE / "_locator_specialist.log"
WEIGHTS = [0.025, 0.05, 0.075, 0.10]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def log(message):
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(str(message) + "\n")
    print(message, flush=True)


def load_manifest():
    with MANIFEST.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    stage5 = manifest["stage5"]
    if not stage5.get("passes_gate"):
        raise RuntimeError("Frozen E116 manifest does not contain a passing Stage 5")
    return stage5


def specialist_predictions(data_seed, seed_data, columns):
    path = pipeline.CACHE_DIR / f"specialist_e116_seed{data_seed}.pkl"
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
    pipeline.dump_pickle(path, {"predictions": predictions, "columns": columns.tolist()})
    return predictions


def markdown(results):
    lines = [
        "# Locator Specialist Against E116",
        "",
        "Baseline: exact E116 allocation, accepted parameters, and fixed-CDF weights.",
        "Specialist: compact LightGBM ranker on white-GLOC7 + WBLOCK8 + calibrated strength12.",
        "",
        "| weight | baseline | candidate | mean_d | min_d | pos | t_agg | seed 42 d / t | seed 7 d / t | seed 123 d / t |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for weight in WEIGHTS:
        row = results[str(weight)]
        seed_cells = []
        for seed in pipeline.DATA_SEEDS:
            seed_row = row["seeds"][str(seed)]
            seed_cells.append(f"{seed_row['delta']:+.6f} / {seed_row['t']:+.2f}")
        lines.append(
            f"| {weight:.3f} | {row['baseline_auc']:.6f} | {row['candidate_auc']:.6f} | "
            f"{row['mean_delta']:+.6f} | {row['min_delta']:+.6f} | "
            f"{row['positive']}/{row['total']} | {row['aggregate_t']:+.2f} | "
            + " | ".join(seed_cells) + " |"
        )
    return "\n".join(lines) + "\n"


def run():
    stage5 = load_manifest()
    winner_mask = int(stage5["winner_mask"])
    accepted_params = list(stage5["accepted_params"])
    e116_weights = dict(stage5["weights"])
    columns = pipeline.specialist_columns()
    if len(columns) != 27:
        raise RuntimeError(f"Expected 27 specialist columns, got {len(columns)}")

    per_weight = {str(weight): [] for weight in WEIGHTS}
    for data_seed in pipeline.DATA_SEEDS:
        stage1 = pipeline.load_pickle(pipeline.stage1_path(data_seed))
        stage2 = pipeline.load_pickle(pipeline.stage2_path(data_seed, 42))
        seed_data = pipeline.CR.prepare_seed(data_seed)
        metas = pipeline.CR.make_folds_meta(seed_data)
        configured = pipeline.configured_predictions(
            stage1, stage2, winner_mask, accepted_params
        )
        e116 = pipeline.blend_predictions(configured, e116_weights)
        baseline = pipeline.fold_aucs(e116, metas)
        specialist = specialist_predictions(data_seed, seed_data, columns)
        for weight in WEIGHTS:
            candidate_predictions = [
                (1.0 - weight) * base + weight * extra
                for base, extra in zip(e116, specialist)
            ]
            candidate = pipeline.fold_aucs(candidate_predictions, metas)
            per_weight[str(weight)].append(pipeline.paired_summary(candidate, baseline))
        log(f"seed {data_seed} complete; E116={float(np.mean(baseline)):.6f}")

    results = {
        str(weight): pipeline.aggregate_seed_rows(per_weight[str(weight)])
        for weight in WEIGHTS
    }
    best = max(results.items(), key=lambda item: item[1]["mean_delta"])
    payload = {
        "baseline": "E116",
        "columns": columns.tolist(),
        "model": "compact_ranker",
        "weights": results,
        "best_weight": float(best[0]),
        "best": best[1],
        "passes_gate": bool(
            best[1]["mean_delta"] >= pipeline.PROMOTION_DELTA
            and best[1]["positive"] == len(pipeline.DATA_SEEDS)
        ),
    }
    with OUTPUT.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    REPORT.write_text(markdown(results), encoding="utf-8")
    log(f"best weight={best[0]} mean_d={best[1]['mean_delta']:+.6f} "
        f"positive={best[1]['positive']}/{best[1]['total']} gate={payload['passes_gate']}")


def main():
    args = parse_args()
    if not args.full:
        print("Locator specialist is prepared but not launched.")
        print("Run: env N_JOBS=15 .venv/bin/python -u _locator_specialist.py --full")
        return
    run()


if __name__ == "__main__":
    main()
