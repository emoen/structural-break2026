"""Checkpoint-only E122/E123 specialist blend against exact E116."""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import _locator_specialist as locator
import _post_white_pipeline as pipeline

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "_curve_checkpoint_blend.json"
REPORT = HERE / "_curve_checkpoint_blend_summary.md"
WEIGHTS = [0.025, 0.05, 0.075, 0.10]


def load_predictions(path):
    saved = pipeline.load_pickle(path)
    return [np.asarray(values, dtype=np.float64) for values in saved["predictions"]]


def exact_e116(seed, stage5):
    stage1 = pipeline.load_pickle(pipeline.stage1_path(seed))
    stage2 = pipeline.load_pickle(pipeline.stage2_path(seed, 42))
    configured = pipeline.configured_predictions(
        stage1, stage2, int(stage5["winner_mask"]), list(stage5["accepted_params"])
    )
    return pipeline.blend_predictions(configured, dict(stage5["weights"]))


def main():
    stage5 = locator.load_manifest()
    correlations = {}
    specialists = {}
    baselines = {}
    metas = {}
    for seed in pipeline.DATA_SEEDS:
        stretched = load_predictions(
            HERE / "_gloc_curve_locator_cache" / f"curve_ranker_seed{seed}.pkl"
        )
        fixed = load_predictions(
            HERE / "_gloc_curve2_locator_cache" / f"curve2_ranker_seed{seed}.pkl"
        )
        if len(stretched) != len(fixed):
            raise RuntimeError(f"seed {seed}: fold count mismatch")
        for fold, (left, right) in enumerate(zip(stretched, fixed)):
            if left.shape != right.shape:
                raise RuntimeError(f"seed {seed} fold {fold}: shape mismatch")
        correlations[str(seed)] = [
            float(spearmanr(left, right).statistic)
            for left, right in zip(stretched, fixed)
        ]
        specialists[seed] = [
            0.5 * left + 0.5 * right for left, right in zip(stretched, fixed)
        ]
        with (HERE / "_calib_cache_gloc6_wblock" / f"seed_{seed}.pkl").open("rb") as stream:
            seed_data = pickle.load(stream)
        metas[seed] = pipeline.CR.make_folds_meta(seed_data)
        baselines[seed] = exact_e116(seed, stage5)

    weights = {}
    for weight in WEIGHTS:
        rows = []
        for seed in pipeline.DATA_SEEDS:
            candidate_predictions = [
                (1.0 - weight) * base + weight * specialist
                for base, specialist in zip(baselines[seed], specialists[seed])
            ]
            candidate = pipeline.fold_aucs(candidate_predictions, metas[seed])
            baseline = pipeline.fold_aucs(baselines[seed], metas[seed])
            rows.append(pipeline.paired_summary(candidate, baseline))
        weights[str(weight)] = pipeline.aggregate_seed_rows(rows)

    best_weight, best = max(weights.items(), key=lambda item: item[1]["mean_delta"])
    all_correlations = np.concatenate([
        np.asarray(values, dtype=np.float64) for values in correlations.values()
    ])
    payload = {
        "experiment": "E124a",
        "baseline": "E116",
        "specialist": "equal_e122_e123_ranker_predictions",
        "correlations": correlations,
        "mean_fold_spearman": float(all_correlations.mean()),
        "min_fold_spearman": float(all_correlations.min()),
        "weights": weights,
        "best_weight": float(best_weight),
        "best": best,
        "passes_gate": bool(
            best["mean_delta"] >= pipeline.PROMOTION_DELTA
            and best["positive"] == len(pipeline.DATA_SEEDS)
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    headers = ["config", "absolute TS-AUC", "mean_d", "min_d", "pos", "t_agg"]
    headers.extend(f"seed {seed} d / t" for seed in pipeline.DATA_SEEDS)
    lines = [
        "# E124a E122/E123 Checkpoint Blend",
        "",
        f"Mean fold Spearman: {payload['mean_fold_spearman']:.6f}; "
        f"minimum fold Spearman: {payload['min_fold_spearman']:.6f}.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for weight in WEIGHTS:
        row = weights[str(weight)]
        cells = [
            f"equal curve blend @ {weight:.3f}",
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
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"best weight={best_weight} auc={best['candidate_auc']:.6f} "
        f"mean_d={best['mean_delta']:+.6f} pos={best['positive']}/{best['total']} "
        f"rho={payload['mean_fold_spearman']:.6f} gate={payload['passes_gate']}"
    )


if __name__ == "__main__":
    main()
