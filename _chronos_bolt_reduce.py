"""E125b cache-only reduction of the Chronos local-minus-global patch contrast."""

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.decomposition import IncrementalPCA

import _chronos_bolt_refine as refine
import _chronos_bolt_smoke as smoke
import _locator_specialist as locator
import _post_white_pipeline as pipeline

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "_chronos_bolt_reduce.json"
REPORT = HERE / "_chronos_bolt_reduce_summary.md"
BATCH_SIZE = 8192


def contrast(matrix):
    return matrix[:, 512:768]


def norm_features(matrix):
    values = contrast(matrix).astype(np.float64)
    return np.column_stack([
        values.mean(axis=1),
        values.std(axis=1),
        np.mean(np.abs(values), axis=1),
        np.sqrt(np.mean(values * values, axis=1)),
        np.max(np.abs(values), axis=1),
    ]).astype(np.float32)


def fit_incremental_pca(train, dimensions):
    reducer = IncrementalPCA(n_components=dimensions, batch_size=BATCH_SIZE)
    for start in range(0, len(train), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(train))
        reducer.partial_fit(contrast(train[start:stop]))
    return reducer


def transform_in_batches(reducer, matrix):
    output = np.empty((len(matrix), reducer.n_components_), dtype=np.float32)
    for start in range(0, len(matrix), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(matrix))
        output[start:stop] = reducer.transform(contrast(matrix[start:stop]))
    return output


def main():
    seed_data = smoke.load_seed_data()
    train = np.load(refine.CACHE / "train_views.npy", mmap_mode="r")
    folds = [
        np.load(refine.CACHE / f"fold{fold}_views.npy", mmap_mode="r")
        for fold in range(2)
    ]
    active_rows = [seed_data["caches"][fold]["active"] for fold in range(2)]

    stage5 = locator.load_manifest()
    configured, baseline_all = smoke.exact_e116(stage5)
    members = {name: values[:2] for name, values in configured.items()}
    baseline = baseline_all[:2]
    metas = pipeline.CR.make_folds_meta(seed_data)[:2]

    representations = {
        "norms5": (norm_features(train), [norm_features(matrix) for matrix in folds]),
    }
    for dimensions in (16, 32, 64):
        reducer = fit_incremental_pca(train, dimensions)
        representations[f"pca{dimensions}"] = (
            transform_in_batches(reducer, train),
            [transform_in_batches(reducer, matrix) for matrix in folds],
        )

    results = {}
    for name, (train_matrix, fold_matrices) in representations.items():
        columns = np.arange(train_matrix.shape[1])
        predictions = refine.fit_predict(
            f"reduce_{name}", columns, train_matrix, fold_matrices, seed_data, active_rows
        )
        results[name] = refine.evaluate(
            name, predictions, baseline, members, metas, active_rows
        )

    payload = {
        "experiment": "E125b_smoke",
        "baseline": "E116",
        "baseline_auc": float(pipeline.fold_aucs(baseline, metas).mean()),
        "source": "last_minus_valid_patch_mean",
        "views": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# E125b Chronos Contrast Reduction",
        "",
        f"Exact E116 baseline: **{payload['baseline_auc']:.6f}**.",
        "",
        "| representation | standalone | residual | Spearman E116 | best weight | candidate | mean_d | fold 0 d | fold 1 d |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in results.items():
        best = row["best"]
        lines.append(
            f"| {name} | {row['standalone_auc']:.6f} | {row['residual_auc']:.6f} | "
            f"{row['spearman_e116']:.6f} | {row['best_weight']:.3f} | "
            f"{best['candidate_auc']:.6f} | {best['delta']:+.6f} | "
            f"{best['fold_deltas'][0]:+.6f} | {best['fold_deltas'][1]:+.6f} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()