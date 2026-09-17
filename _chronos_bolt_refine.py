"""E125 Chronos-Bolt encoder-geometry refinement on the E124b smoke folds.

Tests whether patch-local encoder states recover stable break evidence discarded by
the global [REG] token. Rows, labels, fold maps, model family, and blend protocol
remain identical to E124b.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import lightgbm as lgb
import numpy as np
import torch
from chronos import BaseChronosPipeline
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

import _chronos_bolt_smoke as smoke
import _locator_specialist as locator
import _post_white_pipeline as pipeline

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_chronos_bolt_refine_cache"
OUTPUT = HERE / "_chronos_bolt_refine.json"
REPORT = HERE / "_chronos_bolt_refine_summary.md"
LOG = HERE / "_chronos_bolt_refine.log"
VIEWS = {
    "reg": np.arange(0, 256),
    "last": np.arange(256, 512),
    "last_minus_mean": np.arange(512, 768),
    "all": np.arange(0, 768),
}


def log(message):
    line = str(message)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def embed_views_to_cache(path, count, windows, chronos, batch_size):
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape != (count, 768):
            raise RuntimeError(f"Unexpected embedding cache shape at {path}: {cached.shape}")
        reconstructed = sum(1 for _ in windows)
        if reconstructed != count:
            raise RuntimeError(f"{path.name}: expected {count} rows, got {reconstructed}")
        log(f"reuse {path.name} {cached.shape}")
        return cached

    temporary = path.with_suffix(".tmp")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(count, 768)
    )
    batch = []
    written = 0
    started = time.time()

    def flush():
        nonlocal written
        if not batch:
            return
        context = np.full((len(batch), smoke.WINDOW), np.nan, dtype=np.float32)
        for row, values in enumerate(batch):
            context[row, -len(values):] = values
        tensor = torch.from_numpy(context).to(chronos.model.device, dtype=torch.float32)
        with torch.no_grad():
            embeddings, _loc_scale, _inputs, attention = chronos.model.encode(tensor)
        patches = embeddings[:, :-1, :]
        patch_mask = attention[:, :-1].to(patches.dtype).unsqueeze(-1)
        patch_mean = (patches * patch_mask).sum(dim=1) / patch_mask.sum(dim=1).clamp_min(1.0)
        reg = embeddings[:, -1, :]
        last = patches[:, -1, :]
        features = torch.cat([reg, last, last - patch_mean], dim=1).cpu().numpy()
        output[written:written + len(features)] = features
        written += len(features)
        batch.clear()
        if written % (batch_size * 100) == 0:
            log(f"{path.stem}: {written}/{count}")

    for values in windows:
        batch.append(np.asarray(values, dtype=np.float32))
        if len(batch) == batch_size:
            flush()
    flush()
    if written != count:
        raise RuntimeError(f"{path.name}: expected {count} rows, wrote {written}")
    output.flush()
    del output
    temporary.replace(path)
    log(f"built {path.name} {count} rows in {(time.time() - started) / 60:.1f}m")
    return np.load(path, mmap_mode="r")


def fit_predict(name, columns, train_embedding, fold_embeddings, seed_data, active_rows):
    rows, labels, groups = pipeline.CR.sort_into_groups(
        train_embedding[:, columns], seed_data["Rl"], seed_data["Rt"]
    )
    model = lgb.LGBMRanker(
        objective="rank_xendcg",
        n_estimators=200,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=30,
        reg_lambda=1.0,
        colsample_bytree=0.7,
        random_state=smoke.MODEL_SEED,
        n_jobs=int(os.environ.get("N_JOBS", "15")),
        verbose=-1,
    )
    with threadpool_limits(limits=1):
        model.fit(rows, labels, group=groups)
    reference = np.sort(model.predict(rows))
    predictions = []
    for fold, embedding in enumerate(fold_embeddings):
        scores = np.zeros(len(seed_data["caches"][fold]["index"]), dtype=np.float64)
        raw = model.predict(embedding[:, columns])
        scores[active_rows[fold]] = smoke.percentile(raw, reference)
        predictions.append(scores)
    joblib.dump(model, CACHE / f"chronos_{name}_ranker.joblib", compress=3)
    return predictions


def evaluate(name, specialist, baseline, members, metas, active_rows):
    standalone_folds = pipeline.fold_aucs(specialist, metas)
    residual = smoke.cross_fit_residuals(specialist, members, active_rows)
    residual_folds = pipeline.fold_aucs(residual, metas)
    correlations = [
        spearmanr(specialist[fold][active_rows[fold]], baseline[fold][active_rows[fold]]).statistic
        for fold in range(len(metas))
    ]
    baseline_folds = pipeline.fold_aucs(baseline, metas)
    weights = {}
    for weight in smoke.WEIGHTS:
        candidate = [
            (1.0 - weight) * base + weight * extra
            for base, extra in zip(baseline, specialist)
        ]
        candidate_folds = pipeline.fold_aucs(candidate, metas)
        delta = candidate_folds - baseline_folds
        weights[str(weight)] = {
            "candidate_auc": float(candidate_folds.mean()),
            "delta": float(delta.mean()),
            "fold_deltas": delta.tolist(),
        }
    best_weight, best = max(weights.items(), key=lambda item: item[1]["delta"])
    log(
        f"{name}: standalone={standalone_folds.mean():.6f} "
        f"residual={residual_folds.mean():.6f} rho={np.mean(correlations):.6f} "
        f"best@{best_weight}={best['candidate_auc']:.6f} d={best['delta']:+.6f}"
    )
    return {
        "standalone_auc": float(standalone_folds.mean()),
        "standalone_fold_aucs": standalone_folds.tolist(),
        "residual_auc": float(residual_folds.mean()),
        "residual_fold_aucs": residual_folds.tolist(),
        "spearman_e116": float(np.mean(correlations)),
        "best_weight": float(best_weight),
        "best": best,
        "weights": weights,
    }


def markdown(payload):
    lines = [
        "# E125 Chronos-Bolt Encoder-Geometry Refinement",
        "",
        f"Exact E116 baseline on two seed-42 folds: **{payload['baseline_auc']:.6f}**.",
        "",
        "| representation | standalone | residual | Spearman E116 | best weight | candidate | mean_d | fold 0 d | fold 1 d |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in VIEWS:
        row = payload["views"][name]
        best = row["best"]
        lines.append(
            f"| {name} | {row['standalone_auc']:.6f} | {row['residual_auc']:.6f} | "
            f"{row['spearman_e116']:.6f} | {row['best_weight']:.3f} | "
            f"{best['candidate_auc']:.6f} | {best['delta']:+.6f} | "
            f"{best['fold_deltas'][0]:+.6f} | {best['fold_deltas'][1]:+.6f} |"
        )
    return "\n".join(lines) + "\n"


def run(batch_size):
    CACHE.mkdir(exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    seed_data = smoke.load_seed_data()
    X, yi, _y_val, train_ids, folds = smoke.load_pool()
    selected = smoke.rank_steps(X, train_ids)
    chronos = BaseChronosPipeline.from_pretrained(smoke.MODEL_ID, device_map="cpu")
    chronos.model.eval()

    train_generator = smoke.train_windows(X, yi, selected)
    train_embedding = embed_views_to_cache(
        CACHE / "train_views.npy", len(seed_data["Rl"]), train_generator, chronos, batch_size
    )
    if not np.array_equal(smoke.train_windows.labels, seed_data["Rl"]):
        raise RuntimeError("Train labels do not match E116")
    if not np.array_equal(smoke.train_windows.times, seed_data["Rt"]):
        raise RuntimeError("Train times do not match E116")
    old_reg = np.load(smoke.CACHE / "train_reg.npy", mmap_mode="r")
    reg_max_difference = float(np.max(np.abs(train_embedding[:, :256] - old_reg)))
    if reg_max_difference != 0.0:
        raise RuntimeError(f"Regression-token replay changed: {reg_max_difference}")
    log("exact train-row and regression-token replay passed")

    fold_embeddings = []
    active_rows = []
    for fold in range(2):
        generator = smoke.validation_windows(X, folds[fold])
        embedding = embed_views_to_cache(
            CACHE / f"fold{fold}_views.npy",
            len(seed_data["caches"][fold]["active"]),
            generator,
            chronos,
            batch_size,
        )
        expected = seed_data["caches"][fold]["active"]
        if not np.array_equal(smoke.validation_windows.active, expected):
            raise RuntimeError(f"fold {fold}: active-row map changed")
        old_fold = np.load(smoke.CACHE / f"fold{fold}_reg.npy", mmap_mode="r")
        difference = float(np.max(np.abs(embedding[:, :256] - old_fold)))
        if difference != 0.0:
            raise RuntimeError(f"fold {fold}: regression-token replay changed: {difference}")
        fold_embeddings.append(embedding)
        active_rows.append(expected)
        log(f"fold {fold} exact active-row and regression-token replay passed")

    stage5 = locator.load_manifest()
    configured, baseline_all = smoke.exact_e116(stage5)
    members = {name: values[:2] for name, values in configured.items()}
    baseline = baseline_all[:2]
    metas = pipeline.CR.make_folds_meta(seed_data)[:2]
    views = {}
    for name, columns in VIEWS.items():
        predictions = fit_predict(
            name, columns, train_embedding, fold_embeddings, seed_data, active_rows
        )
        views[name] = evaluate(name, predictions, baseline, members, metas, active_rows)

    payload = {
        "experiment": "E125_smoke",
        "baseline": "E116",
        "baseline_auc": float(pipeline.fold_aucs(baseline, metas).mean()),
        "data_seed": smoke.DATA_SEED,
        "fold_count": 2,
        "train_rows": int(len(train_embedding)),
        "reg_replay_max_difference": reg_max_difference,
        "views": views,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT.write_text(markdown(payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not args.run:
        print("E125 Chronos refinement is prepared but not launched.")
        return
    run(args.batch_size)


if __name__ == "__main__":
    main()