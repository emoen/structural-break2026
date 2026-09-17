"""Frozen Chronos-Bolt Tiny representation smoke against exact E116.

The smoke uses the exact seed-42 ranker rows and the first two untouched validation
folds. It extracts only the frozen encoder [REG] token from each causal trailing
window, fits a compact grouped LightGBM ranker, and checks small fixed-CDF blends.
It never edits the champion notebook, builds a submission artifact, or submits.
"""

import argparse
import importlib.metadata
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

import joblib
import lightgbm as lgb
import numpy as np
import torch
from chronos import BaseChronosPipeline
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

import _locator_specialist as locator
import _post_white_pipeline as pipeline
from _residual_overnight import _active

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_chronos_bolt_smoke_cache"
OUTPUT = HERE / "_chronos_bolt_smoke.json"
REPORT = HERE / "_chronos_bolt_smoke_summary.md"
LOG = HERE / "_chronos_bolt_smoke.log"
MODEL_ID = "amazon/chronos-bolt-tiny"
DATA_SEED = 42
MODEL_SEED = 42
WINDOW = 250
EMBED_DIM = 256
WEIGHTS = (0.025, 0.05, 0.075, 0.10)


def log(message):
    line = str(message)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def exact_e116(stage5):
    stage1 = pipeline.load_pickle(pipeline.stage1_path(DATA_SEED))
    stage2 = pipeline.load_pickle(pipeline.stage2_path(DATA_SEED, MODEL_SEED))
    configured = pipeline.configured_predictions(
        stage1, stage2, int(stage5["winner_mask"]), list(stage5["accepted_params"])
    )
    baseline = pipeline.blend_predictions(configured, dict(stage5["weights"]))
    return configured, baseline


def load_seed_data():
    path = HERE / "_calib_cache_gloc6_wblock" / f"seed_{DATA_SEED}.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def load_pool():
    pipeline.CR.ET.SEED = DATA_SEED
    return pipeline.CR.load_pool_strat()


def rank_steps(X, train_ids):
    if os.environ.get("STEP_BIAS", "uniform") != "uniform":
        raise RuntimeError("Smoke requires the champion's uniform rank-row sampling")
    rng = np.random.default_rng(DATA_SEED + 202)
    selected = {}
    for sid in train_ids:
        _history, online = pipeline.CR.split_series(X.loc[sid])
        length = len(online)
        if length <= pipeline.CR.MIN_N:
            continue
        candidates = np.arange(pipeline.CR.MIN_N, length)
        selected[sid] = rng.choice(
            candidates,
            size=min(pipeline.CR.DENSE_K, len(candidates)),
            replace=False,
        )
    return selected


def train_windows(X, yi, selected):
    labels = []
    times = []
    for sid, steps in selected.items():
        _history, online_frame = pipeline.CR.split_series(X.loc[sid])
        online = online_frame.to_numpy(dtype=np.float64)
        active = _active(online)
        if not len(active):
            continue
        base = active[0]
        tau = int(yi.loc[sid, "tau_index"])
        for step in steps:
            relative = step - base
            if relative < 0 or relative >= len(active):
                continue
            labels.append(1 if tau >= 0 and step >= tau else 0)
            times.append(int(step))
            yield online[max(0, step - WINDOW + 1):step + 1]
    train_windows.labels = np.asarray(labels, dtype=np.int64)
    train_windows.times = np.asarray(times, dtype=np.int64)


def validation_windows(X, fold_ids):
    active_rows = []
    offset = 0
    for sid in fold_ids:
        _history, online_frame = pipeline.CR.split_series(X.loc[sid])
        online = online_frame.to_numpy(dtype=np.float64)
        active = _active(online)
        active_rows.extend((active + offset).tolist())
        for step in active:
            yield online[max(0, step - WINDOW + 1):step + 1]
        offset += len(online_frame.index)
    validation_windows.active = np.asarray(active_rows, dtype=np.int64)


def embed_to_cache(path, count, windows, chronos, batch_size):
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape != (count, EMBED_DIM):
            raise RuntimeError(f"Unexpected embedding cache shape at {path}: {cached.shape}")
        reconstructed = sum(1 for _ in windows)
        if reconstructed != count:
            raise RuntimeError(
                f"{path.name}: expected {count} reconstructed rows, got {reconstructed}"
            )
        log(f"reuse {path.name} {cached.shape}")
        return cached

    temporary = path.with_suffix(".tmp")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(count, EMBED_DIM)
    )
    batch = []
    written = 0
    started = time.time()

    def flush():
        nonlocal written
        if not batch:
            return
        context = np.full((len(batch), WINDOW), np.nan, dtype=np.float32)
        for row, values in enumerate(batch):
            context[row, -len(values):] = values
        embeddings, _ = chronos.embed(torch.from_numpy(context))
        values = embeddings[:, -1, :].numpy()
        output[written:written + len(values)] = values
        written += len(values)
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


def percentile(values, reference):
    return np.searchsorted(reference, values, side="right") / len(reference)


def fit_specialist(train_embedding, seed_data):
    rows, labels, groups = pipeline.CR.sort_into_groups(
        train_embedding, seed_data["Rl"], seed_data["Rt"]
    )
    model = lgb.LGBMRanker(
        objective="rank_xendcg",
        n_estimators=200,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=30,
        reg_lambda=1.0,
        colsample_bytree=0.7,
        random_state=MODEL_SEED,
        n_jobs=int(os.environ.get("N_JOBS", "15")),
        verbose=-1,
    )
    with threadpool_limits(limits=1):
        model.fit(rows, labels, group=groups)
    reference = np.sort(model.predict(rows))
    joblib.dump(model, CACHE / "chronos_reg_ranker.joblib", compress=3)
    return model, reference


def cross_fit_residuals(specialist, members, active_rows):
    residual_predictions = []
    for target in range(len(specialist)):
        source = 1 - target
        source_rows = active_rows[source]
        target_rows = active_rows[target]
        source_matrix = np.column_stack([
            np.ones(len(source_rows)),
            *[members[name][source][source_rows] for name in pipeline.MEMBERS],
        ])
        beta = np.linalg.lstsq(source_matrix, specialist[source][source_rows], rcond=None)[0]
        target_matrix = np.column_stack([
            np.ones(len(target_rows)),
            *[members[name][target][target_rows] for name in pipeline.MEMBERS],
        ])
        values = np.zeros_like(specialist[target])
        values[target_rows] = specialist[target][target_rows] - target_matrix @ beta
        residual_predictions.append(values)
    return residual_predictions


def markdown(payload):
    lines = [
        "# Chronos-Bolt Tiny Frozen-Embedding Smoke",
        "",
        f"Model: `{MODEL_ID}`; representation: final encoder `[REG]` token "
        f"({EMBED_DIM} dimensions); seed {DATA_SEED}; first {payload['fold_count']} folds.",
        "",
        f"E116 absolute TS-AUC: **{payload['baseline_auc']:.6f}**. "
        f"Standalone Chronos specialist: **{payload['standalone_auc']:.6f}**. "
        f"Cross-fit residual TS-AUC: **{payload['residual_auc']:.6f}**.",
        "",
        "| config | absolute TS-AUC | mean_d | fold 0 d | fold 1 d |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for weight in WEIGHTS:
        row = payload["weights"][str(weight)]
        lines.append(
            f"| Chronos ranker @ {weight:.3f} | {row['candidate_auc']:.6f} | "
            f"{row['delta']:+.6f} | {row['fold_deltas'][0]:+.6f} | "
            f"{row['fold_deltas'][1]:+.6f} |"
        )
    lines.extend([
        "",
        "| prediction | Spearman with Chronos |",
        "| --- | ---: |",
    ])
    for name, value in payload["spearman"].items():
        lines.append(f"| {name} | {value:.6f} |")
    return "\n".join(lines) + "\n"


def run(fold_count, batch_size):
    if fold_count != 2:
        raise ValueError("The guarded smoke currently requires exactly two folds")
    CACHE.mkdir(exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    torch.manual_seed(0)
    torch.set_num_threads(4)

    seed_data = load_seed_data()
    X, yi, _y_val, train_ids, folds = load_pool()
    selected = rank_steps(X, train_ids)

    train_generator = train_windows(X, yi, selected)
    train_embedding = embed_to_cache(
        CACHE / "train_reg.npy", len(seed_data["Rl"]), train_generator,
        BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu"), batch_size,
    )
    if not np.array_equal(train_windows.labels, seed_data["Rl"]):
        raise RuntimeError("Chronos train labels do not match the frozen E116 cache")
    if not np.array_equal(train_windows.times, seed_data["Rt"]):
        raise RuntimeError("Chronos train times do not match the frozen E116 cache")
    log("exact train-row alignment passed")

    chronos = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu")
    chronos.model.eval()
    fold_embeddings = []
    active_rows = []
    for fold_index in range(fold_count):
        generator = validation_windows(X, folds[fold_index])
        embedding = embed_to_cache(
            CACHE / f"fold{fold_index}_reg.npy",
            len(seed_data["caches"][fold_index]["active"]),
            generator,
            chronos,
            batch_size,
        )
        expected = seed_data["caches"][fold_index]["active"]
        if not np.array_equal(validation_windows.active, expected):
            raise RuntimeError(f"fold {fold_index}: active-row map changed")
        fold_embeddings.append(embedding)
        active_rows.append(expected)
        log(f"fold {fold_index} active-row alignment passed")

    model, reference = fit_specialist(train_embedding, seed_data)
    specialist = []
    for fold_index, embedding in enumerate(fold_embeddings):
        scores = np.zeros(len(seed_data["caches"][fold_index]["index"]), dtype=np.float64)
        scores[active_rows[fold_index]] = percentile(model.predict(embedding), reference)
        specialist.append(scores)

    stage5 = locator.load_manifest()
    configured, baseline_all = exact_e116(stage5)
    members = {name: values[:fold_count] for name, values in configured.items()}
    baseline = baseline_all[:fold_count]
    metas = pipeline.CR.make_folds_meta(seed_data)[:fold_count]
    baseline_folds = pipeline.fold_aucs(baseline, metas)
    standalone_folds = pipeline.fold_aucs(specialist, metas)
    residual = cross_fit_residuals(specialist, members, active_rows)
    residual_folds = pipeline.fold_aucs(residual, metas)

    correlations = {}
    for name, values in {**members, "E116": baseline}.items():
        correlations[name] = float(np.mean([
            spearmanr(specialist[fold][active_rows[fold]], values[fold][active_rows[fold]]).statistic
            for fold in range(fold_count)
        ]))

    weight_rows = {}
    for weight in WEIGHTS:
        candidate = [
            (1.0 - weight) * base + weight * extra
            for base, extra in zip(baseline, specialist)
        ]
        candidate_folds = pipeline.fold_aucs(candidate, metas)
        deltas = candidate_folds - baseline_folds
        weight_rows[str(weight)] = {
            "candidate_auc": float(candidate_folds.mean()),
            "delta": float(deltas.mean()),
            "fold_deltas": deltas.tolist(),
        }

    payload = {
        "experiment": "E124b_smoke",
        "model": MODEL_ID,
        "chronos_version": importlib.metadata.version("chronos-forecasting"),
        "parameter_count": int(sum(value.numel() for value in chronos.model.parameters())),
        "representation": "encoder_reg_token",
        "embedding_dimensions": EMBED_DIM,
        "train_rows": int(len(train_embedding)),
        "fold_count": fold_count,
        "validation_active_rows": [int(len(rows)) for rows in active_rows],
        "baseline_auc": float(baseline_folds.mean()),
        "standalone_auc": float(standalone_folds.mean()),
        "standalone_fold_aucs": standalone_folds.tolist(),
        "residual_auc": float(residual_folds.mean()),
        "residual_fold_aucs": residual_folds.tolist(),
        "spearman": correlations,
        "weights": weight_rows,
        "ranker_artifact_bytes": (CACHE / "chronos_reg_ranker.joblib").stat().st_size,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT.write_text(markdown(payload), encoding="utf-8")
    best_weight, best = max(weight_rows.items(), key=lambda item: item[1]["delta"])
    log(
        f"baseline={payload['baseline_auc']:.6f} standalone={payload['standalone_auc']:.6f} "
        f"residual={payload['residual_auc']:.6f} best_weight={best_weight} "
        f"best_auc={best['candidate_auc']:.6f} delta={best['delta']:+.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not args.run:
        print("Chronos-Bolt smoke is prepared but not launched.")
        print("Run: env N_JOBS=15 .venv/bin/python -u _chronos_bolt_smoke.py --run")
        return
    run(args.folds, args.batch_size)


if __name__ == "__main__":
    main()