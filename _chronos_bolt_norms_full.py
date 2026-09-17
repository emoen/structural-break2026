"""E126 full screen of five Chronos local-minus-global contrast summaries."""

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

import _chronos_bolt_smoke as smoke
import _locator_specialist as locator
import _post_white_pipeline as pipeline
from _residual_overnight import _active

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_chronos_bolt_norms_cache"
OUTPUT = HERE / "_chronos_bolt_norms_full.json"
REPORT = HERE / "_chronos_bolt_norms_full_summary.md"
LOG = HERE / "_chronos_bolt_norms_full.log"
WEIGHTS = (0.025, 0.05, 0.075, 0.10)
BATCH_SIZE = 128


def log(message):
    line = str(message)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def load_seed_data(seed):
    path = HERE / "_calib_cache_gloc6_wblock" / f"seed_{seed}.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def load_pool(seed):
    pipeline.CR.ET.SEED = seed
    return pipeline.CR.load_pool_strat()


def rank_steps(X, train_ids, seed):
    rng = np.random.default_rng(seed + 202)
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
            yield online[max(0, step - smoke.WINDOW + 1):step + 1]
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
            yield online[max(0, step - smoke.WINDOW + 1):step + 1]
        offset += len(online_frame.index)
    validation_windows.active = np.asarray(active_rows, dtype=np.int64)


def encode_norms(path, count, windows, chronos):
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape != (count, 5):
            raise RuntimeError(f"Unexpected cache shape at {path}: {cached.shape}")
        reconstructed = sum(1 for _ in windows)
        if reconstructed != count:
            raise RuntimeError(f"{path.name}: expected {count} rows, got {reconstructed}")
        log(f"reuse {path.name} {cached.shape}")
        return cached

    temporary = path.with_suffix(".tmp")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(count, 5)
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
        mask = attention[:, :-1].to(patches.dtype).unsqueeze(-1)
        mean = (patches * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        values = embeddings[:, -2, :] - mean
        features = torch.stack([
            values.mean(dim=1),
            values.std(dim=1, correction=0),
            values.abs().mean(dim=1),
            (values * values).mean(dim=1).sqrt(),
            values.abs().amax(dim=1),
        ], dim=1).cpu().numpy()
        output[written:written + len(features)] = features
        written += len(features)
        batch.clear()
        if written % (BATCH_SIZE * 200) == 0:
            log(f"{path.stem}: {written}/{count}")

    for values in windows:
        batch.append(np.asarray(values, dtype=np.float32))
        if len(batch) == BATCH_SIZE:
            flush()
    flush()
    if written != count:
        raise RuntimeError(f"{path.name}: expected {count} rows, wrote {written}")
    output.flush()
    del output
    temporary.replace(path)
    log(f"built {path.name} {count} rows in {(time.time() - started) / 60:.1f}m")
    return np.load(path, mmap_mode="r")


def fit_predict(seed, train, folds, seed_data, active_rows):
    rows, labels, groups = pipeline.CR.sort_into_groups(
        train, seed_data["Rl"], seed_data["Rt"]
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
    for fold, matrix in enumerate(folds):
        scores = np.zeros(len(seed_data["caches"][fold]["index"]), dtype=np.float64)
        scores[active_rows[fold]] = smoke.percentile(model.predict(matrix), reference)
        predictions.append(scores)
    joblib.dump(model, CACHE / f"norms_ranker_seed{seed}.joblib", compress=3)
    return predictions


def exact_e116(seed, stage5):
    stage1 = pipeline.load_pickle(pipeline.stage1_path(seed))
    stage2 = pipeline.load_pickle(pipeline.stage2_path(seed, smoke.MODEL_SEED))
    configured = pipeline.configured_predictions(
        stage1, stage2, int(stage5["winner_mask"]), list(stage5["accepted_params"])
    )
    return pipeline.blend_predictions(configured, dict(stage5["weights"]))


def run():
    CACHE.mkdir(exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    chronos = BaseChronosPipeline.from_pretrained(smoke.MODEL_ID, device_map="cpu")
    chronos.model.eval()
    stage5 = locator.load_manifest()

    specialists = {}
    baselines = {}
    metas = {}
    correlations = {}
    standalone = {}
    for seed in pipeline.DATA_SEEDS:
        started = time.time()
        seed_data = load_seed_data(seed)
        X, yi, _y_val, train_ids, folds = load_pool(seed)
        selected = rank_steps(X, train_ids, seed)
        train = encode_norms(
            CACHE / f"train_seed{seed}.npy", len(seed_data["Rl"]),
            train_windows(X, yi, selected), chronos,
        )
        if not np.array_equal(train_windows.labels, seed_data["Rl"]):
            raise RuntimeError(f"seed {seed}: train labels changed")
        if not np.array_equal(train_windows.times, seed_data["Rt"]):
            raise RuntimeError(f"seed {seed}: train times changed")

        fold_features = []
        active_rows = []
        for fold, fold_ids in enumerate(folds):
            matrix = encode_norms(
                CACHE / f"seed{seed}_fold{fold}.npy",
                len(seed_data["caches"][fold]["active"]),
                validation_windows(X, fold_ids),
                chronos,
            )
            expected = seed_data["caches"][fold]["active"]
            if not np.array_equal(validation_windows.active, expected):
                raise RuntimeError(f"seed {seed} fold {fold}: active rows changed")
            fold_features.append(matrix)
            active_rows.append(expected)

        metas[seed] = pipeline.CR.make_folds_meta(seed_data)
        baselines[seed] = exact_e116(seed, stage5)
        specialists[seed] = fit_predict(
            seed, train, fold_features, seed_data, active_rows
        )
        standalone_folds = pipeline.fold_aucs(specialists[seed], metas[seed])
        standalone[str(seed)] = {
            "auc": float(standalone_folds.mean()),
            "fold_aucs": standalone_folds.tolist(),
        }
        correlations[str(seed)] = float(np.mean([
            spearmanr(extra[rows], base[rows]).statistic
            for extra, base, rows in zip(specialists[seed], baselines[seed], active_rows)
        ]))
        log(
            f"seed {seed}: standalone={standalone_folds.mean():.6f} "
            f"rho={correlations[str(seed)]:.6f} complete in {(time.time()-started)/60:.1f}m"
        )

    weights = {}
    for weight in WEIGHTS:
        rows = []
        for seed in pipeline.DATA_SEEDS:
            candidate = [
                (1.0 - weight) * base + weight * extra
                for base, extra in zip(baselines[seed], specialists[seed])
            ]
            candidate_auc = pipeline.fold_aucs(candidate, metas[seed])
            baseline_auc = pipeline.fold_aucs(baselines[seed], metas[seed])
            rows.append(pipeline.paired_summary(candidate_auc, baseline_auc))
        weights[str(weight)] = pipeline.aggregate_seed_rows(rows)

    best_weight, best = max(weights.items(), key=lambda item: item[1]["mean_delta"])
    payload = {
        "experiment": "E126",
        "baseline": "E116",
        "representation": "chronos_last_patch_minus_valid_mean_norms5",
        "standalone": standalone,
        "spearman_e116": correlations,
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
        "# E126 Chronos Contrast-Norm Full Screen",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for weight in WEIGHTS:
        row = weights[str(weight)]
        cells = [
            f"Chronos norms @ {weight:.3f}",
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
    lines.extend([
        "",
        "| seed | standalone TS-AUC | Spearman with E116 |",
        "| --- | ---: | ---: |",
    ])
    for seed in pipeline.DATA_SEEDS:
        lines.append(
            f"| {seed} | {standalone[str(seed)]['auc']:.6f} | "
            f"{correlations[str(seed)]:.6f} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(
        f"best weight={best_weight} auc={best['candidate_auc']:.6f} "
        f"mean_d={best['mean_delta']:+.6f} min_d={best['min_delta']:+.6f} "
        f"pos={best['positive']}/{best['total']} gate={payload['passes_gate']}"
    )


if __name__ == "__main__":
    run()