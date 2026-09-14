"""Sequential post-white-GLOC ensemble refinement pipeline.

Stages implemented from notes/backlog.md item 4:
  1. Fit each member on champion166 and champion166+white-GLOC; score all 16 allocations.
  2. Test the winning allocation and credible hyperparameter finalists over five model seeds.
  3. Retune fixed-CDF blend weights on a bounded 0.025 simplex.
  5. Evaluate one combined candidate exactly against the frozen 63.17 recipe.
  6. If the combined candidate misses the promotion gate, test a compact locator specialist.

The existing feature caches fix data rows and folds. Checkpoints make the run restartable.
No notebook is edited and nothing is submitted.
"""
import argparse
import itertools
import json
import math
import os
import pickle
import select
import time
import gc
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("GLOC6", "1")
os.environ.setdefault("WBLOCK", "1")

import numpy as np
from scipy.stats import ttest_1samp

import _ensemble_hp_white_gloc as HP

CR = HP.CR
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "_post_white_pipeline_cache"
LOG = HERE / "_post_white_pipeline.log"
RESULTS = HERE / "_post_white_pipeline.json"
REPORT = HERE / "_post_white_pipeline_summary.md"

DATA_SEEDS = [42, 7, 123]
MODEL_SEEDS = [42, 7, 123, 2024, 31337]
MEMBERS = ["ranker", "rf", "xgb", "catboost"]
MEMBER_BITS = {name: 1 << index for index, name in enumerate(MEMBERS)}
BASE_WEIGHTS = dict(HP.WEIGHTS)
PROMOTION_DELTA = 0.0015

PARAM_FINALISTS = {
    "ranker_slow500": (
        "ranker",
        {**HP.RANK_BASE, "n_estimators": 500, "learning_rate": 0.03},
    ),
    "xgb_depth8": ("xgb", {**HP.XGB_BASE, "max_depth": 8}),
    "catboost_depth8": ("catboost", {**HP.CAT_BASE, "depth": 8}),
}

WEIGHT_BOUNDS = {
    "ranker": (0.35, 0.55),
    "rf": (0.30, 0.50),
    "xgb": (0.05, 0.20),
    "catboost": (0.00, 0.15),
}
WEIGHT_STEP = 0.025
SPECIALIST_WEIGHTS = [0.025, 0.05, 0.075, 0.10]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int)
    return parser.parse_args()


def wait_for_pid(pid):
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    except AttributeError as exc:
        raise RuntimeError("--wait-pid requires Linux pidfd support") from exc
    log(f"Waiting for PID {pid} before starting the 15-job pipeline...")
    try:
        select.select([descriptor], [], [])
    finally:
        os.close(descriptor)
    log(f"PID {pid} finished; starting pipeline.")


def log(message=""):
    line = str(message)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def load_json(path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
    temporary.replace(path)


def dump_pickle(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_pickle(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def compact_predictions(predictions):
    return [np.asarray(values, dtype=np.float32) for values in predictions]


def member_params(member):
    return {
        "ranker": HP.RANK_BASE,
        "rf": HP.RF_BASE,
        "xgb": HP.XGB_BASE,
        "catboost": HP.CAT_BASE,
    }[member]


def fit_predictions(member, params, model_seed, seed_data, columns):
    HP.MODEL_SEED = model_seed
    model, predictions = HP.fit_member(member, params, seed_data, columns)
    del model
    return compact_predictions(predictions)


def blend_predictions(member_predictions, weights=None):
    if weights is None:
        weights = BASE_WEIGHTS
    folds = len(next(iter(member_predictions.values())))
    return [
        sum(weights[name] * member_predictions[name][fold] for name in MEMBERS)
        for fold in range(folds)
    ]


def fold_aucs(predictions, metas):
    return HP.fold_aucs(predictions, metas)


def paired_summary(candidate, baseline):
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    delta = candidate - baseline
    if len(delta) > 1 and np.std(delta, ddof=1) > 0:
        t_score = float(ttest_1samp(delta, 0.0).statistic)
    else:
        t_score = math.inf if delta.mean() > 0 else -math.inf
    return {
        "baseline_auc": float(baseline.mean()),
        "candidate_auc": float(candidate.mean()),
        "delta": float(delta.mean()),
        "t": t_score,
        "fold_aucs": candidate.tolist(),
        "baseline_fold_aucs": baseline.tolist(),
    }


def mask_name(mask):
    selected = [name for name in MEMBERS if mask & MEMBER_BITS[name]]
    return "+".join(selected) if selected else "none"


def predictions_for_mask(stage, mask):
    return {
        member: stage["rich" if mask & MEMBER_BITS[member] else "base"][member]
        for member in MEMBERS
    }


def stage1_path(seed):
    return CACHE_DIR / f"stage1_seed{seed}.pkl"


def run_stage1_seed(seed):
    path = stage1_path(seed)
    if path.exists():
        log(f"[stage 1 seed {seed}] checkpoint exists")
        return load_pickle(path)

    started = time.time()
    seed_data = CR.prepare_seed(seed)
    metas = CR.make_folds_meta(seed_data)
    base_columns = CR.cand_cols(["czrb_all79", "wblock"])
    rich_columns = CR.cand_cols(["czrb_all79", "wblock", "white_gloc"])
    if len(base_columns) != 166 or len(rich_columns) != 173:
        raise RuntimeError(f"Unexpected widths: base={len(base_columns)} rich={len(rich_columns)}")

    stage = {"seed": seed, "base": {}, "rich": {}, "masks": {}}
    for variant, columns in (("base", base_columns), ("rich", rich_columns)):
        for member in MEMBERS:
            fit_started = time.time()
            stage[variant][member] = fit_predictions(
                member, member_params(member), 42, seed_data, columns
            )
            log(f"[stage 1 seed {seed}] {variant:<4} {member:<8} "
                f"{time.time() - fit_started:.0f}s")

    baseline = fold_aucs(blend_predictions(stage["base"]), metas)
    for mask in range(16):
        candidate = fold_aucs(blend_predictions(predictions_for_mask(stage, mask)), metas)
        stage["masks"][str(mask)] = paired_summary(candidate, baseline)
    dump_pickle(path, stage)
    log(f"[stage 1 seed {seed}] complete in {(time.time() - started) / 60:.1f}m")
    return stage


def aggregate_seed_rows(rows):
    deltas = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    all_fold_delta = np.concatenate([
        np.asarray(row["fold_aucs"]) - np.asarray(row["baseline_fold_aucs"])
        for row in rows
    ])
    aggregate_t = float(ttest_1samp(all_fold_delta, 0.0).statistic)
    return {
        "baseline_auc": float(np.mean([row["baseline_auc"] for row in rows])),
        "candidate_auc": float(np.mean([row["candidate_auc"] for row in rows])),
        "mean_delta": float(deltas.mean()),
        "min_delta": float(deltas.min()),
        "positive": int((deltas > 0).sum()),
        "total": len(deltas),
        "aggregate_t": aggregate_t,
        "seeds": {str(seed): row for seed, row in zip(DATA_SEEDS, rows)},
    }


def summarize_stage1(stages):
    summary = {}
    for mask in range(16):
        rows = [stage["masks"][str(mask)] for stage in stages]
        summary[str(mask)] = {
            "allocation": mask_name(mask),
            **aggregate_seed_rows(rows),
        }
    eligible = [
        (int(mask), row) for mask, row in summary.items()
        if int(mask) != 0 and row["positive"] == len(DATA_SEEDS)
    ]
    winner_mask = max(eligible, key=lambda item: item[1]["mean_delta"])[0] if eligible else 0
    return summary, winner_mask


def stage2_path(data_seed, model_seed):
    return CACHE_DIR / f"stage2_data{data_seed}_model{model_seed}.pkl"


def run_stage2_pair(data_seed, model_seed, winner_mask, stage1):
    path = stage2_path(data_seed, model_seed)
    if path.exists():
        saved = load_pickle(path)
        if saved["winner_mask"] == winner_mask:
            log(f"[stage 2 data {data_seed} model {model_seed}] checkpoint exists")
            return saved

    started = time.time()
    seed_data = CR.prepare_seed(data_seed)
    metas = CR.make_folds_meta(seed_data)
    base_columns = CR.cand_cols(["czrb_all79", "wblock"])
    rich_columns = CR.cand_cols(["czrb_all79", "wblock", "white_gloc"])

    if model_seed == 42:
        base_predictions = stage1["base"]
        allocation_predictions = predictions_for_mask(stage1, winner_mask)
    else:
        base_predictions = {}
        allocation_predictions = {}
        for member in MEMBERS:
            base = fit_predictions(member, member_params(member), model_seed, seed_data, base_columns)
            base_predictions[member] = base
            if winner_mask & MEMBER_BITS[member]:
                allocation_predictions[member] = fit_predictions(
                    member, member_params(member), model_seed, seed_data, rich_columns
                )
            else:
                allocation_predictions[member] = base

    baseline_aucs = fold_aucs(blend_predictions(base_predictions), metas)
    allocation_aucs = fold_aucs(blend_predictions(allocation_predictions), metas)
    saved = {
        "data_seed": data_seed,
        "model_seed": model_seed,
        "winner_mask": winner_mask,
        "allocation": paired_summary(allocation_aucs, baseline_aucs),
        "params": {},
    }

    for config, (member, params) in PARAM_FINALISTS.items():
        columns = rich_columns if winner_mask & MEMBER_BITS[member] else base_columns
        candidate_member = fit_predictions(member, params, model_seed, seed_data, columns)
        candidate_predictions = dict(allocation_predictions)
        candidate_predictions[member] = candidate_member
        candidate_aucs = fold_aucs(blend_predictions(candidate_predictions), metas)
        saved["params"][config] = paired_summary(candidate_aucs, allocation_aucs)
        if model_seed == 42:
            saved.setdefault("candidate_predictions", {})[config] = candidate_member

    dump_pickle(path, saved)
    log(f"[stage 2 data {data_seed} model {model_seed}] complete in "
        f"{(time.time() - started) / 60:.1f}m")
    return saved


def summarize_randomness(stage2):
    allocation = {}
    for data_seed in DATA_SEEDS:
        rows = [stage2[(data_seed, model_seed)]["allocation"] for model_seed in MODEL_SEEDS]
        allocation[str(data_seed)] = {
            "mean_delta": float(np.mean([row["delta"] for row in rows])),
            "min_delta": float(np.min([row["delta"] for row in rows])),
            "positive_models": int(sum(row["delta"] > 0 for row in rows)),
            "model_deltas": {str(model_seed): row["delta"]
                             for model_seed, row in zip(MODEL_SEEDS, rows)},
        }

    params = {}
    accepted = []
    for config in PARAM_FINALISTS:
        by_data = {}
        all_deltas = []
        for data_seed in DATA_SEEDS:
            rows = [stage2[(data_seed, model_seed)]["params"][config]
                    for model_seed in MODEL_SEEDS]
            deltas = np.asarray([row["delta"] for row in rows])
            all_deltas.extend(deltas.tolist())
            by_data[str(data_seed)] = {
                "mean_delta": float(deltas.mean()),
                "min_delta": float(deltas.min()),
                "positive_models": int((deltas > 0).sum()),
                "model_deltas": {str(model_seed): float(delta)
                                 for model_seed, delta in zip(MODEL_SEEDS, deltas)},
            }
        data_means = np.asarray([row["mean_delta"] for row in by_data.values()])
        robust = bool((data_means > 0).all() and np.mean(all_deltas) > 0)
        params[config] = {
            "mean_delta": float(np.mean(all_deltas)),
            "min_data_mean_delta": float(data_means.min()),
            "accepted": robust,
            "data_seeds": by_data,
        }
        if robust:
            accepted.append(config)
    return {"allocation": allocation, "params": params, "accepted_params": accepted}


def integer_weight_grid():
    units = int(round(1 / WEIGHT_STEP))
    bounds = {
        name: (int(round(low / WEIGHT_STEP)), int(round(high / WEIGHT_STEP)))
        for name, (low, high) in WEIGHT_BOUNDS.items()
    }
    for values in itertools.product(*(range(bounds[name][0], bounds[name][1] + 1)
                                      for name in MEMBERS)):
        if sum(values) == units:
            yield {name: value / units for name, value in zip(MEMBERS, values)}


def configured_predictions(stage1, stage2_seed42, winner_mask, accepted_params):
    predictions = predictions_for_mask(stage1, winner_mask)
    for config in accepted_params:
        member, _params = PARAM_FINALISTS[config]
        predictions[member] = stage2_seed42["candidate_predictions"][config]
    return predictions


def run_weight_search(stage1, stage2, winner_mask, accepted_params):
    grid = list(integer_weight_grid())
    rows = []
    contexts = {}
    for data_seed, stage in zip(DATA_SEEDS, stage1):
        seed_data = CR.prepare_seed(data_seed)
        metas = CR.make_folds_meta(seed_data)
        contexts[data_seed] = {
            "metas": metas,
            "baseline": fold_aucs(blend_predictions(stage["base"]), metas),
            "configured": configured_predictions(
                stage, stage2[(data_seed, 42)], winner_mask, accepted_params
            ),
        }
        del seed_data
        gc.collect()

    for weights in grid:
        seed_rows = []
        for data_seed in DATA_SEEDS:
            context = contexts[data_seed]
            candidate = fold_aucs(
                blend_predictions(context["configured"], weights), context["metas"]
            )
            seed_rows.append(paired_summary(candidate, context["baseline"]))
        summary = aggregate_seed_rows(seed_rows)
        rows.append({"weights": weights, **summary})

    eligible = [row for row in rows if row["positive"] == len(DATA_SEEDS)]
    best = max(eligible or rows, key=lambda row: row["mean_delta"])
    rows.sort(key=lambda row: row["mean_delta"], reverse=True)
    return {"grid_size": len(grid), "best": best, "top10": rows[:10]}


def specialist_columns():
    white = CR.CAND_SLICE["white_gloc"]
    wblock = CR.CAND_SLICE["wblock"]
    robust = CR.CAND_SLICE["czrb_all79"]
    break_strength_local = []
    for scale_offset in (0, 15, 30):
        break_strength_local.extend([
            scale_offset + 0,  # mean_z
            scale_offset + 1,  # log variance ratio
            scale_offset + 2,  # KS distance
            scale_offset + 7,  # IQR ratio
        ])
    calibrated = robust[np.asarray(break_strength_local, dtype=int)]
    columns = np.concatenate([white, wblock, calibrated])
    if len(columns) != 27:
        raise RuntimeError(f"Expected 27 specialist columns, got {len(columns)}")
    return columns


def run_specialist(stage1, stage2, winner_mask, accepted_params, best_weights):
    columns = specialist_columns()
    results = {}
    for data_seed, stage in zip(DATA_SEEDS, stage1):
        seed_data = CR.prepare_seed(data_seed)
        metas = CR.make_folds_meta(seed_data)
        path = CACHE_DIR / f"specialist_seed{data_seed}.pkl"
        if path.exists():
            saved = load_pickle(path)
        else:
            params = {
                **HP.RANK_BASE,
                "n_estimators": 200,
                "learning_rate": 0.04,
                "num_leaves": 15,
                "min_child_samples": 30,
            }
            predictions = fit_predictions("ranker", params, 42, seed_data, columns)
            saved = {"predictions": predictions, "columns": columns.tolist()}
            dump_pickle(path, saved)

        configured = configured_predictions(
            stage, stage2[(data_seed, 42)], winner_mask, accepted_params
        )
        main_predictions = blend_predictions(configured, best_weights)
        baseline = fold_aucs(blend_predictions(stage["base"]), metas)
        specialist = saved["predictions"]
        for weight in SPECIALIST_WEIGHTS:
            combined = [(1 - weight) * values + weight * specialist[fold]
                        for fold, values in enumerate(main_predictions)]
            candidate = fold_aucs(combined, metas)
            results.setdefault(str(weight), []).append(paired_summary(candidate, baseline))
        del seed_data
        gc.collect()

    summary = {weight: aggregate_seed_rows(rows) for weight, rows in results.items()}
    eligible = [
        (float(weight), row) for weight, row in summary.items()
        if row["positive"] == len(DATA_SEEDS)
    ]
    best_weight, best = max(eligible or [(float(w), r) for w, r in summary.items()],
                            key=lambda item: item[1]["mean_delta"])
    return {"columns": columns.tolist(), "weights": summary,
            "best_weight": best_weight, "best": best}


def format_float(value):
    return f"{value:+.6f}" if value is not None else "NA"


def result_table(row):
    headers = ["config", "absolute TS-AUC", "mean_d", "min_d", "pos", "t_agg"]
    headers.extend(f"seed {seed} d / t" for seed in DATA_SEEDS)
    values = [
        "candidate",
        f"{row['candidate_auc']:.6f}",
        f"{row['mean_delta']:+.6f}",
        f"{row['min_delta']:+.6f}",
        f"{row['positive']}/{row['total']}",
        f"{row['aggregate_t']:+.2f}",
    ]
    for seed in DATA_SEEDS:
        seed_row = row["seeds"][str(seed)]
        values.append(f"{seed_row['delta']:+.6f} / {seed_row['t']:+.2f}")
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(values) + " |",
    ])


def write_report(results):
    stage1 = sorted(results["stage1"].values(), key=lambda row: row["mean_delta"], reverse=True)
    seed_columns = " ".join(f"| seed {seed} d / t " for seed in DATA_SEEDS)
    lines = [
        "# Post-white-GLOC Ensemble Refinement",
        "",
        f"Completed: `{results.get('completed_at', 'in progress')}`",
        "",
        "## Stage 1: white-GLOC allocation",
        "",
        f"| allocation | absolute TS-AUC | mean_d | min_d | pos | t_agg {seed_columns}|",
        "| --- | ---: | ---: | ---: | ---: | ---: "
        + "| ---: " * len(DATA_SEEDS) + "|",
    ]
    for row in stage1:
        per_seed = " ".join(
            f"| {row['seeds'][str(seed)]['delta']:+.6f} / "
            f"{row['seeds'][str(seed)]['t']:+.2f} "
            for seed in DATA_SEEDS
        )
        lines.append(
            f"| {row['allocation']} | {row['candidate_auc']:.6f} | "
            f"{row['mean_delta']:+.6f} | {row['min_delta']:+.6f} | "
            f"{row['positive']}/{row['total']} | {row['aggregate_t']:+.2f} "
            f"{per_seed}|"
        )

    randomness = results.get("stage2", {})
    if randomness:
        lines.extend([
            "", "## Stage 2: model randomness", "",
            "| candidate | mean_d | min data-seed mean_d | accepted |",
            "| --- | ---: | ---: | --- |",
        ])
        for config, row in randomness["params"].items():
            lines.append(
                f"| {config} | {row['mean_delta']:+.6f} | "
                f"{row['min_data_mean_delta']:+.6f} | {row['accepted']} |"
            )

    weights = results.get("stage3")
    if weights:
        best = weights["best"]
        weight_text = ", ".join(f"{name}={best['weights'][name]:.3f}" for name in MEMBERS)
        lines.extend([
            "", "## Stage 3: fixed-CDF weights", "",
            f"Best of {weights['grid_size']} bounded weights: `{weight_text}`.",
            f"Absolute TS-AUC `{best['candidate_auc']:.6f}`, mean_d "
            f"`{best['mean_delta']:+.6f}`, min_d `{best['min_delta']:+.6f}`, "
            f"positive `{best['positive']}/{best['total']}`, t_agg `{best['aggregate_t']:+.2f}`.",
        ])

    final = results.get("stage5")
    if final:
        lines.extend([
            "", "## Stage 5: combined candidate", "",
            result_table(final),
            f"Promotion gate: **{'PASS' if final['passes_gate'] else 'FAIL'}**.",
        ])

    specialist = results.get("stage6")
    if specialist and "best" in specialist:
        best = specialist["best"]
        lines.extend([
            "", "## Stage 6: locator specialist fallback", "",
            f"Best specialist weight: `{specialist['best_weight']:.3f}`.",
            result_table(best),
        ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    CACHE_DIR.mkdir(exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    if args.wait_pid is not None:
        wait_for_pid(args.wait_pid)
    results = load_json(RESULTS, {})
    log(f"=== POST-WHITE PIPELINE {datetime.now():%F %T} ===")
    log(f"data_seeds={DATA_SEEDS} model_seeds={MODEL_SEEDS} jobs={HP.N_JOBS}")

    stages = [run_stage1_seed(seed) for seed in DATA_SEEDS]
    stage1_summary, winner_mask = summarize_stage1(stages)
    results["stage1"] = stage1_summary
    results["winner_mask"] = winner_mask
    results["winner_allocation"] = mask_name(winner_mask)
    write_json(RESULTS, results)
    write_report(results)
    log(f"[stage 1] winner={mask_name(winner_mask)} mask={winner_mask}")

    stage2 = {}
    for data_seed, stage in zip(DATA_SEEDS, stages):
        for model_seed in MODEL_SEEDS:
            stage2[(data_seed, model_seed)] = run_stage2_pair(
                data_seed, model_seed, winner_mask, stage
            )
    randomness = summarize_randomness(stage2)
    results["stage2"] = randomness
    write_json(RESULTS, results)
    write_report(results)
    log(f"[stage 2] accepted_params={randomness['accepted_params']}")

    weight_search = run_weight_search(
        stages, stage2, winner_mask, randomness["accepted_params"]
    )
    results["stage3"] = weight_search
    write_json(RESULTS, results)
    write_report(results)
    log(f"[stage 3] best_weights={weight_search['best']['weights']} "
        f"mean_d={weight_search['best']['mean_delta']:+.6f}")

    final = dict(weight_search["best"])
    final["winner_mask"] = winner_mask
    final["allocation"] = mask_name(winner_mask)
    final["accepted_params"] = randomness["accepted_params"]
    final["passes_gate"] = bool(
        final["mean_delta"] >= PROMOTION_DELTA
        and final["positive"] == len(DATA_SEEDS)
    )
    results["stage5"] = final
    write_json(RESULTS, results)
    write_report(results)
    log(f"[stage 5] gate={'PASS' if final['passes_gate'] else 'FAIL'}")

    if not final["passes_gate"]:
        specialist = run_specialist(
            stages, stage2, winner_mask, randomness["accepted_params"],
            weight_search["best"]["weights"],
        )
        results["stage6"] = specialist
        write_json(RESULTS, results)
        write_report(results)
        log(f"[stage 6] best_weight={specialist['best_weight']:.3f} "
            f"mean_d={specialist['best']['mean_delta']:+.6f}")
    else:
        results["stage6"] = {"skipped": "Stage 5 passed the promotion gate"}

    results["completed_at"] = f"{datetime.now():%F %T}"
    write_json(RESULTS, results)
    write_report(results)
    log(f"=== DONE {results['completed_at']} ===")


if __name__ == "__main__":
    main()
