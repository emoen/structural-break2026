"""Real held-out EDA of champion score behavior across time regimes.

Uses one existing GLOC6/WBLOCK feature cache, but deliberately selects only the 166
columns shipped by the 63.17 champion (base79 + robust calibration79 + WBLOCK8).
The four baseline members are fit once on the seed's training partition, then scored
on all 2,000 held-out series at every online step.

Usage:
  N_JOBS=15 .venv/bin/python -u _eda_time_regimes.py
  N_JOBS=15 .venv/bin/python -u _eda_time_regimes.py --wait-pid 41864
"""
import argparse
import gc
import json
import os
import select
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("GLOC6", "1")
os.environ.setdefault("WBLOCK", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _ensemble_hp_white_gloc as HP
from _exp_fast import split_series

CR = HP.CR
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "docs" / "time_regime_eda"
POST_REGIMES = [
    ("onset 0-4", 0, 4),
    ("early 5-14", 5, 14),
    ("developing 15-39", 15, 39),
    ("established 40-79", 40, 79),
    ("mature 80+", 80, None),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def wait_for_pid(pid):
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    except AttributeError as exc:
        raise RuntimeError("--wait-pid requires Linux pidfd support") from exc
    print(f"Waiting for PID {pid} to finish before using model-fitting resources...", flush=True)
    try:
        select.select([descriptor], [], [])
    finally:
        os.close(descriptor)
    print(f"PID {pid} finished; starting EDA.", flush=True)


def fit_champion_predictions(seed):
    sd = CR.prepare_seed(seed)
    columns = CR.cand_cols(["czrb_all79", "wblock"])
    if len(columns) != 166:
        raise RuntimeError(f"Expected 166 champion columns, got {len(columns)}")

    member_predictions = {}
    specs = [
        ("ranker", HP.RANK_BASE),
        ("rf", HP.RF_BASE),
        ("xgb", HP.XGB_BASE),
        ("catboost", HP.CAT_BASE),
    ]
    for name, params in specs:
        started = time.time()
        model, predictions = HP.fit_member(name, params, sd, columns)
        member_predictions[name] = predictions
        del model
        gc.collect()
        print(f"Fit and scored {name}: {time.time() - started:.0f}s", flush=True)

    blended = HP.blend(member_predictions)
    return sd, blended


def build_prediction_frame(sd, fold_predictions):
    frames = []
    target = sd["y_val"][["target"]]
    for fold, (cache, scores) in enumerate(zip(sd["caches"], fold_predictions)):
        frame = pd.DataFrame({"score": scores, "fold": fold}, index=cache["index"])
        frame = frame.join(target, how="left")
        if frame["target"].isna().any():
            raise RuntimeError("Validation labels did not align with cached prediction rows")
        frame["time_online"] = frame.groupby(level="id").cumcount()
        frames.append(frame)
    out = pd.concat(frames).sort_index()
    out["target"] = out["target"].astype(np.int8)
    return out


def infer_tau(group):
    positive = np.flatnonzero(group["target"].to_numpy() == 1)
    return int(positive[0]) if len(positive) else -1


def auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def add_tau_and_age(frame):
    tau_by_id = frame.groupby(level="id", sort=False).apply(infer_tau)
    tau_by_id.name = "tau"
    out = frame.join(tau_by_id, on="id")
    out["post_age"] = np.where(
        out["target"].to_numpy() == 1,
        out["time_online"].to_numpy() - out["tau"].to_numpy(),
        -1,
    )
    return out, tau_by_id


def weighted_auc_rows(frame):
    rows = []
    for time_online, group in frame.groupby("time_online", sort=True):
        labels = group["target"].to_numpy(dtype=np.int8)
        scores = group["score"].to_numpy(dtype=np.float64)
        n_pos = int(labels.sum())
        n_neg = int(len(labels) - n_pos)
        step_auc = auc(labels, scores)
        if step_auc is None:
            continue
        rows.append({
            "time_online": int(time_online),
            "auc": step_auc,
            "weight": n_pos * n_neg,
            "n_pos": n_pos,
            "n_neg": n_neg,
        })
    return pd.DataFrame(rows)


def conditional_regime_auc(frame, name, low, high):
    weighted = 0.0
    total_weight = 0
    steps = 0
    positive_rows = 0
    for _, group in frame.groupby("time_online", sort=True):
        ages = group["post_age"].to_numpy()
        labels = group["target"].to_numpy(dtype=np.int8)
        positive = (labels == 1) & (ages >= low)
        if high is not None:
            positive &= ages <= high
        negative = labels == 0
        keep = positive | negative
        n_pos = int(positive.sum())
        n_neg = int(negative.sum())
        if n_pos == 0 or n_neg == 0:
            continue
        regime_labels = positive[keep].astype(np.int8)
        regime_auc = auc(regime_labels, group["score"].to_numpy()[keep])
        weight = n_pos * n_neg
        weighted += weight * regime_auc
        total_weight += weight
        positive_rows += n_pos
        steps += 1
    return {
        "regime": name,
        "min_post_age": low,
        "max_post_age": high,
        "auc": weighted / total_weight if total_weight else np.nan,
        "weight": total_weight,
        "time_steps": steps,
        "positive_rows": positive_rows,
    }


def safe_mean(values):
    return float(np.mean(values)) if len(values) else np.nan


def safe_max(values):
    return float(np.max(values)) if len(values) else np.nan


def series_diagnostics(frame):
    rows = []
    for sid, group in frame.groupby(level="id", sort=True):
        group = group.sort_values("time_online")
        scores = group["score"].to_numpy(dtype=np.float64)
        tau = int(group["tau"].iloc[0])
        if tau >= 0:
            pre = scores[:tau]
            post = scores[tau:]
            crossings = np.flatnonzero(post >= 0.5)
            delay = int(crossings[0]) if len(crossings) else np.nan
            early = post[:20]
            established = post[40:80]
            late = post[80:]
        else:
            pre = scores
            post = np.empty(0)
            delay = np.nan
            early = established = late = np.empty(0)
        diffs = np.abs(np.diff(scores))
        rows.append({
            "id": sid,
            "tau": tau,
            "length": len(scores),
            "has_break": tau >= 0,
            "pre_mean": safe_mean(pre),
            "pre_max": safe_max(pre),
            "pre_high_frac": safe_mean(pre >= 0.5),
            "post_mean": safe_mean(post),
            "post_early_mean": safe_mean(early),
            "post_established_mean": safe_mean(established),
            "post_late_mean": safe_mean(late),
            "detection_delay_050": delay,
            "early_lift": safe_mean(early) - safe_mean(pre[-20:]),
            "decay": safe_mean(early) - safe_mean(late),
            "max_jump": safe_max(diffs),
            "mean_jump": safe_mean(diffs),
            "plateau_fraction": safe_mean(diffs < 1e-12),
            "total_variation": float(diffs.sum()),
        })
    return pd.DataFrame(rows).set_index("id")


def ranked_pick(metrics, mask, column, ascending, used):
    candidates = metrics.loc[mask & ~metrics.index.isin(used)].dropna(subset=[column])
    if candidates.empty:
        return None
    return candidates.sort_values([column, "length"], ascending=[ascending, False]).index[0]


def select_examples(metrics):
    broken = metrics["has_break"]
    enough_pre = broken & (metrics["tau"] >= 20)
    enough_late = broken & ((metrics["length"] - metrics["tau"]) >= 100)
    no_break = ~broken
    rules = [
        ("clean early detection", enough_pre, "early_lift", False),
        ("delayed detection", broken & metrics["detection_delay_050"].notna(),
         "detection_delay_050", False),
        ("missed break", broken, "post_mean", True),
        ("transient / decaying detection", enough_late, "decay", False),
        ("pre-break false alarm", enough_pre, "pre_max", False),
        ("quiet no-break", no_break, "post_mean", True),
        ("no-break false positive", no_break, "pre_max", False),
        ("largest point-to-point jump", pd.Series(True, index=metrics.index), "max_jump", False),
    ]
    selected = []
    used = set()
    for category, mask, column, ascending in rules:
        if category == "quiet no-break":
            column = "pre_mean"
        sid = ranked_pick(metrics, mask, column, ascending, used)
        if sid is None:
            continue
        used.add(sid)
        selected.append((category, sid))
    return selected


def load_selected_raw(ids):
    path = HERE / "data" / "X_train.parquet"
    normalized = [item.item() if hasattr(item, "item") else item for item in ids]
    return pd.read_parquet(path, filters=[("id", "in", normalized)])


def plot_regimes(per_time, regimes, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(per_time["time_online"], per_time["auc"], color="#16697a", lw=1.5)
    axes[0].axhline(0.5, color="#777777", ls=":", lw=1)
    axes[0].set(title="Champion TS-AUC by online step", xlabel="online step", ylabel="TS-AUC",
                ylim=(0.45, 1.0))
    axes[1].barh(regimes["regime"], regimes["auc"], color="#2a9d8f")
    axes[1].axvline(0.5, color="#777777", ls=":", lw=1)
    axes[1].set(title="Conditional AUC by post-break age", xlabel="conditional AUC",
                xlim=(0.45, 1.0))
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "regime_auc.png", dpi=150)
    plt.close(fig)


def plot_examples(frame, metrics, selected, raw, output_dir):
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharey=True)
    for axis, (category, sid) in zip(axes.flat, selected):
        group = frame.xs(sid, level="id").sort_values("time_online")
        time_online = group["time_online"].to_numpy()
        scores = group["score"].to_numpy()
        target = group["target"].to_numpy()
        tau = int(metrics.loc[sid, "tau"])
        axis.plot(time_online, scores, color="#168aad", lw=1.5, label="champion score")
        axis.plot(time_online, target, color="#d1495b", ls="--", lw=1.1, label="ideal")
        if tau >= 0:
            axis.axvline(tau, color="#d1495b", ls=":", lw=1)
        axis.set_title(f"{category}\nid={sid}, tau={tau}", fontsize=9)
        axis.set_ylim(-0.04, 1.04)
        axis.set_xlabel("online step")
        axis.set_ylabel("score")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Representative held-out champion trajectories", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_dir / "trajectory_gallery.png", dpi=150)
    plt.close(fig)

    for number, (category, sid) in enumerate(selected, start=1):
        group = frame.xs(sid, level="id").sort_values("time_online")
        time_online = group["time_online"].to_numpy()
        scores = group["score"].to_numpy()
        target = group["target"].to_numpy()
        tau = int(metrics.loc[sid, "tau"])
        _, online = split_series(raw.loc[sid])
        values = online.to_numpy(dtype=np.float64)

        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        top.plot(np.arange(len(values)), values, color="#345995", lw=1)
        if tau >= 0:
            top.axvline(tau, color="#d1495b", ls="--", lw=1.2)
        top.set(title=f"{category}: series {sid}", ylabel="online value")
        bottom.plot(time_online, scores, color="#168aad", lw=1.5, label="champion score")
        bottom.plot(time_online, target, color="#d1495b", ls="--", lw=1.1, label="ideal score")
        if tau >= 0:
            bottom.axvline(tau, color="#d1495b", ls=":", lw=1)
        bottom.set(xlabel="online step", ylabel="score", ylim=(-0.04, 1.04))
        bottom.legend(loc="best", fontsize=8)
        fig.tight_layout()
        safe_category = category.replace(" ", "_").replace("/", "-")
        fig.savefig(output_dir / f"trajectory_{number:02d}_{safe_category}.png", dpi=150)
        plt.close(fig)


def markdown_table(frame, columns, formats=None):
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("NA")
            elif column in formats:
                values.append(formats[column].format(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(frame, per_time, regimes, metrics, selected, output_dir, seed):
    total_weight = per_time["weight"].sum()
    overall_auc = float(np.average(per_time["auc"], weights=per_time["weight"]))
    weakest = regimes.loc[regimes["auc"].idxmin()]
    broken = metrics[metrics["has_break"]]
    no_break = metrics[~metrics["has_break"]]
    detectable = broken["detection_delay_050"].dropna()
    decay_rate = float((broken["decay"].dropna() > 0.10).mean())
    summary = {
        "seed": seed,
        "series": int(len(metrics)),
        "online_rows": int(len(frame)),
        "overall_ts_auc": overall_auc,
        "metric_weight": int(total_weight),
        "weakest_post_break_regime": str(weakest["regime"]),
        "weakest_regime_auc": float(weakest["auc"]),
        "break_series_crossing_0.5_fraction": float(len(detectable) / len(broken)),
        "median_detection_delay_at_0.5": float(detectable.median()) if len(detectable) else None,
        "break_series_decay_over_0.1_fraction": decay_rate,
        "no_break_pre_score_ge_0.5_fraction": float(no_break["pre_high_frac"].mean()),
        "mean_plateau_fraction": float(metrics["plateau_fraction"].mean()),
        "median_max_jump": float(metrics["max_jump"].median()),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    selected_rows = metrics.loc[[sid for _, sid in selected]].reset_index()
    selected_rows.insert(0, "case", [category for category, _ in selected])
    selected_rows.to_csv(output_dir / "selected_series.csv", index=False)

    report = [
        "# Champion Time-Regime EDA",
        "",
        f"Seed `{seed}` real held-out validation; untouched 166-feature four-member champion.",
        "The conditional age-band AUC compares positives in that post-break age band with all",
        "currently negative series at the same online step, preserving the cross-sectional metric.",
        "",
        "## Main results",
        "",
        f"- Absolute TS-AUC: **{overall_auc:.6f}** across {len(metrics):,} held-out series and "
        f"{len(frame):,} online rows.",
        f"- Weakest positive-age regime: **{weakest['regime']}**, AUC **{weakest['auc']:.6f}**.",
        f"- Break series ever crossing score 0.5: **{len(detectable) / len(broken):.1%}**; "
        f"median first-crossing delay among crossers: **{summary['median_detection_delay_at_0.5']}** steps.",
        f"- Break series with early-to-late score decay above 0.10: **{decay_rate:.1%}**.",
        f"- No-break online points scoring at least 0.5: **{summary['no_break_pre_score_ge_0.5_fraction']:.1%}**.",
        f"- Mean exact plateau fraction: **{summary['mean_plateau_fraction']:.1%}**; "
        f"median per-series maximum jump: **{summary['median_max_jump']:.3f}**.",
        "",
        "## Post-break age bands",
        "",
        markdown_table(regimes, ["regime", "auc", "positive_rows", "time_steps"],
                       {"auc": "{:.6f}", "positive_rows": "{:.0f}", "time_steps": "{:.0f}"}),
        "",
        "## Selected trajectories",
        "",
        markdown_table(selected_rows, ["case", "id", "tau", "length", "pre_max", "post_mean",
                                       "detection_delay_050", "decay", "max_jump"],
                       {"tau": "{:.0f}", "length": "{:.0f}", "pre_max": "{:.3f}",
                        "post_mean": "{:.3f}", "detection_delay_050": "{:.0f}",
                        "decay": "{:+.3f}", "max_jump": "{:.3f}"}),
        "",
        "## Interpretation",
        "",
        "The score need not be temporally smooth: TS-AUC rewards within-step ordering. Treat the",
        "trajectory plots as diagnostics. A temporal feature or specialist is justified only when",
        "a weak age regime is repeatable and the change improves the full cross-sectional metric.",
        "",
        "![TS-AUC regime diagnostics](regime_auc.png)",
        "",
        "![Representative score trajectories](trajectory_gallery.png)",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report))


def main():
    args = parse_args()
    if args.wait_pid is not None:
        wait_for_pid(args.wait_pid)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading seed {args.seed} cache and fitting exact champion members...", flush=True)
    sd, fold_predictions = fit_champion_predictions(args.seed)
    frame = build_prediction_frame(sd, fold_predictions)
    frame, _ = add_tau_and_age(frame)

    per_time = weighted_auc_rows(frame)
    regimes = pd.DataFrame([
        conditional_regime_auc(frame, name, low, high)
        for name, low, high in POST_REGIMES
    ])
    metrics = series_diagnostics(frame)
    selected = select_examples(metrics)
    if len(selected) < 5:
        raise RuntimeError(f"Expected at least five distinct trajectory examples, got {len(selected)}")

    frame.reset_index().to_parquet(args.output_dir / "heldout_predictions.parquet", index=False)
    per_time.to_csv(args.output_dir / "auc_by_online_step.csv", index=False)
    regimes.to_csv(args.output_dir / "auc_by_post_break_age.csv", index=False)
    metrics.reset_index().to_csv(args.output_dir / "series_diagnostics.csv", index=False)

    raw = load_selected_raw([sid for _, sid in selected])
    plot_regimes(per_time, regimes, args.output_dir)
    plot_examples(frame, metrics, selected, raw, args.output_dir)
    write_report(frame, per_time, regimes, metrics, selected, args.output_dir, args.seed)
    print(f"EDA complete: {args.output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
