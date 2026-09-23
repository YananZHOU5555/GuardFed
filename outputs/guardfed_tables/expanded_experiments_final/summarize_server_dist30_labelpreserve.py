import json
import math
import os
from pathlib import Path

import pandas as pd


RAW = Path("/home/yannan/workspace/GuardFed/results/paper_tables/raw_results.jsonl")
OUT = Path("/home/yannan/workspace/GuardFed/results/paper_tables/expanded_experiments")
SUITE = os.environ.get("SERVER_SUITE", "expanded_server_dist30_labelpreserve")
PREFIX = os.environ.get("SERVER_PREFIX") or SUITE.replace("expanded_", "").replace("_v4", "")


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def score(acc, aeod, aspd):
    return float(acc) - 0.5 * (float(aeod) + float(aspd))


def select_joint_last10(record):
    best = None
    for item in record.get("last10_metrics") or []:
        metrics = item.get("metrics", {})
        acc = metrics.get("accuracy")
        aeod = metrics.get("aeod")
        aspd = metrics.get("aspd")
        if not all(finite_number(v) for v in (acc, aeod, aspd)):
            continue
        candidate = {
            "selected_round": item.get("round"),
            "acc": float(acc),
            "aeod": float(aeod),
            "aspd": float(aspd),
            "score": score(acc, aeod, aspd),
            "selection": "best_joint_last10",
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is not None:
        return best
    metrics = record.get("metrics", {})
    return {
        "selected_round": record.get("rounds"),
        "acc": float(metrics.get("accuracy")),
        "aeod": float(metrics.get("aeod")),
        "aspd": float(metrics.get("aspd")),
        "score": score(metrics.get("accuracy"), metrics.get("aeod"), metrics.get("aspd")),
        "selection": "final_round_fallback",
    }


def pct(value):
    return 100.0 * float(value)


def curve_diagnostics(frame, group_cols):
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.sort_values("server_alpha")
        acc = group["acc_mean"].tolist()
        score_values = group["score_mean"].tolist()
        alpha_values = group["server_alpha"].tolist()
        acc_jumps = [abs(b - a) for a, b in zip(acc, acc[1:])]
        score_jumps = [abs(b - a) for a, b in zip(score_values, score_values[1:])]
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "num_alpha": len(group),
                "acc_min": min(acc),
                "acc_max": max(acc),
                "acc_range": max(acc) - min(acc),
                "max_adjacent_acc_jump": max(acc_jumps) if acc_jumps else 0.0,
                "max_adjacent_score_jump": max(score_jumps) if score_jumps else 0.0,
                "best_alpha_by_score": alpha_values[int(group["score_mean"].values.argmax())],
                "best_score": max(score_values),
            }
        )
    return pd.DataFrame(rows)


def main():
    rows = []
    with RAW.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            cfg = record.get("config", {})
            if cfg.get("experiment_suite") != SUITE:
                continue
            selected = select_joint_last10(record)
            rows.append(
                {
                    "dataset": record.get("dataset"),
                    "distribution": record.get("distribution"),
                    "attack": record.get("attack"),
                    "method": record.get("method"),
                    "seed": cfg.get("seed"),
                    "tag": cfg.get("experiment_tag"),
                    "server_alpha": float(cfg.get("server_alpha")),
                    "server_ratio": float(cfg.get("server_ratio") or 0.0),
                    "server_sampling": cfg.get("server_sampling"),
                    **selected,
                }
            )

    if not rows:
        raise SystemExit(f"No rows found for {SUITE}")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    raw_csv = OUT / f"{PREFIX}_joint_raw.csv"
    by_dataset_csv = OUT / f"{PREFIX}_by_alpha_dataset.csv"
    by_slice_csv = OUT / f"{PREFIX}_by_alpha_slice.csv"
    diag_csv = OUT / f"{PREFIX}_curve_diagnostics.csv"
    md_path = OUT / f"{PREFIX}_summary.md"
    plot_acc = OUT / f"{PREFIX}_acc_curve.png"
    plot_fair = OUT / f"{PREFIX}_fairness_curve.png"
    plot_score = OUT / f"{PREFIX}_score_curve.png"

    df.to_csv(raw_csv, index=False)
    agg_spec = {
        "n": ("score", "size"),
        "acc_mean": ("acc", "mean"),
        "acc_std": ("acc", "std"),
        "aeod_mean": ("aeod", "mean"),
        "aeod_std": ("aeod", "std"),
        "aspd_mean": ("aspd", "mean"),
        "aspd_std": ("aspd", "std"),
        "score_mean": ("score", "mean"),
        "score_std": ("score", "std"),
        "round_mean": ("selected_round", "mean"),
    }
    by_dataset = (
        df.groupby(["dataset", "server_alpha"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["dataset", "server_alpha"])
    )
    by_dataset.to_csv(by_dataset_csv, index=False)

    by_slice = (
        df.groupby(["dataset", "distribution", "attack", "server_alpha"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["dataset", "distribution", "attack", "server_alpha"])
    )
    by_slice.to_csv(by_slice_csv, index=False)

    diag_dataset = curve_diagnostics(by_dataset, ["dataset"])
    diag_slice = curve_diagnostics(by_slice, ["dataset", "distribution", "attack"])
    diag = pd.concat([diag_dataset.assign(level="dataset"), diag_slice.assign(level="slice")], ignore_index=True)
    diag.to_csv(diag_csv, index=False)

    try:
        import matplotlib.pyplot as plt

        for metric, path, ylabel in [
            ("acc_mean", plot_acc, "ACC"),
            ("score_mean", plot_score, "Joint score"),
        ]:
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=180)
            for dataset, group in by_dataset.groupby("dataset"):
                group = group.sort_values("server_alpha")
                ax.plot(group["server_alpha"], group[metric], marker="o", linewidth=1.6, label=dataset)
                yerr = group[metric.replace("mean", "std")]
                ax.fill_between(
                    group["server_alpha"],
                    group[metric] - yerr,
                    group[metric] + yerr,
                    alpha=0.12,
                )
            ax.set_xscale("log")
            ax.set_xlabel("Dirichlet alpha of clean server/root data")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=180)
        for dataset, group in by_dataset.groupby("dataset"):
            group = group.sort_values("server_alpha")
            ax.plot(group["server_alpha"], group["aeod_mean"], marker="o", linewidth=1.6, label=f"{dataset} AEOD")
            ax.plot(group["server_alpha"], group["aspd_mean"], marker="s", linewidth=1.3, linestyle="--", label=f"{dataset} ASPD")
        ax.set_xscale("log")
        ax.set_xlabel("Dirichlet alpha of clean server/root data")
        ax.set_ylabel("Fairness gap")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(plot_fair)
        plt.close(fig)
    except Exception as exc:
        print(f"plot_warning={exc}")

    display_dataset = by_dataset.copy()
    for column in ["acc_mean", "acc_std", "score_mean", "score_std"]:
        display_dataset[column] = display_dataset[column].map(pct)

    display_diag = diag.copy()
    for column in ["acc_min", "acc_max", "acc_range", "max_adjacent_acc_jump", "max_adjacent_score_jump", "best_score"]:
        display_diag[column] = display_diag[column].map(pct)

    lines = []
    lines.append(f"# Server/root distribution ablation: {SUITE}")
    lines.append("")
    lines.append(
        "Selection rule: each run uses one real checkpoint from the last 10 rounds that maximizes "
        "`ACC - 0.5*(AEOD+ASPD)`. Dataset-level values average over 3 seeds, 2 distributions, and 2 attacks."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Raw rows: {len(df)}")
    lines.append(f"- Unique alpha values: {df['server_alpha'].nunique()}")
    lines.append(f"- Seeds per alpha tag are expected to cover 123/456/789.")
    lines.append("")
    lines.append("## Dataset-level alpha curve")
    lines.append("")
    lines.append(
        display_dataset[
            [
                "dataset",
                "server_alpha",
                "n",
                "acc_mean",
                "acc_std",
                "aeod_mean",
                "aeod_std",
                "aspd_mean",
                "aspd_std",
                "score_mean",
                "score_std",
            ]
        ].to_markdown(index=False, floatfmt=".4f")
    )
    lines.append("")
    lines.append("## Curve diagnostics")
    lines.append("")
    lines.append(
        display_diag[
            [
                "level",
                "dataset",
                "distribution",
                "attack",
                "num_alpha",
                "acc_min",
                "acc_max",
                "acc_range",
                "max_adjacent_acc_jump",
                "max_adjacent_score_jump",
                "best_alpha_by_score",
                "best_score",
            ]
        ].to_markdown(index=False, floatfmt=".4f")
    )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"- ACC curve: `{plot_acc}`")
    lines.append(f"- Fairness curve: `{plot_fair}`")
    lines.append(f"- Joint score curve: `{plot_score}`")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"rows={len(df)} alphas={df['server_alpha'].nunique()}")
    print(f"wrote={raw_csv}")
    print(f"wrote={by_dataset_csv}")
    print(f"wrote={by_slice_csv}")
    print(f"wrote={diag_csv}")
    print(f"wrote={md_path}")
    print(
        diag_dataset[
            [
                "dataset",
                "num_alpha",
                "acc_min",
                "acc_max",
                "acc_range",
                "max_adjacent_acc_jump",
                "best_alpha_by_score",
                "best_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
