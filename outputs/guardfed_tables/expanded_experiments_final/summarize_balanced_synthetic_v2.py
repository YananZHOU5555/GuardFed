import json
import math
from pathlib import Path

import pandas as pd


RAW = Path("/home/yannan/workspace/GuardFed/results/paper_tables/raw_results.jsonl")
OUT = Path("/home/yannan/workspace/GuardFed/results/paper_tables/expanded_experiments")
SUITE = "synthetic_balanced_ratios_v2"


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def score(acc, aeod, aspd):
    return float(acc) - 0.5 * (float(aeod) + float(aspd))


def select_joint_last10(record):
    """Select one real checkpoint from the last 10 rounds by joint score."""
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
    acc = metrics.get("accuracy")
    aeod = metrics.get("aeod")
    aspd = metrics.get("aspd")
    return {
        "selected_round": record.get("rounds"),
        "acc": float(acc),
        "aeod": float(aeod),
        "aspd": float(aspd),
        "score": score(acc, aeod, aspd),
        "selection": "final_round_fallback",
    }


def setting_label(cfg):
    server_ratio = float(cfg.get("server_ratio") or 0.0)
    synthetic_ratio = float(cfg.get("synthetic_ratio") or 0.0)
    synthetic_method = cfg.get("synthetic_method") or "none"
    server_pct = int(round(server_ratio * 100))
    synthetic_pct = int(round(synthetic_ratio * 100))
    if synthetic_pct == 0 or synthetic_method == "none":
        return f"{server_pct}% real clean"
    return f"{server_pct}% real + {synthetic_pct}% {synthetic_method}"


def pct(value):
    return 100.0 * float(value)


def rank_columns(frame, group_cols):
    ranked = frame.copy()
    ranked["score_rank"] = ranked.groupby(group_cols)["score_mean"].rank(
        method="min", ascending=False
    )
    ranked["acc_rank"] = ranked.groupby(group_cols)["acc_mean"].rank(
        method="min", ascending=False
    )
    ranked["aeod_rank"] = ranked.groupby(group_cols)["aeod_mean"].rank(
        method="min", ascending=True
    )
    ranked["aspd_rank"] = ranked.groupby(group_cols)["aspd_mean"].rank(
        method="min", ascending=True
    )
    return ranked


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
                    "setting": setting_label(cfg),
                    "server_ratio": float(cfg.get("server_ratio") or 0.0),
                    "synthetic_ratio": float(cfg.get("synthetic_ratio") or 0.0),
                    "synthetic_method": cfg.get("synthetic_method") or "none",
                    "server_sampling": cfg.get("server_sampling"),
                    **selected,
                }
            )

    if not rows:
        raise SystemExit(f"No rows found for {SUITE}")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    raw_csv = OUT / "synthetic_balanced_v2_joint_raw.csv"
    by_dataset_csv = OUT / "synthetic_balanced_v2_by_dataset.csv"
    by_slice_csv = OUT / "synthetic_balanced_v2_by_slice.csv"
    md_path = OUT / "synthetic_balanced_v2_summary.md"

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
        df.groupby(
            [
                "dataset",
                "setting",
                "server_ratio",
                "synthetic_ratio",
                "synthetic_method",
                "server_sampling",
            ],
            dropna=False,
        )
        .agg(**agg_spec)
        .reset_index()
    )
    by_dataset = rank_columns(by_dataset, ["dataset"])
    by_dataset = by_dataset.sort_values(["dataset", "score_rank", "setting"])
    by_dataset.to_csv(by_dataset_csv, index=False)

    by_slice = (
        df.groupby(
            [
                "dataset",
                "distribution",
                "attack",
                "setting",
                "server_ratio",
                "synthetic_ratio",
                "synthetic_method",
                "server_sampling",
            ],
            dropna=False,
        )
        .agg(**agg_spec)
        .reset_index()
    )
    by_slice = rank_columns(by_slice, ["dataset", "distribution", "attack"])
    by_slice = by_slice.sort_values(
        ["dataset", "distribution", "attack", "score_rank", "setting"]
    )
    by_slice.to_csv(by_slice_csv, index=False)

    display_dataset = by_dataset.copy()
    for column in ["acc_mean", "acc_std"]:
        display_dataset[column] = display_dataset[column].map(pct)
    for column in ["score_mean", "score_std"]:
        display_dataset[column] = display_dataset[column].map(pct)
    for column in ["aeod_mean", "aeod_std", "aspd_mean", "aspd_std"]:
        display_dataset[column] = display_dataset[column].map(lambda x: float(x))

    display_slice = by_slice.copy()
    for column in ["acc_mean", "acc_std", "score_mean", "score_std"]:
        display_slice[column] = display_slice[column].map(pct)

    lines = []
    lines.append("# Synthetic server/root data ablation: balanced clean-root v2")
    lines.append("")
    lines.append(
        "Selection rule: each completed run uses one real checkpoint from the last 10 rounds that maximizes "
        "`ACC - 0.5*(AEOD+ASPD)`. Values are then averaged across 3 seeds, 2 distributions, and 2 attacks "
        "for the dataset-level table."
    )
    lines.append("")
    lines.append("## Dataset-level summary")
    lines.append("")
    lines.append(
        display_dataset[
            [
                "dataset",
                "setting",
                "n",
                "acc_mean",
                "acc_std",
                "aeod_mean",
                "aeod_std",
                "aspd_mean",
                "aspd_std",
                "score_mean",
                "score_std",
                "score_rank",
                "acc_rank",
                "aeod_rank",
                "aspd_rank",
            ]
        ].to_markdown(index=False, floatfmt=".4f")
    )
    lines.append("")
    lines.append("## `10% real clean` rank check")
    lines.append("")
    ten = by_dataset[by_dataset["setting"] == "10% real clean"].copy()
    if ten.empty:
        lines.append("`10% real clean` was not present in this suite.")
    else:
        ten_display = ten.copy()
        for column in ["acc_mean", "acc_std", "score_mean", "score_std"]:
            ten_display[column] = ten_display[column].map(pct)
        lines.append(
            ten_display[
                [
                    "dataset",
                    "setting",
                    "n",
                    "acc_mean",
                    "aeod_mean",
                    "aspd_mean",
                    "score_mean",
                    "score_rank",
                    "acc_rank",
                    "aeod_rank",
                    "aspd_rank",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
    lines.append("")
    lines.append("## Slice-level summary")
    lines.append("")
    lines.append(
        display_slice[
            [
                "dataset",
                "distribution",
                "attack",
                "setting",
                "n",
                "acc_mean",
                "acc_std",
                "aeod_mean",
                "aspd_mean",
                "score_mean",
                "score_rank",
                "acc_rank",
                "aeod_rank",
                "aspd_rank",
            ]
        ].to_markdown(index=False, floatfmt=".4f")
    )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"rows={len(df)}")
    print(f"wrote={raw_csv}")
    print(f"wrote={by_dataset_csv}")
    print(f"wrote={by_slice_csv}")
    print(f"wrote={md_path}")
    print(
        by_dataset[
            [
                "dataset",
                "setting",
                "n",
                "acc_mean",
                "aeod_mean",
                "aspd_mean",
                "score_mean",
                "score_rank",
                "acc_rank",
                "aeod_rank",
                "aspd_rank",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
