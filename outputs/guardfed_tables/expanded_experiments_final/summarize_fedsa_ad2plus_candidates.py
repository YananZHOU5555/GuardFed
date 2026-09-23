import json
import math
from pathlib import Path

import pandas as pd


RAW = Path("/home/yannan/workspace/GuardFed/results/paper_tables/raw_results.jsonl")
OUT = Path("/home/yannan/workspace/GuardFed/results/paper_tables/expanded_experiments")
BASELINE_SUITE = "fedsa_all_methods"
CANDIDATE_SUITES = {
    "ad2plus_adaptive_fedsa_grid",
    "ad2plus_adaptive_calibration_grid",
    "ad2plus_adaptive_middle_grid",
    "ad2plus_adaptive_budget_refine",
    "ad2plus_adaptive_acc_refine",
    "ad2plus_profile_bank_fedsa",
    "ad2plus_selector_v2_smoke",
}


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


def candidate_label(cfg):
    suite = cfg.get("experiment_suite")
    tag = cfg.get("experiment_tag") or ""
    seed_suffix = f"_seed{cfg.get('seed')}"
    if suite == BASELINE_SUITE:
        return BASELINE_SUITE
    if tag.endswith(seed_suffix):
        tag = tag[: -len(seed_suffix)]
    return tag or suite


def display_method(record, cfg):
    method = record.get("method") or cfg.get("method")
    suite = cfg.get("experiment_suite")
    label = candidate_label(cfg)
    if method == "GuardFed-AD2+" and label and label != BASELINE_SUITE:
        return f"GuardFed-AD2+ [{label}]"
    return method


def pct(value):
    return 100.0 * float(value)


def rank_frame(frame):
    out = frame.copy()
    group_cols = ["dataset", "distribution"]
    out["score_rank"] = out.groupby(group_cols)["score_mean"].rank(method="min", ascending=False)
    out["acc_rank"] = out.groupby(group_cols)["acc_mean"].rank(method="min", ascending=False)
    out["aeod_rank"] = out.groupby(group_cols)["aeod_mean"].rank(method="min", ascending=True)
    out["aspd_rank"] = out.groupby(group_cols)["aspd_mean"].rank(method="min", ascending=True)
    return out


def main():
    rows = []
    with RAW.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            cfg = record.get("config", {})
            suite = cfg.get("experiment_suite") or "main"
            if suite != BASELINE_SUITE and suite not in CANDIDATE_SUITES:
                continue
            if record.get("attack") != "FedSA":
                continue
            selected = select_joint_last10(record)
            rows.append(
                {
                    "suite": suite,
                    "tag": candidate_label(cfg) or cfg.get("experiment_tag"),
                    "dataset": record.get("dataset"),
                    "distribution": record.get("distribution"),
                    "attack": record.get("attack"),
                    "method": display_method(record, cfg),
                    "base_method": record.get("method"),
                    "seed": cfg.get("seed"),
                    "server_ratio": cfg.get("server_ratio"),
                    "server_sampling": cfg.get("server_sampling"),
                    "ad2_plus_mode": cfg.get("ad2_plus_mode"),
                    "ad2_calibration_budget": cfg.get("ad2_calibration_budget"),
                    "ad2_calibration_max_acc_drop": cfg.get("ad2_calibration_max_acc_drop"),
                    "ad2_norm_clip_scale": cfg.get("ad2_norm_clip_scale"),
                    "ad2_norm_mode": cfg.get("ad2_norm_mode"),
                    "ad2_calibration_objective": cfg.get("ad2_calibration_objective"),
                    **selected,
                }
            )

    if not rows:
        raise SystemExit("No FedSA rows found")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    raw_csv = OUT / "fedsa_ad2plus_candidate_joint_raw.csv"
    summary_csv = OUT / "fedsa_ad2plus_candidate_joint_summary.csv"
    compact_csv = OUT / "fedsa_ad2plus_candidate_compact_ranking.csv"
    md_path = OUT / "fedsa_ad2plus_candidate_summary.md"
    df.to_csv(raw_csv, index=False)

    agg = (
        df.groupby(
            [
                "dataset",
                "distribution",
                "attack",
                "method",
                "suite",
                "tag",
                "base_method",
                "server_ratio",
                "server_sampling",
                "ad2_plus_mode",
                "ad2_calibration_budget",
                "ad2_calibration_max_acc_drop",
                "ad2_norm_clip_scale",
                "ad2_norm_mode",
                "ad2_calibration_objective",
            ],
            dropna=False,
        )
        .agg(
            n=("score", "size"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            aeod_mean=("aeod", "mean"),
            aeod_std=("aeod", "std"),
            aspd_mean=("aspd", "mean"),
            aspd_std=("aspd", "std"),
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            round_mean=("selected_round", "mean"),
        )
        .reset_index()
    )
    agg = rank_frame(agg)
    agg = agg.sort_values(["dataset", "distribution", "score_rank", "method"])
    agg.to_csv(summary_csv, index=False)

    compact = agg[
        [
            "dataset",
            "distribution",
            "method",
            "suite",
            "tag",
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
    ].copy()
    compact.to_csv(compact_csv, index=False)

    display = compact.copy()
    for column in ["acc_mean", "score_mean"]:
        display[column] = display[column].map(pct)

    lines = []
    lines.append("# FedSA new attack: all baselines plus AD2+ candidate search")
    lines.append("")
    lines.append(
        "Selection rule: every run uses a real checkpoint from the last 10 rounds maximizing "
        "`ACC - 0.5*(AEOD+ASPD)`. Baselines use their completed 3-seed FedSA suite; AD2+ rows include completed candidate-search suites."
    )
    lines.append("")
    lines.append("## Top 8 by dataset/distribution")
    lines.append("")
    top = display[display["score_rank"] <= 8]
    lines.append(
        top[
            [
                "dataset",
                "distribution",
                "method",
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
    lines.append("## Best AD2+ rows")
    lines.append("")
    ad2 = display[display["method"].str.startswith("GuardFed-AD2+")].copy()
    lines.append(
        ad2.sort_values(["dataset", "distribution", "score_rank"])[
            [
                "dataset",
                "distribution",
                "method",
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
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"raw_rows={len(df)} summary_rows={len(agg)}")
    print(f"wrote={raw_csv}")
    print(f"wrote={summary_csv}")
    print(f"wrote={compact_csv}")
    print(f"wrote={md_path}")
    for dataset in ["adult", "compas"]:
        for distribution in ["IID", "non-IID"]:
            view = agg[(agg["dataset"] == dataset) & (agg["distribution"] == distribution)].sort_values("score_rank")
            print(f"\n== {dataset} {distribution} top 8 ==")
            print(
                view[
                    [
                        "method",
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
                ].head(8).to_string(index=False)
            )


if __name__ == "__main__":
    main()
