from pathlib import Path

import pandas as pd


OUT = Path("results/paper_tables/advisor_experiments")


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    abl = pd.read_csv(OUT / "adult_ad2plus_ablation.csv")
    dist = pd.read_csv(OUT / "server_distribution_ablation.csv")
    syn = pd.read_csv(OUT / "server_generation_ablation.csv")

    abl_sum = abl.groupby("tag").agg(
        n=("tag", "size"),
        acc_mean=("ACC", "mean"),
        acc_min=("ACC", "min"),
        acc_max=("ACC", "max"),
        aeod_mean=("AEOD", "mean"),
        aeod_max=("AEOD", "max"),
        aspd_mean=("ASPD", "mean"),
        aspd_max=("ASPD", "max"),
        fair_mean=("fair_avg", "mean"),
        score_mean=("score", "mean"),
    ).reset_index()
    abl_sum["score_rank"] = abl_sum["score_mean"].rank(ascending=False, method="min").astype(int)
    abl_sum["acc_rank"] = abl_sum["acc_mean"].rank(ascending=False, method="min").astype(int)
    abl_sum["fair_rank"] = abl_sum["fair_mean"].rank(ascending=True, method="min").astype(int)
    abl_sum = abl_sum.sort_values(["score_rank", "tag"])
    abl_sum.to_csv(OUT / "adult_ablation_by_tag_summary.csv", index=False)

    trend = dist.groupby(["dataset", "server_alpha"]).agg(
        n=("tag", "size"),
        acc_mean=("ACC", "mean"),
        acc_min=("ACC", "min"),
        aeod_mean=("AEOD", "mean"),
        aspd_mean=("ASPD", "mean"),
        fair_mean=("fair_avg", "mean"),
        score_mean=("score", "mean"),
    ).reset_index().sort_values(["dataset", "server_alpha"])
    base = trend[trend.server_alpha == 5000.0].set_index("dataset")
    trend_rows = []
    for _, row in trend.iterrows():
        baseline = base.loc[row.dataset]
        record = row.to_dict()
        record["delta_score_vs_5000"] = row.score_mean - baseline.score_mean
        record["delta_acc_vs_5000"] = row.acc_mean - baseline.acc_mean
        record["delta_fair_vs_5000"] = row.fair_mean - baseline.fair_mean
        trend_rows.append(record)
    trend2 = pd.DataFrame(trend_rows)
    trend2.to_csv(OUT / "server_distribution_trend_by_alpha.csv", index=False)

    syn_sum = syn.groupby(["tag", "dataset"]).agg(
        n=("tag", "size"),
        server_ratio=("server_ratio", "first"),
        synthetic_ratio=("synthetic_ratio", "first"),
        synthetic_method=("synthetic_method", "first"),
        acc_mean=("ACC", "mean"),
        acc_min=("ACC", "min"),
        aeod_mean=("AEOD", "mean"),
        aspd_mean=("ASPD", "mean"),
        fair_mean=("fair_avg", "mean"),
        score_mean=("score", "mean"),
    ).reset_index()
    syn_sum["score_rank_in_dataset"] = syn_sum.groupby("dataset")["score_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    syn_sum = syn_sum.sort_values(["dataset", "score_rank_in_dataset", "tag"])
    syn_sum.to_csv(OUT / "server_generation_by_tag_dataset_summary.csv", index=False)

    syn_all = syn.groupby("tag").agg(
        n=("tag", "size"),
        server_ratio=("server_ratio", "first"),
        synthetic_ratio=("synthetic_ratio", "first"),
        synthetic_method=("synthetic_method", "first"),
        acc_mean=("ACC", "mean"),
        acc_min=("ACC", "min"),
        aeod_mean=("AEOD", "mean"),
        aspd_mean=("ASPD", "mean"),
        fair_mean=("fair_avg", "mean"),
        score_mean=("score", "mean"),
    ).reset_index()
    syn_all["score_rank"] = syn_all["score_mean"].rank(ascending=False, method="min").astype(int)
    syn_all = syn_all.sort_values(["score_rank", "tag"])
    syn_all.to_csv(OUT / "server_generation_overall_summary.csv", index=False)

    lines = [
        "# GuardFed-AD2+ Advisor Experiments Summary",
        "",
        "All values are generated from results/paper_tables/raw_results.jsonl. Metrics use the configured last-10-round selection rule: ACC=max, AEOD=min, ASPD=min.",
        "",
        "## Adult AD2+ Ablation: tag-level summary",
        "",
    ]
    cols = ["score_rank", "tag", "acc_mean", "aeod_mean", "aspd_mean", "fair_mean", "score_mean", "acc_rank", "fair_rank"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in abl_sum.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            vals.append(str(int(value)) if col.endswith("rank") else (fmt(value) if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend(["", "## Server/root distribution trend", ""])
    cols = ["dataset", "server_alpha", "acc_mean", "aeod_mean", "aspd_mean", "fair_mean", "score_mean", "delta_score_vs_5000"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in trend2.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            vals.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend(["", "## Synthetic server data: overall ranking", ""])
    cols = ["score_rank", "tag", "server_ratio", "synthetic_ratio", "synthetic_method", "acc_mean", "aeod_mean", "aspd_mean", "fair_mean", "score_mean"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in syn_all.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            vals.append(str(int(value)) if col == "score_rank" else (fmt(value) if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend(["", "## Synthetic server data: by dataset", ""])
    cols = ["dataset", "score_rank_in_dataset", "tag", "acc_mean", "aeod_mean", "aspd_mean", "fair_mean", "score_mean"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in syn_sum.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            vals.append(str(int(value)) if col == "score_rank_in_dataset" else (fmt(value) if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(vals) + " |")

    (OUT / "advisor_experiments_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summaries to {OUT}")
    print(f"adult rows={len(abl)} server_dist rows={len(dist)} synthetic rows={len(syn)}")


if __name__ == "__main__":
    main()
