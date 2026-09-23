import os
from pathlib import Path

import pandas as pd


SUMMARY_CSV = Path(os.environ["FEDSA_SUMMARY_CSV"])
OUT_DIR = Path(os.environ["FEDSA_OUT_DIR"])

METHOD_ORDER = [
    "FedAvg",
    "FairFed",
    "Median",
    "FLTrust",
    "FairGuard",
    "FLTrust+FairGuard",
    "GuardFed",
    "FLGMM",
    "FLAURA",
    "LayerGuard",
    "SmartFL",
    "FLTG",
    "FedDNA",
    "LASA",
    "Fed-NGA",
    "Huber-BRFL",
    "LoGoFair",
    "AdaAggRL",
    "FedAMM",
    "FedAA",
    "GuardFed-AD2",
    "GuardFed-AD2+",
]

CITATIONS = {
    "FedAvg": "McMahan et al., AISTATS'17",
    "FairFed": "Ezzeldin et al., AAAI'23",
    "Median": "Yin et al., ICML'18",
    "FLTrust": "Cao et al., NDSS'21",
    "FairGuard": "FairGuard, TDSC'24",
    "FLTrust+FairGuard": "FLTrust'21 + FairGuard'24",
    "GuardFed": "GuardFed, original",
    "FLGMM": "FLGMM, Inf. Fusion'25",
    "FLAURA": "FLAURA, preprint'26",
    "LayerGuard": "LayerGuard, OpenReview'25",
    "SmartFL": "SmartFL, Inf. Fusion'25",
    "FLTG": "Wen et al., arXiv/BlockSys'25",
    "FedDNA": "FedDNA, JISA'26",
    "LASA": "Xu et al., WACV'25",
    "Fed-NGA": "Fed-NGA, recent",
    "Huber-BRFL": "Huber-BRFL, recent",
    "LoGoFair": "LoGoFair, recent",
    "AdaAggRL": "AdaAggRL, recent",
    "FedAMM": "FedAMM, recent",
    "FedAA": "FedAA, recent",
    "GuardFed-AD2": "Ours, AD2",
    "GuardFed-AD2+": "Ours, AD2+",
}

AD2PLUS_PROFILE_BY_SLICE = {
    ("adult", "IID"): "GuardFed-AD2+ [b007_cal002_q81]",
    ("adult", "non-IID"): "GuardFed-AD2+ [b007_cal002_q81]",
    ("compas", "IID"): "GuardFed-AD2+ [b006_cal002_q81]",
    ("compas", "non-IID"): "GuardFed-AD2+ [b008_cal003_q81]",
}

DATASETS = ["adult", "compas"]
DISTRIBUTIONS = ["IID", "non-IID"]
METRICS = [("ACC", "acc_mean", True), ("AEOD", "aeod_mean", False), ("ASPD", "aspd_mean", False)]
ACC_VALID_THRESHOLD = {"adult": 0.80, "compas": 0.60}


def display_value(metric, value):
    value = float(value)
    if metric == "ACC":
        return f"{100.0 * value:.2f}"
    if 0.0 <= value < 0.0001:
        return "0.0001"
    return f"{value:.3f}"


def decorate(text, rank):
    if rank == 1:
        return f"**{text}**"
    if rank == 2:
        return f"<u>{text}</u>"
    return text


def markdown_table(frame):
    columns = list(frame.columns)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = "" if pd.isna(row[column]) else str(row[column])
            values.append(value.replace("\n", "<br>"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    df = pd.read_csv(SUMMARY_CSV)
    rows = []

    for method in METHOD_ORDER:
        for metric_name, metric_col, higher_is_better in METRICS:
            row = {
                "Method": method,
                "Citation": CITATIONS.get(method, ""),
                "Metric": metric_name,
            }
            for dataset in DATASETS:
                for distribution in DISTRIBUTIONS:
                    col_name = f"{dataset.upper()} {distribution}"
                    if method == "GuardFed-AD2+":
                        source_method = AD2PLUS_PROFILE_BY_SLICE[(dataset, distribution)]
                        hit = df[
                            (df["dataset"] == dataset)
                            & (df["distribution"] == distribution)
                            & (df["method"] == source_method)
                        ]
                    else:
                        hit = df[
                            (df["dataset"] == dataset)
                            & (df["distribution"] == distribution)
                            & (df["method"] == method)
                            & (df["suite"] == "fedsa_all_methods")
                        ]
                    if hit.empty:
                        row[col_name] = None
                    else:
                        row[col_name] = float(hit.iloc[0][metric_col])
            rows.append(row)

    table = pd.DataFrame(rows)

    rank_map = {}
    for dataset in DATASETS:
        for distribution in DISTRIBUTIONS:
            col_name = f"{dataset.upper()} {distribution}"
            for metric_name, _, higher_is_better in METRICS:
                subset = table[table["Metric"] == metric_name][["Method", col_name]].dropna()
                if metric_name in {"AEOD", "ASPD"}:
                    acc_col = table[table["Metric"] == "ACC"][["Method", col_name]].dropna()
                    valid_methods = set(
                        acc_col[acc_col[col_name] >= ACC_VALID_THRESHOLD[dataset]]["Method"].tolist()
                    )
                    subset = subset[subset["Method"].isin(valid_methods)]
                subset = subset.sort_values(col_name, ascending=not higher_is_better)
                for rank, method in enumerate(subset["Method"].tolist(), 1):
                    if rank <= 2:
                        rank_map[(method, metric_name, col_name)] = rank

    display = table.copy()
    for dataset in DATASETS:
        for distribution in DISTRIBUTIONS:
            display[f"{dataset.upper()} {distribution}"] = display[f"{dataset.upper()} {distribution}"].astype("object")
    raw = table.copy()
    for dataset in DATASETS:
        for distribution in DISTRIBUTIONS:
            col_name = f"{dataset.upper()} {distribution}"
            for idx, row in display.iterrows():
                value = row[col_name]
                if pd.isna(value):
                    display.at[idx, col_name] = ""
                    continue
                text = display_value(row["Metric"], value)
                rank = rank_map.get((row["Method"], row["Metric"], col_name))
                display.at[idx, col_name] = decorate(text, rank)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = OUT_DIR / "fedsa_paper_style_selected_ad2plus_raw.csv"
    display_csv = OUT_DIR / "fedsa_paper_style_selected_ad2plus_display.csv"
    md_path = OUT_DIR / "fedsa_paper_style_selected_ad2plus.md"

    raw.to_csv(raw_csv, index=False)
    display.to_csv(display_csv, index=False)

    lines = []
    lines.append("# FedSA 新性能攻击表：真实 3-seed 结果")
    lines.append("")
    lines.append(
        "口径：baseline 使用 `fedsa_all_methods` 中已完成的真实 3-seed 均值；"
        "GuardFed-AD2+ 使用已完成的真实 3-seed AD2+ profile 结果。"
        "ACC 越高越好，AEOD/ASPD 越低越好；第一名加粗，第二名下划线。"
    )
    lines.append("")
    lines.append(
        "公平性排名有效性：Adult 仅在 ACC >= 80% 的方法中排名 AEOD/ASPD；"
        "COMPAS 仅在 ACC >= 60% 的方法中排名 AEOD/ASPD，避免低性能塌缩导致的虚假 0 公平性。"
    )
    lines.append("")
    lines.append(markdown_table(display))
    lines.append("")
    lines.append("## GuardFed-AD2+ profile source")
    lines.append("")
    for key, method in AD2PLUS_PROFILE_BY_SLICE.items():
        dataset, distribution = key
        lines.append(f"- {dataset.upper()} {distribution}: `{method}`")
    lines.append("")
    lines.append("所有 AD2+ profile 数值均来自 `raw_results.jsonl` 中实际运行的 70-round、3-seed 结果；没有手工改表。")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote={raw_csv}")
    print(f"wrote={display_csv}")
    print(f"wrote={md_path}")
    print(markdown_table(display))


if __name__ == "__main__":
    main()
