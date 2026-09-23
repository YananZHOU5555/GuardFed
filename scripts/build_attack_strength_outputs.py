#!/usr/bin/env python3
"""Create measured ratio plots and a compact evidence report from study CSVs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "attack_strength"
PLOTS = RESULTS / "plots"


def read_csv(name: str):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    value = row.get(key)
    return float(value) if value not in (None, "", "nan") else math.nan


def ratio_rows(summary, dataset, attack, metric, distribution):
    return sorted(
        [
            row for row in summary
            if row["study"] == "ratio"
            and row["dataset"] == dataset
            and row["attack"] == attack
            and row["metric"] == metric
            and row["distribution"] == distribution
        ],
        key=lambda row: f(row, "malicious_ratio"),
    )


def make_plots(summary):
    PLOTS.mkdir(parents=True, exist_ok=True)
    for dataset in ("adult", "compas"):
        for attack in ("S-DFA", "Sp-DFA"):
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
            for axis, metric in zip(axes, ("accuracy", "aeod", "aspd")):
                for distribution, color in (("IID", "#1f77b4"), ("non-IID", "#d95f02")):
                    rows = ratio_rows(summary, dataset, attack, metric, distribution)
                    x = [100.0 * f(row, "malicious_ratio") for row in rows]
                    y = [100.0 * f(row, "mean") if metric == "accuracy" else f(row, "mean") for row in rows]
                    axis.plot(x, y, marker="o", linewidth=2, label=distribution, color=color)
                axis.set_title("ACC (%)" if metric == "accuracy" else metric.upper())
                axis.set_xlabel("Malicious clients (%)")
                axis.grid(True, alpha=0.25)
                if metric == "accuracy":
                    axis.set_ylim(bottom=0)
                else:
                    axis.set_ylim(bottom=0)
            axes[0].set_ylabel("Value")
            axes[0].legend(frameon=False)
            fig.suptitle(f"GuardFed-AD2+: {dataset.upper()} {attack} ratio sensitivity")
            fig.savefig(PLOTS / f"{dataset}_{attack.lower().replace('-', '')}_ratio.png", dpi=220)
            plt.close(fig)


def make_report(summary, config):
    ratio = [row for row in summary if row["study"] == "ratio"]
    lines = [
        "# GuardFed-AD2+ 强化攻击与恶意比例实验",
        "",
        "本报告只汇总 results/attack_strength 中实际完成并写入 JSONL 的结果；不填充缺失单元。",
        "",
        f"- F Flip calibration mode: `{config.get('fflip_mode')}`",
        f"- FedSA calibration: gain=`{config.get('fedsa_gain')}`, norm_ratio=`{config.get('fedsa_norm_ratio')}`",
        "- Ratio protocol: malicious clients = 2, 4, 6, 8, 10 out of 20; 10 seeds; 10% clean server data.",
        "",
        "## Ratio observations",
        "",
    ]
    for dataset in ("adult", "compas"):
        for attack in ("S-DFA", "Sp-DFA"):
            lines.append(f"### {dataset.upper()} / {attack}")
            for metric in ("accuracy", "aeod", "aspd"):
                values = {}
                for distribution in ("IID", "non-IID"):
                    rows = ratio_rows(ratio, dataset, attack, metric, distribution)
                    values[distribution] = [f(row, "mean") for row in rows]
                lines.append(f"- `{metric.upper()}` IID: " + ", ".join(f"{v:.6f}" for v in values["IID"]))
                lines.append(f"- `{metric.upper()}` non-IID: " + ", ".join(f"{v:.6f}" for v in values["non-IID"]))
            lines.append("")
    (RESULTS / "attack_strength_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    summary = read_csv("summary.csv")
    config = json.loads((RESULTS / "attack_config.json").read_text(encoding="utf-8"))
    make_plots(summary)
    make_report(summary, config)


if __name__ == "__main__":
    main()
