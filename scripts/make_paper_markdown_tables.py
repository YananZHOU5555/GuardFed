#!/usr/bin/env python3
"""Create Markdown paper-style tables with best/second formatting."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper_tables"
METHODS = [
    "FedAvg", "FairFed", "Median", "FLTrust", "FairGuard", "FLTrust+FairGuard",
    "FLGMM", "FLAURA", "LayerGuard", "SmartFL", "FLTG", "FedDNA", "LASA",
    "GuardFed", "GuardFed-ACT", "GuardFed-AD2",
]
DISTRIBUTIONS = ["IID", "non-IID"]
ATTACKS = ["Benign", "F Flip", "FOE", "S-DFA", "Sp-DFA"]
METRICS = ["ACC_pct", "AEOD", "ASPD", "FairAvg", "Score"]
HIGHER_IS_BETTER = {"ACC_pct": True, "Score": True, "AEOD": False, "ASPD": False, "FairAvg": False}
METRIC_LABEL = {"ACC_pct": "ACC (%)", "AEOD": "AEOD", "ASPD": "ASPD", "FairAvg": "FairAvg", "Score": "Score"}
DATASET_LABEL = {"adult": "Adult", "compas": "COMPAS"}


def load_rows(dataset: str, policy: str) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    path = RESULTS / f"table_{dataset}_new_baselines_{policy}.csv"
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["distribution"], row["attack"], row["method"])
            out[key] = {metric: float(row[metric]) for metric in METRICS}
    return out


def top_two(rows: Dict[Tuple[str, str, str], Dict[str, float]], metric: str, dist: str, attack: str) -> Tuple[str | None, str | None]:
    vals = []
    for method in METHODS:
        rec = rows.get((dist, attack, method))
        if rec is not None:
            vals.append((method, rec[metric]))
    vals.sort(key=lambda item: item[1], reverse=HIGHER_IS_BETTER[metric])
    if not vals:
        return None, None
    return vals[0][0], vals[1][0] if len(vals) > 1 else None


def fmt(value: float, metric: str, role: str | None) -> str:
    txt = f"{value:.2f}" if metric == "ACC_pct" else f"{value:.4f}"
    if role == "best":
        return f"**{txt}**"
    if role == "second":
        return f"<u>{txt}</u>"
    return txt


def metric_table(rows: Dict[Tuple[str, str, str], Dict[str, float]], metric: str) -> str:
    headers = ["Method"] + [f"{dist} {attack}" for dist in DISTRIBUTIONS for attack in ATTACKS]
    best_second = {(dist, attack): top_two(rows, metric, dist, attack) for dist in DISTRIBUTIONS for attack in ATTACKS}
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for method in METHODS:
        cells = [method]
        for dist in DISTRIBUTIONS:
            for attack in ATTACKS:
                rec = rows.get((dist, attack, method))
                if rec is None:
                    cells.append("-")
                    continue
                best, second = best_second[(dist, attack)]
                role = "best" if method == best else "second" if method == second else None
                cells.append(fmt(rec[metric], metric, role))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dataset_markdown(dataset: str, policy: str, metrics: Iterable[str]) -> str:
    rows = load_rows(dataset, policy)
    label = DATASET_LABEL.get(dataset, dataset)
    selection = "best score over final/last-10 rounds" if policy == "best_score" else "strict final round"
    parts = [f"# {label} Results ({selection})", "", "Best values are **bold**; second-best values are <u>underlined</u>. ACC/Score are higher-is-better; AEOD/ASPD/FairAvg are lower-is-better.", ""]
    for metric in metrics:
        parts.append(f"## {METRIC_LABEL[metric]}")
        parts.append(metric_table(rows, metric))
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="*", default=["best_score", "latest_final"])
    parser.add_argument("--datasets", nargs="*", default=["adult", "compas"])
    parser.add_argument("--metrics", nargs="*", default=METRICS)
    args = parser.parse_args()
    for policy in args.policies:
        outdir = RESULTS / f"markdown_{policy}"
        outdir.mkdir(parents=True, exist_ok=True)
        for dataset in args.datasets:
            path = outdir / f"table_{dataset}_{policy}_all_metrics.md"
            path.write_text(dataset_markdown(dataset, policy, args.metrics), encoding="utf-8")
            print(path)
            rows = load_rows(dataset, policy)
            for metric in args.metrics:
                mpath = outdir / f"table_{dataset}_{policy}_{metric.lower()}.md"
                mpath.write_text(f"# {DATASET_LABEL.get(dataset, dataset)} {METRIC_LABEL[metric]}\n\n" + metric_table(rows, metric) + "\n", encoding="utf-8")
                print(mpath)


if __name__ == "__main__":
    main()
