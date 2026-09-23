#!/usr/bin/env python3
"""Run expanded GuardFed advisor experiments.

Suites:
  server-dist30: 30 Dirichlet alpha values x 3 seeds for GuardFed-AD2+.
  synthetic-ratios: larger real/synthetic root-data ratio grid x 3 seeds.
  fedsa-table: FedSA-only table for all retained methods x 3 seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results" / "paper_tables" / "raw_results.jsonl"
OUT_DIR = ROOT / "results" / "paper_tables" / "expanded_experiments"
RUNNER = ROOT / "scripts" / "reproduce_paper_tables.py"
PYTHON = Path(sys.executable)

SEEDS = [123, 456, 789]
ALPHAS30 = [0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75,
            1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0,
            50.0, 75.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0]

AD2_BASE = [
    "--server-ratio", "0.10",
    "--synthetic-ratio", "0",
    "--synthetic-method", "none",
    "--device", "cuda",
    "--methods", "GuardFed-AD2+",
    "--ad2-plus-mode", "fixed",
    "--act-fairness-budget", "0.12",
    "--act-fairness-metric", "aeod_aspd",
    "--act-risk-weight", "0.10",
    "--act-violation-weight", "0.02",
    "--act-keep-ratio", "1.0",
    "--act-temperature", "0.80",
    "--ad2-utility-weight", "3.0",
    "--ad2-centrality-weight", "0.2",
    "--ad2-alignment-weight", "1.5",
    "--ad2-score-clip", "5",
    "--ad2-norm-mode", "root",
    "--ad2-calibration-objective", "original",
    "--sdfa-foe-mode", "fedsa",
    "--spdfa-foe-mode", "fedsa",
]

ALL_METHODS = [
    "FedAvg", "FairFed", "Median", "FLTrust", "FairGuard", "FLTrust+FairGuard", "GuardFed",
    "FLGMM", "FLAURA", "LayerGuard", "SmartFL", "FLTG", "FedDNA", "LASA",
    "Fed-NGA", "Huber-BRFL", "LoGoFair", "AdaAggRL", "FedAMM", "FedAA",
    "GuardFed-AD2", "GuardFed-AD2+",
]

SYNTH_METHODS = ["gaussian_copula", "smote", "ctgan", "tvae"]
REAL_ONLY_RATIOS = [0.01, 0.03, 0.05, 0.07, 0.10]
SYNTH_RATIO_GRID = [(0.01, 0.09), (0.02, 0.08), (0.03, 0.07), (0.05, 0.05), (0.07, 0.03), (0.09, 0.01)]


def run_command(args: Sequence[str], dry_run: bool) -> None:
    cmd = [str(PYTHON), str(RUNNER), *args]
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def base_args(suite: str, tag: str, seed: int, rounds: int, extra: Sequence[str]) -> List[str]:
    return [
        "--full",
        "--rounds", str(rounds),
        "--seed", str(seed),
        "--experiment-suite", suite,
        "--experiment-tag", tag,
        *AD2_BASE,
        *extra,
    ]


def run_server_dist30(rounds: int, dry_run: bool) -> None:
    for seed in SEEDS:
        for alpha in ALPHAS30:
            tag = f"alpha_{alpha:g}_seed{seed}"
            extra = [
                "--datasets", "adult", "compas",
                "--distributions", "IID", "non-IID",
                "--attacks", "Benign", "FedSA",
                "--server-sampling", "dirichlet_label_preserved",
                "--server-alpha", f"{alpha:g}",
            ]
            run_command(base_args("expanded_server_dist30_labelpreserve", tag, seed, rounds, extra), dry_run=dry_run)


def run_synthetic_ratios(rounds: int, dry_run: bool) -> None:
    common = ["--datasets", "adult", "compas", "--distributions", "IID", "non-IID", "--attacks", "Benign", "FedSA"]
    for seed in SEEDS:
        for real_ratio in REAL_ONLY_RATIOS:
            tag = f"real{int(round(real_ratio * 100))}_none_seed{seed}"
            extra = [*common, "--server-ratio", f"{real_ratio:g}", "--synthetic-ratio", "0", "--synthetic-method", "none"]
            run_command(base_args("expanded_synthetic_ratios", tag, seed, rounds, extra), dry_run=dry_run)
        for real_ratio, synth_ratio in SYNTH_RATIO_GRID:
            for method in SYNTH_METHODS:
                tag = f"real{int(round(real_ratio * 100))}_{method}_synth{int(round(synth_ratio * 100))}_seed{seed}"
                extra = [
                    *common,
                    "--server-ratio", f"{real_ratio:g}",
                    "--synthetic-ratio", f"{synth_ratio:g}",
                    "--synthetic-method", method,
                    "--synthetic-epochs", "50",
                ]
                run_command(base_args("expanded_synthetic_ratios", tag, seed, rounds, extra), dry_run=dry_run)


def run_fedsa_table(rounds: int, dry_run: bool) -> None:
    for seed in SEEDS:
        tag = f"fedsa_all_methods_seed{seed}"
        extra = [
            "--datasets", "adult", "compas",
            "--distributions", "IID", "non-IID",
            "--attacks", "FedSA",
            "--methods", *ALL_METHODS,
            "--server-ratio", "0.10",
            "--synthetic-ratio", "0",
            "--synthetic-method", "none",
        ]
        run_command(base_args("fedsa_all_methods", tag, seed, rounds, extra), dry_run=dry_run)


def selected_metric(record: Dict[str, Any], metric: str) -> float:
    key = "accuracy" if metric == "ACC" else metric.lower()
    vals: List[float] = []
    for item in record.get("last10_metrics", []):
        value = item.get("metrics", {}).get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            vals.append(float(value))
    if not vals:
        value = record.get("metrics", {}).get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            vals.append(float(value))
    if not vals:
        return float("nan")
    return max(vals) if metric == "ACC" else min(vals)


def load_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not RAW_PATH.exists():
        return records
    with RAW_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def rows_for_suite(records: Iterable[Dict[str, Any]], suite: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in records:
        cfg = r.get("config", {})
        if cfg.get("experiment_suite") != suite:
            continue
        acc = selected_metric(r, "ACC")
        aeod = selected_metric(r, "AEOD")
        aspd = selected_metric(r, "ASPD")
        rows.append({
            "suite": suite,
            "tag": cfg.get("experiment_tag", ""),
            "dataset": r.get("dataset"),
            "distribution": r.get("distribution"),
            "attack": r.get("attack"),
            "method": r.get("method"),
            "seed": int(r.get("seed")),
            "rounds": int(r.get("rounds")),
            "server_sampling": cfg.get("server_sampling"),
            "server_alpha": cfg.get("server_alpha"),
            "server_ratio": cfg.get("server_ratio"),
            "synthetic_ratio": cfg.get("synthetic_ratio"),
            "synthetic_method": cfg.get("synthetic_method"),
            "ACC": acc,
            "ACC_pct": acc * 100.0,
            "AEOD": aeod,
            "ASPD": aspd,
            "fair_avg": 0.5 * (aeod + aspd),
            "score": acc - 0.5 * (aeod + aspd),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset", "distribution", "attack", "method", "seed", "server_alpha", "tag"])


def add_rank(group: pd.DataFrame, metric_col: str, rank_col: str, ascending: bool) -> pd.DataFrame:
    group[rank_col] = group[metric_col].rank(ascending=ascending, method="min").astype(int)
    return group


def summarize() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()

    dist = rows_for_suite(records, "expanded_server_dist30_labelpreserve")
    syn = rows_for_suite(records, "expanded_synthetic_ratios")
    fedsa = rows_for_suite(records, "fedsa_all_methods")

    if not dist.empty:
        dist.to_csv(OUT_DIR / "expanded_server_dist30_raw.csv", index=False)
        dist_curve = dist.groupby(["dataset", "distribution", "attack", "server_alpha"], dropna=False).agg(
            n=("ACC", "size"),
            acc_mean=("ACC", "mean"), acc_std=("ACC", "std"),
            aeod_mean=("AEOD", "mean"), aeod_std=("AEOD", "std"),
            aspd_mean=("ASPD", "mean"), aspd_std=("ASPD", "std"),
            fair_mean=("fair_avg", "mean"), fair_std=("fair_avg", "std"),
            score_mean=("score", "mean"), score_std=("score", "std"),
        ).reset_index().sort_values(["dataset", "distribution", "attack", "server_alpha"])
        dist_curve.to_csv(OUT_DIR / "expanded_server_dist30_curve.csv", index=False)

    if not syn.empty:
        syn.to_csv(OUT_DIR / "expanded_synthetic_ratios_raw.csv", index=False)
        syn_sum = syn.groupby(["dataset", "distribution", "attack", "server_ratio", "synthetic_ratio", "synthetic_method"], dropna=False).agg(
            n=("ACC", "size"),
            acc_mean=("ACC", "mean"), acc_std=("ACC", "std"),
            aeod_mean=("AEOD", "mean"), aeod_std=("AEOD", "std"),
            aspd_mean=("ASPD", "mean"), aspd_std=("ASPD", "std"),
            fair_mean=("fair_avg", "mean"), fair_std=("fair_avg", "std"),
            score_mean=("score", "mean"), score_std=("score", "std"),
        ).reset_index()
        syn_sum["setting"] = syn_sum.apply(lambda r: f"real{int(round(float(r.server_ratio)*100))}_" + ("none" if float(r.synthetic_ratio) == 0 else f"{r.synthetic_method}_synth{int(round(float(r.synthetic_ratio)*100))}"), axis=1)
        syn_sum["score_rank_in_slice"] = syn_sum.groupby(["dataset", "distribution", "attack"])["score_mean"].rank(ascending=False, method="min").astype(int)
        syn_sum = syn_sum.sort_values(["dataset", "distribution", "attack", "score_rank_in_slice", "setting"])
        syn_sum.to_csv(OUT_DIR / "expanded_synthetic_ratios_summary.csv", index=False)
        syn_dataset = syn.groupby(["dataset", "server_ratio", "synthetic_ratio", "synthetic_method"], dropna=False).agg(
            n=("ACC", "size"),
            acc_mean=("ACC", "mean"), acc_std=("ACC", "std"),
            aeod_mean=("AEOD", "mean"), aeod_std=("AEOD", "std"),
            aspd_mean=("ASPD", "mean"), aspd_std=("ASPD", "std"),
            fair_mean=("fair_avg", "mean"), fair_std=("fair_avg", "std"),
            score_mean=("score", "mean"), score_std=("score", "std"),
        ).reset_index()
        syn_dataset["setting"] = syn_dataset.apply(lambda r: f"real{int(round(float(r.server_ratio)*100))}_" + ("none" if float(r.synthetic_ratio) == 0 else f"{r.synthetic_method}_synth{int(round(float(r.synthetic_ratio)*100))}"), axis=1)
        syn_dataset["score_rank_in_dataset"] = syn_dataset.groupby("dataset")["score_mean"].rank(ascending=False, method="min").astype(int)
        syn_dataset = syn_dataset.sort_values(["dataset", "score_rank_in_dataset", "setting"])
        syn_dataset.to_csv(OUT_DIR / "expanded_synthetic_by_dataset_summary.csv", index=False)

    if not fedsa.empty:
        fedsa.to_csv(OUT_DIR / "fedsa_all_methods_raw.csv", index=False)
        fedsa_sum = fedsa.groupby(["dataset", "distribution", "attack", "method"], dropna=False).agg(
            n=("ACC", "size"),
            acc_mean=("ACC", "mean"), acc_std=("ACC", "std"),
            aeod_mean=("AEOD", "mean"), aeod_std=("AEOD", "std"),
            aspd_mean=("ASPD", "mean"), aspd_std=("ASPD", "std"),
            fair_mean=("fair_avg", "mean"), fair_std=("fair_avg", "std"),
            score_mean=("score", "mean"), score_std=("score", "std"),
        ).reset_index()
        fedsa_sum["acc_rank"] = fedsa_sum.groupby(["dataset", "distribution"])["acc_mean"].rank(ascending=False, method="min").astype(int)
        fedsa_sum["fair_rank"] = fedsa_sum.groupby(["dataset", "distribution"])["fair_mean"].rank(ascending=True, method="min").astype(int)
        fedsa_sum["score_rank"] = fedsa_sum.groupby(["dataset", "distribution"])["score_mean"].rank(ascending=False, method="min").astype(int)
        fedsa_sum = fedsa_sum.sort_values(["dataset", "distribution", "score_rank", "method"])
        fedsa_sum.to_csv(OUT_DIR / "fedsa_all_methods_summary.csv", index=False)

    lines = [
        "# Expanded GuardFed Advisor Experiments Summary",
        "",
        "All rows are derived from results/paper_tables/raw_results.jsonl. ACC=max over last 10 rounds; AEOD/ASPD=min over last 10 rounds.",
        "",
        f"- expanded_server_dist30 rows: {len(dist)}",
        f"- expanded_synthetic_ratios rows: {len(syn)}",
        f"- fedsa_all_methods rows: {len(fedsa)}",
    ]
    if not dist.empty:
        lines += ["", "## Server Distribution Coverage", ""]
        lines.append(dist.groupby(["dataset", "distribution", "attack"])["server_alpha"].nunique().reset_index(name="num_alphas").to_markdown(index=False))
    if not syn.empty:
        lines += ["", "## Synthetic Dataset-Level Top 12", ""]
        top = pd.read_csv(OUT_DIR / "expanded_synthetic_by_dataset_summary.csv").groupby("dataset").head(12)
        lines.append(top[["dataset", "score_rank_in_dataset", "setting", "acc_mean", "aeod_mean", "aspd_mean", "score_mean", "n"]].to_markdown(index=False, floatfmt=".4f"))
    if not fedsa.empty:
        lines += ["", "## FedSA All-Methods Top 8 Per Dataset/Distribution", ""]
        topf = pd.read_csv(OUT_DIR / "fedsa_all_methods_summary.csv").groupby(["dataset", "distribution"]).head(8)
        lines.append(topf[["dataset", "distribution", "score_rank", "method", "acc_mean", "aeod_mean", "aspd_mean", "score_mean", "n"]].to_markdown(index=False, floatfmt=".4f"))
    (OUT_DIR / "expanded_experiments_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote expanded summaries to {OUT_DIR}")
    print(f"server_dist30 rows={len(dist)} synthetic rows={len(syn)} fedsa rows={len(fedsa)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["server-dist30", "synthetic-ratios", "fedsa-table", "all"], default="server-dist30")
    p.add_argument("--rounds", type=int, default=70)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    if not args.summarize_only:
        if args.suite in {"server-dist30", "all"}:
            run_server_dist30(args.rounds, args.dry_run)
        if args.suite in {"synthetic-ratios", "all"}:
            run_synthetic_ratios(args.rounds, args.dry_run)
        if args.suite in {"fedsa-table", "all"}:
            run_fedsa_table(args.rounds, args.dry_run)
    if not args.dry_run:
        summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
