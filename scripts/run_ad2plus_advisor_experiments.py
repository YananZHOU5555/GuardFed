#!/usr/bin/env python3
"""Run and summarize advisor-requested GuardFed-AD2+ experiments.

Suites:
  ablation: Adult AD2+ score component ablations.
  server-dist: AD2+ sensitivity to 10% clean server/root data distribution.
  synthetic: AD2+ sensitivity to generated root data composition.
  fedsa-smoke: one short FedSA sanity check.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results" / "paper_tables" / "raw_results.jsonl"
OUT_DIR = ROOT / "results" / "paper_tables" / "advisor_experiments"
RUNNER = ROOT / "scripts" / "reproduce_paper_tables.py"
PYTHON = Path(sys.executable)

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

NEW_ATTACKS = ["Benign", "F Flip", "FedSA", "S-DFA", "Sp-DFA"]
SERVER_ALPHAS = [0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 50.0, 5000.0]
SYNTH_METHODS = ["gaussian_copula", "ctgan", "tvae", "smote", "pca_gaussian"]


def run_command(args: Sequence[str], dry_run: bool) -> None:
    cmd = [str(PYTHON), str(RUNNER)] + list(args)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def base_args(suite: str, tag: str, rounds: int, extra: Sequence[str]) -> List[str]:
    return [
        "--full",
        "--rounds", str(rounds),
        "--experiment-suite", suite,
        "--experiment-tag", tag,
        *AD2_BASE,
        *extra,
    ]


def run_ablation(rounds: int, dry_run: bool) -> None:
    component_ablation = [
        ("full_score", {}),
        ("no_utility_U", {"--ad2-utility-weight": "0"}),
        ("no_centrality_C", {"--ad2-centrality-weight": "0"}),
        ("no_alignment_A", {"--ad2-alignment-weight": "0"}),
        ("no_fairness_risk_F", {"--act-risk-weight": "0"}),
        ("no_violation_V", {"--act-violation-weight": "0"}),
        ("reward_only_R", {"--act-risk-weight": "0", "--act-violation-weight": "0"}),
        ("penalty_only_P", {"--ad2-utility-weight": "0", "--ad2-centrality-weight": "0", "--ad2-alignment-weight": "0"}),
        ("utility_only_U", {"--ad2-centrality-weight": "0", "--ad2-alignment-weight": "0", "--act-risk-weight": "0", "--act-violation-weight": "0"}),
        ("geo_only_CA", {"--ad2-utility-weight": "0", "--act-risk-weight": "0", "--act-violation-weight": "0"}),
        ("fair_only_FV", {"--ad2-utility-weight": "0", "--ad2-centrality-weight": "0", "--ad2-alignment-weight": "0"}),
    ]
    macro_ablation = []
    for reward_scale, penalty_scale in [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]:
        macro_ablation.append((
            f"macro_R{reward_scale:.2f}_P{penalty_scale:.2f}",
            {
                "--ad2-utility-weight": f"{3.0 * reward_scale:.6g}",
                "--ad2-centrality-weight": f"{0.2 * reward_scale:.6g}",
                "--ad2-alignment-weight": f"{1.5 * reward_scale:.6g}",
                "--act-risk-weight": f"{0.10 * penalty_scale:.6g}",
                "--act-violation-weight": f"{0.02 * penalty_scale:.6g}",
            },
        ))
    for tag, overrides in component_ablation + macro_ablation:
        extra = ["--datasets", "adult", "--distributions", "IID", "non-IID", "--attacks", *NEW_ATTACKS]
        for key, value in overrides.items():
            extra.extend([key, value])
        run_command(base_args("adult_ad2plus_ablation", tag, rounds, extra), dry_run=dry_run)


def run_server_distribution(rounds: int, dry_run: bool) -> None:
    for alpha in SERVER_ALPHAS:
        tag = f"server_alpha_{alpha:g}"
        extra = [
            "--datasets", "adult", "compas",
            "--distributions", "IID", "non-IID",
            "--attacks", *NEW_ATTACKS,
            "--server-sampling", "dirichlet_strata",
            "--server-alpha", f"{alpha:g}",
        ]
        run_command(base_args("server_distribution_ablation", tag, rounds, extra), dry_run=dry_run)


def run_synthetic(rounds: int, dry_run: bool) -> None:
    real_only = ["--datasets", "adult", "compas", "--distributions", "IID", "non-IID", "--attacks", *NEW_ATTACKS]
    run_command(base_args("server_generation_ablation", "real10_none", rounds, real_only), dry_run=dry_run)
    for real_ratio, synth_ratio in [(0.01, 0.09), (0.05, 0.05)]:
        for method in SYNTH_METHODS:
            tag = f"real{int(real_ratio * 100)}_{method}_synth{int(synth_ratio * 100)}"
            extra = [
                "--datasets", "adult", "compas",
                "--distributions", "IID", "non-IID",
                "--attacks", *NEW_ATTACKS,
                "--server-ratio", f"{real_ratio:g}",
                "--synthetic-ratio", f"{synth_ratio:g}",
                "--synthetic-method", method,
                "--synthetic-epochs", "50",
            ]
            run_command(base_args("server_generation_ablation", tag, rounds, extra), dry_run=dry_run)


def run_fedsa_smoke(rounds: int, dry_run: bool) -> None:
    extra = ["--datasets", "adult", "compas", "--distributions", "IID", "--attacks", "FedSA"]
    run_command(base_args("fedsa_smoke", "fedsa_single_attack", rounds, extra), dry_run=dry_run)


def selected_metric(record: Dict[str, Any], metric: str) -> float:
    key = "accuracy" if metric == "ACC" else metric.lower()
    values = []
    for item in record.get("last10_metrics", []):
        value = item.get("metrics", {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values and isinstance(record.get("metrics", {}).get(key), (int, float)):
        values.append(float(record["metrics"][key]))
    if not values:
        return float("nan")
    return max(values) if metric == "ACC" else min(values)


def load_records() -> List[Dict[str, Any]]:
    if not RAW_PATH.exists():
        return []
    records = []
    with RAW_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize_suite(records: Iterable[Dict[str, Any]], suite: str) -> List[Dict[str, Any]]:
    rows = []
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
            "seed": r.get("seed"),
            "rounds": r.get("rounds"),
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
    rows.sort(key=lambda x: (str(x["tag"]), str(x["dataset"]), str(x["distribution"]), str(x["attack"])))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, Any]], title: str) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No matching records found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    cols = ["tag", "dataset", "distribution", "attack", "ACC_pct", "AEOD", "ASPD", "fair_avg", "score"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.4f}" if col != "ACC_pct" else f"{value:.2f}"
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    suites = [
        ("adult_ad2plus_ablation", "Adult AD2+ Score Ablation"),
        ("server_distribution_ablation", "AD2+ Server Distribution Ablation"),
        ("server_generation_ablation", "AD2+ Server Generation Ablation"),
        ("fedsa_smoke", "FedSA Smoke Results"),
    ]
    for suite, title in suites:
        rows = summarize_suite(records, suite)
        write_csv(OUT_DIR / f"{suite}.csv", rows)
        write_markdown(OUT_DIR / f"{suite}.md", rows, title)
        print(f"{suite}: {len(rows)} rows -> {OUT_DIR / (suite + '.csv')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["ablation", "server-dist", "synthetic", "fedsa-smoke", "all"], default="fedsa-smoke")
    parser.add_argument("--rounds", type=int, default=70)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.summarize_only:
        if args.suite in {"ablation", "all"}:
            run_ablation(args.rounds, args.dry_run)
        if args.suite in {"server-dist", "all"}:
            run_server_distribution(args.rounds, args.dry_run)
        if args.suite in {"synthetic", "all"}:
            run_synthetic(args.rounds, args.dry_run)
        if args.suite in {"fedsa-smoke", "all"}:
            run_fedsa_smoke(args.rounds, args.dry_run)
    if not args.dry_run:
        summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
