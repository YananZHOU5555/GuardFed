#!/usr/bin/env python3
"""Benign add-on runs for a readable ablation table.

The earlier v3 stress ablation covered attacks only. This add-on fills the
normal/Benign condition for the same six profiles, datasets, distributions and
seeds so the final table can show normal-vs-attack behavior directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path("/home/yannan/workspace/GuardFed")
RAW_PATH = ROOT / "results" / "paper_tables" / "raw_results.jsonl"
OUT_DIR = ROOT / "results" / "paper_tables" / "goal_revision_v4"
RUNNER = ROOT / "scripts" / "reproduce_paper_tables.py"
PY = ROOT / ".venv" / "bin" / "python"
SUITE = "goal_ablation_v4_benign"

SEEDS = [123, 456, 789]

COMMON = [
    "--server-ratio", "0.10",
    "--synthetic-ratio", "0",
    "--synthetic-method", "none",
    "--device", "cuda",
    "--methods", "GuardFed-AD2+",
    "--ad2-plus-mode", "fixed",
    "--server-sampling", "stratified_sensitive",
    "--act-fairness-budget", "0.06",
    "--act-temperature", "0.25",
    "--act-keep-ratio", "0.75",
    "--act-fairness-metric", "aeod_aspd",
    "--ad2-score-clip", "5",
    "--ad2-norm-mode", "root",
    "--sdfa-foe-mode", "fedsa",
    "--spdfa-foe-mode", "fedsa",
    "--fedsa-gain", "5.00",
    "--fedsa-norm-ratio", "5.00",
]

PROFILES = [
    ("full", [
        "--act-risk-weight", "1.40",
        "--act-violation-weight", "0.70",
        "--ad2-utility-weight", "3.00",
        "--ad2-centrality-weight", "0.80",
        "--ad2-alignment-weight", "1.80",
    ]),
    ("no_performance_UCA", [
        "--act-risk-weight", "1.40",
        "--act-violation-weight", "0.70",
        "--ad2-utility-weight", "0.00",
        "--ad2-centrality-weight", "0.00",
        "--ad2-alignment-weight", "0.00",
    ]),
    ("no_fairness_FC", [
        "--disable-ad2-calibration",
        "--act-risk-weight", "0.00",
        "--act-violation-weight", "0.00",
        "--ad2-utility-weight", "3.00",
        "--ad2-centrality-weight", "0.80",
        "--ad2-alignment-weight", "1.80",
    ]),
    ("no_geometry_CA", [
        "--act-risk-weight", "1.40",
        "--act-violation-weight", "0.70",
        "--ad2-utility-weight", "3.00",
        "--ad2-centrality-weight", "0.00",
        "--ad2-alignment-weight", "0.00",
    ]),
    ("utility_only_U", [
        "--disable-ad2-calibration",
        "--act-risk-weight", "0.00",
        "--act-violation-weight", "0.00",
        "--ad2-utility-weight", "3.00",
        "--ad2-centrality-weight", "0.00",
        "--ad2-alignment-weight", "0.00",
    ]),
    ("fairness_only_FC", [
        "--act-risk-weight", "1.40",
        "--act-violation-weight", "0.70",
        "--ad2-utility-weight", "0.00",
        "--ad2-centrality-weight", "0.00",
        "--ad2-alignment-weight", "0.00",
    ]),
]


def run_cmd(args: Sequence[str], dry_run: bool = False) -> None:
    cmd = [str(PY), str(RUNNER), *args]
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_benign(rounds: int, seeds: Sequence[int], dry_run: bool) -> None:
    for seed in seeds:
        for profile, profile_args in PROFILES:
            tag = f"{profile}_Benign_seed{seed}"
            run_cmd([
                "--full",
                "--rounds", str(rounds),
                "--seed", str(seed),
                "--experiment-suite", SUITE,
                "--experiment-tag", tag,
                *COMMON,
                "--datasets", "adult", "compas",
                "--distributions", "IID", "non-IID",
                "--attacks", "Benign",
                *profile_args,
            ], dry_run=dry_run)


def load_records() -> List[Dict[str, Any]]:
    if not RAW_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with RAW_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def rows_for_suite(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        cfg = r.get("config", {})
        if cfg.get("experiment_suite") != SUITE:
            continue
        tag = str(cfg.get("experiment_tag", ""))
        profile = tag.split("_Benign_seed")[0]
        acc = selected_metric(r, "ACC")
        aeod = selected_metric(r, "AEOD")
        aspd = selected_metric(r, "ASPD")
        out.append({
            "suite": SUITE,
            "tag": tag,
            "profile": profile,
            "dataset": r.get("dataset"),
            "distribution": r.get("distribution"),
            "attack": r.get("attack"),
            "seed": r.get("seed"),
            "rounds": r.get("rounds"),
            "ad2_calibration_enabled": cfg.get("ad2_calibration_enabled"),
            "server_sampling": cfg.get("server_sampling"),
            "ACC": acc,
            "ACC_pct": acc * 100 if math.isfinite(acc) else float("nan"),
            "AEOD": aeod,
            "ASPD": aspd,
            "fair_avg": (aeod + aspd) / 2 if math.isfinite(aeod) and math.isfinite(aspd) else float("nan"),
            "score": acc - (aeod + aspd) / 2 if math.isfinite(acc) and math.isfinite(aeod) and math.isfinite(aspd) else float("nan"),
        })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--seeds", default="123,456,789")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.run:
        run_benign(args.rounds, seeds, args.dry_run)
    if args.summarize:
        rows = rows_for_suite(load_records())
        write_csv(OUT_DIR / "goal_ablation_v4_benign_raw.csv", rows)
        print(f"rows={len(rows)} wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
