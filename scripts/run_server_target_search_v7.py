#!/usr/bin/env python3
"""Target-group server/root skew search for the server distribution ablation.

The v5 dense run only tested two Adult target groups. This script tests all four
Adult (sensitive, label) target groups so the final 02 workbook can be based on
the direction that is actually supported by raw runs.
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


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results" / "paper_tables" / "raw_results.jsonl"
OUT_DIR = ROOT / "results" / "paper_tables" / "goal_revision_v3"
RUNNER = ROOT / "scripts" / "reproduce_paper_tables.py"
PY = Path(sys.executable)

SUITE = "goal_server_target_search_v7"

TARGETS = [(0, 0), (0, 1), (1, 0), (1, 1)]
SKEWS = [0.00, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50]

COMMON = [
    "--server-ratio", "0.10",
    "--synthetic-ratio", "0",
    "--synthetic-method", "none",
    "--device", "cuda",
    "--methods", "GuardFed-AD2+",
    "--ad2-plus-mode", "fixed",
    "--act-fairness-budget", "0.06",
    "--act-temperature", "0.25",
    "--act-keep-ratio", "0.75",
    "--act-fairness-metric", "aeod_aspd",
    "--act-risk-weight", "1.40",
    "--act-violation-weight", "0.70",
    "--ad2-utility-weight", "3.00",
    "--ad2-centrality-weight", "0.80",
    "--ad2-alignment-weight", "1.80",
    "--ad2-score-clip", "5",
    "--ad2-norm-mode", "root",
    "--sdfa-foe-mode", "fedsa",
    "--spdfa-foe-mode", "fedsa",
    "--fedsa-gain", "5.00",
    "--fedsa-norm-ratio", "5.00",
]


def run_cmd(args: Sequence[str], dry_run: bool = False) -> None:
    cmd = [str(PY), str(RUNNER), *args]
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_search(rounds: int, seeds: Sequence[int], dry_run: bool) -> None:
    for seed in seeds:
        for target_sensitive, target_label in TARGETS:
            for skew in SKEWS:
                tag = f"adult_target{target_sensitive}{target_label}_skew{skew:.2f}_seed{seed}"
                run_cmd(
                    [
                        "--full",
                        "--rounds", str(rounds),
                        "--seed", str(seed),
                        "--experiment-suite", SUITE,
                        "--experiment-tag", tag,
                        *COMMON,
                        "--datasets", "adult",
                        "--distributions", "IID", "non-IID",
                        "--attacks", "Benign", "FedSA",
                        "--server-sampling", "controlled_target_group_skew",
                        "--server-alpha", f"{skew:.2f}",
                        "--server-target-sensitive", str(target_sensitive),
                        "--server-target-label", str(target_label),
                    ],
                    dry_run=dry_run,
                )


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


def suite_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in records:
        cfg = r.get("config", {})
        if cfg.get("experiment_suite") != SUITE:
            continue
        audit = r.get("data_contract", {}).get("server_sampling_audit", {}) or {}
        rows.append({
            "suite": SUITE,
            "tag": cfg.get("experiment_tag"),
            "dataset": r.get("dataset"),
            "distribution": r.get("distribution"),
            "attack": r.get("attack"),
            "seed": r.get("seed"),
            "server_target_sensitive": cfg.get("server_target_sensitive", audit.get("server_target_sensitive")),
            "server_target_label": cfg.get("server_target_label", audit.get("server_target_label")),
            "server_alpha": cfg.get("server_alpha"),
            "group_tvd": audit.get("group_tvd"),
            "sensitive_tvd": audit.get("sensitive_tvd"),
            "label_tvd": audit.get("label_tvd"),
            "acc": selected_metric(r, "ACC"),
            "aeod": selected_metric(r, "AEOD"),
            "aspd": selected_metric(r, "ASPD"),
        })
    return rows


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if isinstance(x, (int, float)) and isinstance(y, (int, float))
        and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return float("nan")
    mx = mean([p[0] for p in pairs])
    my = mean([p[1] for p in pairs])
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((p[0] - mx) * (p[1] - my) for p in pairs) / (sx * sy)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((
            row["dataset"],
            row["distribution"],
            row["attack"],
            row["server_target_sensitive"],
            row["server_target_label"],
        ), []).append(row)
    out: List[Dict[str, Any]] = []
    for key, vals in sorted(grouped.items()):
        xs = [v["group_tvd"] for v in vals]
        out.append({
            "dataset": key[0],
            "distribution": key[1],
            "attack": key[2],
            "server_target_sensitive": key[3],
            "server_target_label": key[4],
            "n": len(vals),
            "corr_tvd_acc": corr(xs, [v["acc"] for v in vals]),
            "corr_tvd_aeod": corr(xs, [v["aeod"] for v in vals]),
            "corr_tvd_aspd": corr(xs, [v["aspd"] for v in vals]),
            "mean_acc": mean([v["acc"] for v in vals]),
            "mean_aeod": mean([v["aeod"] for v in vals]),
            "mean_aspd": mean([v["aspd"] for v in vals]),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seeds", default="123")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.run:
        run_search(args.rounds, seeds, args.dry_run)
    if args.summarize:
        rows = suite_rows(load_records())
        write_csv(OUT_DIR / f"{SUITE}_raw.csv", rows)
        write_csv(OUT_DIR / f"{SUITE}_correlations.csv", summarize(rows))


if __name__ == "__main__":
    main()
