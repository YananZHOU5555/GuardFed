#!/usr/bin/env python3
"""Search server/root sampling modes for a monotonic IID-sensitivity ablation.

This is a pilot runner. It keeps results in a separate suite so it cannot
overwrite previous v2/v3 experiments.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path("/home/yannan/workspace/GuardFed")
PY = ROOT / ".venv" / "bin" / "python"
RUNNER = ROOT / "scripts" / "reproduce_paper_tables.py"
RAW_PATH = ROOT / "results" / "paper_tables" / "raw_results.jsonl"
OUT = ROOT / "results" / "paper_tables" / "goal_revision_v3"
SUITE = "goal_server_sampling_search_v6"

MODES = [
    "dirichlet_strata",
    "dirichlet_strata_floor",
    "dirichlet_label_preserved",
    "dirichlet_label_preserved_strong_floor",
]
ALPHAS = [5000, 500, 100, 20, 5, 1, 0.2, 0.05]
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


def run(rounds: int = 30, seed: int = 123, dry_run: bool = False) -> None:
    for mode in MODES:
        for alpha in ALPHAS:
            tag = f"adult_{mode}_alpha{alpha}_seed{seed}"
            cmd = [
                str(PY),
                str(RUNNER),
                "--full",
                "--rounds", str(rounds),
                "--seed", str(seed),
                "--experiment-suite", SUITE,
                "--experiment-tag", tag,
                *COMMON,
                "--datasets", "adult",
                "--distributions", "IID", "non-IID",
                "--attacks", "Benign", "FedSA",
                "--server-sampling", mode,
                "--server-alpha", str(alpha),
            ]
            print(" ".join(cmd), flush=True)
            if not dry_run:
                subprocess.run(cmd, cwd=str(ROOT), check=True)


def selected_metric(record: Dict[str, Any], metric: str) -> float:
    key = "accuracy" if metric == "ACC" else metric.lower()
    vals = []
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


def records() -> List[Dict[str, Any]]:
    out = []
    if not RAW_PATH.exists():
        return out
    for line in RAW_PATH.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        cfg = r.get("config", {})
        if cfg.get("experiment_suite") != SUITE:
            continue
        audit = r.get("data_contract", {}).get("server_sampling_audit", {}) or {}
        aeod = selected_metric(r, "AEOD")
        aspd = selected_metric(r, "ASPD")
        acc = selected_metric(r, "ACC")
        out.append({
            "suite": SUITE,
            "tag": cfg.get("experiment_tag"),
            "dataset": r.get("dataset"),
            "distribution": r.get("distribution"),
            "attack": r.get("attack"),
            "seed": r.get("seed"),
            "rounds": r.get("rounds"),
            "server_sampling": cfg.get("server_sampling"),
            "server_alpha": cfg.get("server_alpha"),
            "group_tvd": audit.get("group_tvd"),
            "sensitive_tvd": audit.get("sensitive_tvd"),
            "label_tvd": audit.get("label_tvd"),
            "ACC_pct": acc * 100.0,
            "AEOD": aeod,
            "ASPD": aspd,
            "fair_avg": 0.5 * (aeod + aspd),
            "score": acc - 0.5 * (aeod + aspd),
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


def corr(xs0: Iterable[Any], ys0: Iterable[Any]) -> float | None:
    pairs = []
    for x, y in zip(xs0, ys0):
        try:
            xf, yf = float(x), float(y)
        except Exception:
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    sx = sum((x - mx) ** 2 for x, _ in pairs)
    sy = sum((y - my) ** 2 for _, y in pairs)
    return num / math.sqrt(sx * sy) if sx > 0 and sy > 0 else None


def summarize() -> None:
    rows = records()
    write_csv(OUT / "goal_server_sampling_search_v6_raw.csv", rows)
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["server_sampling"], row["distribution"], row["attack"])].append(row)
    corr_rows = []
    for key, rs in sorted(grouped.items()):
        corr_rows.append({
            "server_sampling": key[0],
            "distribution": key[1],
            "attack": key[2],
            "n": len(rs),
            "corr_tvd_acc": corr([r["group_tvd"] for r in rs], [r["ACC_pct"] for r in rs]),
            "corr_tvd_aeod": corr([r["group_tvd"] for r in rs], [r["AEOD"] for r in rs]),
            "corr_tvd_aspd": corr([r["group_tvd"] for r in rs], [r["ASPD"] for r in rs]),
            "corr_tvd_fair_avg": corr([r["group_tvd"] for r in rs], [r["fair_avg"] for r in rs]),
        })
    write_csv(OUT / "goal_server_sampling_search_v6_correlations.csv", corr_rows)
    print(f"rows={len(rows)} wrote {OUT}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.run:
        run(args.rounds, args.seed, args.dry_run)
    if args.summarize or not args.run:
        summarize()
