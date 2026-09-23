#!/usr/bin/env python3
"""Goal revision v3: stronger diagnostic experiments for 01/02/03.

This script is intentionally separate from v2 so old CSVs remain auditable.
It uses reproduce_paper_tables.py for every run and only summarizes raw JSONL.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path("/home/yannan/workspace/GuardFed")
RAW_PATH = ROOT / "results" / "paper_tables" / "raw_results.jsonl"
OUT_DIR = ROOT / "results" / "paper_tables" / "goal_revision_v3"
RUNNER = ROOT / "scripts" / "reproduce_paper_tables.py"
PY = ROOT / ".venv" / "bin" / "python"

ABLATION_SUITE = "goal_ablation_v3_stress"
SERVER_SUITE = "goal_server_dense_v5"

SEEDS = [123, 456, 789]
SERVER_SEEDS = [123, 456]

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

ABLATION_PROFILES = [
    (
        "full",
        [
            "--act-risk-weight", "1.40",
            "--act-violation-weight", "0.70",
            "--ad2-utility-weight", "3.00",
            "--ad2-centrality-weight", "0.80",
            "--ad2-alignment-weight", "1.80",
        ],
    ),
    (
        "no_performance_UCA",
        [
            "--act-risk-weight", "1.40",
            "--act-violation-weight", "0.70",
            "--ad2-utility-weight", "0.00",
            "--ad2-centrality-weight", "0.00",
            "--ad2-alignment-weight", "0.00",
        ],
    ),
    (
        "no_fairness_FC",
        [
            "--disable-ad2-calibration",
            "--act-risk-weight", "0.00",
            "--act-violation-weight", "0.00",
            "--ad2-utility-weight", "3.00",
            "--ad2-centrality-weight", "0.80",
            "--ad2-alignment-weight", "1.80",
        ],
    ),
    (
        "no_geometry_CA",
        [
            "--act-risk-weight", "1.40",
            "--act-violation-weight", "0.70",
            "--ad2-utility-weight", "3.00",
            "--ad2-centrality-weight", "0.00",
            "--ad2-alignment-weight", "0.00",
        ],
    ),
    (
        "utility_only_U",
        [
            "--disable-ad2-calibration",
            "--act-risk-weight", "0.00",
            "--act-violation-weight", "0.00",
            "--ad2-utility-weight", "3.00",
            "--ad2-centrality-weight", "0.00",
            "--ad2-alignment-weight", "0.00",
        ],
    ),
    (
        "fairness_only_FC",
        [
            "--act-risk-weight", "1.40",
            "--act-violation-weight", "0.70",
            "--ad2-utility-weight", "0.00",
            "--ad2-centrality-weight", "0.00",
            "--ad2-alignment-weight", "0.00",
        ],
    ),
]

SERVER_DENSE_JOBS = [
    ("adult", 0, 0),
    ("adult", 1, 1),
    ("compas", 1, 0),
]
SERVER_SKEWS = [0.00, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35]


def run_cmd(args: Sequence[str], dry_run: bool = False) -> None:
    cmd = [str(PY), str(RUNNER), *args]
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_ablation(rounds: int, seeds: Sequence[int], dry_run: bool) -> None:
    for seed in seeds:
        for attack in ["FedSA", "F Flip", "S-DFA"]:
            for profile, profile_args in ABLATION_PROFILES:
                tag = f"{profile}_{attack.replace(' ', '')}_seed{seed}"
                run_cmd(
                    [
                        "--full",
                        "--rounds", str(rounds),
                        "--seed", str(seed),
                        "--experiment-suite", ABLATION_SUITE,
                        "--experiment-tag", tag,
                        *COMMON,
                        "--datasets", "adult", "compas",
                        "--distributions", "IID", "non-IID",
                        "--attacks", attack,
                        *profile_args,
                    ],
                    dry_run=dry_run,
                )


def run_server_dense(rounds: int, seeds: Sequence[int], dry_run: bool) -> None:
    for seed in seeds:
        for dataset, target_sensitive, target_label in SERVER_DENSE_JOBS:
            for skew in SERVER_SKEWS:
                tag = f"{dataset}_target{target_sensitive}{target_label}_skew{skew:.2f}_seed{seed}"
                run_cmd(
                    [
                        "--full",
                        "--rounds", str(rounds),
                        "--seed", str(seed),
                        "--experiment-suite", SERVER_SUITE,
                        "--experiment-tag", tag,
                        *COMMON,
                        "--datasets", dataset,
                        "--distributions", "IID", "non-IID",
                        "--attacks", "Benign", "FedSA",
                        "--server-sampling", "controlled_target_group_skew",
                        "--server-alpha", f"{skew:.2f}",
                        "--server-target-sensitive", str(target_sensitive),
                        "--server-target-label", str(target_label),
                        "--act-risk-weight", "1.40",
                        "--act-violation-weight", "0.70",
                        "--ad2-utility-weight", "3.00",
                        "--ad2-centrality-weight", "0.80",
                        "--ad2-alignment-weight", "1.80",
                    ],
                    dry_run=dry_run,
                )


def load_records() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not RAW_PATH.exists():
        return rows
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


def suite_rows(records: Iterable[Dict[str, Any]], suite: str) -> List[Dict[str, Any]]:
    rows = []
    for r in records:
        cfg = r.get("config", {})
        if cfg.get("experiment_suite") != suite:
            continue
        audit = r.get("data_contract", {}).get("server_sampling_audit", {}) or {}
        acc = selected_metric(r, "ACC")
        aeod = selected_metric(r, "AEOD")
        aspd = selected_metric(r, "ASPD")
        tag = str(cfg.get("experiment_tag", ""))
        profile = tag
        for attack in ["FedSA", "FFlip", "S-DFA", "SDFA"]:
            profile = profile.replace(f"_{attack}", "")
        profile = profile.split("_seed")[0]
        rows.append({
            "suite": suite,
            "tag": tag,
            "profile": profile,
            "dataset": r.get("dataset"),
            "distribution": r.get("distribution"),
            "attack": r.get("attack"),
            "seed": r.get("seed"),
            "rounds": r.get("rounds"),
            "ad2_calibration_enabled": cfg.get("ad2_calibration_enabled"),
            "server_sampling": cfg.get("server_sampling"),
            "server_alpha": cfg.get("server_alpha"),
            "server_target_sensitive": cfg.get("server_target_sensitive", audit.get("server_target_sensitive")),
            "server_target_label": cfg.get("server_target_label", audit.get("server_target_label")),
            "group_tvd": audit.get("group_tvd"),
            "sensitive_tvd": audit.get("sensitive_tvd"),
            "label_tvd": audit.get("label_tvd"),
            "ACC": acc,
            "ACC_pct": acc * 100.0,
            "AEOD": aeod,
            "ASPD": aspd,
            "fair_avg": 0.5 * (aeod + aspd),
            "score": acc - 0.5 * (aeod + aspd),
        })
    rows.sort(key=lambda x: (str(x["dataset"]), str(x["distribution"]), str(x["attack"]), str(x["tag"]), str(x["seed"])))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[Dict[str, Any]], keys: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    out = []
    for key, group_rows in grouped.items():
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(group_rows)
        for metric in metrics:
            vals = [float(r[metric]) for r in group_rows if r.get(metric) not in (None, "") and math.isfinite(float(r[metric]))]
            if vals:
                row[f"{metric}_mean"] = sum(vals) / len(vals)
                row[f"{metric}_min"] = min(vals)
                row[f"{metric}_max"] = max(vals)
        out.append(row)
    out.sort(key=lambda r: tuple(str(r.get(k)) for k in keys))
    return out


def summarize_ablation(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    full = {}
    for row in rows:
        if row["profile"] == "full":
            full[(row["dataset"], row["distribution"], row["attack"], row["seed"])] = row
    out = []
    for row in rows:
        base = full.get((row["dataset"], row["distribution"], row["attack"], row["seed"]))
        if not base:
            continue
        out.append({
            **row,
            "delta_acc_pp_vs_full": (float(row["ACC"]) - float(base["ACC"])) * 100.0,
            "delta_aeod_vs_full": float(row["AEOD"]) - float(base["AEOD"]),
            "delta_aspd_vs_full": float(row["ASPD"]) - float(base["ASPD"]),
            "delta_fair_avg_vs_full": float(row["fair_avg"]) - float(base["fair_avg"]),
            "delta_score_vs_full": float(row["score"]) - float(base["score"]),
        })
    return out


def corr(xs0: Sequence[Any], ys0: Sequence[Any]) -> float | None:
    pairs = []
    for x, y in zip(xs0, ys0):
        try:
            xf = float(x)
            yf = float(y)
        except Exception:
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    return num / math.sqrt(sx * sy) if sx > 0 and sy > 0 else None


def summarize_server(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg = aggregate(
        rows,
        ["dataset", "distribution", "attack", "server_target_sensitive", "server_target_label", "server_alpha"],
        ["group_tvd", "ACC_pct", "AEOD", "ASPD", "fair_avg", "score", "sensitive_tvd", "label_tvd"],
    )
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in agg:
        grouped.setdefault((row["dataset"], row["distribution"], row["attack"], row["server_target_sensitive"], row["server_target_label"]), []).append(row)
    corr_rows = []
    for key, group_rows in grouped.items():
        corr_rows.append({
            "dataset": key[0],
            "distribution": key[1],
            "attack": key[2],
            "server_target_sensitive": key[3],
            "server_target_label": key[4],
            "n": len(group_rows),
            "corr_tvd_acc": corr([r.get("group_tvd_mean") for r in group_rows], [r.get("ACC_pct_mean") for r in group_rows]),
            "corr_tvd_aeod": corr([r.get("group_tvd_mean") for r in group_rows], [r.get("AEOD_mean") for r in group_rows]),
            "corr_tvd_aspd": corr([r.get("group_tvd_mean") for r in group_rows], [r.get("ASPD_mean") for r in group_rows]),
        })
    write_csv(OUT_DIR / "goal_server_dense_v5_by_tvd_slice.csv", agg)
    return corr_rows


def summarize() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    ablation_rows = suite_rows(records, ABLATION_SUITE)
    write_csv(OUT_DIR / "goal_ablation_v3_stress_raw.csv", ablation_rows)
    ablation_delta = summarize_ablation(ablation_rows)
    write_csv(OUT_DIR / "goal_ablation_v3_stress_deltas.csv", ablation_delta)
    write_csv(OUT_DIR / "goal_ablation_v3_stress_profile_summary.csv", aggregate(
        ablation_delta,
        ["dataset", "attack", "profile"],
        ["ACC_pct", "AEOD", "ASPD", "fair_avg", "score", "delta_acc_pp_vs_full", "delta_fair_avg_vs_full", "delta_score_vs_full"],
    ))

    server_rows = suite_rows(records, SERVER_SUITE)
    write_csv(OUT_DIR / "goal_server_dense_v5_raw.csv", server_rows)
    write_csv(OUT_DIR / "goal_server_dense_v5_correlations.csv", summarize_server(server_rows))

    print(f"wrote {OUT_DIR}")
    print(f"ablation rows={len(ablation_rows)} server rows={len(server_rows)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["ablation", "server", "summarize", "all"], default="summarize")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seeds = args.seeds or SEEDS
    t0 = time.time()
    if args.phase in {"ablation", "all"}:
        run_ablation(args.rounds, seeds, args.dry_run)
    if args.phase in {"server", "all"}:
        run_server_dense(args.rounds, args.seeds or SERVER_SEEDS, args.dry_run)
    summarize()
    print(f"elapsed_sec={time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
