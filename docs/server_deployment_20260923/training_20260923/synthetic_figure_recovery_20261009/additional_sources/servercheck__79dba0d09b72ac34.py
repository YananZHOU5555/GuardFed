#!/usr/bin/env python3
"""Run/summarize additional goal-aligned GuardFed experiments on 5090.

This wrapper does not implement algorithms. It calls scripts/reproduce_paper_tables.py
with auditable experiment_suite names and then summarizes completed raw_results rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results" / "paper_tables" / "raw_results.jsonl"
OUT_DIR = ROOT / "results" / "paper_tables" / "goal_revision_v2"
RUNNER = ROOT / "scripts" / "reproduce_paper_tables.py"
PY = Path(sys.executable)

ABLATION_SUITE = "goal_ablation_v2"
SERVER_SUITE = "goal_server_skew_v2"
POS_SERVER_SUITE = "goal_server_positive_skew_v3"
TARGET_SERVER_SUITE = "goal_server_target_skew_v4"
SYNTH_SUITE = "server_generation_ablation"

SEEDS_PILOT = [123]
SEEDS_FULL = [123, 456, 789]

COMMON_AD2 = [
    "--server-ratio", "0.10",
    "--synthetic-ratio", "0",
    "--synthetic-method", "none",
    "--device", "cuda",
    "--methods", "GuardFed-AD2+",
    "--ad2-plus-mode", "fixed",
    "--act-fairness-budget", "0.08",
    "--act-temperature", "0.50",
    "--act-keep-ratio", "0.80",
    "--act-fairness-metric", "aeod_aspd",
    "--ad2-score-clip", "5",
    "--ad2-norm-mode", "root",
    "--sdfa-foe-mode", "fedsa",
    "--spdfa-foe-mode", "fedsa",
    "--fedsa-gain", "3.00",
    "--fedsa-norm-ratio", "3.50",
]

ABLATION_PROFILES = [
    ("full", ["--ad2-utility-weight", "2.50", "--ad2-centrality-weight", "0.30", "--ad2-alignment-weight", "1.20", "--act-risk-weight", "0.80", "--act-violation-weight", "0.25"]),
    ("no_utility_U", ["--ad2-utility-weight", "0.00", "--ad2-centrality-weight", "0.30", "--ad2-alignment-weight", "1.20", "--act-risk-weight", "0.80", "--act-violation-weight", "0.25"]),
    ("no_fairness_FV", ["--ad2-utility-weight", "2.50", "--ad2-centrality-weight", "0.30", "--ad2-alignment-weight", "1.20", "--act-risk-weight", "0.00", "--act-violation-weight", "0.00"]),
    ("no_geometry_CA", ["--ad2-utility-weight", "2.50", "--ad2-centrality-weight", "0.00", "--ad2-alignment-weight", "0.00", "--act-risk-weight", "0.80", "--act-violation-weight", "0.25"]),
    ("utility_only_U", ["--ad2-utility-weight", "2.50", "--ad2-centrality-weight", "0.00", "--ad2-alignment-weight", "0.00", "--act-risk-weight", "0.00", "--act-violation-weight", "0.00"]),
    ("fairness_only_FV", ["--ad2-utility-weight", "0.00", "--ad2-centrality-weight", "0.00", "--ad2-alignment-weight", "0.00", "--act-risk-weight", "0.80", "--act-violation-weight", "0.25"]),
]

SERVER_SKEWS = [0.00, 0.03, 0.06, 0.10, 0.18, 0.30, 0.45, 0.60, 0.78, 0.92]


def run_cmd(args: Sequence[str], dry_run: bool = False) -> None:
    cmd = [str(PY), str(RUNNER), *args]
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def base_args(suite: str, tag: str, seed: int, rounds: int) -> List[str]:
    return [
        "--full",
        "--rounds", str(rounds),
        "--seed", str(seed),
        "--experiment-suite", suite,
        "--experiment-tag", tag,
        *COMMON_AD2,
    ]


def run_ablation(rounds: int, seeds: Sequence[int], dry_run: bool) -> None:
    attacks = ["FedSA", "F Flip", "S-DFA"]
    for seed in seeds:
        for attack in attacks:
            for profile, extra_profile in ABLATION_PROFILES:
                tag = f"{profile}_{attack.replace(' ', '')}_seed{seed}"
                extra = [
                    "--datasets", "adult", "compas",
                    "--distributions", "IID", "non-IID",
                    "--attacks", attack,
                    "--server-sampling", "stratified_sensitive",
                    *extra_profile,
                ]
                run_cmd([*base_args(ABLATION_SUITE, tag, seed, rounds), *extra], dry_run=dry_run)


def run_server_skew(rounds: int, seeds: Sequence[int], dry_run: bool) -> None:
    for seed in seeds:
        for skew in SERVER_SKEWS:
            tag = f"skew_{skew:.2f}_seed{seed}"
            extra = [
                "--datasets", "adult", "compas",
                "--distributions", "IID", "non-IID",
                "--attacks", "Benign", "FedSA",
                "--server-sampling", "controlled_group_skew",
                "--server-alpha", f"{skew:.2f}",
                "--ad2-utility-weight", "2.50",
                "--ad2-centrality-weight", "0.30",
                "--ad2-alignment-weight", "1.20",
                "--act-risk-weight", "0.80",
                "--act-violation-weight", "0.25",
            ]
            run_cmd([*base_args(SERVER_SUITE, tag, seed, rounds), *extra], dry_run=dry_run)


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


def suite_rows(records: Iterable[Dict[str, Any]], suite: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in records:
        cfg = r.get("config", {})
        if cfg.get("experiment_suite") != suite:
            continue
        acc = selected_metric(r, "ACC")
        aeod = selected_metric(r, "AEOD")
        aspd = selected_metric(r, "ASPD")
        audit = r.get("data_contract", {}).get("server_sampling_audit", {}) or {}
        rows.append({
            "suite": suite,
            "tag": cfg.get("experiment_tag", ""),
            "dataset": r.get("dataset"),
            "distribution": r.get("distribution"),
            "attack": r.get("attack"),
            "method": r.get("method"),
            "seed": r.get("seed"),
            "rounds": r.get("rounds"),
            "server_sampling": cfg.get("server_sampling"),
            "server_alpha": cfg.get("server_alpha"),
            "server_target_sensitive": cfg.get("server_target_sensitive", audit.get("server_target_sensitive")),
            "server_target_label": cfg.get("server_target_label", audit.get("server_target_label")),
            "synthetic_ratio": cfg.get("synthetic_ratio"),
            "synthetic_method": cfg.get("synthetic_method"),
            "ACC": acc,
            "ACC_pct": acc * 100.0,
            "AEOD": aeod,
            "ASPD": aspd,
            "fair_avg": 0.5 * (aeod + aspd),
            "score": acc - 0.5 * (aeod + aspd),
            "group_tvd": audit.get("group_tvd"),
            "group_kl": audit.get("group_kl"),
            "max_group_abs_delta": audit.get("max_group_abs_delta"),
            "sensitive_tvd": audit.get("sensitive_tvd"),
            "label_tvd": audit.get("label_tvd"),
            "server_group_counts": json.dumps(audit.get("server_group_counts", {}), sort_keys=True),
            "global_group_counts": json.dumps(audit.get("global_group_counts", {}), sort_keys=True),
        })
    rows.sort(key=lambda x: (str(x["suite"]), str(x["dataset"]), str(x["distribution"]), str(x["attack"]), str(x["tag"]), str(x["seed"])))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize_ablation(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    full: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        if str(r["tag"]).startswith("full_"):
            full[(r["dataset"], r["distribution"], r["attack"], r["seed"])] = r
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (r["dataset"], r["distribution"], r["attack"], r["seed"])
        b = full.get(key)
        if not b:
            continue
        profile = str(r["tag"]).split("_seed")[0]
        for attack in ["FedSA", "F Flip", "S-DFA"]:
            profile = profile.replace(f"_{attack.replace(' ', '')}", "")
        out.append({
            **r,
            "profile": profile,
            "delta_acc_pp_vs_full": (float(r["ACC"]) - float(b["ACC"])) * 100.0,
            "delta_aeod_vs_full": float(r["AEOD"]) - float(b["AEOD"]),
            "delta_aspd_vs_full": float(r["ASPD"]) - float(b["ASPD"]),
            "delta_fair_avg_vs_full": float(r["fair_avg"]) - float(b["fair_avg"]),
            "delta_score_vs_full": float(r["score"]) - float(b["score"]),
        })
    return out


def aggregate(rows: List[Dict[str, Any]], keys: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    out: List[Dict[str, Any]] = []
    for key, rs in grouped.items():
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(rs)
        for m in metrics:
            vals = [float(r[m]) for r in rs if r.get(m) not in (None, "") and math.isfinite(float(r[m]))]
            if vals:
                avg = sum(vals) / len(vals)
                row[f"{m}_mean"] = avg
                row[f"{m}_min"] = min(vals)
                row[f"{m}_max"] = max(vals)
        out.append(row)
    out.sort(key=lambda r: tuple(str(r.get(k)) for k in keys))
    return out


def summarize() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()

    ab_rows = suite_rows(records, ABLATION_SUITE)
    write_csv(OUT_DIR / "goal_ablation_v2_raw.csv", ab_rows)
    ab_delta = summarize_ablation(ab_rows)
    write_csv(OUT_DIR / "goal_ablation_v2_deltas.csv", ab_delta)
    write_csv(OUT_DIR / "goal_ablation_v2_profile_summary.csv", aggregate(
        ab_delta,
        ["dataset", "attack", "profile"],
        ["ACC_pct", "AEOD", "ASPD", "fair_avg", "score", "delta_acc_pp_vs_full", "delta_fair_avg_vs_full", "delta_score_vs_full"],
    ))

    sv_rows = suite_rows(records, SERVER_SUITE)
    write_csv(OUT_DIR / "goal_server_skew_v2_raw.csv", sv_rows)
    write_csv(OUT_DIR / "goal_server_skew_v2_by_tvd_dataset.csv", aggregate(
        sv_rows,
        ["dataset", "server_alpha"],
        ["group_tvd", "ACC_pct", "AEOD", "ASPD", "fair_avg", "score", "sensitive_tvd", "label_tvd"],
    ))
    write_csv(OUT_DIR / "goal_server_skew_v2_by_tvd_slice.csv", aggregate(
        sv_rows,
        ["dataset", "distribution", "attack", "server_alpha"],
        ["group_tvd", "ACC_pct", "AEOD", "ASPD", "fair_avg", "score", "sensitive_tvd", "label_tvd"],
    ))

    pos_rows = suite_rows(records, POS_SERVER_SUITE)
    write_csv(OUT_DIR / "goal_server_positive_skew_v3_raw.csv", pos_rows)
    write_csv(OUT_DIR / "goal_server_positive_skew_v3_by_tvd_dataset.csv", aggregate(
        pos_rows,
        ["dataset", "server_alpha"],
        ["group_tvd", "ACC_pct", "AEOD", "ASPD", "fair_avg", "score", "sensitive_tvd", "label_tvd"],
    ))
    write_csv(OUT_DIR / "goal_server_positive_skew_v3_by_tvd_slice.csv", aggregate(
        pos_rows,
        ["dataset", "distribution", "attack", "server_alpha"],
        ["group_tvd", "ACC_pct", "AEOD", "ASPD", "fair_avg", "score", "sensitive_tvd", "label_tvd"],
    ))

    target_rows = suite_rows(records, TARGET_SERVER_SUITE)
    write_csv(OUT_DIR / "goal_server_target_skew_v4_raw.csv", target_rows)
    write_csv(OUT_DIR / "goal_server_target_skew_v4_by_tvd_dataset.csv", aggregate(
        target_rows,
        ["dataset", "server_target_sensitive", "server_target_label", "server_alpha"],
        ["group_tvd", "ACC_pct", "AEOD", "ASPD", "fair_avg", "score", "sensitive_tvd", "label_tvd"],
    ))
    write_csv(OUT_DIR / "goal_server_target_skew_v4_by_tvd_slice.csv", aggregate(
        target_rows,
        ["dataset", "distribution", "attack", "server_target_sensitive", "server_target_label", "server_alpha"],
        ["group_tvd", "ACC_pct", "AEOD", "ASPD", "fair_avg", "score", "sensitive_tvd", "label_tvd"],
    ))

    synth_rows = suite_rows(records, SYNTH_SUITE)
    write_csv(OUT_DIR / "server_generation_ablation_raw_from_existing.csv", synth_rows)
    write_csv(OUT_DIR / "server_generation_ablation_summary_from_existing.csv", aggregate(
        synth_rows,
        ["dataset", "tag", "synthetic_method", "synthetic_ratio"],
        ["ACC_pct", "AEOD", "ASPD", "fair_avg", "score"],
    ))

    print(f"Wrote summaries to {OUT_DIR}")
    print(f"ablation rows={len(ab_rows)} server rows={len(sv_rows)} positive_server rows={len(pos_rows)} target_server rows={len(target_rows)} synth rows={len(synth_rows)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ablation", "server", "summarize", "all"], default="summarize")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--full-seeds", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seeds = SEEDS_FULL if args.full_seeds else SEEDS_PILOT
    t0 = time.time()
    if args.phase in {"ablation", "all"}:
        run_ablation(args.rounds, seeds, args.dry_run)
    if args.phase in {"server", "all"}:
        run_server_skew(args.rounds, seeds, args.dry_run)
    summarize()
    print(f"elapsed_sec={time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
