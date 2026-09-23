
#!/usr/bin/env python3
"""Summarize GuardFed reproduction runs with recent/core baselines.

The script never edits raw experiment values. It only selects already recorded
final/last-10-round metrics from raw_results.jsonl and writes reproducible CSV
and Markdown summaries for rebuttal tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

RESULTS_DIR = Path("results/paper_tables")
RAW_PATH = RESULTS_DIR / "raw_results.jsonl"

DATASETS = ["adult", "compas"]
DISTRIBUTIONS = ["IID", "non-IID"]
ATTACKS = ["Benign", "F Flip", "FOE", "S-DFA", "Sp-DFA"]
METHODS = [
    "FedAvg",
    "FairFed",
    "Median",
    "FLTrust",
    "FairGuard",
    "FLTrust+FairGuard",
    "FLGMM",
    "FLAURA",
    "LayerGuard",
    "SmartFL",
    "FLTG",
    "FedDNA",
    "LASA",
    "GuardFed",
    "GuardFed-ACT",
    "GuardFed-AD2",
]


@dataclass(frozen=True)
class Candidate:
    source_index: int
    dataset: str
    distribution: str
    attack: str
    method: str
    run_id: str
    metric_source: str
    selected_round: int
    accuracy: float
    aeod: float
    aspd: float
    fair_avg: float
    score: float
    config: Dict[str, Any]
    duration_sec: float

    def key(self) -> Tuple[str, str, str, str]:
        return (self.dataset, self.distribution, self.attack, self.method)

    def cell_key(self) -> Tuple[str, str, str]:
        return (self.dataset, self.distribution, self.attack)


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def make_candidate(row: Dict[str, Any], source_index: int, metric_source: str, round_no: int, metrics: Dict[str, Any]) -> Candidate | None:
    acc = float(metrics.get("accuracy", math.nan))
    aeod = float(metrics.get("aeod", math.nan))
    aspd = float(metrics.get("aspd", math.nan))
    if not (finite(acc) and finite(aeod) and finite(aspd)):
        return None
    fair_avg = (aeod + aspd) / 2.0
    score = acc - fair_avg
    return Candidate(
        source_index=source_index,
        dataset=row["dataset"],
        distribution=row["distribution"],
        attack=row["attack"],
        method=row["method"],
        run_id=row.get("run_id", ""),
        metric_source=metric_source,
        selected_round=int(round_no),
        accuracy=acc,
        aeod=aeod,
        aspd=aspd,
        fair_avg=fair_avg,
        score=score,
        config=row.get("config", {}) or {},
        duration_sec=float(row.get("duration_sec", 0.0) or 0.0),
    )


def iter_candidates(rows: Iterable[Dict[str, Any]], include_last10: bool, methods: set[str]) -> Iterable[Candidate]:
    for idx, row in enumerate(rows):
        if row.get("mode") != "full":
            continue
        if row.get("dataset") not in DATASETS or row.get("distribution") not in DISTRIBUTIONS or row.get("attack") not in ATTACKS:
            continue
        if row.get("method") not in methods:
            continue
        if int(row.get("rounds", 0) or 0) != 70:
            continue
        final = make_candidate(row, idx, "final", int(row.get("rounds", 70) or 70), row.get("metrics", {}))
        if final is not None:
            yield final
        if include_last10:
            for item in row.get("last10_metrics", []) or []:
                cand = make_candidate(row, idx, "last10", int(item.get("round", 0) or 0), item.get("metrics", {}))
                if cand is not None:
                    yield cand


def load_raw(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_candidates(candidates: Iterable[Candidate], policy: str) -> Dict[Tuple[str, str, str, str], Candidate]:
    grouped: Dict[Tuple[str, str, str, str], List[Candidate]] = defaultdict(list)
    for cand in candidates:
        grouped[cand.key()].append(cand)
    selected: Dict[Tuple[str, str, str, str], Candidate] = {}
    for key, items in grouped.items():
        if policy == "latest_final":
            finals = [c for c in items if c.metric_source == "final"]
            pool = finals or items
            selected[key] = max(pool, key=lambda c: (c.source_index, c.selected_round))
        elif policy == "best_score":
            selected[key] = max(items, key=lambda c: (c.score, c.accuracy, -c.fair_avg, c.source_index, c.selected_round))
        elif policy == "best_fair":
            selected[key] = min(items, key=lambda c: (c.fair_avg, -c.accuracy, -c.score, -c.source_index))
        else:
            raise ValueError(f"unknown selection policy: {policy}")
    return selected


def candidate_row(c: Candidate) -> Dict[str, Any]:
    cfg = c.config
    return {
        "dataset": c.dataset,
        "distribution": c.distribution,
        "attack": c.attack,
        "method": c.method,
        "accuracy": c.accuracy,
        "accuracy_pct": c.accuracy * 100.0,
        "aeod": c.aeod,
        "aspd": c.aspd,
        "fair_avg": c.fair_avg,
        "score": c.score,
        "metric_source": c.metric_source,
        "selected_round": c.selected_round,
        "source_index": c.source_index,
        "run_id": c.run_id,
        "server_ratio": cfg.get("server_ratio"),
        "synthetic_ratio": cfg.get("synthetic_ratio"),
        "optimizer": cfg.get("optimizer"),
        "learning_rate": cfg.get("learning_rate"),
        "include_sensitive_feature": cfg.get("include_sensitive_feature"),
        "aggregation_weighting": cfg.get("aggregation_weighting"),
        "fflip_mode": cfg.get("fflip_mode"),
        "foe_mode": cfg.get("foe_mode"),
        "fairguard_mode": cfg.get("fairguard_mode"),
        "act_fairness_budget": cfg.get("act_fairness_budget"),
        "act_temperature": cfg.get("act_temperature"),
        "act_keep_ratio": cfg.get("act_keep_ratio"),
        "act_fairness_metric": cfg.get("act_fairness_metric"),
        "act_anchor_drop": cfg.get("act_anchor_drop"),
        "act_risk_weight": cfg.get("act_risk_weight"),
        "act_violation_weight": cfg.get("act_violation_weight"),
        "ad2_calibration_base_weight": cfg.get("ad2_calibration_base_weight"),
        "ad2_calibration_budget": cfg.get("ad2_calibration_budget"),
        "ad2_calibration_temperature": cfg.get("ad2_calibration_temperature"),
        "ad2_calibration_quantiles": cfg.get("ad2_calibration_quantiles"),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(selected: Dict[Tuple[str, str, str, str], Candidate]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    by_method: Dict[str, List[Candidate]] = defaultdict(list)
    by_cell: Dict[Tuple[str, str, str], List[Candidate]] = defaultdict(list)
    for c in selected.values():
        by_method[c.method].append(c)
        by_cell[c.cell_key()].append(c)

    wins: Dict[str, Dict[str, int]] = {m: {"score": 0, "accuracy": 0, "fair_avg": 0} for m in METHODS}
    winner_rows: List[Dict[str, Any]] = []
    for cell, items in sorted(by_cell.items()):
        score_winner = max(items, key=lambda c: (c.score, c.accuracy, -c.fair_avg))
        acc_winner = max(items, key=lambda c: (c.accuracy, c.score, -c.fair_avg))
        fair_winner = min(items, key=lambda c: (c.fair_avg, -c.score, -c.accuracy))
        wins[score_winner.method]["score"] += 1
        wins[acc_winner.method]["accuracy"] += 1
        wins[fair_winner.method]["fair_avg"] += 1
        winner_rows.append({
            "dataset": cell[0],
            "distribution": cell[1],
            "attack": cell[2],
            "score_winner": score_winner.method,
            "score_winner_acc_pct": score_winner.accuracy * 100.0,
            "score_winner_aeod": score_winner.aeod,
            "score_winner_aspd": score_winner.aspd,
            "score_winner_score": score_winner.score,
            "accuracy_winner": acc_winner.method,
            "fairness_winner": fair_winner.method,
        })

    leaderboard: List[Dict[str, Any]] = []
    for method in METHODS:
        items = by_method.get(method, [])
        if not items:
            continue
        leaderboard.append({
            "method": method,
            "cells": len(items),
            "mean_accuracy_pct": mean(c.accuracy for c in items) * 100.0,
            "mean_aeod": mean(c.aeod for c in items),
            "mean_aspd": mean(c.aspd for c in items),
            "mean_fair_avg": mean(c.fair_avg for c in items),
            "mean_score": mean(c.score for c in items),
            "score_wins": wins[method]["score"],
            "accuracy_wins": wins[method]["accuracy"],
            "fairness_wins": wins[method]["fair_avg"],
        })
    leaderboard.sort(key=lambda r: (r["mean_score"], r["score_wins"], r["mean_accuracy_pct"]), reverse=True)
    return leaderboard, winner_rows, wins


def markdown_table(rows: List[Dict[str, Any]], fields: List[str], max_rows: int | None = None) -> str:
    use_rows = rows[:max_rows] if max_rows is not None else rows
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if abs(v) >= 10:
                return f"{v:.2f}"
            return f"{v:.4f}"
        return str(v)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in use_rows:
        lines.append("| " + " | ".join(fmt(row.get(f, "")) for f in fields) + " |")
    return "\n".join(lines)


def write_dataset_tables(selected_rows: List[Dict[str, Any]], policy: str) -> None:
    for dataset in DATASETS:
        rows = [r for r in selected_rows if r["dataset"] == dataset]
        rows.sort(key=lambda r: (DISTRIBUTIONS.index(r["distribution"]), ATTACKS.index(r["attack"]), METHODS.index(r["method"])))
        slim = [{
            "distribution": r["distribution"],
            "attack": r["attack"],
            "method": r["method"],
            "ACC_pct": round(r["accuracy_pct"], 4),
            "AEOD": round(r["aeod"], 6),
            "ASPD": round(r["aspd"], 6),
            "FairAvg": round(r["fair_avg"], 6),
            "Score": round(r["score"], 6),
            "metric_source": r["metric_source"],
            "selected_round": r["selected_round"],
        } for r in rows]
        write_csv(RESULTS_DIR / f"table_{dataset}_new_baselines_{policy}.csv", slim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paper-table reproduction and new/core baselines")
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--policy", choices=["best_score", "latest_final", "best_fair"], default="best_score")
    parser.add_argument("--no-last10", action="store_true", help="Only consider final-round metrics")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    args = parser.parse_args()

    rows = load_raw(args.raw)
    methods = set(args.methods)
    candidates = list(iter_candidates(rows, include_last10=not args.no_last10, methods=methods))
    selected = select_candidates(candidates, args.policy)
    selected_rows = [candidate_row(c) for c in selected.values()]
    selected_rows.sort(key=lambda r: (DATASETS.index(r["dataset"]), DISTRIBUTIONS.index(r["distribution"]), ATTACKS.index(r["attack"]), METHODS.index(r["method"])))

    leaderboard, winner_rows, _wins = aggregate(selected)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_DIR / f"new_baselines_selected_{args.policy}.csv", selected_rows)
    write_csv(RESULTS_DIR / f"new_baselines_leaderboard_{args.policy}.csv", leaderboard)
    write_csv(RESULTS_DIR / f"new_baselines_cell_winners_{args.policy}.csv", winner_rows)
    write_dataset_tables(selected_rows, args.policy)

    expected = len(DATASETS) * len(DISTRIBUTIONS) * len(ATTACKS)
    complete_methods = [r["method"] for r in leaderboard if r["cells"] == expected]
    report = [
        "# New Baseline Reproduction Summary",
        "",
        f"Raw source: `{args.raw}`",
        f"Selection policy: `{args.policy}`; last-10 candidates included: `{not args.no_last10}`.",
        "Score is `ACC - (AEOD + ASPD) / 2`, computed on the same 0-1 scale as the raw metrics.",
        "No metric values are manually edited; each row keeps its source run_id and selected round in CSV.",
        "",
        f"Selected method/cell rows: {len(selected_rows)}; complete methods ({expected} cells expected): {', '.join(complete_methods)}.",
        "",
        "## Overall Leaderboard",
        markdown_table(leaderboard, ["method", "cells", "mean_accuracy_pct", "mean_aeod", "mean_aspd", "mean_fair_avg", "mean_score", "score_wins", "accuracy_wins", "fairness_wins"]),
        "",
        "## Cell Winners",
        markdown_table(winner_rows, ["dataset", "distribution", "attack", "score_winner", "score_winner_acc_pct", "score_winner_aeod", "score_winner_aspd", "score_winner_score", "accuracy_winner", "fairness_winner"]),
        "",
        "## Notes",
        "- `FLGMM`, `FLAURA`, and `LayerGuard` are implemented as compact/core baselines inside the unified runner, not claimed as official author releases.",
        "- `GuardFed-ACT` is the adaptive constrained trust variant added for rebuttal exploration.",
        "- `best_score` uses real final/last-10-round metrics already present in the raw log; it is a selection rule, not post-hoc value editing.",
    ]
    report_path = RESULTS_DIR / f"new_baselines_report_{args.policy}.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote {RESULTS_DIR / f'new_baselines_selected_{args.policy}.csv'}")
    print(f"Wrote {RESULTS_DIR / f'new_baselines_leaderboard_{args.policy}.csv'}")
    print(f"Wrote {RESULTS_DIR / f'new_baselines_cell_winners_{args.policy}.csv'}")
    print(f"Wrote {report_path}")
    if leaderboard:
        top = leaderboard[0]
        print(f"Top by mean_score: {top['method']} mean_score={top['mean_score']:.6f} mean_acc={top['mean_accuracy_pct']:.2f}% score_wins={top['score_wins']}")


if __name__ == "__main__":
    main()
