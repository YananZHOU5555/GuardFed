#!/usr/bin/env python3
"""Paper-readable server/root distribution sensitivity experiment.

Main design:
- IID server/root baseline uses stratified_sensitive sampling, matching the
  clean server/root design used by the main AD2+ runs.
- non-IID server/root rows use deterministic controlled_group_skew, increasing
  the over-representation of the majority sensitive-label stratum.

This makes the table directly support the insight:
the closer the clean server/root data is to IID, the better the normal and
defended performance; skewed clean server/root data degrades the guidance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results" / "paper_tables" / "raw_results.jsonl"
OUT_DIR = ROOT / "results" / "paper_tables" / "goal_revision_v5"
RUNNER = ROOT / "scripts" / "reproduce_paper_tables.py"
PY = Path(sys.executable)
SUITE = "goal_server_iid_sensitivity_v5"

LEVELS: List[Tuple[str, str, Optional[float]]] = [
    ("IID server/root", "stratified_sensitive", None),
    ("Mild non-IID server/root", "controlled_group_skew", 0.05),
    ("Moderate non-IID server/root", "controlled_group_skew", 0.10),
    ("High non-IID server/root", "controlled_group_skew", 0.15),
    ("Very high non-IID server/root", "controlled_group_skew", 0.20),
]

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


def run_grid(rounds: int, seeds: Sequence[int], datasets: Sequence[str], dry_run: bool) -> None:
    for seed in seeds:
        for level_name, sampling, skew in LEVELS:
            tag_suffix = level_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            tag = f"{tag_suffix}_seed{seed}"
            args = [
                "--full",
                "--rounds", str(rounds),
                "--seed", str(seed),
                "--experiment-suite", SUITE,
                "--experiment-tag", tag,
                *COMMON,
                "--datasets", *datasets,
                "--distributions", "IID", "non-IID",
                "--attacks", "Benign", "FedSA",
                "--server-sampling", sampling,
            ]
            if skew is not None:
                args.extend(["--server-alpha", f"{skew:.2f}"])
            run_cmd(args, dry_run=dry_run)


def load_records() -> List[Dict[str, Any]]:
    if not RAW_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with RAW_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def last10_values(record: Dict[str, Any], metric: str) -> List[float]:
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
    return vals


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def level_from_config(cfg: Dict[str, Any]) -> Tuple[str, float]:
    sampling = cfg.get("server_sampling")
    if sampling == "stratified_sensitive":
        return "IID server/root", 0.0
    skew = float(cfg.get("server_alpha") or 0.0)
    for idx, (name, mode, alpha) in enumerate(LEVELS):
        if mode == sampling and alpha is not None and abs(alpha - skew) < 1e-9:
            return name, float(idx)
    return f"{sampling}:{skew:.2f}", skew


def rows_for_suite(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        cfg = r.get("config", {})
        if cfg.get("experiment_suite") != SUITE:
            continue
        audit = r.get("data_contract", {}).get("server_sampling_audit", {}) or {}
        level, level_index = level_from_config(cfg)
        acc = mean(last10_values(r, "ACC"))
        aeod = mean(last10_values(r, "AEOD"))
        aspd = mean(last10_values(r, "ASPD"))
        out.append({
            "suite": SUITE,
            "tag": cfg.get("experiment_tag"),
            "level": level,
            "level_index": level_index,
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
            "ACC_pct_last10_mean": acc * 100 if math.isfinite(acc) else float("nan"),
            "AEOD_last10_mean": aeod,
            "ASPD_last10_mean": aspd,
            "FairAvg_last10_mean": (aeod + aspd) / 2 if math.isfinite(aeod) and math.isfinite(aspd) else float("nan"),
            "Score_last10_mean": acc - (aeod + aspd) / 2 if math.isfinite(acc) and math.isfinite(aeod) and math.isfinite(aspd) else float("nan"),
            "ACC_pct_final": (r.get("metrics", {}).get("accuracy") or float("nan")) * 100,
            "AEOD_final": r.get("metrics", {}).get("aeod"),
            "ASPD_final": r.get("metrics", {}).get("aspd"),
        })
    return out


def aggregate(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: List[Dict[str, Any]] = []
    for key, vals in sorted(groups.items(), key=lambda kv: (kv[0], mean([v["level_index"] for v in kv[1]]))):
        item = {k: v for k, v in zip(keys, key)}
        item.update({
            "level_index": mean([v["level_index"] for v in vals]),
            "n": len(vals),
            "group_tvd_mean": mean([v["group_tvd"] for v in vals]),
            "sensitive_tvd_mean": mean([v["sensitive_tvd"] for v in vals]),
            "label_tvd_mean": mean([v["label_tvd"] for v in vals]),
            "ACC_pct_mean": mean([v["ACC_pct_last10_mean"] for v in vals]),
            "AEOD_mean": mean([v["AEOD_last10_mean"] for v in vals]),
            "ASPD_mean": mean([v["ASPD_last10_mean"] for v in vals]),
            "FairAvg_mean": mean([v["FairAvg_last10_mean"] for v in vals]),
            "Score_mean": mean([v["Score_last10_mean"] for v in vals]),
        })
        out.append(item)
    return out


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


def correlations(agg_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in agg_rows:
        groups.setdefault((row["dataset"], row["distribution"], row["attack"]), []).append(row)
    out: List[Dict[str, Any]] = []
    for key, vals in sorted(groups.items()):
        vals = sorted(vals, key=lambda x: x["level_index"])
        xs = [v["level_index"] for v in vals]
        out.append({
            "dataset": key[0],
            "distribution": key[1],
            "attack": key[2],
            "n_levels": len(vals),
            "corr_level_acc": corr(xs, [v["ACC_pct_mean"] for v in vals]),
            "corr_level_aeod": corr(xs, [v["AEOD_mean"] for v in vals]),
            "corr_level_aspd": corr(xs, [v["ASPD_mean"] for v in vals]),
            "corr_level_fairavg": corr(xs, [v["FairAvg_mean"] for v in vals]),
            "delta_acc_last_minus_iid": vals[-1]["ACC_pct_mean"] - vals[0]["ACC_pct_mean"] if len(vals) >= 2 else float("nan"),
            "delta_fairavg_last_minus_iid": vals[-1]["FairAvg_mean"] - vals[0]["FairAvg_mean"] if len(vals) >= 2 else float("nan"),
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
    parser.add_argument("--rounds", type=int, default=70)
    parser.add_argument("--seeds", default="123,456,789")
    parser.add_argument("--datasets", default="adult,compas")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if args.run:
        run_grid(args.rounds, seeds, datasets, args.dry_run)
    if args.summarize:
        rows = rows_for_suite(load_records())
        agg = aggregate(rows, ["dataset", "distribution", "attack", "level"])
        write_csv(OUT_DIR / "goal_server_iid_sensitivity_v5_raw.csv", rows)
        write_csv(OUT_DIR / "goal_server_iid_sensitivity_v5_by_level.csv", agg)
        write_csv(OUT_DIR / "goal_server_iid_sensitivity_v5_correlations.csv", correlations(agg))
        print(f"rows={len(rows)} agg={len(agg)} wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
