#!/usr/bin/env python3
"""Attack-strength and malicious-ratio study for GuardFed-AD2+.

This orchestration layer imports the existing reproduction core. It keeps the
paper-table runner unchanged while adding calibration, resumable multi-seed
runs, validity gates, trajectories, and machine-readable summaries.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "reproduce_paper_tables.py"
RESULTS_DIR = ROOT / "results" / "attack_strength"
RAW_PATH = RESULTS_DIR / "raw_results.jsonl"
CALIBRATION_PATH = RESULTS_DIR / "calibration_results.jsonl"
CONFIG_PATH = RESULTS_DIR / "attack_config.json"
FAILURES_PATH = RESULTS_DIR / "failures.jsonl"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"
SEED_RESULTS_PATH = RESULTS_DIR / "seed_results.csv"
TRAJECTORY_PATH = RESULTS_DIR / "trajectory.csv"
AUDIT_PATH = RESULTS_DIR / "audit.csv"

SEEDS = [123, 456, 789, 1001, 2024, 3141, 4242, 5050, 6060, 7070]
CALIBRATION_SEED = 314159
RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50]
RATIO_CLIENTS = {0.10: 2, 0.20: 4, 0.30: 6, 0.40: 8, 0.50: 10}
FFLIP_CANDIDATES = ["invert", "label_conditioned", "label_conditioned_reverse", "all_privileged", "all_unprivileged"]
FEDSA_GAIN_CANDIDATES = [1.75, 2.5, 3.5, 4.5]
FEDSA_NORM_CANDIDATES = [2.0, 3.0, 4.0]


def load_core():
    spec = importlib.util.spec_from_file_location("guardfed_reproduction_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reproduction core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core()
METHODS = list(core.METHODS)
DATASETS = list(core.DATASETS)
DISTRIBUTIONS = dict(core.DISTRIBUTIONS)
METRICS = ["accuracy", "aeod", "aspd"]


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, default=json_default) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def done_ids(path: Path) -> set[str]:
    return {str(row.get("run_id")) for row in read_jsonl(path) if row.get("run_id")}


def make_config(
    seed: int,
    malicious: int,
    rounds: int,
    fflip_mode: str,
    fedsa_gain: float,
    fedsa_norm_ratio: float,
    tag: str,
) -> Any:
    return core.ExperimentConfig(
        seed=seed,
        num_clients=20,
        num_malicious=malicious,
        local_epochs=1,
        batch_size=256,
        learning_rate=0.005,
        rounds=rounds,
        device="cuda",
        server_ratio=0.10,
        synthetic_ratio=0.0,
        server_sampling="stratified_sensitive",
        synthetic_method="none",
        include_sensitive_feature=False,
        aggregation_weighting="count",
        fflip_mode=fflip_mode,
        foe_mode="state",
        sdfa_foe_mode="fedsa",
        spdfa_foe_mode="fedsa",
        fedsa_gain=fedsa_gain,
        fedsa_norm_ratio=fedsa_norm_ratio,
        fairguard_mode="server_aeod",
        use_reweighting=True,
        experiment_suite="attack_strength_ratio",
        experiment_tag=tag,
    )


def validity(result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = result.get("metrics", {})
    stats = result.get("evaluation_stats", {})
    acc = float(metrics.get("accuracy", math.nan))
    majority = float(stats.get("majority_accuracy", math.nan))
    positive = float(stats.get("positive_rate", math.nan))
    reasons: List[str] = []
    if not math.isfinite(acc) or not math.isfinite(majority) or acc <= majority:
        reasons.append("accuracy_not_above_majority_baseline")
    if not math.isfinite(positive) or positive < 0.01 or positive > 0.99:
        reasons.append("degenerate_positive_rate")
    warnings = list(result.get("warnings", []))
    if warnings:
        reasons.extend(str(item) for item in warnings)
    return {
        "valid_for_fairness": not reasons,
        "reasons": reasons,
        "accuracy": acc,
        "majority_accuracy": majority,
        "positive_rate": positive,
    }


def attach_metadata(result: Dict[str, Any], study: str, ratio: float | None = None) -> Dict[str, Any]:
    result["study"] = study
    result["malicious_ratio"] = ratio
    result["validity"] = validity(result)
    result["protocol"] = {
        "server_ratio": 0.10,
        "synthetic_ratio": 0.0,
        "num_clients": result.get("num_clients"),
        "num_malicious": result.get("num_malicious"),
        "rounds": result.get("rounds"),
        "seed": result.get("seed"),
        "class_b_fl": False,
        "act_included": False,
    }
    return result


def run_one(
    dataset: str,
    distribution: str,
    method: str,
    attack: str,
    seed: int,
    malicious: int,
    rounds: int,
    fflip_mode: str,
    fedsa_gain: float,
    fedsa_norm_ratio: float,
    study: str,
    ratio: float | None,
    device: Any,
) -> Dict[str, Any]:
    tag = f"{study}|{dataset}|{distribution}|{method}|{attack}|seed={seed}|m={malicious}|ratio={ratio}"
    config = make_config(seed, malicious, rounds, fflip_mode, fedsa_gain, fedsa_norm_ratio, tag)
    result = core.run_experiment(dataset, distribution, method, attack, config, study, device)
    return attach_metadata(result, study, ratio)


def run_and_record(
    result_path: Path,
    completed: set[str],
    **kwargs: Any,
) -> Dict[str, Any] | None:
    preview_config = make_config(
        kwargs["seed"], kwargs["malicious"], kwargs["rounds"], kwargs["fflip_mode"],
        kwargs["fedsa_gain"], kwargs["fedsa_norm_ratio"],
        f"{kwargs['study']}|{kwargs['dataset']}|{kwargs['distribution']}|{kwargs['method']}|{kwargs['attack']}|seed={kwargs['seed']}|m={kwargs['malicious']}|ratio={kwargs['ratio']}",
    )
    run_id = core.make_run_id(
        kwargs["study"], kwargs["dataset"], kwargs["distribution"], kwargs["method"], kwargs["attack"], preview_config
    )
    if run_id in completed:
        return None
    print(
        f"RUN {kwargs['study']} {kwargs['dataset']} {kwargs['distribution']} {kwargs['method']} "
        f"{kwargs['attack']} seed={kwargs['seed']} malicious={kwargs['malicious']} ratio={kwargs['ratio']}",
        flush=True,
    )
    started = time.time()
    try:
        result = run_one(**kwargs)
    except Exception as exc:
        failure = {
            "study": kwargs["study"],
            "dataset": kwargs["dataset"],
            "distribution": kwargs["distribution"],
            "method": kwargs["method"],
            "attack": kwargs["attack"],
            "seed": kwargs["seed"],
            "malicious": kwargs["malicious"],
            "ratio": kwargs["ratio"],
            "error": repr(exc),
        }
        append_jsonl(FAILURES_PATH, failure)
        raise
    append_jsonl(result_path, result)
    completed.add(str(result["run_id"]))
    metrics = result["metrics"]
    valid = result["validity"]["valid_for_fairness"]
    print(
        f"DONE ACC={float(metrics['accuracy']):.6f} AEOD={float(metrics['aeod']):.6f} "
        f"ASPD={float(metrics['aspd']):.6f} fairness_valid={valid} elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return result


def calibration_score(rows: Sequence[Mapping[str, Any]], benign: Mapping[Tuple[str, str], Mapping[str, Any]], kind: str) -> float:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["distribution"])), []).append(row)
    scores: List[float] = []
    for key, group in grouped.items():
        base = benign[key]["metrics"]
        for row in group:
            if not row.get("validity", {}).get("valid_for_fairness", False):
                continue
            metrics = row["metrics"]
            if kind == "fflip":
                scores.append((float(metrics["aeod"]) + float(metrics["aspd"])) - (float(base["aeod"]) + float(base["aspd"])))
            else:
                scores.append(float(base["accuracy"]) - float(metrics["accuracy"]))
    return float(statistics.fmean(scores)) if scores else -math.inf


def run_calibration(rounds: int, device: Any) -> Dict[str, Any]:
    existing = read_jsonl(CALIBRATION_PATH)
    if CONFIG_PATH.exists() and existing:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    completed = done_ids(CALIBRATION_PATH)
    base: Dict[Tuple[str, str], Dict[str, Any]] = {}
    base_rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for distribution in DISTRIBUTIONS:
            result = run_one(dataset, distribution, "FedAvg", "Benign", CALIBRATION_SEED, 4, rounds, "invert", 1.75, 2.0, "calibration", None, device)
            base[(dataset, distribution)] = result
            result = dict(result)
            result["calibration_kind"] = "benign_reference"
            append_jsonl(CALIBRATION_PATH, result)
            base_rows.append(result)
    for mode in FFLIP_CANDIDATES:
        for dataset in DATASETS:
            for distribution in DISTRIBUTIONS:
                result = run_one(dataset, distribution, "FedAvg", "F Flip", CALIBRATION_SEED, 4, rounds, mode, 1.75, 2.0, "calibration", None, device)
                result["calibration_kind"] = "fflip_candidate"
                result["candidate"] = mode
                append_jsonl(CALIBRATION_PATH, result)
    for gain in FEDSA_GAIN_CANDIDATES:
        for norm_ratio in FEDSA_NORM_CANDIDATES:
            for dataset in DATASETS:
                for distribution in DISTRIBUTIONS:
                    result = run_one(dataset, distribution, "FedAvg", "FedSA", CALIBRATION_SEED, 4, rounds, "invert", gain, norm_ratio, "calibration", None, device)
                    result["calibration_kind"] = "fedsa_candidate"
                    result["candidate"] = {"gain": gain, "norm_ratio": norm_ratio}
                    append_jsonl(CALIBRATION_PATH, result)
    rows = read_jsonl(CALIBRATION_PATH)
    fflip_rows = [row for row in rows if row.get("calibration_kind") == "fflip_candidate"]
    fedsa_rows = [row for row in rows if row.get("calibration_kind") == "fedsa_candidate"]
    fflip_scores = {mode: calibration_score([row for row in fflip_rows if row.get("candidate") == mode], base, "fflip") for mode in FFLIP_CANDIDATES}
    fedsa_scores: Dict[str, float] = {}
    for gain in FEDSA_GAIN_CANDIDATES:
        for norm_ratio in FEDSA_NORM_CANDIDATES:
            key = json.dumps({"gain": gain, "norm_ratio": norm_ratio}, sort_keys=True)
            candidates = [row for row in fedsa_rows if row.get("candidate") == {"gain": gain, "norm_ratio": norm_ratio}]
            fedsa_scores[key] = calibration_score(candidates, base, "fedsa")
    selected_fflip = max(fflip_scores, key=fflip_scores.get)
    selected_fedsa_key = max(fedsa_scores, key=fedsa_scores.get)
    selected_fedsa = json.loads(selected_fedsa_key)
    config = {
        "calibration_seed": CALIBRATION_SEED,
        "fflip_mode": selected_fflip,
        "fflip_scores": fflip_scores,
        "fedsa_gain": float(selected_fedsa["gain"]),
        "fedsa_norm_ratio": float(selected_fedsa["norm_ratio"]),
        "fedsa_scores": fedsa_scores,
        "selection_rule": "global FedAvg calibration; maximize fairness increase for F Flip and ACC drop for FedSA; invalid/degenerate runs are excluded",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def run_main(rounds: int, device: Any, config: Mapping[str, Any]) -> None:
    completed = done_ids(RAW_PATH)
    for seed in SEEDS:
        for dataset in DATASETS:
            for distribution in DISTRIBUTIONS:
                for method in METHODS:
                    for attack in ["Benign", "F Flip", "FedSA"]:
                        run_and_record(
                            RAW_PATH,
                            completed,
                            dataset=dataset,
                            distribution=distribution,
                            method=method,
                            attack=attack,
                            seed=seed,
                            malicious=4,
                            rounds=rounds,
                            fflip_mode=str(config["fflip_mode"]),
                            fedsa_gain=float(config["fedsa_gain"]),
                            fedsa_norm_ratio=float(config["fedsa_norm_ratio"]),
                            study="main",
                            ratio=None,
                            device=device,
                        )


def run_ratio(rounds: int, device: Any, config: Mapping[str, Any]) -> None:
    completed = done_ids(RAW_PATH)
    for seed in SEEDS:
        for ratio in RATIOS:
            for malicious in [RATIO_CLIENTS[ratio]]:
                for dataset in DATASETS:
                    for distribution in DISTRIBUTIONS:
                        for attack in ["S-DFA", "Sp-DFA"]:
                            run_and_record(
                                RAW_PATH,
                                completed,
                                dataset=dataset,
                                distribution=distribution,
                                method="GuardFed-AD2+",
                                attack=attack,
                                seed=seed,
                                malicious=malicious,
                                rounds=rounds,
                                fflip_mode=str(config["fflip_mode"]),
                                fedsa_gain=float(config["fedsa_gain"]),
                                fedsa_norm_ratio=float(config["fedsa_norm_ratio"]),
                                study="ratio",
                                ratio=ratio,
                                device=device,
                            )


def run_smoke(rounds: int, device: Any, config: Mapping[str, Any]) -> None:
    completed = done_ids(RAW_PATH)
    for dataset in DATASETS:
        for distribution in DISTRIBUTIONS:
            for method in ["FedAvg", "GuardFed-AD2+"]:
                for attack in ["Benign", "F Flip", "FedSA", "S-DFA", "Sp-DFA"]:
                    run_and_record(
                        RAW_PATH,
                        completed,
                        dataset=dataset,
                        distribution=distribution,
                        method=method,
                        attack=attack,
                        seed=123,
                        malicious=4,
                        rounds=rounds,
                        fflip_mode=str(config["fflip_mode"]),
                        fedsa_gain=float(config["fedsa_gain"]),
                        fedsa_norm_ratio=float(config["fedsa_norm_ratio"]),
                        study="smoke",
                        ratio=None,
                        device=device,
                    )


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return math.nan, math.nan
    return float(statistics.fmean(clean)), float(statistics.stdev(clean)) if len(clean) > 1 else 0.0


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def summarize() -> None:
    rows = [row for row in read_jsonl(RAW_PATH) if row.get("study") in {"main", "ratio"}]
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row.get("study"), row.get("dataset"), row.get("distribution"), row.get("method"), row.get("attack"), row.get("malicious_ratio"))
        grouped.setdefault(key, []).append(row)
    summary_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        study, dataset, distribution, method, attack, ratio = key
        for metric in METRICS:
            values = [float(row["metrics"].get(metric, math.nan)) for row in group]
            mean, std = mean_std(values)
            valid_count = sum(bool(row.get("validity", {}).get("valid_for_fairness", False)) for row in group)
            summary_rows.append({
                "study": study, "dataset": dataset, "distribution": distribution,
                "method": method, "attack": attack, "malicious_ratio": ratio,
                "metric": metric, "mean": mean, "std": std, "n": len(group),
                "valid_n": valid_count, "fairness_eligible": valid_count >= 8 or metric == "accuracy",
                "min": min(values) if values else math.nan, "max": max(values) if values else math.nan,
            })
        for row in group:
            for metric in METRICS:
                seed_rows.append({
                    "study": study, "dataset": dataset, "distribution": distribution,
                    "method": method, "attack": attack, "malicious_ratio": ratio,
                    "seed": row.get("seed"), "metric": metric,
                    "value": row["metrics"].get(metric),
                    "valid_for_fairness": row.get("validity", {}).get("valid_for_fairness"),
                    "validity_reasons": ";".join(row.get("validity", {}).get("reasons", [])),
                })
            for point in row.get("trajectory_metrics", []):
                for metric in METRICS:
                    trajectory_rows.append({
                        "study": study, "dataset": dataset, "distribution": distribution,
                        "method": method, "attack": attack, "malicious_ratio": ratio,
                        "seed": row.get("seed"), "round": point.get("round"),
                        "metric": metric, "value": point.get("metrics", {}).get(metric),
                    })
            for audit in row.get("attack_audit", []):
                audit_rows.append({
                    "study": study, "dataset": dataset, "distribution": distribution,
                    "method": method, "attack": attack, "malicious_ratio": ratio,
                    "seed": row.get("seed"), "client_id": audit.get("client_id"),
                    "is_malicious": audit.get("is_malicious"), "attack_types": "+".join(audit.get("attack_types", [])),
                    "samples": audit.get("samples"), "fflip_mode": audit.get("fflip_mode"),
                    "fflip_changed_ratio": audit.get("fflip_ratio"),
                    "fflip_corr_before": audit.get("fflip_label_corr_before"),
                    "fflip_corr_after": audit.get("fflip_label_corr_after"),
                    "fflip_corr_delta": (
                        float(audit["fflip_label_corr_after"]) - float(audit["fflip_label_corr_before"])
                        if audit.get("fflip_label_corr_before") is not None and audit.get("fflip_label_corr_after") is not None
                        else None
                    ),
                    "label_changed_count": audit.get("label_changed_count"),
                    "foe_mode": audit.get("foe_mode"), "foe_norm_ratio": audit.get("foe_norm_ratio"),
                    "foe_pre_post_cosine": audit.get("foe_pre_post_cosine"),
                })
    write_csv(SUMMARY_PATH, summary_rows, ["study", "dataset", "distribution", "method", "attack", "malicious_ratio", "metric", "mean", "std", "n", "valid_n", "fairness_eligible", "min", "max"])
    write_csv(SEED_RESULTS_PATH, seed_rows, ["study", "dataset", "distribution", "method", "attack", "malicious_ratio", "seed", "metric", "value", "valid_for_fairness", "validity_reasons"])
    write_csv(TRAJECTORY_PATH, trajectory_rows, ["study", "dataset", "distribution", "method", "attack", "malicious_ratio", "seed", "round", "metric", "value"])
    write_csv(AUDIT_PATH, audit_rows, ["study", "dataset", "distribution", "method", "attack", "malicious_ratio", "seed", "client_id", "is_malicious", "attack_types", "samples", "fflip_mode", "fflip_changed_ratio", "fflip_corr_before", "fflip_corr_after", "fflip_corr_delta", "label_changed_count", "foe_mode", "foe_norm_ratio", "foe_pre_post_cosine"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["calibrate", "smoke", "main", "ratio", "summarize"], required=True)
    parser.add_argument("--rounds", type=int, default=70)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    args = parser.parse_args()
    if args.phase == "summarize":
        summarize()
        return 0
    device = core.choose_device(args.device)
    core.run_validation()
    config = run_calibration(args.rounds, device)
    print(json.dumps(config, indent=2), flush=True)
    if args.phase == "calibrate":
        summarize()
    elif args.phase == "smoke":
        run_smoke(min(args.rounds, 1), device, config)
        summarize()
    elif args.phase == "main":
        run_main(args.rounds, device, config)
        summarize()
    elif args.phase == "ratio":
        run_ratio(args.rounds, device, config)
        summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
