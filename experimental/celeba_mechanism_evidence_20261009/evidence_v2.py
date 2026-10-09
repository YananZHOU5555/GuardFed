"""Inspect accepted terminal evidence, summarize it, and back up only verified deltas.

No training, protocol freezing, test access, monitoring, or prediction postprocessing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import platform
import socket
import statistics
import sys
import tarfile

VARIANTS = ["Full", "minus_U", "minus_C", "minus_A", "minus_F", "minus_V", "minus_N", "no_hard_screen", "fixed_balanced"]
SEEDS = list(range(91001, 91011))
DISTRIBUTIONS = {"IID": 5000.0, "non-IID": 5.0}
ATTACKS = ["Benign", "F Flip", "FedSA", "S-DFA", "Sp-DFA"]
METRICS = ["accuracy_pct", "aeod", "aspd"]
CORE_SHA = "cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed"
FULL_INVENTORY_SHA = "3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd"
VALID_IDS_SHA = "64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf"
VALID_SUPPORT = {"0": {"n": 11409, "positives": 6157}, "1": {"n": 8458, "positives": 3445}}
SUPPORT_REFERENCE_SHA = "374c55aa5788bd2a068a7f15cad2b9c31d5637f75ee1925ae9b57348977a3107"
SCIENCE_FILES = {
    "scripts/reproduce_paper_tables.py", "src/data_loader.py", "src/celeba_data.py", "scripts/build_celeba_cache.py",
    "data/celeba/derived/rgb64_v1/manifest.json", "data/celeba/derived/rgb64_v1/metadata.npz",
    "data/celeba/derived/rgb64_v1/images.npy", "data/celeba/derived/rgb64_v1/available.npy",
    "data/celeba/list_attr_celeba.txt", "data/celeba/list_eval_partition.txt",
}
IGNORE_RECIPE = {"seed", "client_alpha", "ablation_component", "experiment_suite", "experiment_tag", "full_round_diagnostics"}
DISCLOSURES = [
    "Full reuses100 accepted models: 2 trained with torch2.11.0+cu130,98 with cu128; new controls require cu128. Current driver595 is a different runtime context; no CUDA/driver equivalence or identical trajectories is claimed.",
    "Seed91001 participated in recipe selection. All10 declared seeds remain visible; the other9 were also previously observed validation seeds, not prospectively untouched confirmation.",
    "fixed_balanced removes candidate inference calls and can change the later RNG stream; compare complete procedures, not coefficient-only causal effects.",
    "Negative and constant predictions are retained. These controls do not establish that every component is necessary.",
    "raw/native/shared calibration of all900 terminal models remains a separate PENDING stage; neither prepared calibration nor a successful process exit completes it or the rebuttal.",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validators(repo, adapter_dir, manifest):
    require(not sys.flags.optimize, "Existing checked validators require assertions enabled")
    for name, expected in manifest["adapter_hashes"].items():
        require(digest(adapter_dir / Path(name).name) == expected, "Adapter source drift: " + name)
    original_path = repo / "scripts/run_revision_ablation.py"
    require(digest(original_path) == manifest["source_hashes"]["scripts/run_revision_ablation.py"], "Original checked_result source drift")
    sys.path.insert(0, str(adapter_dir))
    original = load("mechanism_evidence_original", original_path)
    worker = load("mechanism_evidence_checked_worker", adapter_dir / "worker.py")
    return original, worker


def cell(job, variant=None):
    return (variant or job["variant"], job["distribution"], job["attack"], job.get("seed", job.get("config", {}).get("seed")))


def validate_design(manifest, jobs):
    require(manifest["variants"] == VARIANTS and manifest["seeds"] == SEEDS, "Variant/seed cohort differs from the prepared design")
    require(manifest["distributions"] == DISTRIBUTIONS and manifest["attacks"] == ATTACKS, "Scenario cohort drift")
    require(manifest["rounds"] == 70 and manifest["evaluation_split"] == "valid", "Only terminal70 valid evidence is accepted")
    require((manifest["planned_new"], manifest["planned_reused"], manifest["total_model_records"]) == (800, 100, 900), "Wrong design totals")
    entries = manifest["jobs"] + manifest["reused_full"]
    require(len(entries) == 900 and len({entry["id"] for entry in entries}) == 900, "Duplicate or missing model IDs")
    require(len(jobs) == 800 and len(manifest["jobs"]) == 800 and len(manifest["reused_full"]) == 100, "Missing declared job/Full record")
    expected = set(itertools.product(VARIANTS[1:], DISTRIBUTIONS, ATTACKS, SEEDS))
    require(len({cell(job) for job in jobs}) == 800 and {cell(job) for job in jobs} == expected, "New controls contain duplicate cells or missing seeds/scenes")
    full = {cell(entry, "Full") for entry in manifest["reused_full"]}
    require(len(full) == 100 and full == set(itertools.product(["Full"], DISTRIBUTIONS, ATTACKS, SEEDS)), "Full reuse contains duplicate cells or missing seeds/scenes")
    template = {key: value for key, value in jobs[0]["config"].items() if key not in IGNORE_RECIPE}
    for entry, job in zip(manifest["jobs"], jobs):
        require(job["id"] == entry["id"] and job["variant"] == entry["variant"] and job["output"] == entry["output"], "Declared job identity mismatch")
        require(job["dataset"] == "celeba" and job["method"] == "GuardFed-AD2+", "Wrong dataset/method")
        require(job["source_hashes"] == manifest["source_hashes"] and job["adapter_hashes"] == manifest["adapter_hashes"], "Job source/adapter identity mismatch")
        require(job["protocol_sha256"] == manifest["protocol_sha256"], "Job protocol identity mismatch")
        require(job["config"]["client_alpha"] == DISTRIBUTIONS[job["distribution"]], "Distribution alpha mismatch")
        require(job["config"]["ablation_component"] == (job["variant"][-1] if job["variant"].startswith("minus_") else "none"), "Variant mask/config mismatch")
        require({key: value for key, value in job["config"].items() if key not in IGNORE_RECIPE} == template, "Non-variant recipe drift")
    require(template["rounds"] == 70 and template["celeba_evaluation_split"] == "valid", "Invalid recipe horizon/split")


def metrics_ok(metrics):
    require(all(type(metrics.get(key)) in (int, float) and math.isfinite(metrics[key]) and 0 <= metrics[key] <= 1
                for key in ("accuracy", "aeod", "aspd")), "Incomplete, undefined or nonfinite metrics")


def terminal_checks(result, job, manifest, variant):
    """Supplement the existing checked path with explicit protocol support/round checks."""
    require(result["dataset"] == "celeba" and result["method"] == "GuardFed-AD2+" and result["rounds"] == 70, "Wrong result dataset/method/horizon")
    require(result["seed"] == job["config"]["seed"] and result["distribution"] == job["distribution"] and result["attack"] == job["attack"], "Seed/scenario result drift")
    require(result["alpha"] == DISTRIBUTIONS[result["distribution"]], "Result alpha drift")
    require([row["round"] for row in result["round_summaries"]] == list(range(1, 71)), "Diagnostic rounds are incomplete or duplicated")
    require(result["metrics"] == result["trajectory_metrics"][-1]["metrics"], "Terminal metrics do not share the final checkpoint")
    for row in result["trajectory_metrics"]:
        metrics_ok(row["metrics"])
    metrics_ok(result["metrics"])
    cfg = result["config"]
    require(cfg["celeba_evaluation_split"] == "valid" and cfg["rounds"] == 70 and cfg["num_clients"] == 20 and cfg["num_malicious"] == 4, "Result protocol configuration drift")
    contract = result["data_contract"]["image_data_contract"]
    require((contract["evaluation_split"], contract["actual_train_rows"], contract["actual_evaluation_rows"]) == ("valid", 162770, 19867), "Wrong official train/valid support")
    require(contract["train_eval_disjoint"] and contract["root_client_disjoint"], "Data partition overlap")
    require(contract["evaluation_image_ids_sha256"] == VALID_IDS_SHA, "Validation sample/order identity drift")
    require(contract["cache_manifest_sha256"] == manifest["source_hashes"]["data/celeba/derived/rgb64_v1/manifest.json"], "Cache identity drift")
    stats = result["evaluation_stats"]
    require(stats["prediction_count"] == 19867 and 0 <= stats["positive_rate"] <= 1 and math.isfinite(stats["positive_rate"]), "Incomplete prediction support")
    expected_component = variant[-1] if variant.startswith("minus_") else "none"
    for row in result["round_summaries"]:
        aggregate = row["aggregate"]
        weights = aggregate["client_weights"]
        require(len(weights) == 20 and all(math.isfinite(w) and w >= 0 for w in weights) and abs(sum(weights) - 1) < 1e-6, "Invalid round aggregation weights")
        if expected_component in {"U", "C", "A", "F", "V"}:
            require(len(aggregate["component_contributions"]) == 20 and all(t[expected_component] == 0 for t in aggregate["component_contributions"]), "Stored selected candidate violates its component mask")
        if expected_component == "N":
            require(aggregate["norm_clip_scales"] and all(scale == 1 for scale in aggregate["norm_clip_scales"]), "Norm control retained a scale")
        if variant == "no_hard_screen":
            require(aggregate["hard_gate_clients"] == aggregate["selected_clients"] == list(range(20)), "No-screen control excluded clients")
        if variant == "fixed_balanced":
            require(aggregate.get("ad2_plus_mode") == "fixed_balanced_without_candidate_selection" and aggregate.get("fixed_candidate") == "balanced", "Fixed-balanced selector identity drift")
    return contract


def partition_identity(result, full_record):
    require(isinstance(full_record, dict), "Independent paired Full inventory record is required")
    expected = full_record["data_contract"]
    actual = result["data_contract"]["image_data_contract"]
    for key in ("root_image_ids_sha256", "train_image_ids_sha256", "evaluation_image_ids_sha256", "cache_manifest_sha256",
                "actual_train_rows", "actual_evaluation_rows", "official_split_sizes", "client_sample_counts"):
        require(key in actual and actual[key] == expected[key], "Paired Full partition identity mismatch: " + key)
    counts = expected["client_sample_counts"]
    require(len(counts) == 20 and all(type(value) is int and value >= 0 for value in counts), "Independent client sample support is incomplete")
    expected_root_n = expected["actual_train_rows"] - sum(counts)
    data = result["data_contract"]
    require(data.get("root_clean_rows") == expected_root_n and expected_root_n > 0 and data.get("root_synthetic_rows") == 0, "Root clean/synthetic sample support differs from Full")
    require(data.get("train_rows") == 162770 and data.get("test_rows") == 19867, "Top-level partition sample counts drifted")
    for key in ("root_counts", "valid_counts", "sensitive_group_counts", "root_group_counts", "evaluation_group_counts"):
        if key in actual or key in expected:
            require(key in actual and key in expected and actual[key] == expected[key], "Stored sensitive support differs from Full: " + key)


def accept_new(original, worker, job, entry, manifest, full_record):
    path = Path(entry["job"])
    require(digest(path) == entry["job_sha256"], "Job bytes changed")
    result = worker.checked(original, job, path)
    if result is None:
        require(not Path(job["output"]).exists(), "Partial output must be preserved and reviewed")
        return None
    terminal_checks(result, job, manifest, job["variant"])
    require((full_record["distribution"], full_record["attack"], full_record["seed"], full_record["method"]) ==
            (job["distribution"], job["attack"], job["config"]["seed"], "GuardFed-AD2+"), "Paired Full record is from a different seed/scenario")
    partition_identity(result, full_record)
    provenance = result["revision_job"]
    require(provenance["adapter_hashes"] == manifest["adapter_hashes"] and provenance["protocol_sha256"] == manifest["protocol_sha256"], "Result adapter/protocol drift")
    require(provenance["torch_version"] == "2.11.0+cu128", "New formal controls require the isolated cu128 environment")
    ledger = read(Path(job["output"]) / "candidate_mask_audit.json")
    component = job["variant"][-1] if job["variant"].startswith("minus_") else "none"
    require(all(item["component"] == component and item["clients"] == 20 and 1 <= item["selected_count"] <= 20 for item in ledger), "Candidate ledger fields contradict the frozen intervention")
    if job["variant"] == "no_hard_screen":
        require(all(item["selected_count"] == 20 for item in ledger), "A candidate excluded clients in no-screen")
    return result


def accept_full(original, entry, record, manifest, template):
    out = Path(entry["output"])
    require(not list(out.glob("failure*.json")), "Preserved Full failure requires review")
    if not (out / "result.json").exists():
        return None
    require(digest(out / "result.json") == record["result"]["sha256"], "Reused Full result differs from its accepted backup chain")
    require(entry["checkpoint_sha256"] == record["checkpoint"]["sha256"] == digest(out / "model.pt"), "Reused Full checkpoint drift")
    job = {"id": PurePosixPath(record["raw_job"]["member"]).stem, "output": str(out), "config": record["config"],
           "source_hashes": record["source_hashes"], "distribution": entry["distribution"], "attack": entry["attack"]}
    result = original.checked_result(job)
    require(result is not None, "Missing reused Full checked result")
    terminal_checks(result, job, manifest, "Full")
    require(result["config"]["ablation_component"] == "none", "Reused Full contains an ablation mask")
    contract = result["data_contract"]["image_data_contract"]
    require(all(contract[key] == record["data_contract"][key] for key in
                ("root_image_ids_sha256", "train_image_ids_sha256", "evaluation_image_ids_sha256", "cache_manifest_sha256")), "Reused Full root/train/valid identity drift")
    require((record["method"], record["distribution"], record["attack"], record["seed"], record["actual_alpha"]) ==
            (entry["method"], entry["distribution"], entry["attack"], entry["seed"], DISTRIBUTIONS[entry["distribution"]]), "Full inventory scientific identity mismatch")
    require(all(result["metrics"][key] == entry[key] for key in ("accuracy", "aeod", "aspd")), "Full metrics differ from accepted snapshot")
    require(SCIENCE_FILES <= set(record["source_hashes"]) and
            all(manifest["source_hashes"].get(name) == record["source_hashes"][name] for name in SCIENCE_FILES), "Full scientific source/data lineage differs")
    require({key: value for key, value in record["config"].items() if key not in IGNORE_RECIPE} ==
            {key: value for key, value in template.items() if key not in IGNORE_RECIPE}, "Reused Full numerical recipe differs")
    import torch
    state = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    require(state and all(torch.isfinite(value).all().item() for value in state.values()), "Nonfinite reused checkpoint tensors")
    return result


def row_from_result(entry, result, role):
    out = Path(entry["output"])
    files = [out / "result.json", out / "model.pt"]
    if role == "new":
        files += [Path(entry["job"]), out / "candidate_mask_audit.json", out / "mechanism_acceptance.json"]
        for path in [out / "progress.json", Path(entry["job"]).parents[1] / "logs" / (entry["id"] + ".log")]:
            if path.is_file():
                files.append(path)
    warnings = list(result.get("warnings", []))
    if result["evaluation_stats"]["positive_rate"] in (0, 1):
        warnings.append("Constant terminal predictions retained")
    return {"id": entry["id"], "role": role, "variant": entry.get("variant", "Full"),
            "distribution": result["distribution"], "attack": result["attack"], "seed": result["seed"],
            "accuracy_pct": 100 * result["metrics"]["accuracy"], "aeod": result["metrics"]["aeod"], "aspd": result["metrics"]["aspd"],
            "checkpoint_sha256": result["revision_job"]["checkpoint_sha256"], "torch_version": result["revision_job"]["torch_version"],
            "prediction_support": result["evaluation_stats"], "group_denominator_support": VALID_SUPPORT,
            "warnings": warnings, "output": str(out), "job": entry.get("job"),
            "files": {str(path): digest(path) for path in files}}


def statistic(rows, expected_n):
    require(len({row["seed"] for row in rows}) == len(rows), "Duplicate seeds would inflate statistics")
    return {"n": len(rows), "expected_n": expected_n, "complete": len(rows) == expected_n,
            "seeds": sorted(row["seed"] for row in rows),
            **{metric: {"mean": statistics.mean(row[metric] for row in rows) if rows else None,
                        "sample_sd_ddof1": statistics.stdev(row[metric] for row in rows) if len(rows) > 1 else None}
               for metric in METRICS}}


def summarize(rows):
    keys = [(row["variant"], row["distribution"], row["attack"], row["seed"]) for row in rows]
    require(len(set(keys)) == len(keys), "Duplicate accepted cells")
    scene = []
    for variant, dist, attack in itertools.product(VARIANTS, DISTRIBUTIONS, ATTACKS):
        selected = [row for row in rows if (row["variant"], row["distribution"], row["attack"]) == (variant, dist, attack)]
        scene.append({"variant": variant, "distribution": dist, "attack": attack, **statistic(selected, 10)})
    full = {(row["distribution"], row["attack"], row["seed"]): row for row in rows if row["variant"] == "Full"}
    paired = []
    for row in rows:
        if row["variant"] == "Full":
            continue
        control = full.get((row["distribution"], row["attack"], row["seed"]))
        if control is not None:
            paired.append({key: row[key] for key in ("id", "variant", "distribution", "attack", "seed")} |
                          {metric: row[metric] - control[metric] for metric in METRICS})
    paired_scene = [{"variant": variant, "distribution": dist, "attack": attack,
                     **statistic([row for row in paired if (row["variant"], row["distribution"], row["attack"]) == (variant, dist, attack)], 10)}
                    for variant, dist, attack in itertools.product(VARIANTS[1:], DISTRIBUTIONS, ATTACKS)]
    cross_seed, cross_summary = [], []
    for name, scenarios in [("all10scenarios", list(itertools.product(DISTRIBUTIONS, ATTACKS))),
                             *[(dist + "_all5attacks", [(dist, attack) for attack in ATTACKS]) for dist in DISTRIBUTIONS]]:
        for kind, source, variants in [("reported_native", rows, VARIANTS), ("paired_delta_to_Full", paired, VARIANTS[1:])]:
            for variant in variants:
                complete = []
                for seed in SEEDS:
                    selected = [row for row in source if row["variant"] == variant and row["seed"] == seed and (row["distribution"], row["attack"]) in scenarios]
                    item = {"panel": name, "kind": kind, "variant": variant, "seed": seed,
                            "n_scenarios": len(selected), "expected_scenarios": len(scenarios), "complete": len(selected) == len(scenarios)}
                    if item["complete"]:
                        item.update({metric: statistics.mean(row[metric] for row in selected) for metric in METRICS})
                        complete.append(item)
                    cross_seed.append(item)
                cross_summary.append({"panel": name, "kind": kind, "variant": variant, **statistic(complete, 10)})
    return {"per_scene": scene, "paired_per_seed": paired, "paired_per_scene": paired_scene,
            "cross_scenario_per_seed": cross_seed, "cross_scenario_summary": cross_summary,
            "units": {"accuracy_pct": "percent; paired difference is percentage points", "aeod": "absolute TPR gap", "aspd": "absolute positive-rate gap"},
            "pair_direction": "variant minus Full; negative AEOD/ASPD difference is lower disparity",
            "partial_policy": "Each summary exposes n. Cross-scenario statistics include only seeds with every declared scenario in that panel; partial panels stay visible without a mean."}


def inspect(repo, stage, adapter_dir, inventory_path, output):
    require(not output.exists(), "Use a fresh inspection directory to preserve prior snapshots")
    output.mkdir(parents=True)
    manifest = read(stage / "manifest.json")
    report = {"status": "INVALID", "accepted_new_ids": [], "accepted_reused_ids": [], "records": [], "invalid": [], "pending": [],
              "manifest": str(stage / "manifest.json"), "manifest_sha256": digest(stage / "manifest.json"), "stage": str(stage), "repo": str(repo),
              "source_script_sha256": digest(__file__), "whole_rebuttal_complete": False, "postprocessing_raw_native_shared": "PENDING", "disclosures": DISCLOSURES}
    try:
        require(digest(stage / "PROTOCOL.md") == manifest["protocol_sha256"], "Protocol bytes drifted")
        require(digest(inventory_path) == FULL_INVENTORY_SHA, "Full inventory differs from the previously accepted identity chain")
        records = {record["id"]: record for record in read(inventory_path)["records"]}
        jobs = []
        for entry in manifest["jobs"]:
            require(digest(entry["job"]) == entry["job_sha256"], "Manifest/job bytes drift: " + entry["id"])
            jobs.append(read(entry["job"]))
        validate_design(manifest, jobs)
        require(manifest["source_hashes"]["scripts/reproduce_paper_tables.py"] == CORE_SHA, "Wrong frozen core")
        for name, expected in manifest["source_hashes"].items():
            require(digest(repo / name) == expected, "Live source/data drift: " + name)
        original, worker = validators(repo, adapter_dir, manifest)
        report["full_inventory"] = str(inventory_path)
        report["full_inventory_sha256"] = FULL_INVENTORY_SHA
        report["adapter_dir"] = str(adapter_dir)
        report["frozen_source_hashes"] = manifest["source_hashes"]
        report["adapter_hashes"] = manifest["adapter_hashes"]
        report["support_reference"] = {"published_accepted_json_sha256": SUPPORT_REFERENCE_SHA, "validation_ids_sha256": VALID_IDS_SHA,
                                       "valid_counts": VALID_SUPPORT, "new_confusion_matrices_computed": False}
        paired_full = {(entry["distribution"], entry["attack"], entry["seed"]): records[entry["id"]] for entry in manifest["reused_full"]}
        for role, entry, job in [("new", entry, job) for entry, job in zip(manifest["jobs"], jobs)] + [("reused", entry, None) for entry in manifest["reused_full"]]:
            try:
                result = accept_new(original, worker, job, entry, manifest, paired_full[(job["distribution"], job["attack"], job["config"]["seed"])]) if role == "new" else accept_full(original, entry, records[entry["id"]], manifest, jobs[0]["config"])
                if result is None:
                    report["pending"].append({"id": entry["id"], "role": role})
                    continue
                report["records"].append(row_from_result(entry, result, role))
                report["accepted_new_ids" if role == "new" else "accepted_reused_ids"].append(entry["id"])
            except Exception as error:
                out = Path(entry["output"])
                failures = {str(path): digest(path) for path in sorted(out.glob("failure*.json"))}
                for filename in ("result.json", "mechanism_acceptance.json", "candidate_mask_audit.json", "progress.json"):
                    if (out / filename).is_file():
                        failures[str(out / filename)] = digest(out / filename)
                if entry.get("job") and Path(entry["job"]).is_file():
                    failures[entry["job"]] = digest(entry["job"])
                log = stage / "logs" / (entry["id"] + ".log")
                if log.is_file():
                    failures[str(log)] = digest(log)
                report["invalid"].append({"id": entry["id"], "role": role, "error": repr(error), "preserved_files": failures})
        report["accepted_new_ids"].sort()
        report["accepted_reused_ids"].sort()
        report["new_count"] = len(report["accepted_new_ids"])
        report["reused_count"] = len(report["accepted_reused_ids"])
        report["status"] = "COMPLETE_MODEL_EVIDENCE_ONLY" if (report["new_count"], report["reused_count"], len(report["invalid"]), len(report["pending"])) == (800, 100, 0, 0) else "PARTIAL"
        report["torch_counts_by_role"] = {role: dict(Counter(row["torch_version"] for row in report["records"] if row["role"] == role)) for role in ("new", "reused")}
        report["historical_worker_sources"] = {entry["id"]: records[entry["id"]]["source_hashes"].get("scripts/run_revision_ablation.py") for entry in manifest["reused_full"]}
        report["reused_full_restore_chain"] = [{"id": entry["id"], **{key: records[entry["id"]][key] for key in ("checkpoint", "result", "raw_job")}} for entry in manifest["reused_full"]]
        if (stage / "dispatch_receipt.json").is_file():
            receipt = read(stage / "dispatch_receipt.json")
            require(receipt["manifest_sha256"] == report["manifest_sha256"], "Dispatch receipt manifest drift")
            report["dispatch_environment"] = {key: receipt.get(key) for key in ("torch_version", "cuda_build", "gpu_snapshot", "guide_sha256")}
        summary = summarize(report["records"])
        excluded = [entry for entry in manifest["reused_full"] if entry["torch_version"] != "2.11.0+cu128"]
        require(len(excluded) == 2 and all(entry["torch_version"] == "2.11.0+cu130" for entry in excluded), "Declared Full runtime mixture changed")
        excluded_ids = {entry["id"] for entry in excluded}
        sensitivity_rows = [row for row in report["records"] if row["id"] not in excluded_ids]
        summary["runtime_sensitivity"] = {
            "excluded_Full_records": [{key: entry[key] for key in ("id", "distribution", "attack", "seed", "torch_version")} for entry in excluded],
            "boundary": "Exclude the2 cu130 Full models and each corresponding variant-to-Full pair. Keep new-control raw observations. Driver equivalence remains unverified.",
            "full_cohort_complete_boundary": {"Full_records": 98, "paired_new_records": 784, "affected_scene_seed_n": 9,
                                               "all_scenarios_complete_seed_n": 9, "non_IID_all_attacks_complete_seed_n": 9, "IID_all_attacks_complete_seed_n": 10},
            "statistics": summarize(sensitivity_rows)}
        save(output / "statistics.json", summary)
        flat = [{key: row[key] for key in ("variant", "distribution", "attack", "n", "expected_n", "complete")} |
                {metric + "_" + field: row[metric][field] for metric in METRICS for field in ("mean", "sample_sd_ddof1")}
                for row in summary["per_scene"]]
        with (output / "per_scene_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        for name, rows in [("per_seed.csv", report["records"]), ("paired_per_seed.csv", summary["paired_per_seed"])]:
            fields = ["id", "variant", "distribution", "attack", "seed", *METRICS]
            with (output / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
    except Exception as error:
        report.update(status="INVALID", fatal_error=repr(error))
        report["accepted_new_ids"] = []
        report["accepted_reused_ids"] = []
    save(output / "inspection.json", report)
    (output / "inspection.sha256").write_text(digest(output / "inspection.json") + "\n", encoding="utf-8")
    return report


def inspect_full(repo, stage, inventory_path, output):
    """Read-only prerequisite for dispatch; the original100 models stay in place."""
    require(not output.exists(), "Use a fresh Full inspection directory")
    output.mkdir(parents=True)
    manifest = read(stage / "manifest.json")
    report = {"status": "INVALID", "accepted_reused_ids": [], "invalid": [], "pending": [],
              "manifest_sha256": digest(stage / "manifest.json"), "inventory_sha256": digest(inventory_path),
              "inspector_sha256": digest(__file__), "new_training_started": False, "test_accessed": False,
              "postprocessing_raw_native_shared": "PENDING", "disclosures": DISCLOSURES}
    try:
        require(report["inventory_sha256"] == FULL_INVENTORY_SHA, "Independent Full inventory SHA mismatch")
        require(digest(stage / "PROTOCOL.md") == manifest["protocol_sha256"], "Protocol bytes drift")
        expected = set(itertools.product(["Full"], DISTRIBUTIONS, ATTACKS, SEEDS))
        entries = manifest["reused_full"]
        require(len(entries) == 100 and len({entry["id"] for entry in entries}) == 100 and
                {cell(entry, "Full") for entry in entries} == expected, "Full100 cohort has duplicate IDs or missing cells/seeds")
        require(manifest["source_hashes"]["scripts/reproduce_paper_tables.py"] == CORE_SHA, "Wrong frozen core")
        actual_sources = {}
        for name in sorted(SCIENCE_FILES | {"scripts/run_revision_ablation.py"}):
            actual_sources[name] = digest(repo / name)
            require(actual_sources[name] == manifest["source_hashes"][name], "Live scientific input hash drift: " + name)
        report["live_scientific_input_hashes"] = actual_sources
        original = load("mechanism_full_strict_original", repo / "scripts/run_revision_ablation.py")
        records = {record["id"]: record for record in read(inventory_path)["records"]}
        template = read(manifest["jobs"][0]["job"])["config"]
        report["full_identity_receipts"] = []
        for entry in entries:
            try:
                record = records[entry["id"]]
                result = accept_full(original, entry, record, manifest, template)
                if result is None:
                    report["pending"].append(entry["id"])
                    continue
                report["accepted_reused_ids"].append(entry["id"])
                report["full_identity_receipts"].append({"id": entry["id"], "seed": result["seed"],
                      "distribution": result["distribution"], "attack": result["attack"], "alpha": result["alpha"],
                      "checkpoint_sha256": record["checkpoint"]["sha256"], "result_sha256": record["result"]["sha256"],
                      "raw_job_identity": record["raw_job"], "torch_version": result["revision_job"]["torch_version"],
                      "historical_worker_sha256": record["source_hashes"].get("scripts/run_revision_ablation.py"),
                      "root_image_ids_sha256": record["data_contract"]["root_image_ids_sha256"], "metrics": result["metrics"]})
            except Exception as error:
                report["invalid"].append({"id": entry["id"], "error": repr(error),
                      "preserved_failures": {str(path): digest(path) for path in Path(entry["output"]).glob("failure*.json")}})
        report["accepted_reused_ids"].sort()
        report["accepted_reused_count"] = len(report["accepted_reused_ids"])
        report["status"] = "FULL_REUSE_VERIFIED_100" if report["accepted_reused_count"] == 100 and not report["invalid"] and not report["pending"] else "PARTIAL_OR_INVALID_FULL_REUSE"
        report["torch_counts"] = dict(Counter(row["torch_version"] for row in report["full_identity_receipts"]))
        report["runtime_sensitivity_exclusions"] = [{key: row[key] for key in ("id", "distribution", "attack", "seed", "torch_version")}
                for row in report["full_identity_receipts"] if row["torch_version"] != "2.11.0+cu128"]
    except Exception as error:
        report.update(status="INVALID", fatal_error=repr(error))
    save(output / "full_inspection.json", report)
    (output / "full_inspection.sha256").write_text(digest(output / "full_inspection.json") + "\n", encoding="utf-8")
    return report


def safe_member(name):
    path = PurePosixPath(name)
    require(name and not name.startswith("/") and ".." not in path.parts and "\\" not in name and ":" not in name, "Unsafe archive member: " + name)


def verify_archive(archive, receipt):
    require(digest(archive) == receipt["archive_sha256"], "Archive SHA mismatch")
    seen, inventory = set(), None
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            safe_member(member.name)
            require(member.isfile() and member.name not in seen, "Duplicate or nonregular archive member")
            seen.add(member.name)
            stream = handle.extractfile(member)
            if member.name == "backup_inventory.json":
                data = stream.read()
                require(hashlib.sha256(data).hexdigest() == receipt["inventory_sha256"], "Inventory bytes drift")
                inventory = json.loads(data)
                require(inventory["accepted_new_ids"] == receipt["accepted_new_ids"], "Receipt/archive ID mismatch")
                require(len(set(inventory["accepted_new_ids"])) == len(inventory["accepted_new_ids"]), "Duplicate backed-up IDs")
            else:
                require(inventory is not None and member.name in inventory["members"], "Undeclared archive member")
                expected = inventory["members"][member.name]
                hasher = hashlib.sha256()
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    hasher.update(block)
                require(hasher.hexdigest() == expected["sha256"] and member.size == expected["bytes"], "Archive member SHA/size mismatch: " + member.name)
    require(inventory is not None and seen == {"backup_inventory.json", *inventory["members"]}, "Archive is incomplete")
    return {"pass": True, "archive_sha256": receipt["archive_sha256"], "members_verified": len(seen),
            "accepted_new_ids": receipt["accepted_new_ids"], "verification_host": socket.gethostname(),
            "source_host": receipt["source_host"], "different_host_observed": socket.gethostname() != receipt["source_host"]}


def verified_ledger(path, manifest_sha, planned_ids, initialize=False):
    if not path.exists():
        require(initialize, "Missing backup ledger; first use requires --initialize-ledger")
        save(path, {"schema": "celeba_mechanism_incremental_backup_ledger_v1", "manifest_sha256": manifest_sha, "entries": []})
    ledger = read(path)
    require(ledger["schema"] == "celeba_mechanism_incremental_backup_ledger_v1" and ledger["manifest_sha256"] == manifest_sha, "Backup ledger cohort drift")
    ids, failures, previous = set(), set(), None
    for entry in ledger["entries"]:
        require(digest(entry["receipt"]) == entry["receipt_sha256"], "Backup receipt SHA mismatch")
        receipt = read(entry["receipt"])
        require(receipt["previous_receipt_sha256"] == previous and receipt["manifest_sha256"] == manifest_sha, "Broken restore chain")
        require(set(receipt["accepted_new_ids"]) <= planned_ids and not ids.intersection(receipt["accepted_new_ids"]), "Duplicate or undeclared backed-up ID")
        verify_archive(Path(entry["archive"]), receipt)
        ids.update(receipt["accepted_new_ids"])
        failures.update(receipt.get("failure_identities", []))
        previous = entry["receipt_sha256"]
    return ledger, ids, failures, previous


def backup(inspection_path, ledger_path, archive_path, initialize=False):
    require(digest(inspection_path) == inspection_path.with_name("inspection.sha256").read_text().strip(), "Inspection snapshot was changed")
    report = read(inspection_path)
    require(report["status"] in ("PARTIAL", "COMPLETE_MODEL_EVIDENCE_ONLY") and report["source_script_sha256"] == digest(__file__), "Inspection is invalid or used different acceptance source")
    manifest = read(report["manifest"])
    require(digest(report["manifest"]) == report["manifest_sha256"], "Manifest drift after inspection")
    planned = {entry["id"] for entry in manifest["jobs"]}
    accepted = set(report["accepted_new_ids"])
    require(len(accepted) == len(report["accepted_new_ids"]) and accepted <= planned, "Duplicate/undeclared accepted IDs")
    new_rows = [row for row in report["records"] if row["role"] == "new"]
    rows = {row["id"]: row for row in new_rows}
    require(set(rows) == accepted and len(rows) == len(new_rows), "Inspection accepted IDs do not match unique new records")
    entries = {entry["id"]: entry for entry in manifest["jobs"]}
    for identity, row in rows.items():
        require(row["output"] == entries[identity]["output"] and row["job"] == entries[identity]["job"], "Backup row refers to a different job/model path")
        require(row["files"].get(str(Path(row["output"]) / "model.pt")) == row["checkpoint_sha256"], "Backup row mixes checkpoints")
    ledger, already, prior_failures, previous = verified_ledger(ledger_path, report["manifest_sha256"], planned, initialize)
    delta = sorted(accepted - already)
    failure_files = {path: expected for row in report["invalid"] for path, expected in row["preserved_files"].items()}
    fresh_failures = {path: expected for path, expected in failure_files.items() if path + "=" + expected not in prior_failures}
    if not delta and not fresh_failures:
        return {"status": "NO_NEW_VERIFIED_EVIDENCE", "accepted_new_ids": [], "already_backed_up": len(already)}
    require(not archive_path.exists() and not Path(str(archive_path) + ".partial").exists()
            and not Path(str(archive_path) + ".receipt.json").exists(), "Never overwrite an archive, receipt or partial attempt")
    stage, repo = Path(report["stage"]), Path(report["repo"])
    dispatch = stage / "dispatch_receipt.json"
    require(dispatch.is_file() and read(dispatch)["manifest_sha256"] == report["manifest_sha256"], "Formal sourcefreeze requires the existing dispatch receipt")
    contents = {}
    def add_file(name, path, expected=None):
        safe_member(name)
        require(name not in contents and path.is_file(), "Missing/duplicate backup input: " + str(path))
        actual = digest(path)
        require(expected is None or actual == expected, "Input changed since acceptance: " + str(path))
        contents[name] = (path, actual, path.stat().st_size)
    def add_bytes(name, data):
        safe_member(name)
        require(name not in contents, "Duplicate generated member")
        contents[name] = (data, hashlib.sha256(data).hexdigest(), len(data))
    for identity in delta:
        row = rows[identity]
        for path_string, expected in row["files"].items():
            path = Path(path_string)
            name = "jobs/" + identity + ".json" if path == Path(row["job"]) else "runs/" + identity + "/" + path.name
            add_file(name, path, expected)
        log = stage / "logs" / (identity + ".log")
        require(str(log) in row["files"], "Accepted run log is absent from inspection; inspect again after the worker closes it")
        add_bytes("runs/" + identity + "/config.json", encoded(read(row["job"])["config"]))
    for index, (path, expected) in enumerate(sorted(fresh_failures.items())):
        add_file("preserved_failures/" + str(index) + "_" + Path(path).name, Path(path), expected)
    for path in [stage / "manifest.json", stage / "PROTOCOL.md", dispatch, inspection_path,
                 inspection_path.with_name("statistics.json"), inspection_path.with_name("per_seed.csv"), inspection_path.with_name("paired_per_seed.csv"),
                 inspection_path.with_name("per_scene_summary.csv"),
                 Path(report["full_inventory"]), Path(__file__)]:
        add_file("sourcefreeze/" + path.name, path)
    for name, expected in report["frozen_source_hashes"].items():
        if not name.startswith("data/"):
            add_file("sourcefreeze/repo/" + name, repo / name, expected)
    for name, expected in report["adapter_hashes"].items():
        add_file("sourcefreeze/adapter/" + Path(name).name, Path(report["adapter_dir"]) / Path(name).name, expected)
    add_bytes("sourcefreeze/source_and_data_hashes.json", encoded(report["frozen_source_hashes"]))
    add_bytes("sourcefreeze/reused_full_restore_chain.json", encoded(report["reused_full_restore_chain"]))
    inventory = {"schema": "celeba_mechanism_incremental_backup_v1", "accepted_new_ids": delta,
                 "manifest_sha256": report["manifest_sha256"], "members": {name: {"sha256": value[1], "bytes": value[2]} for name, value in sorted(contents.items())},
                 "reused_full_weights_repacked": 0, "failure_identities": sorted(path + "=" + expected for path, expected in fresh_failures.items())}
    inventory_data = encoded(inventory)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(archive_path) + ".partial")
    try:
        with tarfile.open(partial, "w:gz") as archive:
            for name, source, size in [("backup_inventory.json", inventory_data, len(inventory_data)),
                                       *[(name, data[0], data[2]) for name, data in sorted(contents.items())]]:
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = size, 0o644, 0
                stream = source.open("rb") if isinstance(source, Path) else io.BytesIO(source)
                with stream:
                    archive.addfile(info, stream)
        receipt = {"schema": "celeba_mechanism_backup_receipt_v1", "archive_sha256": digest(partial), "inventory_sha256": hashlib.sha256(inventory_data).hexdigest(),
                   "manifest_sha256": report["manifest_sha256"], "inspection_sha256": digest(inspection_path), "accepted_new_ids": delta,
                   "previous_receipt_sha256": previous, "failure_identities": inventory["failure_identities"], "source_host": socket.gethostname(),
                   "reused_full_weights_repacked": 0, "off_server_verification": "PENDING_COPY_AND_VERIFY", "original_full_chain_preserved": True}
        verify_archive(partial, receipt)
        partial.replace(archive_path)
        receipt_path = Path(str(archive_path) + ".receipt.json")
        require(not receipt_path.exists(), "Never overwrite an existing receipt")
        save(receipt_path, receipt)
        ledger["entries"].append({"archive": str(archive_path), "receipt": str(receipt_path), "receipt_sha256": digest(receipt_path)})
        save(ledger_path, ledger)
        return {"status": "VERIFIED_LOCAL_BACKUP", "archive": str(archive_path), "receipt": str(receipt_path), "accepted_new_ids": delta,
                "members": len(contents) + 1, "off_server_verification": "PENDING_COPY_AND_VERIFY"}
    except Exception as error:
        save(Path(str(archive_path) + ".failure.json"), {"error": repr(error), "preserved_partial": str(partial), "accepted_new_ids": delta})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect_parser = sub.add_parser("inspect")
    for name in ("repo", "stage", "adapter-dir", "full-inventory", "output"):
        inspect_parser.add_argument("--" + name, type=Path, required=True)
    full_parser = sub.add_parser("inspect-full")
    for name in ("repo", "stage", "full-inventory", "output"):
        full_parser.add_argument("--" + name, type=Path, required=True)
    backup_parser = sub.add_parser("backup")
    for name in ("inspection", "ledger", "archive"):
        backup_parser.add_argument("--" + name, type=Path, required=True)
    backup_parser.add_argument("--initialize-ledger", action="store_true")
    verify_parser = sub.add_parser("verify")
    for name in ("archive", "receipt", "output"):
        verify_parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not sys.flags.optimize, "Assertions must remain enabled for reused validators")
    if args.action == "inspect-full":
        report = inspect_full(args.repo.resolve(), args.stage.resolve(), args.full_inventory.resolve(), args.output.resolve())
        print(json.dumps({key: report.get(key) for key in ("status", "accepted_reused_count", "fatal_error")}))
        if report["status"] != "FULL_REUSE_VERIFIED_100":
            raise SystemExit(1)
    elif args.action == "inspect":
        report = inspect(args.repo.resolve(), args.stage.resolve(), args.adapter_dir.resolve(), args.full_inventory.resolve(), args.output.resolve())
        print(json.dumps({key: report.get(key) for key in ("status", "new_count", "reused_count", "fatal_error")}))
        if report["status"] == "INVALID":
            raise SystemExit(1)
    elif args.action == "backup":
        print(json.dumps(backup(args.inspection.resolve(), args.ledger.resolve(), args.archive.resolve(), args.initialize_ledger)))
    else:
        report = verify_archive(args.archive.resolve(), read(args.receipt))
        require(not args.output.exists(), "Preserve previous off-server verification receipts")
        save(args.output, report)
        print(json.dumps(report))


if __name__ == "__main__":
    main()
