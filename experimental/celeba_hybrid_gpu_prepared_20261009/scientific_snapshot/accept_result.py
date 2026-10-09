"""Strict hybrid screen-result acceptance; never train, retry or select peaks."""
import json
import math
from pathlib import Path

from worker import ADAPTER_SOURCES, HERE, LABEL, METHOD, digest, validate_job


def checked_result(job_path, output):
    job_path, output = Path(job_path), Path(output)
    job = json.loads(job_path.read_text(encoding="utf8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    validate_job(job, protocol)
    if list(output.glob("failure*.json")):
        raise ValueError("Preserved failure evidence; diagnose before considering reuse")
    if not (output / "result.json").exists():
        return None
    result = json.loads((output / "result.json").read_text(encoding="utf8"))
    evidence = json.loads((output / "acceptance.json").read_text(encoding="utf8"))
    if evidence["status"] != "PASS" or result["status"] != "screen_complete":
        raise ValueError("Hybrid result has no complete acceptance")
    if (evidence["rounds"], evidence["evaluation_split"], evidence["train_rows"],
            evidence["evaluation_rows"], evidence["tuning_candidate"]) != (
            70, "valid", 162770, 19867, job["tuning_candidate"]):
        raise ValueError("Acceptance identity differs from frozen screen")
    names = {"model.pt", "diagnostics.json", "result.json", "provenance.json"}
    if set(evidence["artifact_hashes"]) != names:
        raise ValueError("Incomplete or unexpected result artifact hash set")
    for name, expected in evidence["artifact_hashes"].items():
        if digest(output / name) != expected:
            raise ValueError("Artifact hash mismatch: " + name)
    if json.loads((output / "job.json").read_text(encoding="utf8")) != job:
        raise ValueError("Stored input job differs from expected job")
    if result["revision_job"] != job or result["config"] != job["config"]:
        raise ValueError("Result/job config mismatch")
    if any(result[key] != job[key] for key in ("dataset", "method", "distribution", "attack", "tuning_candidate")):
        raise ValueError("Result/job metadata mismatch")
    if (result["method"], result["method_impl_note"], result["seed"], result["rounds"],
            result["alpha"], result["num_clients"], result["num_malicious"], result["evidence_stage"]) != (
            METHOD, LABEL, 91001, 70, job["config"]["client_alpha"], 20, 4, "validation_screen"):
        raise ValueError("Method, seed, alpha or horizon mismatch")
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf8"))
    if result["provenance"] != provenance or provenance["job_sha256"] != digest(job_path):
        raise ValueError("Input/checkpoint provenance mismatch")
    for key in ("source_hashes", "adapter_source_hashes"):
        if provenance[key] != job[key]:
            raise ValueError("Frozen source identity mismatch")
    if set(job["adapter_source_hashes"]) != ADAPTER_SOURCES:
        raise ValueError("Missing adapter identities")
    for name, expected in job["adapter_source_hashes"].items():
        if digest(HERE / name) != expected:
            raise ValueError("Adapter code/protocol changed: " + name)
    for name in ("trajectory_metrics", "round_summaries"):
        if [row["round"] for row in result[name]] != list(range(1, 71)):
            raise ValueError("Incomplete/duplicate round coverage")
    metrics = result["trajectory_metrics"]
    if result["metrics"] != metrics[-1]["metrics"] or not all(
            math.isfinite(row["metrics"][name]) and 0 <= row["metrics"][name] <= 1
            for row in metrics for name in ("accuracy", "aeod", "aspd")):
        raise ValueError("Invalid same-terminal-checkpoint metrics")
    contract = result["data_contract"]["image_data_contract"]
    if (contract["evaluation_split"], contract["actual_train_rows"], contract["actual_evaluation_rows"],
            contract["train_eval_disjoint"], contract["root_client_disjoint"],
            result["evaluation_stats"]["prediction_count"]) != ("valid", 162770, 19867, True, True, 19867):
        raise ValueError("Unexpected split/sample count/disjointness")
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf8"))
    if len(diagnostics) != 70 or any(row["round"] != i + 1 or row["aggregate"] != result["round_summaries"][i]["aggregate"]
                                   for i, row in enumerate(diagnostics)):
        raise ValueError("Saved diagnostics differ from observed aggregation")
    for row in diagnostics:
        info = row["aggregate"]
        if info["actual_parameters"] != job["adapter"] or info["aggregation_weighting"] != "equal_after_threshold":
            raise ValueError("Changed hybrid mechanism")
        if info["client_order"] != list(range(20)) or len(info["root_aeod"]) != 20 or len(info["trust_scores"]) != 20:
            raise ValueError("Incomplete client/root diagnostics")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in info["root_aeod"]) or any(
                not math.isfinite(value) or not 0 <= value <= 1 + 1e-6 for value in info["trust_scores"]):
            raise ValueError("Invalid root fairness/trust diagnostic")
        selected = [i for i, value in enumerate(info["trust_scores"]) if value > job["adapter"]["threshold"]]
        if not selected:
            selected = [max(range(20), key=lambda i: info["trust_scores"][i])]
        if info["selected_clients"] != selected:
            raise ValueError("Threshold/fallback selection mismatch")
    audits = result["attack_audit"]
    if [row["client_id"] for row in audits] != list(range(20)) or any(row.get("label_changed_count", 0) != 0 for row in audits):
        raise ValueError("Client or unchanged-label audit mismatch")
    if job["attack"] == "S-DFA" and any(
            row["attack_types"] != ["fflip", "foe"] or "foe_impl" not in row for row in audits[:4]):
        raise ValueError("S-DFA attack audit incomplete")
    import torch
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint or any(
            not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in checkpoint.values()):
        raise ValueError("Invalid/nonfinite terminal checkpoint")
    return result
