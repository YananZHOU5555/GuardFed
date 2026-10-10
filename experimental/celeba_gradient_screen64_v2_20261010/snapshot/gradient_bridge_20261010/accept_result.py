"""Strict acceptance of frozen gradient screens; no training or peak selection."""
import json
import math
from pathlib import Path

import torch

from worker import COMPONENTS, HERE, digest, validate_job, verify_hashes


def checked_result(job_path, output):
    job_path, output = Path(job_path), Path(output)
    job = json.loads(job_path.read_text(encoding="utf8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    validate_job(job, protocol)
    if list(output.glob("failure*.json")):
        raise ValueError("Preserved failure blocks reuse")
    if not (output / "result.json").exists():
        return None
    result = json.loads((output / "result.json").read_text(encoding="utf8"))
    receipt = json.loads((output / "acceptance.json").read_text(encoding="utf8"))
    names = {"model.pt", "diagnostics.json", "result.json", "provenance.json"}
    if receipt["status"] != "PASS" or set(receipt["artifact_hashes"]) != names:
        raise ValueError("Missing complete artifact receipt")
    for name, expected in receipt["artifact_hashes"].items():
        if digest(output / name) != expected:
            raise ValueError("Artifact SHA mismatch: " + name)
    if result["status"] != "screen_complete" or result["evidence_stage"] != "validation_screen":
        raise ValueError("Incomplete/non-screen result")
    if json.loads((output / "job.json").read_text(encoding="utf8")) != job or result["revision_job"] != job:
        raise ValueError("Stored job identity mismatch")
    if result["config"] != job["config"] or result["gradient_recipe"] != job["adapter"]:
        raise ValueError("Actual gradient parameters differ from frozen recipe")
    for key in ("dataset", "method", "distribution", "attack", "tuning_candidate"):
        if result[key] != job[key]:
            raise ValueError("Method/condition identity mismatch")
    if (result["seed"], result["rounds"], result["alpha"], result["num_clients"], result["num_malicious"]) != (
            91001, 70, job["config"]["client_alpha"], 20, 4):
        raise ValueError("Seed/round/alpha/client contract mismatch")
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf8"))
    if result["provenance"] != provenance or provenance["job_sha256"] != digest(job_path):
        raise ValueError("Checkpoint/job provenance mismatch")
    for key in ("source_hashes", "local_hashes", "component_hashes"):
        if provenance[key] != job[key]:
            raise ValueError("Source/data/component provenance mismatch")
    verify_hashes(HERE, job["local_hashes"])
    verify_hashes(HERE.parent, COMPONENTS)
    for field in ("trajectory_metrics", "round_summaries"):
        if [row["round"] for row in result[field]] != list(range(1, 71)):
            raise ValueError("Incomplete or duplicate gradient rounds")
    if result["metrics"] != result["trajectory_metrics"][-1]["metrics"] or any(
            not math.isfinite(row["metrics"][key]) or not 0 <= row["metrics"][key] <= 1
            for row in result["trajectory_metrics"] for key in ("accuracy", "aeod", "aspd")):
        raise ValueError("Invalid or mixed-checkpoint metrics")
    image = result["data_contract"]["image_data_contract"]
    if (image["evaluation_split"], image["actual_train_rows"], image["actual_evaluation_rows"],
        image["train_eval_disjoint"], image["root_client_disjoint"], result["evaluation_stats"]["prediction_count"]) != (
            "valid", 162770, 19867, True, True, 19867):
        raise ValueError("Wrong evaluation split/sample identity")
    contract = result["data_contract"]
    if (contract["train_rows"], contract["test_rows"], contract["root_synthetic_rows"],
        contract["feature_includes_label"], contract["feature_includes_sensitive"]) != (162770, 19867, 0, False, False):
        raise ValueError("Wrong full image data contract")
    diagnostic = json.loads((output / "diagnostics.json").read_text(encoding="utf8"))
    if diagnostic != result["round_summaries"]:
        raise ValueError("Diagnostic trajectory mismatch")
    root_expected = job["attack"] == "S-DFA"
    for row in diagnostic:
        info = row["aggregate"]
        if (info["actual_parameters"], info["aggregation_weighting"], info["client_order"],
            info["local_optimizer_steps"], info["projection"], info["defense_root_fairness_used"], info["defense_root_reference_used"]) != (
                job["adapter"], "raw_client_sample_count", list(range(20)), 0, "identity_Rp", False, False):
            raise ValueError("Changed gradient mechanism")
        if len(info["counts"]) != 20 or any(count <= 0 for count in info["counts"]) or len(info["raw_gradient_norms"]) != 20 or len(info["uploaded_norms"]) != 20:
            raise ValueError("Incomplete gradient/count diagnostics")
        if info["counts"] != image["client_sample_counts"] or sum(info["counts"]) + contract["root_clean_rows"] != 162770:
            raise ValueError("Aggregation sample weights differ from the frozen data partition")
        if any(not math.isfinite(value) or value < 0 for value in info["raw_gradient_norms"] + info["uploaded_norms"] + [info["server_step_norm"]]):
            raise ValueError("Nonfinite gradient diagnostics")
        reference = info["root_threat_reference"]
        if reference["used"] != root_expected or reference.get("defense_input", False):
            raise ValueError("Unexpected root access")
        if root_expected and (reference["mode"] != job["adapter"]["root_reference"] or reference["evaluation_labels_used"] or reference["root_rows"] != contract["root_clean_rows"]):
            raise ValueError("Changed root threat reference")
        if root_expected and (not math.isfinite(reference["norm"]) or reference["norm"] < 0 or len(reference["sha256"]) != 64):
            raise ValueError("Invalid root threat reference diagnostic")
        for name in ("global_point_sha256", "uploaded_sha256", "step_sha256"):
            if len(info[name]) != 64 or any(ch not in "0123456789abcdef" for ch in info[name]):
                raise ValueError("Missing gradient/step identity")
        if job["method"] == "Huber-BRFL-gradient":
            expected = [job["adapter"]["t0"] + job["adapter"]["m"] / math.sqrt(count) for count in info["counts"]]
            if not info["converged"] or not math.isfinite(info["stationarity_l2"]) or info["stationarity_l2"] > job["adapter"]["tolerance"] or len(info["fixed_thresholds"]) != 20 or any(
                    not math.isclose(actual, wanted, rel_tol=1e-14, abs_tol=1e-15)
                    for actual, wanted in zip(info["fixed_thresholds"], expected)):
                raise ValueError("Unconverged or altered fixed Huber objective")
            trace = info["objective_trace"]
            if not trace or any(not math.isfinite(value) or value < 0 for value in trace) or any(
                    b > a + 1e-10 * max(1., abs(a)) for a, b in zip(trace, trace[1:])):
                raise ValueError("Invalid fixed Huber objective trajectory")
            weights = info["normalized_final_weights"]
            if (info["objective"] != trace[-1] or info["iterations"] != len(trace) - 1 or
                info["iterations"] > job["adapter"]["max_iter"] or len(weights) != 20 or
                any(not math.isfinite(value) or not 0 <= value <= 1 for value in weights) or
                not math.isclose(sum(weights), 1., rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("Invalid Huber convergence/objective/weight identity")
    audits = result["attack_audit"]
    if [row["client_id"] for row in audits] != list(range(20)) or any(row.get("label_changed_count", 0) for row in audits):
        raise ValueError("Client/label audit mismatch")
    if root_expected and any(row["attack_types"] != ["fflip", "foe"] or "gradient sign-conjugacy" not in row["foe_impl"] for row in audits[:4]):
        raise ValueError("Incomplete S-DFA gradient upload audit")
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in checkpoint.values()):
        raise ValueError("Invalid terminal checkpoint")
    return result
