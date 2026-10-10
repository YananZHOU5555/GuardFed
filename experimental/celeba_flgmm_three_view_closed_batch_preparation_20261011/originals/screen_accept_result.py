"""Strict terminal-result check for parent runners; no retry or training action."""
import json
import math
from pathlib import Path

from flgmm_adapter import FLGMMAdapter
from worker import HERE, LABEL, METHOD, digest, validate_job


def checked_result(job_path, output):
    job_path, output = Path(job_path), Path(output)
    job = json.loads(job_path.read_text(encoding="utf8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    validate_job(job, protocol)
    if list(output.glob("failure*.json")):
        raise ValueError("Preserved failure evidence; parent must diagnose before reuse")
    path = output / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf8"))
    evidence = json.loads((output / "acceptance.json").read_text(encoding="utf8"))
    if evidence["status"] != "PASS" or result["status"] != "screen_complete":
        raise ValueError("FLGMM result has no terminal acceptance")
    for name, sha in evidence["artifact_hashes"].items():
        if name not in {"model.pt", "state.json", "diagnostics.json", "result.json", "provenance.json"}:
            raise ValueError("Unexpected result artifact")
        if digest(output / name) != sha:
            raise ValueError("Artifact hash mismatch: " + name)
    if set(evidence["artifact_hashes"]) != {"model.pt", "state.json", "diagnostics.json", "result.json", "provenance.json"}:
        raise ValueError("Incomplete result artifact hash set")
    if result["revision_job"] != job or result["config"] != job["config"]:
        raise ValueError("Result/job config identity mismatch")
    if any(result[key] != job[key] for key in ("dataset", "method", "distribution", "attack", "tuning_candidate")):
        raise ValueError("Result/job metadata mismatch")
    if result["method"] != METHOD or result["method_impl_note"] != LABEL or result["seed"] != 91001 or result["rounds"] != 70:
        raise ValueError("Method, seed or horizon mismatch")
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf8"))
    if result["provenance"] != provenance or provenance["job_sha256"] != digest(job_path):
        raise ValueError("Input/checkpoint provenance mismatch")
    for name in ("source_hashes", "adapter_source_hashes"):
        if provenance[name] != job[name]:
            raise ValueError("Frozen source identity mismatch")
    for name, sha in job["adapter_source_hashes"].items():
        if digest(HERE / name) != sha:
            raise ValueError("Adapter code/protocol changed")
    for name in ("trajectory_metrics", "round_summaries"):
        if [row["round"] for row in result[name]] != list(range(1, 71)):
            raise ValueError("Incomplete/duplicate final-round coverage")
    metrics = result["trajectory_metrics"]
    if result["metrics"] != metrics[-1]["metrics"] or not all(
            math.isfinite(row["metrics"][name]) and 0 <= row["metrics"][name] <= 1
            for row in metrics for name in ("accuracy", "aeod", "aspd")):
        raise ValueError("Invalid same-terminal-checkpoint metrics")
    contract = result["data_contract"]["image_data_contract"]
    if (contract["evaluation_split"], contract["actual_train_rows"], contract["actual_evaluation_rows"],
            contract["train_eval_disjoint"], contract["root_client_disjoint"],
            result["evaluation_stats"]["prediction_count"]) != ("valid", 162770, 19867, True, True, 19867):
        raise ValueError("Unexpected split/row count/disjointness")
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf8"))
    if len(diagnostics) != 70 or any(row["round"] != i + 1 or row["aggregate"] != result["round_summaries"][i]["aggregate"]
                                     for i, row in enumerate(diagnostics)):
        raise ValueError("Stored diagnostics differ from observed aggregation")
    state = json.loads((output / "state.json").read_text(encoding="utf8"))
    controller = FLGMMAdapter(range(20), **job["adapter"])
    controller.load_state_dict(state)
    if controller.round_index != 70 or controller.ucl != diagnostics[-1]["aggregate"]["ucl"]:
        raise ValueError("Incomplete FLGMM history or control-limit identity")
    return result
