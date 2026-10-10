"""Bounded source/job/mapping/storage regression; no postprocessing fit or CNN."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STAGE = HERE / "snapshot/logofair_bridge_20261010"
sys.path.insert(0, str(STAGE))
import bridge
from stage_inputs import bulk_path, digest, read, require
from summarize import summarize


def refused(action, label):
    try:
        action()
    except (ValueError, KeyError, RuntimeError):
        return label
    raise AssertionError("Invalid fixture accepted: " + label)


def check():
    original = HERE.parent / "celeba_baselines/logofair_bridge_20261009"
    protocol, manifest = read(STAGE / "protocol.json"), read(HERE / "jobs/manifest.json")
    old = read(original / "protocol.json")
    assert protocol["status"] == "FROZEN" and all(d["status"] == "APPROVED" for d in protocol["decisions"].values())
    assert protocol["candidates"] == old["candidates"] and {c["settings"]["post_rounds"] for c in protocol["candidates"]} == {30}
    assert (STAGE / "bridge.py").read_bytes() == (original / "bridge.py").read_bytes()
    assert digest(HERE / "snapshot/logofair/adapter.py") == bridge.ADAPTER_SHA
    assert __import__("hashlib").sha256((HERE / "snapshot/logofair/upstream/fedlearn/models/FedFairPostClient.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest() == bridge.OFFICIAL_LF_SHA
    references = read(STAGE / "reuse_manifest.json")["entries"]
    old_refs = {e["id"]: e for e in read(original / "reuse_manifest.json")["entries"]}
    assert len(references) == 4 and all(e == old_refs[e["id"]] for e in references)
    inputs = read(HERE / "REFERENCE_INPUTS.json")
    with np.load(inputs["mapping_source"], allow_pickle=False) as source:
        mapping = {k: source[k].copy() for k in source.files}
    metadata = read(HERE / "mapping_metadata.json")
    assert digest(inputs["mapping_source"]) == metadata["mapping_sha256"]
    assert metadata["true_training_client_identity"] is False
    assert bridge.validate_mapping(mapping, metadata, metadata["root_image_ids_sha256"], metadata["valid_image_ids_sha256"])
    assert len(manifest["jobs"]) == len({e["id"] for e in manifest["jobs"]}) == 32
    candidate_conditions = {}
    for entry in manifest["jobs"]:
        path = HERE / "jobs" / entry["job"]; job = read(path)
        assert digest(path) == entry["job_sha256"]
        bridge.validate_job(job, protocol)
        assert job["mapping_sha256"] == metadata["mapping_sha256"] and job["mapping_metadata_sha256"] == digest(HERE / "mapping_metadata.json")
        assert all(digest(STAGE / n) == wanted for n, wanted in job["local_hashes"].items())
        ref = old_refs[job["baseline_id"]]
        candidate_conditions.setdefault(job["candidate"], set()).add((ref["source_job"]["distribution"], ref["source_job"]["attack"]))
    conditions = {("IID", "Benign"), ("IID", "S-DFA"), ("non-IID", "Benign"), ("non-IID", "S-DFA")}
    assert len(candidate_conditions) == 8 and all(v == conditions for v in candidate_conditions.values())
    first = read(HERE / "jobs" / manifest["jobs"][0]["job"])
    checks = [refused(lambda: bridge.validate_job(first, old), "old PREPARED protocol")]
    for key, value in (("fit_seed", 1720), ("seed", 91002), ("evaluation_split", "test"), ("mapping_sha256", None)):
        bad = copy.deepcopy(first); bad[key] = value
        checks.append(refused(lambda: bridge.validate_job(bad, protocol), key))
    bad = copy.deepcopy(first); bad["settings"]["post_rounds"] = 100
    checks.append(refused(lambda: bridge.validate_job(bad, protocol), "post_rounds100"))
    assert bridge.checked_output(HERE / "jobs" / manifest["jobs"][0]["job"], HERE / "NO_OUTPUT_CREATED", None, None) is None
    checks.append(refused(lambda: bulk_path(HERE / "must_not_be_created", 0), "E model/array destination"))
    target, actual_volume = bulk_path("F:/YananResearchStorage/GuardFed/logofair_screen32_20261010/inputs", 0)
    assert target.drive.upper() == "F:"
    mock = [dict(id=c["id"] + str(i), candidate=c["id"], distribution=d, attack=a,
                 metrics=dict(accuracy=.5, aeod=0., aspd=0.))
            for c in protocol["candidates"] for i, (d, a) in enumerate(sorted(conditions))]
    result = summarize(mock)
    assert result["selected_per_method"]["LoGoFair-DP-official-adapted"]["candidate"] == "LoGoFair-DP_00"
    assert len(result["three_metric_pareto"]) == 8
    checks.append(refused(lambda: summarize(mock[:-1]), "partial31 cannot select"))
    gate = read(HERE / "GATE_ACCEPTANCE.json")
    assert digest(HERE / "GATE_ACCEPTANCE.json") == protocol["real_score_gate_acceptance_sha256"]
    assert (gate["settings"]["post_rounds"], gate["beta_fits"], gate["valid_rows"], gate["new_CNN_calls"]) == (3, 40, 19867, 0)
    assert gate["serialized_predictions_exact"] and gate["metrics"]["positive_rate"] == 0
    old_gate = read(original / "acceptance_gate.json")
    assert old_gate["status"] == "PASS" and old_gate["bridge_sha256"] == digest(STAGE / "bridge.py")
    for path, wanted in read(HERE / "INPUT_PINS.json")["source_pins"].items():
        assert digest(path) == wanted
    return dict(status="PASS_SOURCE32_ONLY_NO_EXECUTION", jobs32=True, candidates8_exact=True, post_rounds30=True,
        four_existing_accepted_references_exact=True, original_bridge_adapter_and_official_source_exact=True,
        approved_actual_mapping_identity=True, old_sources_unchanged=True, original_acceptance_gate_reused=True,
        metadata_refusals=checks, selection_tie_Pareto_MOCK_only=True, actual_F_volume=actual_volume,
        new_CNN_calls=0, new_fits=0, new_bulk_writes=0, new_scientific_results=0, actual_dispatch=False)


if __name__ == "__main__":
    proof = check()
    (HERE / "SELF_CHECK.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: proof[k] for k in ("status", "jobs32", "post_rounds30", "new_fits", "actual_dispatch")}))
