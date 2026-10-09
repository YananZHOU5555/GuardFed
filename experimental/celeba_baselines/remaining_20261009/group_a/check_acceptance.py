"""Acceptance structural unit fixtures, explicitly not seventy-round evidence.

Temporary mock metadata is removed after tests. No prepared protocol is frozen,
no experiment output is accepted for a paper, and no training is launched here.
"""
import copy
import json
from pathlib import Path
import tempfile

import accept_result
from worker import ADAPTER_SOURCES, HERE, LABEL, METHOD, digest, validate_job, write_json


def must_reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Invalid acceptance fixture accepted")


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    manifest = json.loads((HERE / "screen_jobs_draft/manifest.json").read_text())
    source = next(row for row in manifest["jobs"] if "lam20.0_tau0.2_lr0.001" in row["id"] and
                  row["distribution"] == "non-IID" and row["attack"] == "S-DFA")
    job = json.loads((HERE / "screen_jobs_draft" / source["job"]).read_text())
    for key, wrong in [("seed", 123), ("rounds", 3), ("celeba_evaluation_split", "test"),
                       ("client_alpha", 5000.), ("trust_threshold", .1)]:
        altered = copy.deepcopy(job)
        altered["config"][key] = wrong
        must_reject(lambda: validate_job(altered, protocol, require_frozen=False))
    must_reject(lambda: accept_result.checked_result(HERE / "screen_jobs_draft" / source["job"], HERE / "integration_toy"))
    original_here = accept_result.HERE
    checks = ["prepared result acceptance refused", "seed/horizon/split/alpha/parameter job mutations refused"]
    try:
        with tempfile.TemporaryDirectory(prefix=".acceptance_unit_fixture_", dir=HERE) as directory:
            root = Path(directory)
            # FROZEN exists only in this ephemeral mock fixture, never protocol.json.
            mock_protocol = copy.deepcopy(protocol)
            mock_protocol["status"] = "FROZEN"
            write_json(root / "protocol.json", mock_protocol)
            for name in ADAPTER_SOURCES - {"protocol.json"}:
                (root / name).write_bytes((HERE / name).read_bytes())
            mock_job = copy.deepcopy(job)
            mock_job["adapter_source_hashes"] = {name: digest(root / name) for name in ADAPTER_SOURCES}
            job_path = root / "input_job.json"
            write_json(job_path, mock_job)
            output = root / "mock_output_not_scientific"
            output.mkdir()
            write_json(output / "job.json", mock_job)
            (output / "model.pt").write_bytes((HERE / "integration_toy/model.pt").read_bytes())
            toy = json.loads((HERE / "integration_toy/result.json").read_text())["result"]
            result = copy.deepcopy(toy)
            result.update(status="screen_complete", evidence_stage="validation_screen", dataset="celeba",
                          method=METHOD, method_impl_note=LABEL, seed=91001, rounds=70,
                          config=mock_job["config"], revision_job=mock_job,
                          tuning_candidate=mock_job["tuning_candidate"])
            # Repeated three-round diagnostic/metric rows are MOCK DATA only.
            result["trajectory_metrics"] = [dict(round=i + 1, metrics=toy["metrics"]) for i in range(70)]
            result["round_summaries"] = [dict(copy.deepcopy(toy["round_summaries"][-1]), round=i + 1) for i in range(70)]
            result["data_contract"]["image_data_contract"] = dict(evaluation_split="valid", actual_train_rows=162770,
                actual_evaluation_rows=19867, train_eval_disjoint=True, root_client_disjoint=True)
            result["evaluation_stats"]["prediction_count"] = 19867
            provenance = dict(job_sha256=digest(job_path), source_hashes=mock_job["source_hashes"],
                              adapter_source_hashes=mock_job["adapter_source_hashes"])
            result["provenance"] = provenance
            diagnostics = [dict(round=i + 1, aggregate=row["aggregate"]) for i, row in enumerate(result["round_summaries"])]

            def store(candidate, diagnostic_rows):
                write_json(output / "provenance.json", provenance)
                write_json(output / "result.json", candidate)
                write_json(output / "diagnostics.json", diagnostic_rows)
                names = ("model.pt", "diagnostics.json", "result.json", "provenance.json")
                write_json(output / "acceptance.json", dict(status="PASS", rounds=70, evaluation_split="valid",
                    train_rows=162770, evaluation_rows=19867, tuning_candidate=mock_job["tuning_candidate"],
                    artifact_hashes={name: digest(output / name) for name in names}))

            accept_result.HERE = root
            store(result, diagnostics)
            assert accept_result.checked_result(job_path, output)["seed"] == 91001
            checks.append("complete mock metadata structure passes (not real seventy-round evidence)")
            for name in ["trajectory_metrics", "round_summaries"]:
                broken = copy.deepcopy(result)
                broken[name] = broken[name][:-1]
                store(broken, diagnostics)
                must_reject(lambda: accept_result.checked_result(job_path, output))
            for changed in ["split", "final_metrics", "selection", "parameter", "checkpoint_sha", "failure"]:
                broken, diagnostic_rows = copy.deepcopy(result), copy.deepcopy(diagnostics)
                if changed == "split":
                    broken["data_contract"]["image_data_contract"]["evaluation_split"] = "test"
                elif changed == "final_metrics":
                    broken["metrics"] = dict(broken["metrics"], accuracy=.1234567)
                elif changed in {"selection", "parameter"}:
                    key, value = ("selected_clients", []) if changed == "selection" else ("actual_parameters", {})
                    diagnostic_rows[-1]["aggregate"][key] = value
                    broken["round_summaries"][-1]["aggregate"] = diagnostic_rows[-1]["aggregate"]
                store(broken, diagnostic_rows)
                if changed == "checkpoint_sha":
                    (output / "model.pt").write_bytes((output / "model.pt").read_bytes() + b"mock_corruption")
                if changed == "failure":
                    write_json(output / "failure.json", dict(evidence="intentional unit fixture"))
                must_reject(lambda: accept_result.checked_result(job_path, output))
                if changed == "checkpoint_sha":
                    (output / "model.pt").write_bytes((HERE / "integration_toy/model.pt").read_bytes())
            checks.append("missing rounds, test split, final-metric mismatch, wrong selected set/parameters, corrupted checkpoint SHA and preserved failure refused")
    finally:
        accept_result.HERE = original_here
    assert json.loads((HERE / "protocol.json").read_text())["status"] == "PREPARED_NOT_FROZEN"
    assert not list(HERE.glob(".acceptance_unit_fixture_*"))
    report = dict(status="PASS", evidence_stage="structural_unit_fixtures_only", scientific_results=0,
                  protocol_remains="PREPARED_NOT_FROZEN", temporary_mock_data_removed=True, checks=checks,
                  checker_sha256=digest(HERE / "accept_result.py"), check_sha256=digest(HERE / "check_acceptance.py"))
    write_json(HERE / "acceptance_interface_gate.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
