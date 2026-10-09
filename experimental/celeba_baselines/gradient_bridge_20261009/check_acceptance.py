"""Structural mocked-70-row acceptance tests, never seventy-round evidence.

All FROZEN/APPROVED metadata lives only in a temporary fixture, removed on exit.
The real protocol remains PREPARED/UNRESOLVED throughout this test.
"""
import copy
import json
from pathlib import Path
import tempfile

import torch

import accept_result
import worker
from worker import HERE, digest, validate_job, write_json


def expect_refused(action, reason):
    try:
        action()
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        return dict(check=reason, refused=True, error_type=type(error).__name__)
    raise AssertionError("Accepted invalid fixture: " + reason)


def main():
    protocol_path = HERE / "protocol.json"
    before = digest(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    manifest = json.loads((HERE / "screen_jobs_draft/manifest.json").read_text(encoding="utf8"))
    assert protocol["status"] == "PREPARED_NOT_FROZEN" and not manifest["execution_started"]
    assert len(manifest["jobs"]) == len({row["id"] for row in manifest["jobs"]}) == 64
    for row in manifest["jobs"]:
        path = HERE / "screen_jobs_draft" / row["job"]
        assert digest(path) == row["job_sha256"]
        job = json.loads(path.read_text(encoding="utf8"))
        validate_job(job, protocol, require_frozen=False)
        worker.verify_hashes(HERE, job["local_hashes"])
    row = next(row for row in manifest["jobs"] if row["method"] == "Fed-NGA-gradient" and "_S-DFA_" in row["id"])
    job_path = HERE / "screen_jobs_draft" / row["job"]
    job = json.loads(job_path.read_text(encoding="utf8"))
    out = HERE / "must_not_exist_unfrozen_launch"
    assert not out.exists()
    checks = [expect_refused(lambda: worker.run(HERE.parents[1] / "revision-publish-20260928", job_path, out), "PREPARED production launch")]
    assert not out.exists()
    approved = copy.deepcopy(protocol)
    approved["status"] = "FROZEN"
    checks.append(expect_refused(lambda: validate_job(job, approved), "FROZEN status with UNRESOLVED decisions"))
    for decision in approved["protocol_decisions"].values():
        decision["status"] = "APPROVED"
    validate_job(job, approved)
    for field, replacement in (("seed", 42), ("rounds", 3), ("client_alpha", 17.), ("celeba_evaluation_split", "test")):
        bad = copy.deepcopy(job)
        bad["config"][field] = replacement
        checks.append(expect_refused(lambda: validate_job(bad, approved), "job mutation " + field))
    with tempfile.TemporaryDirectory(prefix="acceptance_fixture_", dir=HERE) as name:
        fixture = Path(name)
        write_json(fixture / "protocol.json", approved)
        local_hashes = {source: digest(fixture / source) if source == "protocol.json" else digest(HERE / source) for source in worker.LOCAL_SOURCES}
        for source in worker.LOCAL_SOURCES - {"protocol.json"}:
            (fixture / source).write_bytes((HERE / source).read_bytes())
        fixture_job = copy.deepcopy(job)
        fixture_job["local_hashes"] = local_hashes
        fixture_job_path = fixture / "job.json"
        write_json(fixture_job_path, fixture_job)
        raw = json.loads((HERE / "synthetic_run_1/result.json").read_text(encoding="utf8"))["result"]
        result = copy.deepcopy(raw)
        result.update(status="screen_complete", evidence_stage="validation_screen", config=fixture_job["config"],
            gradient_recipe=fixture_job["adapter"], seed=91001, rounds=70, alpha=fixture_job["config"]["client_alpha"],
            revision_job=fixture_job, tuning_candidate=fixture_job["tuning_candidate"],
            distribution=fixture_job["distribution"])
        # These counts/rows exist only to exercise the strict metadata checks;
        # this deliberately constructed structure is NOT a scientific result.
        counts = [7324] * 19 + [7337]
        result["data_contract"].update(train_rows=162770, test_rows=19867, root_clean_rows=16277)
        result["data_contract"]["image_data_contract"].update(evaluation_split="valid", actual_train_rows=162770,
            actual_evaluation_rows=19867, client_sample_counts=counts, evidence_stage="MOCK_STRUCTURE_ONLY")
        result["evaluation_stats"]["prediction_count"] = 19867
        result["trajectory_metrics"] = [dict(round=number, metrics=copy.deepcopy(raw["metrics"])) for number in range(1, 71)]
        result["round_summaries"] = [dict(round=number, aggregate=copy.deepcopy(raw["round_summaries"][-1]["aggregate"])) for number in range(1, 71)]
        for row in result["round_summaries"]:
            row["aggregate"].update(actual_parameters=fixture_job["adapter"], counts=counts)
            row["aggregate"]["root_threat_reference"]["root_rows"] = 16277
        provenance = dict(job_sha256=digest(fixture_job_path), **{key: fixture_job[key] for key in ("source_hashes", "local_hashes", "component_hashes")})
        result["provenance"] = provenance
        output = fixture / "mock_output"
        output.mkdir()
        torch.save(torch.load(HERE / "synthetic_run_1/model.pt", weights_only=True), output / "model.pt")
        write_json(output / "job.json", fixture_job)
        write_json(output / "provenance.json", provenance)
        def save(value):
            write_json(output / "result.json", value)
            write_json(output / "diagnostics.json", value["round_summaries"])
            write_json(output / "acceptance.json", dict(status="PASS", artifact_hashes={source: digest(output / source) for source in
                ("model.pt", "diagnostics.json", "result.json", "provenance.json")}))
        real_here, real_verify = accept_result.HERE, accept_result.verify_hashes
        def fixture_verify(root, hashes):
            # Code is copied verbatim; external audited components stay real.
            return real_verify(HERE.parent if Path(root) == fixture.parent and hashes == worker.COMPONENTS else root, hashes)
        accept_result.HERE, accept_result.verify_hashes = fixture, fixture_verify
        try:
            save(result)
            assert accept_result.checked_result(fixture_job_path, output)["rounds"] == 70
            checks.append(dict(check="complete temporary MOCK metadata structure", accepted=True, scientific_results=0))
            mutations = {
                "missing round": lambda value: value["trajectory_metrics"].pop(),
                "test split": lambda value: value["data_contract"]["image_data_contract"].update(evaluation_split="test"),
                "final metric mismatch": lambda value: value["metrics"].update(accuracy=.1234567),
                "client Adam substituted": lambda value: value["round_summaries"][0]["aggregate"].update(local_optimizer_steps=1),
                "sample weight mismatch": lambda value: value["round_summaries"][0]["aggregate"]["counts"].__setitem__(0, 4),
                "root used by defense": lambda value: value["round_summaries"][0]["aggregate"].update(defense_root_reference_used=True),
                "wrong root reference": lambda value: value["round_summaries"][0]["aggregate"]["root_threat_reference"].update(mode="same_point_unweighted_gradient"),
            }
            for label, mutate in mutations.items():
                bad = copy.deepcopy(result)
                mutate(bad)
                save(bad)
                checks.append(expect_refused(lambda: accept_result.checked_result(fixture_job_path, output), label))
            save(result)
            with (output / "model.pt").open("ab") as model:
                model.write(b"corruption")
            checks.append(expect_refused(lambda: accept_result.checked_result(fixture_job_path, output), "checkpoint SHA corruption"))
            write_json(output / "failure.json", dict(error="preserved synthetic failure"))
            checks.append(expect_refused(lambda: accept_result.checked_result(fixture_job_path, output), "preserved failure"))
            huber_row = next(row for row in manifest["jobs"] if row["method"] == "Huber-BRFL-gradient" and "_S-DFA_" in row["id"])
            huber_job = json.loads((HERE / "screen_jobs_draft" / huber_row["job"]).read_text(encoding="utf8"))
            huber_job["local_hashes"] = local_hashes
            huber_job_path = fixture / "huber_job.json"
            write_json(huber_job_path, huber_job)
            huber_result = copy.deepcopy(result)
            for key in ("method", "tuning_candidate", "config", "distribution"):
                huber_result[key] = copy.deepcopy(huber_job[key])
            huber_result.update(gradient_recipe=huber_job["adapter"], revision_job=huber_job)
            huber_raw = json.loads((HERE / "synthetic_run_3/result.json").read_text(encoding="utf8"))["result"]
            for row in huber_result["round_summaries"]:
                info = copy.deepcopy(huber_raw["round_summaries"][-1]["aggregate"])
                info.update(actual_parameters=huber_job["adapter"], counts=counts,
                    fixed_thresholds=[huber_job["adapter"]["t0"] + huber_job["adapter"]["m"] / (count ** .5) for count in counts])
                info["root_threat_reference"]["root_rows"] = 16277
                row["aggregate"] = info
            huber_provenance = dict(job_sha256=digest(huber_job_path), **{key: huber_job[key] for key in ("source_hashes", "local_hashes", "component_hashes")})
            huber_result["provenance"] = huber_provenance
            output = fixture / "mock_huber_output"
            output.mkdir()
            torch.save(torch.load(HERE / "synthetic_run_3/model.pt", weights_only=True), output / "model.pt")
            write_json(output / "job.json", huber_job)
            write_json(output / "provenance.json", huber_provenance)
            save(huber_result)
            assert accept_result.checked_result(huber_job_path, output)["method"] == "Huber-BRFL-gradient"
            checks.append(dict(check="complete temporary Huber MOCK metadata structure", accepted=True, scientific_results=0))
            for label, mutate in {
                "Huber nonconvergence": lambda info: info.update(converged=False),
                "Huber stationarity over tolerance": lambda info: info.update(stationarity_l2=1.),
                "Huber threshold changed": lambda info: info["fixed_thresholds"].__setitem__(0, 999.),
                "Huber objective identity changed": lambda info: info.update(objective=999.),
                "Huber count weights invalid": lambda info: info["normalized_final_weights"].__setitem__(0, -1.),
            }.items():
                bad = copy.deepcopy(huber_result)
                mutate(bad["round_summaries"][0]["aggregate"])
                save(bad)
                checks.append(expect_refused(lambda: accept_result.checked_result(huber_job_path, output), label))
        finally:
            accept_result.HERE, accept_result.verify_hashes = real_here, real_verify
    assert digest(protocol_path) == before and not list(HERE.glob("acceptance_fixture_*"))
    write_json(HERE / "acceptance_gate.json", dict(status="PASS", evidence_stage="structural_unit_fixtures_only",
        scientific_results=0, actual_protocol_unchanged=True, temporary_FROZEN_mock_removed=True,
        prepared_jobs=64, checks=checks, checker_sha256=digest(HERE / "accept_result.py"),
        check_sha256=digest(HERE / "check_acceptance.py")))
    print(json.dumps(dict(status="PASS", checks=len(checks), prepared_jobs=64, scientific_results=0)))


if __name__ == "__main__":
    main()
