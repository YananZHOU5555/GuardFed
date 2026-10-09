"""Bounded local evidence checks; never accesses real images/test labels or trains."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tarfile
import tempfile
import traceback

import torch

import evidence as v1
import evidence_v4 as ev

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGE = REPO / "docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1"
INVENTORY = REPO / "docs/server_deployment_20260923/training_20260923/final_evaluation_prepared_20261009/model_inventory.json"
CHECKS = []


def rejected(function):
    try:
        function()
    except (ValueError, AssertionError, RuntimeError) as error:
        return {"exception": type(error).__name__, "reason": str(error)}
    raise AssertionError("Invalid evidence was accepted")


def actual_archived_full_gate():
    manifest = ev.read(STAGE / "manifest.json")
    ev.require(ev.digest(INVENTORY) == ev.FULL_INVENTORY_SHA, "Inventory SHA drift")
    records = {record["id"]: record for record in ev.read(INVENTORY)["records"]}
    original = ev.load("mechanism_evidence_selfcheck_original", REPO / "tmp/revision-publish-20260928/scripts/run_revision_ablation.py")
    entry = copy.deepcopy(manifest["reused_full"][0])
    record = records[entry["id"]]
    template = ev.read(STAGE / "jobs" / (manifest["jobs"][0]["id"] + ".json"))["config"]
    with tempfile.TemporaryDirectory(dir=HERE, prefix="full_gate_") as directory:
        out = Path(directory).resolve()
        out.relative_to(HERE.resolve())
        for key, filename in [("checkpoint", "model.pt"), ("result", "result.json")]:
            ref = record[key]
            with tarfile.open(REPO / ref["archive"], "r:gz") as archive:
                data = archive.extractfile(ref["member"]).read()
            assert hashlib.sha256(data).hexdigest() == ref["sha256"]
            (out / filename).write_bytes(data)
        entry["output"] = str(out)
        accepted = ev.accept_full(original, entry, record, manifest, template)
        assert accepted is not None
        assert v1.accept_full(original, entry, record, manifest, template)["metrics"] == accepted["metrics"]
        failures = {}
        for key in sorted(ev.SCIENCE_FILES):
            changed = copy.deepcopy(record)
            changed["source_hashes"][key] = "d" * 64
            failures["scientific_source_" + key] = rejected(lambda changed=changed: ev.accept_full(original, entry, changed, manifest, template))
        for key in ["root_image_ids_sha256", "train_image_ids_sha256", "evaluation_image_ids_sha256", "cache_manifest_sha256"]:
            changed = copy.deepcopy(record)
            changed["data_contract"][key] = "d" * 64
            failures[key] = rejected(lambda changed=changed: ev.accept_full(original, entry, changed, manifest, template))
        (out / "failure.json").write_text('{"preserve":true}', encoding="utf-8")
        failures["preserved_failure"] = rejected(lambda: ev.accept_full(original, entry, record, manifest, template))
        (out / "failure.json").unlink()
        data = (out / "model.pt").read_bytes()
        (out / "model.pt").write_bytes(data + b"tamper")
        failures["mixed_or_tampered_checkpoint"] = rejected(lambda: ev.accept_full(original, entry, record, manifest, template))
        assert all((out / name).is_file() for name in ("model.pt", "result.json"))
    return {"status": "PASS", "id": record["id"], "original_checkpoint_sha256": record["checkpoint"]["sha256"],
            "original_result_sha256": record["result"]["sha256"], "real_images_evaluated": 0,
            "original_checked_result_called": True, "strict_rejections": failures,
            "scope": "one already accepted archived terminal Full checkpoint/result; local read-only software gate, not Full100 server acceptance"}


def design_and_statistics():
    manifest = ev.read(STAGE / "manifest.json")
    jobs = [ev.read(STAGE / "jobs" / (entry["id"] + ".json")) for entry in manifest["jobs"]]
    ev.validate_design(manifest, jobs)
    failed = {}
    bad = copy.deepcopy(manifest)
    bad["reused_full"][1]["id"] = bad["reused_full"][0]["id"]
    failed["duplicate_Full_ID"] = rejected(lambda: ev.validate_design(bad, jobs))
    bad_jobs = copy.deepcopy(jobs)
    bad_jobs[1]["config"]["seed"] = bad_jobs[0]["config"]["seed"]
    failed["duplicate_seed_missing_declared_seed"] = rejected(lambda: ev.validate_design(manifest, bad_jobs))
    failed["missing_seed_job"] = rejected(lambda: ev.validate_design(manifest, jobs[:-1]))
    rows = []
    for variant in ev.VARIANTS:
        for dist in ev.DISTRIBUTIONS:
            for attack_index, attack in enumerate(ev.ATTACKS):
                for seed in ev.SEEDS:
                    rows.append({"id": f"{variant}_{dist}_{attack}_{seed}", "variant": variant, "distribution": dist,
                                 "attack": attack, "seed": seed, "accuracy_pct": 50 + seed - 91001 + attack_index * 2 + (variant != "Full"),
                                 "aeod": 0.1, "aspd": 0.2})
    summary = ev.summarize(rows)
    scene = next(row for row in summary["per_scene"] if (row["variant"], row["distribution"], row["attack"]) == ("Full", "IID", "Benign"))
    assert scene["n"] == 10 and scene["complete"] and scene["accuracy_pct"]["mean"] == 54.5
    assert abs(scene["accuracy_pct"]["sample_sd_ddof1"] - math.sqrt(55 / 6)) < 1e-14
    assert len(summary["paired_per_seed"]) == 800
    assert all(row["accuracy_pct"] == 1 and row["aeod"] == row["aspd"] == 0 for row in summary["paired_per_seed"])
    cross = next(row for row in summary["cross_scenario_per_seed"] if (row["panel"], row["kind"], row["variant"], row["seed"]) == ("all10scenarios", "reported_native", "Full", 91001))
    assert cross["accuracy_pct"] == 54 and cross["n_scenarios"] == 10
    missing = next(row for row in rows if (row["variant"], row["distribution"], row["attack"], row["seed"]) == ("minus_U", "non-IID", "S-DFA", 91001))
    partial = ev.summarize([row for row in rows if row is not missing])
    scene_partial = next(row for row in partial["per_scene"] if (row["variant"], row["distribution"], row["attack"]) == ("minus_U", "non-IID", "S-DFA"))
    assert scene_partial["n"] == 9 and not scene_partial["complete"]
    cross_partial = next(row for row in partial["cross_scenario_per_seed"] if (row["panel"], row["kind"], row["variant"], row["seed"]) == ("all10scenarios", "reported_native", "minus_U", 91001))
    assert cross_partial["n_scenarios"] == 9 and not cross_partial["complete"] and "accuracy_pct" not in cross_partial
    failed["duplicate_accepted_cell"] = rejected(lambda: ev.summarize(rows + [rows[0]]))
    excluded_keys = {("non-IID", "Benign", 91001), ("non-IID", "S-DFA", 91001)}
    sensitivity = ev.summarize([row for row in rows if not (row["variant"] == "Full" and (row["distribution"], row["attack"], row["seed"]) in excluded_keys)])
    assert len(sensitivity["paired_per_seed"]) == 784
    full_cross = next(row for row in sensitivity["cross_scenario_summary"] if (row["panel"], row["kind"], row["variant"]) == ("all10scenarios", "reported_native", "Full"))
    assert full_cross["n"] == 9 and not full_cross["complete"]
    return {"status": "PASS", "actual_declared_jobs_checked": 800, "synthetic_complete_rows": 900,
            "hand_computed_mean": 54.5, "hand_computed_sample_sd": math.sqrt(55 / 6), "paired_n": 800,
            "runtime_sensitivity_paired_n": 784, "partial_scene_n": 9, "incomplete_seed_panel_has_no_mean": True, "rejections": failed}


def synthetic_new(directory, variant="minus_U"):
    directory = Path(directory)
    manifest = ev.read(STAGE / "manifest.json")
    template = next(entry for entry in manifest["jobs"] if entry["variant"] == variant)
    job = ev.read(STAGE / "jobs" / (template["id"] + ".json"))
    job["output"] = str(directory / "runs" / job["id"])
    out = Path(job["output"])
    out.mkdir(parents=True)
    path = directory / "jobs" / (job["id"] + ".json")
    ev.save(path, job)
    entry = dict(template, output=job["output"], job=str(path), job_sha256=ev.digest(path))
    torch.save({"weight": torch.tensor([1.0])}, out / "model.pt")
    metrics = {"accuracy": 0.8, "aeod": 0.1, "aspd": 0.2}
    aggregate = {"client_weights": [0.05] * 20, "component_contributions": [{name: 0.0 for name in "UCAFV"}] * 20,
                 "norm_clip_scales": [1.0] * 20, "hard_gate_clients": list(range(20)), "selected_clients": list(range(20)),
                 "ad2_plus_mode": "fixed_balanced_without_candidate_selection", "fixed_candidate": "balanced"}
    result = {"dataset": "celeba", "method": "GuardFed-AD2+", "rounds": 70, "seed": job["config"]["seed"],
              "distribution": job["distribution"], "attack": job["attack"], "alpha": job["config"]["client_alpha"],
              "config": job["config"], "metrics": metrics,
              "trajectory_metrics": [{"round": i, "metrics": metrics} for i in range(1, 71)],
              "round_summaries": [{"round": i, "aggregate": aggregate} for i in range(1, 71)],
              "evaluation_stats": {"positive_rate": 0.5, "majority_accuracy": 0.5166859616449389, "prediction_count": 19867},
              "data_contract": {"image_data_contract": {"evaluation_split": "valid", "actual_train_rows": 162770, "actual_evaluation_rows": 19867,
                    "train_eval_disjoint": True, "root_client_disjoint": True, "evaluation_image_ids_sha256": ev.VALID_IDS_SHA,
                    "cache_manifest_sha256": manifest["source_hashes"]["data/celeba/derived/rgb64_v1/manifest.json"]}},
              "revision_job": dict(job, checkpoint_sha256=ev.digest(out / "model.pt"), torch_version="2.11.0+cu128")}
    full_record = next(record for record in ev.read(INVENTORY)["records"] if record["method"] == "GuardFed-AD2+" and
                       (record["distribution"], record["attack"], record["seed"]) == (job["distribution"], job["attack"], job["config"]["seed"]))
    contract = result["data_contract"]["image_data_contract"]
    contract.update(copy.deepcopy(full_record["data_contract"]))
    result["data_contract"].update(root_clean_rows=162770 - sum(contract["client_sample_counts"]), root_synthetic_rows=0,
                                 train_rows=162770, test_rows=19867)
    component = variant[-1] if variant.startswith("minus_") else "none"
    ledger = [{"variant": variant, "component": component, "clients": 20, "selected_count": 20, "mask_verified": True}
              for _ in range(70 * (1 if variant == "fixed_balanced" else 10))]
    ev.save(out / "result.json", result)
    ev.save(out / "candidate_mask_audit.json", ledger)
    audit = {"variant": variant, "rounds": 70, "candidate_calls_verified": len(ledger), "expected_candidate_calls": len(ledger), "pass": True,
             "job_sha256": ev.digest(path), "result_sha256": ev.digest(out / "result.json"),
             "checkpoint_sha256": result["revision_job"]["checkpoint_sha256"], "candidate_audit_sha256": ev.digest(out / "candidate_mask_audit.json"),
             "adapter_hashes": manifest["adapter_hashes"]}
    ev.save(out / "mechanism_acceptance.json", audit)
    ev.save(directory / "PROTOCOL.md", {"fixture": "not a scientific protocol"})
    (directory / "logs").mkdir(exist_ok=True)
    (directory / "logs" / (job["id"] + ".log")).write_text("synthetic terminal output\n")
    return manifest, job, entry, full_record


def real_checked_new_path():
    adapter_dir = REPO / "tmp/celeba_mechanism_20261009"
    original = ev.load("mechanism_new_selfcheck_original", REPO / "tmp/revision-publish-20260928/scripts/run_revision_ablation.py")
    import sys
    sys.path.insert(0, str(adapter_dir))
    worker = ev.load("mechanism_new_selfcheck_worker", adapter_dir / "worker.py")
    with tempfile.TemporaryDirectory(dir=HERE, prefix="new_gate_") as directory:
        Path(directory).resolve().relative_to(HERE.resolve())
        manifest, job, entry, full_record = synthetic_new(directory)
        out = Path(job["output"])
        assert ev.accept_new(original, worker, job, entry, manifest, full_record) is not None
        failures = {}
        result_bytes = (out / "result.json").read_bytes()
        audit_bytes = (out / "mechanism_acceptance.json").read_bytes()
        # The original checker accepts len70 diagnostics with duplicated round numbers;
        # update the result SHA to reach the supplemental actual-behavior check.
        bad = ev.read(out / "result.json")
        bad["round_summaries"][1]["round"] = 1
        ev.save(out / "result.json", bad)
        audit = ev.read(out / "mechanism_acceptance.json")
        audit["result_sha256"] = ev.digest(out / "result.json")
        ev.save(out / "mechanism_acceptance.json", audit)
        assert worker.checked(original, job, Path(entry["job"])) is not None
        failures["missing_diagnostic_round"] = rejected(lambda: ev.accept_new(original, worker, job, entry, manifest, full_record))
        (out / "result.json").write_bytes(result_bytes)
        (out / "mechanism_acceptance.json").write_bytes(audit_bytes)
        ledger_bytes = (out / "candidate_mask_audit.json").read_bytes()
        ledger = ev.read(out / "candidate_mask_audit.json")
        ledger[0]["component"] = "C"
        ev.save(out / "candidate_mask_audit.json", ledger)
        audit = ev.read(out / "mechanism_acceptance.json")
        audit["candidate_audit_sha256"] = ev.digest(out / "candidate_mask_audit.json")
        ev.save(out / "mechanism_acceptance.json", audit)
        assert worker.checked(original, job, Path(entry["job"])) is not None
        failures["contradictory_candidate_ledger_component"] = rejected(lambda: ev.accept_new(original, worker, job, entry, manifest, full_record))
        (out / "candidate_mask_audit.json").write_bytes(ledger_bytes)
        (out / "mechanism_acceptance.json").write_bytes(audit_bytes)
        for field, value in [("actual_evaluation_rows", 1), ("evaluation_image_ids_sha256", "d" * 64)]:
            bad = ev.read(out / "result.json")
            bad["data_contract"]["image_data_contract"][field] = value
            ev.save(out / "result.json", bad)
            failures[field] = rejected(lambda: ev.accept_new(original, worker, job, entry, manifest, full_record))
            (out / "result.json").write_bytes(result_bytes)
        for field, value in [("root_image_ids_sha256", "d" * 64), ("train_image_ids_sha256", "d" * 64),
                             ("client_sample_counts", [1] * 20)]:
            bad = ev.read(out / "result.json")
            bad["data_contract"]["image_data_contract"][field] = value
            ev.save(out / "result.json", bad)
            audit = ev.read(out / "mechanism_acceptance.json")
            audit["result_sha256"] = ev.digest(out / "result.json")
            ev.save(out / "mechanism_acceptance.json", audit)
            assert worker.checked(original, job, Path(entry["job"])) is not None
            failures[field] = rejected(lambda: ev.accept_new(original, worker, job, entry, manifest, full_record))
            (out / "result.json").write_bytes(result_bytes)
            (out / "mechanism_acceptance.json").write_bytes(audit_bytes)
        model = (out / "model.pt").read_bytes()
        (out / "model.pt").write_bytes(model + b"tamper")
        failures["mixed_checkpoint"] = rejected(lambda: ev.accept_new(original, worker, job, entry, manifest, full_record))
    return {"status": "PASS", "existing_worker_checked_and_original_checked_result_used": True,
            "fixture_rounds": 70, "candidate_calls": 700, "new_training": False, "rejections": failures}


def incremental_backup():
    with tempfile.TemporaryDirectory(dir=HERE, prefix="backup_gate_") as directory:
        root = Path(directory).resolve()
        root.relative_to(HERE.resolve())
        manifest, job, entry, _full_record = synthetic_new(root)
        failed_id = "minus_C_IID_Benign_seed91001"
        manifest = {"jobs": [entry, {"id": failed_id}]}
        ev.save(root / "manifest.json", manifest)
        manifest_sha = ev.digest(root / "manifest.json")
        ev.save(root / "dispatch_receipt.json", {"manifest_sha256": manifest_sha, "fixture_only": True,
                                               "strict_prerequisites": {"Full_inspection_sha256": "a" * 64,
                                                                         "pipeline_offserver_verified": True,
                                                                         "future_schema_field": {"preserve": "exact bytes"}}})
        result = ev.read(Path(job["output"]) / "result.json")
        row = ev.row_from_result(entry, result, "new")
        snapshot = root / "inspection"
        snapshot.mkdir()
        for name in ("statistics.json", "per_seed.csv", "paired_per_seed.csv", "per_scene_summary.csv"):
            (snapshot / name).write_text("{}\n")
        inventory = root / "full_inventory.json"
        ev.save(inventory, {"fixture_only": True})
        report = {"status": "PARTIAL", "source_script_sha256": ev.digest(ev.__file__), "manifest": str(root / "manifest.json"),
                  "manifest_sha256": manifest_sha, "accepted_new_ids": [entry["id"]], "records": [row], "invalid": [],
                  "repo": str(root), "stage": str(root), "full_inventory": str(inventory), "frozen_source_hashes": {},
                  "adapter_hashes": {}, "adapter_dir": str(root), "reused_full_restore_chain": [{"id": "Full_old_chain_no_weights"}]}
        inspection = snapshot / "inspection.json"
        def save_snapshot(value):
            ev.save(inspection, value)
            (snapshot / "inspection.sha256").write_text(ev.digest(inspection) + "\n")
        save_snapshot(report)
        ledger = root / "backup_ledger.json"
        first = ev.backup(inspection, ledger, root / "incremental1.tar.gz", initialize=True)
        assert first["accepted_new_ids"] == [entry["id"]]
        receipt = ev.read(first["receipt"])
        verified = ev.verify_archive(Path(first["archive"]), receipt)
        assert verified["pass"]
        with tarfile.open(first["archive"], "r:gz") as archive:
            names = archive.getnames()
            assert sum(name.endswith("/model.pt") for name in names) == 1
            assert all("Full_old_chain_no_weights/model.pt" not in name for name in names)
            assert archive.extractfile("sourcefreeze/dispatch_receipt.json").read() == (root / "dispatch_receipt.json").read_bytes()
        second = ev.backup(inspection, ledger, root / "incremental2.tar.gz")
        assert second["status"] == "NO_NEW_VERIFIED_EVIDENCE" and not (root / "incremental2.tar.gz").exists()
        failures = {}
        tampered = root / "tampered.tar.gz"
        tampered.write_bytes(Path(first["archive"]).read_bytes() + b"tamper")
        failures["archive_bytes_tampered"] = rejected(lambda: ev.verify_archive(tampered, receipt))
        tampered_member = root / "tampered_member.tar.gz"
        import io
        with tarfile.open(first["archive"], "r:gz") as source, tarfile.open(tampered_member, "w:gz") as target:
            for member in source:
                data = source.extractfile(member).read()
                if member.name.endswith("/result.json"):
                    data += b"tamper"
                    member.size = len(data)
                target.addfile(member, io.BytesIO(data))
        changed_archive_receipt = dict(receipt, archive_sha256=ev.digest(tampered_member))
        failures["member_tamper_with_archive_SHA_refreshed"] = rejected(lambda: ev.verify_archive(tampered_member, changed_archive_receipt))
        fake_receipt = copy.deepcopy(receipt)
        fake_receipt["accepted_new_ids"] = ["other_job"]
        failures["receipt_ID_mismatch"] = rejected(lambda: ev.verify_archive(Path(first["archive"]), fake_receipt))
        duplicate = copy.deepcopy(report)
        duplicate["accepted_new_ids"].append(entry["id"])
        save_snapshot(duplicate)
        failures["duplicate_accepted_ID"] = rejected(lambda: ev.backup(inspection, ledger, root / "duplicate.tar.gz"))
        mixed = copy.deepcopy(report)
        mixed["records"][0]["checkpoint_sha256"] = "d" * 64
        save_snapshot(mixed)
        failures["mixed_report_checkpoint"] = rejected(lambda: ev.backup(inspection, ledger, root / "mixed.tar.gz"))
        save_snapshot(report)
        failure_path = root / "runs" / failed_id / "failure.json"
        ev.save(failure_path, {"preserved": True, "fixture_only": True})
        failure_report = copy.deepcopy(report)
        failure_report["invalid"] = [{"id": failed_id, "preserved_files": {str(failure_path): ev.digest(failure_path)}, "failure_identities": {str(failure_path): str(failure_path) + "=" + ev.digest(failure_path)}}]
        save_snapshot(failure_report)
        failure_only = ev.backup(inspection, ledger, root / "failure_only.tar.gz")
        assert failure_only["accepted_new_ids"] == []
        with tarfile.open(failure_only["archive"], "r:gz") as archive:
            assert sum(member.name.endswith("/model.pt") for member in archive) == 0
        assert ev.backup(inspection, ledger, root / "failure_repeated.tar.gz")["status"] == "NO_NEW_VERIFIED_EVIDENCE"
        ledger_value = ev.read(ledger)
        ledger_value["entries"].append(copy.deepcopy(ledger_value["entries"][0]))
        ev.save(ledger, ledger_value)
        failures["duplicate_backup_ledger_entry"] = rejected(lambda: ev.backup(inspection, ledger, root / "badledger.tar.gz"))
        return {"status": "PASS", "first_backup_new_models": 1, "repeat_backup_new_models": 0,
                "reused_Full_models_repacked": 0, "archive_members_verified": verified["members_verified"],
                "failure_only_incremental_new_models": 0, "strict_prerequisites_preserved_as_exact_bytes": True,
                "off_server_verification_claimed": False, "rejections": failures}


def main():
    success = False
    try:
        assert ev.digest(v1.__file__) == "1c0961ae991d75d32d3269e967ac6bfdcdfb783afafcfcc965445a99a179617b"
        for name, function in [("actual_archived_Full_strict_gate", actual_archived_full_gate),
                               ("design_and_hand_computed_statistics", design_and_statistics),
                               ("existing_checked_new_candidate_evidence", real_checked_new_path),
                               ("incremental_backup_and_corruption_rejection", incremental_backup)]:
            CHECKS.append({"name": name, **function()})
            print("PASS " + name, flush=True)
        success = True
    except Exception:
        CHECKS.append({"status": "FAIL", "traceback": traceback.format_exc()})
        raise
    finally:
        ev.save(HERE / "selfcheck_v4.json", {"status": "PASS" if success else "FAIL", "checks": CHECKS, "evidence_sha256": ev.digest(ev.__file__),
                                        "server_accessed": False, "new_training": False, "test_labels_read": False})
    print(json.dumps({"status": "PASS", "checks": len(CHECKS)}))


if __name__ == "__main__":
    main()
