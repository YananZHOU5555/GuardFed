"""Read existing accepted artifacts only. No torch, model inference or data-label loading."""
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import tarfile

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
REPO = BASE.parents[2]
TABLES = REPO / "outputs/guardfed_tables/celeba_nine_method_final_20261004"
METHODS = ["FedAvg", "FairFed", "Median", "FLTrust", "FairGuard", "FLTrust+FairGuard", "GuardFed-AD2+", "FedAA", "LASA"]
ATTACKS = ["Benign", "F Flip", "FedSA", "S-DFA", "Sp-DFA"]
input_files = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    raw = path.read_bytes()
    input_files[str(path.relative_to(REPO)).replace("\\", "/")] = digest(raw)
    return json.loads(raw)


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def member_path(remote):
    return remote.split("/workspace/", 1)[-1].split("/", 1)[-1] if remote.startswith("/workspace/") else remote.lstrip("/")


def main():
    stage = load(TABLES / "source_stageA700_snapshot.json")
    added = load(TABLES / "source_baseline200_snapshot.json")
    identity = load(TABLES / "identity_evidence.json")
    old_manifest = load(BASE / "celeba_fullcoverage_v1/manifest.json")
    new_manifest = load(BASE / "celeba_baseline_fullcoverage_v1/manifest.json")
    assert stage["complete"] and added["complete"] and identity["pass"]
    assert file_digest(BASE / "celeba_fullcoverage_v1/manifest.json") == stage["source_manifest_sha256"]
    assert file_digest(BASE / "celeba_baseline_fullcoverage_v1/manifest.json") == added["manifest_sha256"]
    rows = [dict(r, cohort="StageA700") for r in stage["all_conditions"]]
    rows += [dict(r, method=r["algorithm"], cohort="FedAA_LASA200") for r in added["records"]]
    expected = set(itertools.product(METHODS, ["IID", "non-IID"], ATTACKS, range(91001, 91011)))
    assert len(rows) == 900 and {(r["method"], r["distribution"], r["attack"], r["seed"]) for r in rows} == expected
    assert len({r["id"] for r in rows}) == 900
    by_id = {r["id"]: r for r in rows}
    targets = {member_path(r["output"]) + "/" + name for r in rows
               for name in ["result.json", "model.pt", "input_record.json", "provenance.json", "acceptance.json"]}
    declared_jobs = {}
    for job_path, sha in old_manifest["job_sha256s"].items():
        declared_jobs[member_path(job_path)] = sha
    for entry in new_manifest["jobs"] + new_manifest["reused_jobs"]:
        declared_jobs[member_path(entry["job"])] = entry["job_sha256"]
    targets.update(declared_jobs)
    members, jobs, archives = {}, collections.defaultdict(list), []

    def scan(path, expected_sha, receipt):
        actual_sha = file_digest(path)
        assert actual_sha == expected_sha, (path.name, "archive hash drift")
        local = str(path.relative_to(REPO)).replace("\\", "/")
        archive_record = {"archive": local, "sha256": actual_sha, "bytes": path.stat().st_size,
                          "expected_sha_source": receipt, "archive_sha_verified_now": True}
        selected, inventories = {}, []
        with tarfile.open(path, "r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                wanted = m.name in targets
                potential_job = m.name.endswith(".json") and Path(m.name).stem in by_id and "/runs/" not in m.name
                inventory = "inventory" in m.name.lower() and m.name.endswith(".json")
                if not (wanted or potential_job or inventory):
                    continue
                raw = tf.extractfile(m).read()
                rec = {"archive": local, "archive_sha256": actual_sha, "member": m.name,
                       "sha256": digest(raw), "bytes": len(raw), "archive_sha_verified_now": True}
                if not m.name.endswith(".pt"):
                    rec["json"] = json.loads(raw)
                if inventory:
                    inv = rec["json"]
                    if isinstance(inv, dict):
                        inventories.append(inv.get("files", inv))
                if wanted:
                    selected[m.name] = rec
                if potential_job and isinstance(rec.get("json"), dict) and rec["json"].get("id") in by_id:
                    jobs[rec["json"]["id"]].append(rec)
        verified_inventory = 0
        for name, rec in selected.items():
            expectations = [inv[name] for inv in inventories if name in inv]
            for exp in expectations:
                sha = exp["sha256"] if isinstance(exp, dict) else exp
                assert rec["sha256"] == sha, (path.name, name, "member inventory drift")
            rec["member_inventory_verified_now"] = bool(expectations)
            verified_inventory += bool(expectations)
            if name in members:
                assert members[name]["sha256"] == rec["sha256"], (name, "conflicting backup copies")
            else:
                members[name] = rec
        archive_record.update(selected_members=len(selected), selected_member_inventory_checks=verified_inventory)
        archives.append(archive_record)
        print(json.dumps({"archive_checked": path.name, "selected_members": len(selected)}), flush=True)

    restore = load(BASE / "celeba_fullcoverage_v1/restore_chain_final.json")
    for item in restore["archives"]:
        scan(BASE / "celeba_fullcoverage_v1" / item["archive"], item["sha256"], "celeba_fullcoverage_v1/restore_chain_final.json")
    restore_new = load(BASE / "celeba_baseline_fullcoverage_v1/restore_chain.json")
    for name in restore_new["increments"]:
        receipt = load(BASE / "celeba_baseline_fullcoverage_v1" / name)
        scan(BASE / "celeba_baseline_fullcoverage_v1" / Path(receipt["archive"]).name, receipt["sha256"], "celeba_baseline_fullcoverage_v1/" + name)
    setup = load(BASE / "celeba_baseline_fullcoverage_v1/setup_receipt.json")
    scan(BASE / "celeba_baseline_fullcoverage_v1/fullcoverage_setup_20261003.tar.gz", setup["sha256"], "celeba_baseline_fullcoverage_v1/setup_receipt.json")
    screen = load(BASE / "celeba_baseline_screen_v1/restore_chain.json")
    for receipt in screen["increments"] + [screen["setup"]]:
        scan(BASE / "celeba_baseline_screen_v1" / Path(receipt["archive"]).name, receipt["sha256"], "celeba_baseline_screen_v1/restore_chain.json")
    runtime = load(BASE / "TRAINING_STATE.json")
    seed_backup = load(BASE / "celeba_seedcheck_v1/latest_backup.json")
    for key, hash_key in [("archive", "sha256"), ("prior_archive", "prior_sha256")]:
        scan(BASE / "celeba_seedcheck_v1" / Path(seed_backup[key]).name, seed_backup[hash_key], "celeba_seedcheck_v1/latest_backup.json")
    expanded_final = load(BASE / "celeba_expanded_v2/final/backup_verified.json")
    scan(BASE / "celeba_expanded_v2" / expanded_final["archive"], expanded_final["sha256"], "celeba_expanded_v2/final/backup_verified.json")
    for filename, sha, ref in [
        ("celeba_expanded_incremental34_total74_failures3_20260924T150941Z.tar.gz", "0d61b2544c060a6c592d9f1c3f94cc990a711833a6ed10fa0183beb2b5aa43b2", "TRAINING_STATE.json historical expanded backup"),
        ("celeba_expanded_incremental40_total40_20260924T120757Z.tar.gz", "3051adf180be921d81d22cb688220182e36869f3e834db7d8947884126175ee3", "RUNNING.md historical 2026-09-24T12:07:57Z backup")]:
        scan(BASE / "celeba_expanded_v2" / filename, sha, ref)
    for filename, sha, ref in [
        ("celeba_tuning_accepted17_20260924T041544Z.tar.gz", "480306517396418a3a3b2cb7f40c796988391d4b34930a0c9b7c3b3db64d652f", "RUNNING.md historical 2026-09-24T04:15:44Z backup"),
        ("celeba_tuning_final_incremental23_total40_20260924.tar.gz", "f09c0501df8e6bc580455db565f3cbf72978c1236fc602b47a24bd69c10a6dbe", "TRAINING_STATE.json celeba_tuning_v1.latest_check")]:
        scan(BASE / filename, sha, ref)

    issues, records = [], []
    for row in rows:
        rid = row["id"]
        prefix = member_path(row["output"])
        required = [prefix + "/result.json", prefix + "/model.pt"]
        absent = [name for name in required if name not in members]
        if absent:
            issues.append({"id": rid, "issue": "missing_backup_member", "members": absent})
            continue
        rr, mr = (members[name] for name in required)
        result = rr["json"]
        assert mr["sha256"] == row["checkpoint_sha256"], (rid, "accepted model identity")
        cfg = result["config"]
        assert result["rounds"] == cfg["rounds"] == 70
        assert cfg["seed"] == row["seed"] and cfg["client_alpha"] == (5000.0 if row["distribution"] == "IID" else 5.0)
        assert cfg["celeba_evaluation_split"] == "valid" and cfg["celeba_train_limit"] == cfg["celeba_eval_limit"] == 0
        assert result["distribution"] == row["distribution"] and result["attack"] == row["attack"]
        assert [x["round"] for x in result["trajectory_metrics"]] == list(range(1, 71))
        assert [x["round"] for x in result["round_summaries"]] == list(range(1, 71))
        assert all(math.isfinite(result["metrics"][k]) and result["metrics"][k] == row[k] == result["trajectory_metrics"][-1]["metrics"][k] for k in ["accuracy", "aeod", "aspd"])
        contract = result["data_contract"]
        contract = contract.get("image_data_contract", contract)
        assert contract["evaluation_split"] == "valid" and contract["actual_train_rows"] == 162770 and contract["actual_evaluation_rows"] == 19867
        assert contract["train_eval_disjoint"] and contract["root_client_disjoint"]
        assert contract["official_split_sizes"]["2"] == 19962
        # Adapter jobs bind their output through the parent manifest rather than
        # embedding an output field in the job; their exact byte hash is declared.
        job_candidates = [j for j in jobs[rid] if
                          member_path(j["json"].get("output", row["output"])) == prefix]
        manifest_bound = [j for j in job_candidates if
                          declared_jobs.get(j["member"]) == j["sha256"]]
        if manifest_bound:
            job_candidates = manifest_bound
        if not job_candidates:
            issues.append({"id": rid, "issue": "raw_job_bytes_not_located", "output": row["output"]})
            job = result.get("revision_job", {})
            jr = None
        else:
            jr = job_candidates[0]
            assert all(j["sha256"] == jr["sha256"] for j in job_candidates), (rid, "conflicting unbound job copies")
            job = jr["json"]
            assert job["config"] == cfg, (rid, "job-result config mismatch")
            if jr["member"] in declared_jobs:
                assert jr["sha256"] == declared_jobs[jr["member"]]
        provenance = members.get(prefix + "/provenance.json", {}).get("json", {})
        input_record = members.get(prefix + "/input_record.json", {}).get("json", {})
        source_hashes = result.get("source_hashes") or job.get("source_hashes") or provenance.get("source_hashes")
        assert source_hashes and source_hashes["scripts/reproduce_paper_tables.py"] == "cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed"
        adapter_hashes = result.get("identity", {}).get("adapter_hashes") or provenance.get("adapter_source_hashes") or job.get("adapter_source_hashes", {})
        torch = result.get("identity", {}).get("environment", {}).get("torch") or result.get("revision_job", {}).get("torch_version")
        torch_source = "original_result.identity.environment.torch" if "identity" in result else "original_result.revision_job.torch_version"
        if not torch:
            torch = row.get("torch_version") or identity["atomic_identity_environment"][rid]["torch_version"]
            torch_source = "previous_strict_acceptance_ledger"
        assert torch in ["2.11.0+cu128", "2.11.0+cu130"]
        if result.get("checkpoint_sha256"):
            assert result["checkpoint_sha256"] == mr["sha256"]
        def reference(rec):
            return {k: v for k, v in rec.items() if k != "json"}
        record = {"id": rid, "method": row["method"], "source_method": result["method"],
                  "distribution": row["distribution"], "actual_alpha": cfg["client_alpha"], "attack": row["attack"],
                  "seed": row["seed"], "candidate": row["candidate"], "cohort": row["cohort"],
                  "terminal_round": 70, "original_split": "valid", "original_n_eval": 19867,
                  "checkpoint": reference(mr), "result": reference(rr), "raw_job": reference(jr) if jr else None,
                  "original_remote_output": row["output"], "config": cfg,
                  "config_canonical_sha256": digest(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()),
                  "source_hashes": source_hashes, "adapter_source_hashes": adapter_hashes,
                  "data_contract": contract, "training_torch": torch, "training_torch_source": torch_source,
                  "native_prediction_rule": "root_fitted_group_thresholds_from_original_recipe" if row["method"] == "GuardFed-AD2+" else "argmax_logits_tie_class0",
                  "prior_validation_metrics": {k: row[k] for k in ["accuracy", "aeod", "aspd"]},
                  "recipe_selection_seed": 91001, "seed_in_recipe_selection": row["seed"] == 91001,
                  "evaluation_status": "PREPARED_NOT_FROZEN", "target_split_candidate": "official_test_partition2",
                  "target_image_ids_sha256": None, "test_evaluation_performed": False,
                  "model_reconstruction": "CelebACNN frozen core state_dict; load weights_only=True; verify keys/shapes/finite tensors at server preflight",
                  "replay_metadata": {"input_record_sha256": members.get(prefix + "/input_record.json", {}).get("sha256"),
                                      "provenance_sha256": members.get(prefix + "/provenance.json", {}).get("sha256"),
                                      "acceptance_sha256": members.get(prefix + "/acceptance.json", {}).get("sha256")}}
        records.append(record)
    records.sort(key=lambda r: (METHODS.index(r["method"]), r["distribution"], ATTACKS.index(r["attack"]), r["seed"]))
    env = dict(collections.Counter(r["training_torch"] for r in records))
    save("identity_issues.json", {"status": "PREPARED_NOT_FROZEN", "record_level_identity_gaps": issues,
          "future_global_gates": ["Current server files and GPU environment not observable; rehash before evaluation.",
                                  "Official test partition2 image-ID/order SHA not yet attested locally; verify metadata without labels before protocol freeze.",
                                  "Evaluation implementation is not frozen or dispatched; native validation replay must pass all900 before test inference.",
                                  "Primary endpoint/views/claim wording require decision; no test result may choose them."]})
    save("model_inventory.json", {"status": "PREPARED_NOT_FROZEN", "scope": "current accepted nine-method 900 only", "records": records,
          "expected": 900, "located": len(records), "methods": METHODS, "training_environment_counts": env,
          "test_called": False, "new_training": False, "archives": archives, "input_files": input_files})
    fields = ["id", "method", "source_method", "distribution", "actual_alpha", "attack", "seed", "candidate", "terminal_round", "original_split", "original_n_eval", "training_torch", "native_prediction_rule", "config_canonical_sha256"]
    with (OUT / "model_inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["checkpoint_sha256", "result_sha256", "job_sha256", "backup_archive"])
        w.writeheader()
        for r in records:
            w.writerow({**{k: r[k] for k in fields}, "checkpoint_sha256": r["checkpoint"]["sha256"], "result_sha256": r["result"]["sha256"],
                        "job_sha256": r["raw_job"]["sha256"] if r["raw_job"] else None, "backup_archive": r["checkpoint"]["archive"]})
    save("prepared_acceptance.json", {"status": "PASS_PREPARATION_ONLY" if len(records) == 900 and not issues else "PREPARATION_HAS_IDENTITY_GAPS",
          "model_result_config_source_data_identity_records": len(records), "round70_same_checkpoint_metrics_checked": len(records),
          "archive_sha256_checked": len(archives), "model_bytes_hashed_without_loading": len(records), "result_bytes_hashed": len(records),
          "raw_job_bytes_located": sum(r["raw_job"] is not None for r in records), "record_identity_issues": len(issues),
          "environment_counts": env, "no_torch_import_or_inference": True, "test_labels_read": False,
          "no_new_training": True, "dispatch_ready": False, "current_server_state_checked": False})
    print(json.dumps({"located": len(records), "issues": len(issues), "archives": len(archives), "environments": env}), flush=True)


if __name__ == "__main__":
    main()
