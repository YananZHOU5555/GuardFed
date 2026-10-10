"""Metadata/source regression only; no CNN, tensor computation or model output."""
import ast
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STAGE = HERE / "snapshot/gradient_bridge_20261010"
sys.path.insert(0, str(STAGE))
import worker
import accept_result


def check():
    protocol = json.loads((STAGE / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "jobs/manifest.json").read_text(encoding="utf-8"))
    original = HERE.parent / "celeba_baselines/gradient_bridge_20261009"
    namespace = {"ValueError": ValueError, "KeyError": KeyError, "FileNotFoundError": FileNotFoundError, "RuntimeError": RuntimeError}
    tree = ast.parse((original / "check_acceptance.py").read_text(encoding="utf-8"))
    refusal = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "expect_refused")
    exec(compile(ast.Module(body=[refusal], type_ignores=[]), "original_expect_refused", "exec"), namespace)
    refused = namespace["expect_refused"]
    assert len(manifest["jobs"]) == len({r["id"] for r in manifest["jobs"]}) == 64
    assert protocol["status"] == "FROZEN" and all(d["status"] == "APPROVED" for d in protocol["protocol_decisions"].values())
    assert protocol["author_decisions_sha256"] == worker.digest(HERE / "AUTHOR_DECISIONS.json")
    assert protocol["real_image_gate_root_sha256"] == worker.digest(HERE / "GRADIENT_FOUR_ROOT_VERIFICATION.json")
    old_protocol = json.loads((original / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["candidates"] == old_protocol["candidates"] and protocol["source_hashes"] == old_protocol["source_hashes"]
    for filename in ("worker.py", "accept_result.py"):
        assert (STAGE / filename).read_bytes() == (original / filename).read_bytes()
    worker.verify_hashes(STAGE.parent, worker.COMPONENTS)
    groups = {}
    for entry in manifest["jobs"]:
        job_path = HERE / "jobs" / entry["job"]
        assert worker.digest(job_path) == entry["job_sha256"]
        job = json.loads(job_path.read_text(encoding="utf-8"))
        worker.validate_job(job, protocol)
        worker.verify_hashes(STAGE, job["local_hashes"])
        groups.setdefault(job["tuning_candidate"], []).append(job)
    assert len(groups) == 16 and all(len(g) == 4 for g in groups.values())
    assert all({(r["distribution"], r["attack"]) for r in g} == {("IID", "Benign"), ("IID", "S-DFA"), ("non-IID", "Benign"), ("non-IID", "S-DFA")} for g in groups.values())
    row = next(e for e in manifest["jobs"] if e["method"] == "Fed-NGA-gradient" and "_S-DFA_" in e["id"])
    path = HERE / "jobs" / row["job"]
    job = json.loads(path.read_text(encoding="utf-8"))
    checks = []
    for field, value in (("seed", 42), ("rounds", 3), ("client_alpha", 17.), ("celeba_evaluation_split", "test"), ("use_reweighting", True)):
        bad = copy.deepcopy(job); bad["config"][field] = value
        checks.append(refused(lambda: worker.validate_job(bad, protocol), "job mutation " + field))
    for field, value in (("objective", "shared_reweighted_ce"), ("root_reference", "same_point_unweighted_gradient"), ("server_eta", .004)):
        bad = copy.deepcopy(job); bad["adapter"][field] = value
        checks.append(refused(lambda: worker.validate_job(bad, protocol), "adapter mutation " + field))
    checks.append(refused(lambda: worker.validate_job(job, old_protocol), "old PREPARED protocol"))
    assert accept_result.checked_result(path, HERE / "fixtures/absent_result") is None
    failure = HERE / "fixtures/preserved_failure"
    assert json.loads((failure / "failure.json").read_text())["mock_only"] is True
    checks.append(refused(lambda: accept_result.checked_result(path, failure), "preserved failure blocks acceptance"))
    assert worker.digest(HERE / "frozen_score.py") == "1b31c0322b06bc1901f2a2f49a2b7b3524fed0164b6191d8bb9a5ed7e5f24b27"
    gate = json.loads((HERE / "GRADIENT_FOUR_ROOT_VERIFICATION.json").read_text())
    assert (gate["canaries"], gate["actual_rounds"], gate["same_point_gradient_checks"], gate["scientific_table_records"]) == (4, 12, 240, 0)
    accepted_snapshot = HERE.parent / "celeba_gradient_realimage_gate_20261009/snapshot"
    for filename in ("worker.py", "accept_result.py"):
        assert (STAGE / filename).read_bytes() == (accepted_snapshot / "gradient_bridge_20261009" / filename).read_bytes()
    old_acceptance = json.loads((original / "acceptance_gate.json").read_text())
    assert old_acceptance["status"] == "PASS" and old_acceptance["scientific_results"] == 0
    sys.path.insert(0, str(HERE))
    import run_queue
    old_digest = run_queue.digest
    run_queue.digest = lambda p: "MOCK_PACKAGE_NOT_REAL" if Path(p).name == "FILES_SHA256.json" else old_digest(p)
    mock = dict(status="ROOT_GRADIENT64_RESOURCE_PREFLIGHT_PASS", at_unix=1000, cpu_ids=[105], cpu_threads=1,
        max_workers=1, cuda_visible_device="1", nice=10, idle_io=True, all_threads_cpu105_idle=True,
        no_duplicate_worker=True, no_restricted_CPU_overlap=True, source_data_hashes_verified=True,
        existing_nominal_compute_threads=20, actual_quota_cores=128, gpu_free_memory_mib=8192,
        gpu_uuid="MOCK_GPU_NOT_REAL", guide_sha256=run_queue.GUIDE_SHA, package_seal_sha256="MOCK_PACKAGE_NOT_REAL",
        author_decisions_sha256=worker.digest(HERE / "AUTHOR_DECISIONS.json"), test_authorized=False,
        automatic_retry_authorized=False)
    try:
        run_queue.validate_resources(mock, 1001)
        for field, value in (("cpu_ids", [104]), ("source_data_hashes_verified", False), ("at_unix", 800),
                             ("existing_nominal_compute_threads", 128)):
            bad = dict(mock, **{field: value})
            checks.append(refused(lambda: run_queue.validate_resources(bad, 1001), "MOCK preflight mutation " + field))
    finally:
        run_queue.digest = old_digest
    for source, expected in json.loads((HERE / "INPUT_PINS.json").read_text())["source_pins"].items():
        assert worker.digest(HERE.parents[1] / source) == expected
    return dict(status="PASS_SOURCE_METADATA_REGRESSION_NOT_EXECUTION", prepared_jobs=64,
                methods={m: sum(e["method"] == m for e in manifest["jobs"]) for m in worker.METHODS},
                candidates_exact_original16=True, worker_and_checker_original_bytes_exact=True,
                component_original_bytes_exact=True, old_sources_unchanged=True, rejection_checks=checks,
                original_acceptance_gate_reused=True, original_checked_missing_result_and_failure_regressed=True,
                preflight_MOCK_positive=True, no_actual_preflight_artifact_created=True,
                existing_real_image_gate_reused=True, actual_new_CNN_or_training=False, scientific_records=0,
                actual_resource_preflight=False, actual_dispatch=False)


if __name__ == "__main__":
    result = check()
    (HERE / "SELF_CHECK.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "prepared_jobs", "scientific_records", "actual_dispatch")}))
