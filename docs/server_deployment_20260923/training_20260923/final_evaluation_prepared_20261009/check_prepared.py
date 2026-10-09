"""Validate the preparation package. This file does not evaluate a model."""
import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def require_dispatchable(protocol, jobs):
    """A future launcher must use a frozen successor, not this preparation."""
    if protocol["status"] != "FROZEN":
        raise RuntimeError("PREPARED_NOT_FROZEN: evaluation dispatch is prohibited")
    if any(value is None for value in protocol["decisions"].values()):
        raise RuntimeError("Unresolved major protocol decisions")
    if protocol.get("target_image_ids_sha256") is None:
        raise RuntimeError("Target split identity has not been frozen")
    if jobs.get("status") != "FROZEN" or not jobs.get("dispatch_authorization_receipt"):
        raise RuntimeError("No frozen job manifest and dispatch authorization receipt")


def main():
    protocol = read("protocol.json")
    inventory = read("model_inventory.json")
    jobs = read("evaluation_jobs_draft.json")
    accepted = read("prepared_acceptance.json")
    issues = read("identity_issues.json")
    assert protocol["status"] == inventory["status"] == jobs["status"] == "PREPARED_NOT_FROZEN"
    assert all(value is None for value in protocol["decisions"].values())
    assert not jobs["dispatch_authorization_receipt"] and not jobs["test_called"]
    assert jobs["model_inventory_sha256"] == hashlib.sha256((HERE / "model_inventory.json").read_bytes()).hexdigest()
    assert accepted["status"] == "PASS_PREPARATION_ONLY" and accepted["record_identity_issues"] == 0
    assert not issues["record_level_identity_gaps"]
    rows = inventory["records"]
    expected = set(itertools.product(inventory["methods"], ["IID", "non-IID"], protocol["attacks"], range(91001, 91011)))
    actual = [(r["method"], r["distribution"], r["attack"], r["seed"]) for r in rows]
    assert len(rows) == len(set(actual)) == len(expected) == 900 and set(actual) == expected
    assert len(jobs["jobs"]) == 900 and {r["id"] for r in rows} == {j["model_id"] for j in jobs["jobs"]}
    for r, j in zip(rows, jobs["jobs"]):
        assert r["id"] == j["model_id"] and r["checkpoint"]["sha256"] == j["checkpoint_sha256"]
        assert r["terminal_round"] == 70 and r["original_split"] == "valid"
        assert r["actual_alpha"] == (5000.0 if r["distribution"] == "IID" else 5.0)
        assert r["raw_job"] and r["config"]["seed"] == r["seed"]
        h = hashlib.sha256(json.dumps(r["config"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert h == r["config_canonical_sha256"]
        assert j["selected_views"] is None and j["target_image_ids_sha256"] is None and not j["dispatchable"]
        assert not r["test_evaluation_performed"]
    assert inventory["training_environment_counts"] == {"2.11.0+cu128": 886, "2.11.0+cu130": 14}
    assert protocol["prior_official_test_exposed"] is True
    assert protocol["recipe_selection_seed"] == 91001
    # Merely changing the status must not silently authorize collection.
    for candidate in [protocol, dict(protocol, status="FROZEN")]:
        try:
            require_dispatchable(candidate, jobs)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Unresolved preparation passed dispatch guard")
    result = {"status": "PASS_PREPARATION_ONLY", "unique_paired_grid": 900,
              "config_and_job_bindings": 900, "dispatch_rejection_cases": 2,
              "test_called": False, "inference_performed": False, "new_training": False,
              "limitations": "Not final-result acceptance; live source/model/data and target split preflight remain pending."}
    (HERE / "independent_prepared_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
