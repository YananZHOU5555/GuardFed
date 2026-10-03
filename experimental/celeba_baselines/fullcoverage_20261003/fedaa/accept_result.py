"""Strict acceptance for fixed-recipe multi-seed coverage/canary; no launches."""
import argparse
import json
import math
from pathlib import Path

from run_fedaa_screen import STAGE_VERSION, digest, validate_job


def checked_result(job_path, out):
    if not (out / "result.json").exists():
        return None
    import torch
    job = json.loads(job_path.read_text(encoding="utf-8"))
    validate_job(job)
    r = json.loads((out / "result.json").read_text(encoding="utf-8"))
    expected_status = "coverage_complete" if job["evidence_stage"] == "multi_seed_validation_coverage" else "pilot_complete"
    assert r["schema"] == "fedaa_validation_screen_v1"
    assert r["stage_version"] == STAGE_VERSION
    for k in ["id", "config", "method", "attack", "distribution", "dataset", "evidence_stage", "policy_config", "aggre_num", "tuning_candidate", "source_hashes", "adapter_hashes"]:
        assert r[k] == job[k], k
    assert r["status"] == expected_status
    rounds = job["config"]["rounds"]
    assert r["rounds"] == r["planned_rounds"] == rounds
    assert r["seed"] == job["config"]["seed"]
    assert [x["round"] for x in r["trajectory_metrics"]] == list(range(1, rounds + 1))
    assert [x["round"] for x in r["round_summaries"]] == list(range(1, rounds + 1))
    assert r["metrics"] == r["trajectory_metrics"][-1]["metrics"]
    for row, diag in zip(r["trajectory_metrics"], r["round_summaries"]):
        assert row["evaluation_stats"]["prediction_count"] == 19867
        assert all(math.isfinite(row["metrics"][k]) for k in ["accuracy", "aeod", "aspd"])
        agg = diag["aggregate"]
        assert agg["policy_updates"] == row["round"]
        assert len(set(agg["selected_client_ids"])) == job["aggre_num"]
        assert set(agg["selected_client_ids"]) <= set(range(20))
        assert agg["reward_split"] == "train_clean_root"
    dc = r["data_contract"]
    assert (dc["evaluation_split"], dc["actual_train_rows"], dc["actual_evaluation_rows"]) == ("valid", 162770, 19867)
    assert dc["train_eval_disjoint"] and dc["root_client_disjoint"]
    assert digest(out / "model.pt") == r["checkpoint_sha256"]
    assert digest(out / "training_state.pt") == r["full_checkpoint_sha256"]
    saved = torch.load(out / "training_state.pt", map_location="cpu", weights_only=False)
    assert saved["identity"] == r["identity"]
    assert saved["identity"]["job_sha256"] == digest(job_path)
    assert saved["identity"]["source_hashes"] == job["source_hashes"]
    assert saved["identity"]["adapter_hashes"] == job["adapter_hashes"]
    assert saved["identity"]["policy_seed"] == job["policy_seed"] == job["config"]["seed"]
    assert saved["identity"]["policy_config"] == job["policy_config"]
    assert saved["trajectory_metrics"] == r["trajectory_metrics"]
    assert saved["round_summaries"] == r["round_summaries"]
    controller = saved["controller"]
    assert controller["rounds"] == controller["policy"]["transitions"] == rounds
    pc = controller["policy"]["config"]
    assert pc["aggre_num"] == job["aggre_num"] and pc["seed"] == job["policy_seed"]
    assert all(pc[k] == v for k, v in job["policy_config"].items())
    assert controller["pending"][2] is not None
    model = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    assert model.keys() == saved["model"].keys()
    assert all(torch.equal(v, saved["model"][k]) and torch.isfinite(v).all() for k, v in model.items())
    assert digest(out / "checkpoint_round1.pt") == r["first_checkpoint_sha256"]
    first = torch.load(out / "checkpoint_round1.pt", map_location="cpu", weights_only=False)
    assert first["identity"] == r["identity"] and first["controller"]["rounds"] == 1
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    r = checked_result(args.job, args.out)
    print(json.dumps(dict(status="accepted" if r else "missing", id=r["id"] if r else None,
                         rounds=r["rounds"] if r else None, metrics=r["metrics"] if r else None)))
