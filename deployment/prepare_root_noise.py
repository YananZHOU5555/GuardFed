#!/usr/bin/env python3
import contextlib
import hashlib
import io
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import torch
import reproduce_paper_tables as core
import run_revision_ablation as runner

OUT = ROOT / "results/revision_20260923/adult_root_noise_v1"
SOURCE = Path("/workspace/GuardFed/results/attack_strength/raw_results.jsonl")
ATTACKS = ("Benign", "S-DFA")
RATES = (0.1, 0.2, 0.4)
PROVENANCE_ONLY = ("experiment_suite", "experiment_tag")


def protocol_config(seed):
    return core.ExperimentConfig(
        seed=seed, rounds=70, device="cuda", server_ratio=0.1,
        synthetic_ratio=0.0, synthetic_method="none",
        fflip_mode="all_unprivileged", sdfa_foe_mode="fedsa", spdfa_foe_mode="fedsa",
        fedsa_gain=4.5, fedsa_norm_ratio=3.0)


def semantic_config(config):
    return {k: v for k, v in config.items() if k not in PROVENANCE_ONLY}


def prepare():
    torch.set_num_threads(1)
    assert not (OUT / "manifest.json").exists(), "Existing frozen manifest is not overwritten"
    controls = {}
    for line_no, line in enumerate(SOURCE.open(), 1):
        result = json.loads(line)
        if (result.get("dataset") == "adult" and result.get("distribution") == "non-IID"
            and result.get("method") == "GuardFed-AD2+" and result.get("attack") in ATTACKS
            and result.get("rounds") == 70 and result.get("num_malicious") == 4
            and result.get("seed") in runner.SEEDS):
            key = (result["attack"], result["seed"])
            assert key not in controls, ("duplicate clean control", key)
            controls[key] = (result, line_no, hashlib.sha256(line.encode()).hexdigest())
    assert len(controls) == 20
    frozen = runner.source_hashes()
    frozen[str(Path(__file__).relative_to(ROOT))] = runner.digest(__file__)
    frozen[str(SOURCE)] = runner.digest(SOURCE)
    assertions = []
    for seed in runner.SEEDS:
        expected = asdict(protocol_config(seed))
        with contextlib.redirect_stdout(io.StringIO()):
            clean = core.load_bundle("adult", 5.0, protocol_config(seed), torch.device("cpu"))
        for attack in ATTACKS:
            old, line_no, line_hash = controls[(attack, seed)]
            complete = asdict(core.ExperimentConfig(**old["config"]))
            assert semantic_config(complete) == semantic_config(expected), (attack, seed, "full config mismatch")
            assert old["seed"] == complete["seed"] == seed and old["alpha"] == 5.0
            assert old["rounds"] == complete["rounds"] == 70
            assert [x["round"] for x in old["trajectory_metrics"]] == list(range(1, 71))
            assert old["metrics"] == old["trajectory_metrics"][-1]["metrics"]
            truth = old["data_contract"]
            for field in ("label_col", "sensitive_col", "feature_includes_label", "feature_includes_sensitive",
                          "num_features", "train_rows", "test_rows", "root_clean_rows", "root_synthetic_rows",
                          "server_sampling_audit", "synthetic_method"):
                assert truth[field] == clean[field], (attack, seed, field)
            assert truth["server_sampling"] == expected["server_sampling"] and truth["server_alpha"] is None
            audits = {a["client_id"]: a for a in old["attack_audit"]}
            assert len(audits) == 20
            for cid, client in clean["clients"].items():
                assert audits[cid]["samples"] == len(client["y"])
                assert audits[cid]["attack_types"] == core.attack_types_for_client(attack, cid, list(range(4)))
                if attack == "S-DFA" and cid < 4:
                    assert audits[cid]["foe_mode"] == "fedsa"
                    assert audits[cid]["fflip_mode"] == "all_unprivileged"
                    assert audits[cid]["label_changed_count"] == 0
            assert old["round_summaries"][-1]["aggregate"]["ad2_plus_mode"] == "adaptive_internal_candidate_selector"
            runner.write_json(OUT / "reused_full" / f"{attack}_{seed}.json", {
                "status": "reused_historical", "source": str(SOURCE), "source_line": line_no,
                "source_line_sha256": line_hash, "result": old, "complete_config": complete,
                "provenance_only_fields": list(PROVENANCE_ONLY),
                "truth_context_checks": "whole configuration, alpha, seed, 70 rounds, final metrics, clean sampling, client counts and attack types",
                "limitation": "Historical CUDA/5090 environment; new noise jobs use CPU. No claim of numerical hardware equivalence."})
            assertions.append({"attack": attack, "seed": seed, "source_line": line_no, "verified": True})

    jobs = []
    for seed in runner.SEEDS:
        for attack in ATTACKS:
            old = controls[(attack, seed)][0]
            for field in ("root_label_noise", "root_sensitive_noise"):
                for rate in RATES:
                    job_id = f"{attack}_{field}_{int(rate*100)}pct_seed{seed}"
                    config = dict(old["config"], device="cpu", full_round_diagnostics=True,
                                  experiment_suite="revision_adult_root_noise_v1", experiment_tag=job_id,
                                  **{field: rate})
                    config = asdict(core.ExperimentConfig(**config))
                    job = {"id": job_id, "dataset": "adult", "distribution": "non-IID",
                           "method": "GuardFed-AD2+", "attack": attack, "config": config,
                           "output": str(OUT / "runs" / job_id), "source_hashes": frozen,
                           "evidence_stage": "formal_supplement",
                           "interpretation": "End-to-end root sensitivity; FedSA-inspired attack also uses the corrupted-root update."}
                    path = OUT / "jobs" / f"{job_id}.json"
                    runner.write_json(path, job)
                    jobs.append(str(path))
    manifest = {
        "protocol": "revision_adult_root_noise_v1", "jobs": jobs, "output": str(OUT),
        "new_run_count": 120, "reused_full_count": 20, "seeds": runner.SEEDS,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_hashes": frozen, "old_experiments_rerun": False,
        "statistics": "Fixed round 70, all metrics same checkpoint, mean/sample std, paired seeds, no test selection",
        "config_changes_from_control": {
            "intervention": ["root_label_noise OR root_sensitive_noise"],
            "provenance_only": list(PROVENANCE_ONLY),
            "diagnostics_only": ["full_round_diagnostics"],
            "execution_environment": {"device": "cuda historical -> cpu new; report numerical limitation"},
            "default_extensions": {"ablation_component": "none", "client_alpha": None,
                                   "compas_preprocessing_version": "legacy"}},
        "interpretation": "Single root input perturbation with both defense and FedSA-inspired attack downstream; not defense-only attribution.",
        "status": "prepared_not_started"}
    runner.write_json(OUT / "manifest.json", manifest)
    runner.write_json(OUT / "control_validation.json", assertions)
    verify(manifest)


def verify(manifest):
    assert len(manifest["jobs"]) == len(set(manifest["jobs"])) == 120
    identities, outputs, grouped = set(), set(), {}
    for path in manifest["jobs"]:
        job = json.loads(Path(path).read_text())
        config = job["config"]
        assert asdict(core.ExperimentConfig(**config)) == config
        assert config["device"] == "cpu" and config["rounds"] == 70
        assert config["ablation_component"] == "none" and config["compas_preprocessing_version"] == "legacy"
        assert config["client_alpha"] is None and job["distribution"] == "non-IID"
        rates = (config["root_label_noise"], config["root_sensitive_noise"])
        assert sum(r > 0 for r in rates) == 1 and max(rates) in RATES
        identity = (job["attack"], *rates, config["seed"])
        assert identity not in identities and job["output"] not in outputs
        identities.add(identity); outputs.add(job["output"])
        grouped.setdefault(identity[:-1], []).append(config["seed"])
        assert job["source_hashes"] == manifest["source_hashes"]
        for path_key, expected in job["source_hashes"].items():
            assert runner.digest(ROOT / path_key) == expected, path_key
    assert len(grouped) == 12
    assert all(sorted(seeds) == sorted(runner.SEEDS) for seeds in grouped.values())
    pilots = []
    for field in ("root_label_noise", "root_sensitive_noise"):
        path = ROOT / "results/root_noise_checks" / field / "result.json"
        pilot = json.loads(path.read_text())
        assert pilot["source_sha256"] == manifest["source_hashes"]["scripts/reproduce_paper_tables.py"]
        assert pilot["seed"] == 123 and pilot["rounds"] == 2 and len(pilot["trajectory_metrics"]) == 2
        assert pilot["config"][field] == 0.2
        audit = pilot["data_contract"]["root_noise_audit"]
        key = field.removeprefix("root_").removesuffix("_noise")
        assert audit[key]["flipped_count"] == int(0.2 * audit["root_rows"]) == 603
        assert all(r["root_noise_audit"] == audit for r in pilot["round_summaries"])
        pilots.append(pilot)
    # Exercise the real summary with the two measured 2-round pilots, never mark them formal.
    check_dir = OUT / "verification" / "pilot_summary_only"
    lookup = {}
    paths = []
    for i, pilot in enumerate(pilots):
        path = check_dir / f"pilot_job_{i}.json"
        runner.write_json(path, {"id": str(i), "output": str(check_dir / f"no_formal_run_{i}")})
        lookup[str(i)] = pilot
        paths.append(str(path))
    with patch.object(runner, "checked_result", side_effect=lambda job: lookup[job["id"]]):
        runner.summarize({"output": str(check_dir), "jobs": paths, "reused_full_count": 0})
    summary = json.loads((check_dir / "summary.json").read_text())
    assert len(summary) == 6 and all(r["n_runs"] == 1 for r in summary)
    assert {(r["root_label_noise"], r["root_sensitive_noise"]) for r in summary} == {(0.2, 0.0), (0.0, 0.2)}
    status = runner.summarize(manifest)
    assert status["new_complete"] == status["failed"] == 0
    runner.write_json(OUT / "verification.json", {
        "status": "passed_no_formal_jobs_started", "unique_new_jobs": 120, "unique_condition_groups": 12,
        "unique_seeds_per_condition": 10, "reused_clean_controls": 20,
        "config_and_source_hashes": "verified", "pilot_summary_grouping": "verified_using_two_measured_2round_pilots",
        "historical_device": "cuda", "new_device": "cpu"})
    print(json.dumps({"manifest": str(OUT / "manifest.json"), **status, "validation": "passed"}))


if __name__ == "__main__":
    if "--verify-only" in sys.argv:
        verify(json.loads((OUT / "manifest.json").read_text()))
    else:
        prepare()
