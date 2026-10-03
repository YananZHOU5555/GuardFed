"""CPU-only protocol checks: complete matrix, rejected drift, unchanged adapters.

Does not import torch, create jobs, connect to a server, or run training.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(validator, job, mutate):
    altered = copy.deepcopy(job)
    mutate(altered)
    try:
        validator(altered)
    except (ValueError, KeyError):
        return
    raise AssertionError("Drift was accepted: " + repr(altered))


def function_node(path, name):
    return next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def main():
    # Source-layout evidence is local; deployed workers are separately SHA-pinned.
    old = HERE.parents[1] / "screen_20260928"
    new = HERE.parent
    fedaa = load("coverage_fedaa_worker", HERE / "run_fedaa_screen.py")
    lasa = load("coverage_lasa_worker", new / "lasa/worker.py")
    prepare = load("coverage_lasa_prepare", new / "lasa/prepare_jobs.py")
    template = json.loads((HERE / "selected_recipe_template.json").read_text(encoding="utf-8"))
    assert template == json.loads((old / "fedaa/screen_jobs/jobs/FedAA-DDPG_policy0.001_keep16_local0.001_IID_Benign_seed91001.json").read_text(encoding="utf-8"))
    for component in ["fedaa/fedaa_round_adapter.py", "lasa/adapter.py"]:
        assert (new / component).read_bytes() == (old / component).read_bytes(), component
    for path in new.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    # All numerical training and state-saving statements within FedAA's loop remain identical.
    old_run = function_node(old / "fedaa/run_fedaa_screen.py", "run")
    new_run = function_node(HERE / "run_fedaa_screen.py", "run")
    def training_loop(node):
        return next(n for n in ast.walk(node) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == "rnd")
    assert ast.dump(training_loop(old_run)) == ast.dump(training_loop(new_run))
    assert ast.dump(function_node(old / "lasa/worker.py", "aggregation_wrapper")) == ast.dump(function_node(new / "lasa/worker.py", "aggregation_wrapper"))
    fedaa_jobs = []
    for distribution, alpha in [("IID", 5000.), ("non-IID", 5.)]:
        for attack in fedaa.ATTACKS:
            for seed in range(91001, 91011):
                job = copy.deepcopy(template)
                identity = f"FedAA-DDPG_{fedaa.SELECTED_CANDIDATE}_{distribution}_{attack}_seed{seed}"
                job.update(id=identity, distribution=distribution, attack=attack, policy_seed=seed,
                           evidence_stage="multi_seed_validation_coverage")
                job["config"].update(seed=seed, client_alpha=alpha, rounds=70,
                                     experiment_suite=fedaa.STAGE_VERSION, experiment_tag=identity)
                job["adapter_hashes"]["screen_runner"] = fedaa.digest(HERE / "run_fedaa_screen.py")
                fedaa.validate_job(job)
                fedaa_jobs.append(job)
    assert len(fedaa_jobs) == 100
    fedaa_canary = copy.deepcopy(next(j for j in fedaa_jobs if j["distribution"] == "non-IID" and j["attack"] == "Sp-DFA" and j["config"]["seed"] == 91001))
    fedaa_canary["id"] += "_canary"
    fedaa_canary["evidence_stage"] = "pipeline_canary_only"
    fedaa_canary["config"].update(rounds=3, experiment_tag=fedaa_canary["id"])
    fedaa.validate_job(fedaa_canary)
    job = fedaa_jobs[-1]
    for mutate in [lambda j: j.update(evidence_stage="validation_screen"),
                   lambda j: j["config"].update(seed=91011),
                   lambda j: j.update(policy_seed=91001),
                   lambda j: j.update(attack="unknown"),
                   lambda j: j.update(aggre_num=10),
                   lambda j: j["policy_config"].update(actor_lr=.01),
                   lambda j: j["config"].update(learning_rate=.0005),
                   lambda j: j["config"].update(ad2_calibration_enabled=True),
                   lambda j: j["source_hashes"].update({"scripts/reproduce_paper_tables.py": "0" * 64}),
                   lambda j: j["source_hashes"].clear()]:
        rejected(fedaa.validate_job, job, mutate)
    coverage, canaries = prepare.make_jobs("fullcoverage"), prepare.make_jobs("preflight")
    assert len(coverage) == 100 and len(canaries) == 5
    assert len({j["id"] for j in coverage}) == 100
    for job in coverage + canaries:
        lasa.validate_job(job)
        assert job["config"]["ad2_calibration_enabled"] is True
    old_protocol = json.loads((old / "lasa/protocol.json").read_text(encoding="utf-8"))
    new_protocol = json.loads((new / "lasa/protocol.json").read_text(encoding="utf-8"))
    assert old_protocol["base_config"] == new_protocol["base_config"]
    assert old_protocol["source_hashes"] == new_protocol["source_hashes"]
    for mutate in [lambda j: j.update(phase="screen"),
                   lambda j: j.update(evidence_stage="single_seed_validation_search_not_formal_test"),
                   lambda j: j["config"].update(seed=91011),
                   lambda j: j.update(attack="unknown"),
                   lambda j: j["adapter"].update(lambda_n=1.),
                   lambda j: j["config"].update(learning_rate=.0005),
                   lambda j: j["config"].update(ad2_calibration_enabled=False),
                   lambda j: j["source_hashes"].update({"scripts/reproduce_paper_tables.py": "0" * 64}),
                   lambda j: j.update(id=j["id"] + "_drift")]:
        rejected(lasa.validate_job, coverage[-1], mutate)
    print(json.dumps(dict(status="PASS", fedaa_coverage_jobs=100, lasa_coverage_jobs=100,
                         lasa_canaries=5, drift_cases_rejected=19,
                         adapters_byte_identical=True, numerical_training_ast_unchanged=True,
                         jobs_written=0, gpu_training_run=False)))


if __name__ == "__main__":
    main()
