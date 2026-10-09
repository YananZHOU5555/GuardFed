"""Three-round actual CNN/localAdam/attack/root/eval synthetic RGB64 pipeline gate.

This deliberately substitutes only a synthetic bundle in the frozen core. It is
not a real-data CelebA gate, a 70-round run, or a performance comparison.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from worker import HERE, LABEL, METHOD, aggregation_wrapper, digest, load_core, run, validate_job, write_json


def toy_bundle():
    rng = np.random.default_rng(812)

    def cohort(count):
        return (torch.from_numpy(rng.integers(0, 256, (count, 3, 64, 64), dtype=np.uint8)),
                torch.tensor([0, 0, 1, 1] * (count // 4)),
                np.array([0, 1, 0, 1] * (count // 4)))

    clients = {}
    for cid in range(20):
        x, y, sensitive = cohort(4)
        clients[cid] = dict(X=x, y=y, sensitive=sensitive, sensitive_feature_index=None)
    root_x, root_y, root_s = cohort(8)
    eval_x, eval_y, eval_s = cohort(16)
    return dict(dataset="celeba", clients=clients, num_features=3 * 64 * 64,
                server_X=root_x, server_y=root_y, server_sensitive=root_s,
                X_test=eval_x, y_test=eval_y, test_sensitive=eval_s,
                label_col="Smiling", sensitive_col="Male", feature_includes_label=False,
                feature_includes_sensitive=False, root_clean_rows=8, root_synthetic_rows=0,
                rw_weights={(0, 0): 1., (1, 0): 2., (0, 1): 2., (1, 1): 1.},
                train_rows=88, test_rows=16,
                image_data_contract=dict(evaluation_split="synthetic_pipeline_only",
                    actual_train_rows=88, actual_evaluation_rows=16,
                    train_eval_disjoint=True, root_client_disjoint=True, real_celeba_data=False))


def equal_state(a, b):
    return tuple(a) == tuple(b) and all(torch.equal(a[k], b[k]) for k in a)


def pipeline(core, protocol, output, method=METHOD):
    bundle = toy_bundle()
    originals = {name: getattr(core, name) for name in (
        "load_bundle", "aggregate_round", "apply_foe_if_needed", "evaluate_state_on_server",
        "train_local_model", "make_optimizer", "evaluate_model_calibrated")}
    core.load_bundle = lambda *args: bundle
    parameters = dict(fairness_lambda=20., threshold=.2)
    wrapper = aggregation_wrapper(originals["aggregate_round"], parameters, output)
    config = core.ExperimentConfig(**dict(protocol["base_config"], device="cpu", seed=812,
        batch_size=4, rounds=3, client_alpha=5., guardfed_fairness_lambda=20., trust_threshold=.2,
        experiment_suite="synthetic_pipeline_gate_only", experiment_tag="hybrid_RGB64_not_CelebA",
        full_round_diagnostics=True))
    uploads, root_evaluations, local_calls, adam_calls, root_checks = [], [], [], [], []
    expected_post_state = []

    def attack(local_state, global_state, client, audit, server_update, cfg):
        assert torch.equal(client["y"], bundle["clients"][client["cid"]]["y"])
        delta, audit = originals["apply_foe_if_needed"](local_state, global_state, client, audit, server_update, cfg)
        uploads.append(dict(cid=client["cid"], delta=copy.deepcopy(delta),
                            pre_delta=core.state_delta(local_state, global_state)))
        expected_post_state.append({key: global_state[key] + delta[key] for key in global_state})
        return delta, audit

    def evaluate_root(state, *args):
        assert len(expected_post_state) == len(root_evaluations) + 1
        assert equal_state(state, expected_post_state[-1])
        metrics = originals["evaluate_state_on_server"](state, *args)
        root_evaluations.append(metrics)
        root_checks.append(True)
        return metrics

    def local(model, client, cfg):
        assert model.__class__.__name__ == "CelebACNN"
        local_calls.append(client["cid"])
        return originals["train_local_model"](model, client, cfg)

    def optimizer(model, cfg):
        value = originals["make_optimizer"](model, cfg)
        assert isinstance(value, torch.optim.Adam)
        adam_calls.append(True)
        return value

    def aggregate(label, updates, counts, fairness, server_update, cfg, **kwargs):
        assert all(equal_state(row, audit["delta"]) for row, audit in zip(updates, uploads[-20:]))
        expected = [float(item["aeod"]) for item in root_evaluations[-20:]]
        expected = [1. if np.isnan(v) else v for v in expected]
        assert fairness == expected
        if label == METHOD:
            delta, info = wrapper(label, updates, counts, fairness, server_update, cfg, **kwargs)
            original_delta, original_info = originals["aggregate_round"](
                "GuardFed", updates, counts, fairness, server_update, cfg, **kwargs)
            assert equal_state(delta, original_delta)
            assert info["selected_clients"] == original_info["selected_clients"]
            assert info["trust_scores"] == original_info["trust_scores"]
            return delta, info
        return originals["aggregate_round"](label, updates, counts, fairness, server_update, cfg, **kwargs)

    def no_group_calibration(*args, **kwargs):
        raise AssertionError("Custom hybrid must use native evaluation, never AD2+ calibration")

    core.apply_foe_if_needed, core.evaluate_state_on_server = attack, evaluate_root
    core.train_local_model, core.make_optimizer = local, optimizer
    core.aggregate_round, core.evaluate_model_calibrated = aggregate, no_group_calibration
    progress = []
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        result = core.run_experiment("celeba", "non-IID", method, "S-DFA", config,
            "synthetic_pipeline_only", torch.device("cpu"), progress_callback=progress.append,
            checkpoint_path=output / "model.pt")
    finally:
        for name, original in originals.items():
            setattr(core, name, original)
    assert len(local_calls) == len(uploads) == len(root_checks) == 60
    assert len(adam_calls) == 63  # sixty local clients plus three root models.
    assert local_calls == list(range(20)) * 3
    assert [item["round"] for item in progress] == [1, 2, 3]
    assert result["metrics"] == result["trajectory_metrics"][-1]["metrics"]
    assert result["evaluation_stats"]["prediction_count"] == 16
    assert all(np.isfinite(list(item["metrics"].values())).all() for item in result["trajectory_metrics"])
    assert result["config"]["optimizer"] == "adam" and not result["config"]["ad2_calibration_enabled"]
    assert all("foe_impl" in item and item["attack_types"] == ["fflip", "foe"]
               for item in result["attack_audit"][:4])
    assert all(item.get("label_changed_count", 0) == 0 for item in result["attack_audit"])
    assert any(not equal_state(row["pre_delta"], row["delta"]) for row in uploads[:4])
    saved = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    reloaded = core.make_model(bundle, config, torch.device("cpu"))
    reloaded.load_state_dict(saved)
    final_metrics = core.evaluate_for_reporting(method, reloaded, bundle, config)
    assert {key: final_metrics[key] for key in core.METRICS} == result["metrics"]
    if method == METHOD:
        assert len(wrapper.diagnostics) == 3
        assert json.loads((output / "diagnostics.json").read_text()) == wrapper.diagnostics
    write_json(output / "result.json", dict(evidence="synthetic RGB64 CPU only; not real CelebA results",
               result=result))
    return result, saved, dict(elapsed_seconds=time.time() - started, post_attack_root_model_checks=60,
                              local_adam_calls=60, root_adam_calls=3, checkpoint_reevaluation_equal=True)


def main(repo):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    repo = repo.resolve()
    core = load_core(repo)
    assert digest(repo / "scripts/reproduce_paper_tables.py") == "cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed"
    protocol = json.loads((HERE / "protocol.json").read_text())
    manifest = json.loads((HERE / "screen_jobs_draft/manifest.json").read_text())
    assert manifest["status"] == "PREPARED_NOT_FROZEN" and not manifest["execution_started"]
    assert len(manifest["jobs"]) == len({row["id"] for row in manifest["jobs"]}) == 32
    for item in manifest["jobs"]:
        job_path = HERE / "screen_jobs_draft" / item["job"]
        assert digest(job_path) == item["job_sha256"]
        job = json.loads(job_path.read_text())
        validate_job(job, protocol, require_frozen=False)
        for source, expected in job["adapter_source_hashes"].items():
            assert digest(HERE / source) == expected
    draft = HERE / "screen_jobs_draft" / manifest["jobs"][0]["job"]
    refused_output = HERE / "unfrozen_launch_must_not_create"
    assert not refused_output.exists()
    try:
        run(repo, draft, refused_output)
    except ValueError as error:
        assert "not frozen" in str(error)
    else:
        raise AssertionError("Prepared draft could launch")
    assert not refused_output.exists()
    first, first_state, evidence = pipeline(core, protocol, HERE / "integration_toy")
    replay, replay_state, _ = pipeline(core, protocol, HERE / "integration_replay")
    legacy, legacy_state, _ = pipeline(core, protocol, HERE / "integration_legacy", "GuardFed")
    assert equal_state(first_state, replay_state) and equal_state(first_state, legacy_state)
    assert first["trajectory_metrics"] == replay["trajectory_metrics"] == legacy["trajectory_metrics"]
    assert first["attack_audit"] == replay["attack_audit"] == legacy["attack_audit"]
    receipt = dict(status="PASS", evidence_stage="component_synthetic_pipeline_only", real_celeba_training=False,
        core_sha256=digest(repo / "scripts/reproduce_paper_tables.py"),
        worker_sha256=digest(HERE / "worker.py"), adapter_sha256=digest(HERE / "adapters.py"),
        check_sha256=digest(HERE / "check_worker.py"),
        rounds_per_run=3, pipeline_runs=3, total_pipeline_rounds=9, device="cpu",
        clients=20, images_per_client=4, root_images=8, evaluation_images=16,
        attack="S-DFA", attack_path="unchanged frozen client sensitive metadata flip and FedSA delta attack",
        training_path="unchanged frozen CelebACNN, localAdam, root localAdam, native argmax evaluation",
        post_attack_clean_root_aeod_source_verified=True, equal_selected_mean_verified=True,
        deterministic_replay_bitwise_equal=True, legacy_GuardFed_checkpoint_trajectory_attack_bitwise_equal=True,
        unfrozen_entrypoint_rejected_before_output=True, prepared_screen_jobs=32,
        per_run_evidence=evidence,
        output_hashes={name: digest(HERE / "integration_toy" / name) for name in
                       ["model.pt", "diagnostics.json", "result.json"]},
        limits=["Synthetic uint8 RGB64 only; no real CelebA/GPU or 70-round scientific evidence",
                "Draft protocol remains PREPARED_NOT_FROZEN; no server queue launched",
                "No mid-round resume guarantee; production worker preserves partial output and failures"])
    write_json(HERE / "integration_acceptance.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2] / "revision-publish-20260928")
    main(parser.parse_args().repo)
