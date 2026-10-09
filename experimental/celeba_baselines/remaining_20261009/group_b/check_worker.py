"""Full-state/delta bridge regression and synthetic RGB64 frozen-core CPU gate."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from flgmm_adapter import FLGMMAdapter
from worker import HERE, LABEL, METHOD, aggregation_wrapper, digest, load_core, validate_job, write_json


def check_bridge(core):
    config = core.ExperimentConfig(num_clients=20, device="cpu")
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[32., -64.]]))
        model.bias.copy_(torch.tensor([0.125]))
    global_state = core.clone_state(model)
    updates = []
    for client in range(20):
        value = client if client < 16 else 100 + client
        updates.append({"weight": torch.tensor([[value / 16, -value / 32]]),
                        "bias": torch.tensor([value / 64])})
    local_models = [{key: global_state[key] + row[key] for key in global_state} for row in updates]
    independent = FLGMMAdapter(range(20), warmup_rounds=2, control_width=3)
    full, expected = independent.step(local_models)
    wrapped = aggregation_wrapper(core.aggregate_round, dict(warmup_rounds=2, control_width=3))
    delta, diagnostics = wrapped(METHOD, updates, [4] * 20, [0.] * 20, {}, config,
                                 global_state=global_state)
    core.apply_update(model, delta)
    actual = core.clone_state(model)
    assert all(torch.equal(actual[key], full[key]) for key in full)
    assert all(torch.equal(delta[key], full[key] - global_state[key]) for key in full)
    assert diagnostics["selected_indices"] == expected["selected_indices"]
    assert not torch.allclose(global_state["weight"] + full["weight"], full["weight"])
    assert diagnostics["model_delta_reconstruction_max_abs"] == 0
    return dict(status="PASS", fixture="nonzero dyadic global state and two local-model clusters",
                formula="local_i=global+attacked_delta_i; returned_delta=aggregate_full-global; apply_update=global+returned_delta",
                actual_full_state_bitwise_equal=True, wrong_full_as_delta_detected=True,
                selected=diagnostics["selected_indices"])


def toy_bundle():
    rng = np.random.default_rng(812)
    def cohort(count):
        return (torch.from_numpy(rng.integers(0, 256, (count, 3, 64, 64), dtype=np.uint8)),
                torch.tensor(([0, 0, 1, 1] * (count // 4))),
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
                    train_eval_disjoint=True, root_client_disjoint=True,
                    real_celeba_data=False))


def main(repo):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    repo = repo.resolve()
    core = load_core(repo)
    assert digest(repo / "scripts/reproduce_paper_tables.py") == "cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed"
    protocol = json.loads((HERE / "protocol.json").read_text())
    draft_job = json.loads(next((HERE / "screen_jobs_draft").glob("FLGMM*.json")).read_text())
    validate_job(draft_job, protocol, require_frozen=False)
    try:
        validate_job(draft_job, protocol)
        raise AssertionError("Unfrozen draft could launch")
    except ValueError as error:
        assert "not frozen" in str(error)
    bridge = check_bridge(core)
    original_load, original_aggregate = core.load_bundle, core.aggregate_round
    bundle = toy_bundle()
    core.load_bundle = lambda *args: bundle
    output = HERE / "integration_toy"
    output.mkdir(exist_ok=False)
    parameters = dict(warmup_rounds=1, control_width=3)
    wrapped = aggregation_wrapper(original_aggregate, parameters, output)
    assert wrapped.controller.round_index == 0
    core.aggregate_round = wrapped
    cfg = core.ExperimentConfig(**dict(protocol["base_config"], device="cpu", seed=812,
        batch_size=4, rounds=3, client_alpha=5., experiment_suite="synthetic_pipeline_gate_only",
        experiment_tag="FLGMM_synthetic_RGB64_CPU_not_CelebA", full_round_diagnostics=True))
    progress = []
    started = time.time()
    try:
        result = core.run_experiment("celeba", "non-IID", METHOD, "S-DFA", cfg,
            "synthetic_pipeline_only", torch.device("cpu"), progress_callback=progress.append,
            checkpoint_path=output / "model.pt")
    finally:
        core.load_bundle, core.aggregate_round = original_load, original_aggregate
    assert wrapped.controller.round_index == len(wrapped.diagnostics) == len(progress) == 3
    assert [row["aggregate"]["stage"] for row in result["round_summaries"]] == [
        "per_round_gmm", "fit_control_limit", "monitor"]
    assert result["metrics"] == result["trajectory_metrics"][-1]["metrics"]
    assert result["evaluation_stats"]["prediction_count"] == 16
    assert all(np.isfinite(list(row["metrics"].values())).all() for row in result["trajectory_metrics"])
    assert len(result["attack_audit"]) == 20
    assert all("foe_impl" in row and row["label_changed_count"] == 0 for row in result["attack_audit"][:4])
    assert all(row["client_id"] == i for i, row in enumerate(result["attack_audit"]))
    assert json.loads((output / "state.json").read_text())["round_index"] == 3
    assert len(json.loads((output / "diagnostics.json").read_text())) == 3
    fresh = aggregation_wrapper(original_aggregate, parameters)
    assert fresh.controller.round_index == 0 and not any(fresh.controller.history)
    result["method_impl_note"] = LABEL
    write_json(output / "result.json", dict(evidence="synthetic RGB64 CPU pipeline only; not CelebA data/results", result=result))
    acceptance = dict(status="PASS", evidence_stage="component_synthetic_pipeline_only", bridge=bridge,
        core_sha256=digest(repo / "scripts/reproduce_paper_tables.py"), worker_sha256=digest(HERE / "worker.py"),
        adapter_sha256=digest(HERE / "flgmm_adapter.py"), toy_rounds=3, clients=20,
        device="cpu", images_per_client=4, root_images=8, evaluation_images=16,
        attack="S-DFA", attack_path="unchanged core actual client metadata flip and FedSA delta attack",
        training_path="unchanged frozen CelebACNN + localAdam + train-root update + native metrics",
        full_stage_sequence=["per_round_gmm", "fit_control_limit", "monitor"],
        separate_run_state_reset=True, unfrozen_launch_rejected=True,
        elapsed_seconds=time.time() - started,
        output_hashes={name: digest(output / name) for name in ["model.pt", "state.json", "diagnostics.json", "result.json"]},
        limits=["Synthetic uint8 RGB64 only; no real CelebA/GPU performance evidence",
                "Tg1/three rounds used only to check all stages, not a search recipe",
                "Controller JSON is not a full training checkpoint or resume guarantee"])
    write_json(HERE / "integration_acceptance.json", acceptance)
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2] / "revision-publish-20260928")
    main(parser.parse_args().repo)
