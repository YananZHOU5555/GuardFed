"""Bounded synthetic CNN/gradient/upload/evaluation gates, never CelebA evidence."""
import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

import worker
from worker import HERE, attack_gradient, digest, load_components, load_core, train_pipeline, write_json


def bundle_fixture():
    rng = np.random.default_rng(1704)
    def cohort(count):
        return dict(X=torch.from_numpy(rng.integers(0, 256, (count, 3, 64, 64), dtype=np.uint8)),
                    y=torch.tensor([0, 0, 1, 1] * (count // 4)),
                    sensitive=np.array([0, 1, 0, 1] * (count // 4)), sensitive_feature_index=None)
    clients = {cid: cohort(4 if cid % 2 == 0 else 8) for cid in range(20)}
    root, evaluation = cohort(8), cohort(16)
    return dict(dataset="celeba", clients=clients, num_features=3 * 64 * 64,
        server_X=root["X"], server_y=root["y"], server_sensitive=root["sensitive"],
        X_test=evaluation["X"], y_test=evaluation["y"], test_sensitive=evaluation["sensitive"],
        label_col="Smiling", sensitive_col="Male", feature_includes_label=False,
        feature_includes_sensitive=False, root_clean_rows=8, root_synthetic_rows=0,
        rw_weights={(0, 0): 1., (1, 0): 2., (0, 1): 2., (1, 1): 1.},
        train_rows=128, test_rows=16, image_data_contract=dict(real_celeba_data=False,
            evaluation_split="synthetic_only", train_eval_disjoint=True, root_client_disjoint=True,
            actual_train_rows=128, actual_evaluation_rows=16))


def equal_state(a, b):
    return tuple(a) == tuple(b) and all(torch.equal(a[key], b[key]) for key in a)


def attack_oracles(core, config):
    generator = torch.Generator().manual_seed(105)
    checks = 0
    for index in range(36):
        gradient = torch.randn(17, generator=generator)
        root = torch.randn(17, generator=generator)
        if index % 6 == 0:
            root.zero_()
        if index % 7 == 0:
            gradient.zero_()
        for mode in ("fedsa", "delta", "zero"):
            client = dict(attack_types=["foe"], foe_mode=mode)
            upload, audit = attack_gradient(gradient, client, root, config, {})
            # Pure zero-state oracle avoids subtracting large real model values;
            # production bridge never fabricates or trains such a local state.
            template = {"weight": -gradient[:12].reshape(3, 4), "bias": -gradient[12:]}
            zero = {key: torch.zeros_like(value) for key, value in template.items()}
            reference = {"weight": root[:12].reshape(3, 4), "bias": root[12:]}
            delta, _ = core.apply_foe_if_needed(template, zero, client, {}, reference, config)
            assert torch.equal(upload, -core.vectorize(delta))
            if mode == "fedsa" and torch.linalg.vector_norm(root) > 0:
                assert float(torch.linalg.vector_norm(upload)) <= max(1., config.fedsa_norm_ratio) * float(torch.linalg.vector_norm(gradient)) + 3e-6
            assert audit["upload_semantics"].startswith("positive empirical")
            checks += 1
    try:
        attack_gradient(torch.ones(3), dict(attack_types=["foe"], foe_mode="state"), torch.ones(3), config, {})
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("Ambiguous state-dict attack was accepted")
    # Positive gradient reference gives the expected sign, before cap.
    sample = torch.tensor([1., 0.])
    config_small = copy.deepcopy(config)
    config_small.fedsa_gain, config_small.fedsa_norm_ratio = .5, 3.
    upload, _ = attack_gradient(sample, dict(attack_types=["foe"], foe_mode="fedsa"), torch.tensor([0., -1.]), config_small, {})
    torch.testing.assert_close(upload, torch.tensor([1., -.5]))
    return dict(frozen_attack_exact_oracles=checks, sign_handcheck=True, ambiguous_state_refused=True,
                zero_gradient_and_zero_reference_fallback_checked=True)


def huber_oracle(components):
    gradients = torch.tensor([[.1, .2], [1., -.3], [.2, .4], [9., -8.]], dtype=torch.float64)
    counts = [1, 4, 2, 1]
    threshold = components.huber_thresholds(counts, .1, .4)
    center, info = components.huber_center(gradients, counts, threshold, tolerance=1e-11)
    x, weight, cutoff = gradients.numpy(), np.array(counts) / sum(counts), threshold.numpy()
    def objective(point):
        residual = point[None] - x
        radius = np.linalg.norm(residual, axis=1)
        factor = np.minimum(1., cutoff / np.maximum(radius, 1e-300))
        return (weight * np.where(radius <= cutoff, .5 * radius**2, cutoff * radius - .5 * cutoff**2)).sum(), (weight[:, None] * factor[:, None] * residual).sum(axis=0)
    oracle = minimize(lambda point: objective(point)[0], x.mean(axis=0), jac=lambda point: objective(point)[1], method="BFGS", options=dict(gtol=1e-10))
    np.testing.assert_allclose(center.numpy(), oracle.x, atol=2e-8, rtol=2e-8)
    assert info["converged"] and np.linalg.norm(objective(center.numpy())[1]) <= 1e-11
    adapter = dict(server_eta=.1, objective="original_unweighted_ce", root_reference="same_point_unweighted_gradient",
                   projection="identity_Rp", t0=.1, m=.4, max_iter=1, tolerance=1e-30)
    try:
        worker.aggregate_gradients("Huber-BRFL-gradient", gradients, counts, adapter, components)
    except RuntimeError as error:
        assert not error.solver_diagnostics["converged"]
    else:
        raise AssertionError("Unconverged Huber solve was accepted")
    return dict(independent_scipy_objective_matches=True, unconverged_aggregate_refused=True)


def null_sensitive_gradient(core, components, config):
    bundle = bundle_fixture()
    core.set_seed(1704, deterministic_image=True)
    model = core.make_model(bundle, config, torch.device("cpu"))
    raw = bundle["clients"][0]
    before, _ = core.client_runtime_data(raw, 0, "Benign", [0], bundle["rw_weights"], torch.device("cpu"),
        config.fflip_mode, config.foe_mode, config.sdfa_foe_mode, config.spdfa_foe_mode)
    after, _ = core.client_runtime_data(raw, 0, "F Flip", [0], bundle["rw_weights"], torch.device("cpu"),
        config.fflip_mode, config.foe_mode, config.sdfa_foe_mode, config.spdfa_foe_mode)
    assert torch.equal(before["X"], after["X"]) and torch.equal(before["y"], after["y"])
    unweighted_before = components.empirical_gradient(model, before, 3, use_reweighting=False)
    unweighted_after = components.empirical_gradient(model, after, 3, use_reweighting=False)
    assert torch.equal(unweighted_before, unweighted_after)
    weighted_before = components.empirical_gradient(model, before, 3, use_reweighting=True)
    weighted_after = components.empirical_gradient(model, after, 3, use_reweighting=True)
    assert not torch.equal(weighted_before, weighted_after)
    return dict(unweighted_FFlip_is_exact_null=True, shared_reweight_FFlip_changes_gradient=True,
                images_and_labels_unchanged=True)


def pipeline(core, components, protocol, method, reference, objective, output):
    bundle = bundle_fixture()
    adapter = dict(server_eta=.03, objective=objective, root_reference=reference, projection="identity_Rp")
    if method == "Huber-BRFL-gradient":
        adapter.update(t0=.1, m=1., max_iter=1000, tolerance=1e-9)
    config = core.ExperimentConfig(**dict(protocol["base_config"], device="cpu", seed=1704, rounds=3,
        batch_size=3, client_alpha=5., use_reweighting=objective == "shared_reweighted_ce",
        experiment_suite="gradient_synthetic_only", experiment_tag="RGB64_not_CelebA"))
    originals = {name: getattr(core, name) for name in ("load_bundle", "train_local_model", "evaluate_state_on_server", "make_optimizer", "evaluate_model_calibrated")}
    gradient_original, aggregate_original, attack_original, step_original = (
        components.empirical_gradient, worker.aggregate_gradients, worker.attack_gradient, components.apply_parameter_step)
    gradients, uploads, steps, progress, adam_calls = [], [], [], [], []
    states_before_step = []
    calls = dict(client_gradient=0, root_gradient=0)
    core.load_bundle = lambda *args: bundle
    def forbidden(*args, **kwargs):
        raise AssertionError("Gradient method used localAdam client training/root-fairness inference/group calibration")
    core.train_local_model = core.evaluate_state_on_server = core.evaluate_model_calibrated = forbidden
    def optimizer(model, cfg):
        result = originals["make_optimizer"](model, cfg)
        assert isinstance(result, torch.optim.Adam)
        adam_calls.append(True)
        return result
    core.make_optimizer = optimizer
    def gradient(model, client, batch_size, *, use_reweighting=False):
        before = worker.parameters_vector(model).clone()
        answer = gradient_original(model, client, batch_size, use_reweighting=use_reweighting)
        assert torch.equal(before, worker.parameters_vector(model))
        # Independent whole-cohort autograd oracle; unequal mini-batches use a
        # single full-cohort denominator, never a mean of minibatch means.
        loss = F.cross_entropy(model(client["X"]), client["y"], reduction="none")
        weights = client["weights"] if use_reweighting else torch.ones(client["n"])
        oracle_loss = (loss * weights).sum() / weights.sum()
        oracle = torch.cat([value.reshape(-1) for value in torch.autograd.grad(oracle_loss, tuple(model.parameters()))])
        torch.testing.assert_close(answer, oracle, rtol=2e-4, atol=2e-7)
        if "cid" in client:
            assert torch.equal(client["y"], bundle["clients"][client["cid"]]["y"])
            calls["client_gradient"] += 1
            gradients.append(answer.clone())
        else:
            calls["root_gradient"] += 1
        return answer
    def upload(gradient_value, client, root, cfg, audit):
        value, saved = attack_original(gradient_value, client, root, cfg, audit)
        assert torch.equal(gradient_value, gradients[-1])
        template = {"gradient": -gradient_value}
        expected, _ = core.apply_foe_if_needed(template, {"gradient": torch.zeros_like(gradient_value)}, client,
                                              copy.deepcopy(audit), {"gradient": root}, cfg)
        assert torch.equal(value, -expected["gradient"])
        uploads.append(value.clone())
        return value, saved
    def aggregate(label, matrix, counts, actual, source):
        assert torch.equal(matrix, torch.stack(uploads[-20:]))
        assert counts == [4 if cid % 2 == 0 else 8 for cid in range(20)]
        step, info = aggregate_original(label, matrix, counts, actual, source)
        x = matrix.double()
        n = torch.tensor(counts, dtype=torch.float64) / sum(counts)
        if label == "Fed-NGA-gradient":
            norm = torch.linalg.vector_norm(x, dim=1)
            expected = -actual["server_eta"] * (n[:, None] * x / torch.where(norm > 0, norm, torch.ones_like(norm))[:, None]).sum(0)
            assert torch.equal(step, expected.to(step.dtype))
        else:
            center = -step.double() / actual["server_eta"]
            radius = torch.linalg.vector_norm(center[None] - x, dim=1)
            cutoff = actual["t0"] + actual["m"] / torch.tensor(counts, dtype=torch.float64).sqrt()
            stationarity = torch.linalg.vector_norm((n[:, None] * torch.minimum(torch.ones_like(radius), cutoff / torch.clamp(radius, min=1e-300))[:, None] * (center[None] - x)).sum(0))
            # Step cast to model float32 gives a finite, bounded oracle residual.
            assert float(stationarity) <= max(2e-7, 10 * actual["tolerance"])
            assert info["fixed_thresholds"] == cutoff.tolist()
        steps.append(step.clone())
        return step, info
    def apply(model, step):
        point = worker.parameters_vector(model).clone()
        assert torch.equal(step, steps[-1])
        step_original(model, step)
        assert torch.equal(worker.parameters_vector(model), point + step)
        states_before_step.append(point)
    components.empirical_gradient, components.apply_parameter_step = gradient, apply
    worker.aggregate_gradients, worker.attack_gradient = aggregate, upload
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = train_pipeline(core, components, method, "non-IID", "S-DFA", config, adapter, torch.device("cpu"),
                                progress_callback=progress.append, checkpoint_path=output / "model.pt")
    finally:
        for name, original in originals.items():
            setattr(core, name, original)
        components.empirical_gradient, components.apply_parameter_step = gradient_original, step_original
        worker.aggregate_gradients, worker.attack_gradient = aggregate_original, attack_original
    assert calls["client_gradient"] == len(uploads) == 60 and len(steps) == 3
    assert calls["root_gradient"] == (3 if reference == "same_point_unweighted_gradient" else 0)
    assert len(adam_calls) == (3 if reference == "frozen_root_localadam_delta" else 0)
    assert [row["round"] for row in progress] == [1, 2, 3]
    assert result["metrics"] == result["trajectory_metrics"][-1]["metrics"]
    assert result["evaluation_stats"]["prediction_count"] == 16
    assert all(math.isfinite(value) for row in progress for value in row["metrics"].values())
    saved = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model = core.make_model(bundle, config, torch.device("cpu"))
    model.load_state_dict(saved)
    metric = core.evaluate_for_reporting(method, model, bundle, config)
    assert result["metrics"] == {key: metric[key] for key in core.METRICS}
    write_json(output / "result.json", dict(evidence_stage="synthetic_component_pipeline_only", scientific_results=0, result=result))
    return result, saved, dict(method=method, root_reference=reference, objective=objective,
        rounds=3, exact_upload_oracle_checks=60, same_point_full_gradient_checks=60,
        whole_cohort_autograd_oracle_checks=60 + calls["root_gradient"], client_optimizer_steps=0,
        root_only_adam_calls=len(adam_calls), checkpoint_reevaluation_equal=True)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    repo = HERE.parents[1] / "revision-publish-20260928"
    core, components = load_core(repo), load_components()
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    assert digest(repo / "scripts/reproduce_paper_tables.py") == protocol["source_hashes"]["scripts/reproduce_paper_tables.py"]
    config = core.ExperimentConfig(**protocol["base_config"])
    algebra = attack_oracles(core, config)
    huber = huber_oracle(components)
    null = null_sensitive_gradient(core, components, config)
    evidence, records, states = [], [], []
    choices = [(method, reference, "original_unweighted_ce") for method in sorted(worker.METHODS) for reference in sorted(worker.ROOT_REFERENCES)]
    choices += [("Fed-NGA-gradient", "same_point_unweighted_gradient", "shared_reweighted_ce"), choices[0]]
    for index, (method, reference, objective) in enumerate(choices):
        result, state, check = pipeline(core, components, protocol, method, reference, objective, HERE / f"synthetic_run_{index + 1}")
        evidence.append(check)
        records.append(result)
        states.append(state)
    assert equal_state(states[0], states[-1]) and records[0]["trajectory_metrics"] == records[-1]["trajectory_metrics"]
    write_json(HERE / "pipeline_gate.json", dict(status="PASS", scientific_results=0, real_celeba_training=False,
        protocol_status=protocol["status"], algebra=algebra, huber=huber, sensitive_null_check=null,
        pipeline_runs=len(choices), total_synthetic_rounds=3 * len(choices), deterministic_replay_bitwise_equal=True,
        per_run=evidence, worker_sha256=digest(HERE / "worker.py"), check_sha256=digest(HERE / "check_pipeline.py"),
        component_hashes=worker.COMPONENTS, core_sha256=digest(repo / "scripts/reproduce_paper_tables.py"),
        limits=["Synthetic uint8 RGB64 CPU only; no real CelebA/GPU, seventy-round or performance evidence",
                "Protocol decisions remain unresolved; alternative reference/objective gates are component checks, not approvals"]))
    print(json.dumps(dict(status="PASS", pipeline_runs=len(choices), synthetic_rounds=3 * len(choices), scientific_results=0)))


if __name__ == "__main__":
    main()
