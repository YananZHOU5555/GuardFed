"""Actual accepted CNN + synthetic RGB64/IDs official calibrated pipeline gate.

Only the checkpoint is real accepted evidence. All images/client IDs used for
LoGoFair fitting below are synthetic; no real CelebA mapping or fit is created.
"""
import copy
import json
from pathlib import Path
import tempfile

import netcal
import numpy as np
import torch

import bridge
from bridge import HERE, array_digest, digest, fit_predict, load_core, load_post, model_scores, save_post, validate_mapping, verify_external, write_json


def refused(action, label):
    try:
        action()
    except (ValueError, KeyError, RuntimeError) as error:
        return dict(check=label, refused=True, error_type=type(error).__name__)
    raise AssertionError("Invalid input accepted: " + label)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    repo = HERE.parents[1] / "revision-publish-20260928"
    core = load_core(repo)
    protocol_before = digest(HERE / "protocol.json")
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    reuse = json.loads((HERE / "reuse_manifest.json").read_text(encoding="utf8"))
    entry = next(row for row in reuse["entries"] if row["id"] == "FedAvg_IID_Benign_seed91001")
    reference = HERE / "accepted_fedavg_reference"
    original, accepted_cache, checkpoint = verify_external(reference, entry, core)
    checks = []
    changed = copy.deepcopy(entry)
    changed["accepted_record"]["checkpoint_sha256"] = "0" * 64
    checks.append(refused(lambda: verify_external(reference, changed, core), "external checkpoint identity changed"))
    changed_config = copy.deepcopy(entry)
    changed_config["source_job"]["config"]["learning_rate"] = 999.
    checks.append(refused(lambda: verify_external(reference, changed_config, core), "external training config changed"))
    accepted_probability = bridge.scores_from_accepted_margins(accepted_cache)
    assert all(np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all() for value in accepted_probability.values())
    config = core.ExperimentConfig(**entry["source_job"]["config"])
    model = core.make_model(dict(dataset="celeba", num_features=3), config, torch.device("cpu"))
    model.load_state_dict(checkpoint)
    rng = np.random.default_rng(8129)
    n_root, n_valid = 1600, 320
    root_images = torch.from_numpy(rng.integers(0, 256, (n_root, 3, 64, 64), dtype=np.uint8))
    valid_images = torch.from_numpy(rng.integers(0, 256, (n_valid, 3, 64, 64), dtype=np.uint8))
    root_probability, root_native = model_scores(model, root_images, 64)
    valid_probability, valid_native = model_scores(model, valid_images, 64)
    replay_probability, replay_native = model_scores(model, valid_images, 64)
    assert np.array_equal(valid_probability, replay_probability) and np.array_equal(valid_native, replay_native)
    with torch.no_grad():
        assert np.array_equal(valid_probability[:64], torch.softmax(model(valid_images[:64]), dim=1)[:, 1].numpy())
    mapping = dict(root_image_id=np.arange(1, n_root + 1, dtype=np.int64),
        root_client_id=np.repeat(np.arange(20, dtype=np.int64), 80),
        valid_image_id=np.arange(100001, 100001 + n_valid, dtype=np.int64),
        valid_client_id=np.repeat(np.arange(20, dtype=np.int64), 16))
    sensitive = dict(root_sensitive=np.tile(np.repeat([0, 1], 40), 20),
                     valid_sensitive=np.tile(np.repeat([0, 1], 8), 20))
    root_y = np.zeros(n_root, dtype=np.int64)
    # Synthetic labels use within-cohort score ranks solely to exercise finite
    # two-label beta fits, not to estimate accuracy or train the CNN.
    for cid in range(20):
        for group in (0, 1):
            positions = np.flatnonzero((mapping["root_client_id"] == cid) & (sensitive["root_sensitive"] == group))
            ranked = positions[np.argsort(root_probability[positions], kind="stable")]
            root_y[ranked[len(ranked) // 2:]] = 1
    valid_y = rng.integers(0, 2, n_valid, dtype=np.int64)
    metadata = dict(status="SYNTHETIC_GATE_ONLY", approved=False, semantics="synthetic_cohorts_only",
        root_image_ids_sha256=array_digest(mapping["root_image_id"]), valid_image_ids_sha256=array_digest(mapping["valid_image_id"]))
    validate_mapping(mapping, metadata, metadata["root_image_ids_sha256"], metadata["valid_image_ids_sha256"], require_frozen=False)
    checks.append(refused(lambda: validate_mapping(mapping, metadata, metadata["root_image_ids_sha256"], metadata["valid_image_ids_sha256"]), "synthetic IDs cannot authorize real mapping"))
    wrong = copy.deepcopy(mapping)
    wrong["root_image_id"] = wrong["root_image_id"][::-1].copy()
    checks.append(refused(lambda: validate_mapping(wrong, metadata, metadata["root_image_ids_sha256"], metadata["valid_image_ids_sha256"], False), "mapping row/sample order changed"))
    wrong_cid = copy.deepcopy(mapping)
    wrong_cid["valid_client_id"][0] = 99
    checks.append(refused(lambda: validate_mapping(wrong_cid, metadata, metadata["root_image_ids_sha256"], metadata["valid_image_ids_sha256"], False), "unknown evaluation client"))
    settings = dict(global_delta=.02, local_delta=.02, post_rounds=3, local_steps=3,
                    global_steps=3, post_lr=.005, beta=1000., calibration=True)
    probability = dict(root_probability=root_probability, valid_probability=valid_probability)
    labels = dict(root_y=root_y)
    post, prediction = fit_predict(probability, labels, sensitive, mapping, settings, 1719)
    replay, replay_prediction = fit_predict(probability, labels, sensitive, mapping, settings, 1719)
    assert np.array_equal(prediction, replay_prediction)
    assert all(torch.equal(post.thresholds[cid], replay.thresholds[cid]) for cid in post.clients)
    assert len(post.clients) == 20 and len(post.history) == 3
    np.random.seed(999)
    torch.manual_seed(999)
    assert np.array_equal(prediction, post.predict(valid_probability, sensitive["valid_sensitive"], mapping["valid_client_id"]))
    order = rng.permutation(n_valid)
    shuffled = post.predict(valid_probability[order], sensitive["valid_sensitive"][order], mapping["valid_client_id"][order])
    recovered = np.empty_like(shuffled)
    recovered[order] = shuffled
    assert np.array_equal(prediction, recovered)
    output = HERE / "synthetic_pipeline"
    output.mkdir(exist_ok=False)
    save_post(post, output / "post_state.pkl")
    restored = load_post(output / "post_state.pkl", digest(output / "post_state.pkl"))
    assert np.array_equal(prediction, restored.predict(valid_probability, sensitive["valid_sensitive"], mapping["valid_client_id"]))
    checks.append(refused(lambda: load_post(output / "post_state.pkl", "0" * 64), "fitted classifier state SHA changed"))
    bad_group = copy.deepcopy(sensitive)
    bad_group["root_sensitive"][:] = 0
    checks.append(refused(lambda: fit_predict(probability, labels, bad_group, mapping, settings, 1719), "missing calibration sensitive group"))
    bad_labels = dict(root_y=np.zeros_like(root_y))
    checks.append(refused(lambda: fit_predict(probability, bad_labels, sensitive, mapping, settings, 1719), "missing beta-calibration binary-label support"))
    bad_probability = copy.deepcopy(probability)
    bad_probability["root_probability"][0] = np.nan
    checks.append(refused(lambda: fit_predict(bad_probability, labels, sensitive, mapping, settings, 1719), "nonfinite score"))
    # Freeze no new tie policy: use a legitimate one-point calibrated score to
    # construct exact equality, which the official adapter explicitly refuses.
    cid, group = int(mapping["valid_client_id"][0]), int(sensitive["valid_sensitive"][0])
    calibrator = (post.clients[cid].score_calibrated_0, post.clients[cid].score_calibrated_1)[group]
    old = post.thresholds[cid].clone()
    post.thresholds[cid][group] = torch.tensor(calibrator.transform(valid_probability[:1]), dtype=torch.float32)[0]
    checks.append(refused(lambda: post.predict(valid_probability[:1], sensitive["valid_sensitive"][:1], mapping["valid_client_id"][:1]), "exact calibrated-score threshold tie"))
    post.thresholds[cid] = old
    metrics = core.compute_metrics(valid_y, prediction, sensitive["valid_sensitive"])
    assert all(np.isfinite(metrics[key]) for key in bridge.METRICS)
    prediction_reloaded = restored.predict(valid_probability, sensitive["valid_sensitive"], mapping["valid_client_id"])
    assert all(core.compute_metrics(valid_y, prediction_reloaded, sensitive["valid_sensitive"])[key] == metrics[key] for key in bridge.METRICS)
    np.savez_compressed(output / "scores_mapping_predictions.npz", **probability, **mapping, **sensitive, root_y=root_y, valid_y=valid_y, prediction=prediction)
    write_json(output / "mapping_metadata.json", metadata)
    # These are synthetic metrics and are kept solely as gate evidence.
    write_json(output / "result.json", dict(evidence_stage="synthetic_RGB64_only", scientific_results=0,
        accepted_checkpoint_sha256=entry["accepted_record"]["checkpoint_sha256"], settings=settings,
        metrics=metrics, history=post.history, thresholds={str(cid): value.tolist() for cid, value in post.thresholds.items()}))
    manifest = json.loads((HERE / "screen_jobs_draft/manifest.json").read_text(encoding="utf8"))
    for row in manifest["jobs"]:
        path = HERE / "screen_jobs_draft" / row["job"]
        assert digest(path) == row["job_sha256"]
        job = json.loads(path.read_text(encoding="utf8"))
        bridge.validate_job(job, protocol, require_frozen=False)
        assert job["mapping_sha256"] is None and job["mapping_metadata_sha256"] is None
        for source, wanted in job["local_hashes"].items():
            assert digest(HERE / source) == wanted
    first_job = HERE / "screen_jobs_draft" / manifest["jobs"][0]["job"]
    refused_out = HERE / "must_not_exist_unfrozen"
    checks.append(refused(lambda: bridge.run(repo, reference, first_job, Path("missing.npz"), Path("missing.json"), refused_out), "PREPARED execution before output"))
    assert not refused_out.exists() and digest(HERE / "protocol.json") == protocol_before
    write_json(HERE / "pipeline_gate.json", dict(status="PASS", evidence_stage="actual_accepted_CNN_synthetic_images_and_IDs_only",
        scientific_results=0, real_CelebA_fit=False, gpu_execution=False, referenced_accepted_FedAvg_models=100,
        selected_external_artifacts_verified=1, synthetic_root_images=n_root, synthetic_valid_images=n_valid,
        clients=20, beta_calibrators=40, post_rounds=3, fresh_fit_replay=True,
        actual_CNN_softmax_extractor_pass=True, model_unmodified=True,
        serialized_prediction_replay=True, RNG_perturbation_and_permutation_prediction_replay=True,
        all_metrics_same_serialized_classifier=True, exact_ties_refused=True, checks=checks,
        settings=settings, environment=dict(torch=torch.__version__, numpy=np.__version__, netcal=netcal.__version__),
        bridge_sha256=digest(HERE / "bridge.py"), check_sha256=digest(HERE / "check_pipeline.py"),
        accepted_checkpoint_sha256=entry["accepted_record"]["checkpoint_sha256"],
        artifact_hashes={name: digest(output / name) for name in ("post_state.pkl", "scores_mapping_predictions.npz", "mapping_metadata.json", "result.json")},
        limitation="Synthetic label ranks support beta-fit gate only; no real image/map/postprocessing performance evidence. Protocol remains PREPARED and mapping unresolved."))
    print(json.dumps(dict(status="PASS", synthetic_images=n_root + n_valid, clients=20, beta_calibrators=40, scientific_results=0)))


if __name__ == "__main__":
    main()
