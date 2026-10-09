"""Temporary synthetic structural fixtures for strict post-result acceptance.

No real mapping is created or fitted. FROZEN metadata, synthetic 19867-row
arrays and the mocked external verifier exist only inside the removed fixture.
Real external artifacts are separately authenticated in check_pipeline.py.
"""
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

import bridge
from bridge import HERE, array_digest, digest, load_core, load_post, write_json


def refused(action, label):
    try:
        action()
    except (ValueError, KeyError, RuntimeError) as error:
        return dict(check=label, refused=True, error_type=type(error).__name__)
    raise AssertionError("Invalid acceptance fixture passed: " + label)


def main():
    protocol_before = digest(HERE / "protocol.json")
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    manifest = json.loads((HERE / "screen_jobs_draft/manifest.json").read_text(encoding="utf8"))
    source_job = json.loads((HERE / "screen_jobs_draft" / manifest["jobs"][0]["job"]).read_text(encoding="utf8"))
    checks = [refused(lambda: bridge.validate_job(source_job, protocol), "PREPARED result acceptance")]
    only_status = copy.deepcopy(protocol)
    only_status["status"] = "FROZEN"
    checks.append(refused(lambda: bridge.validate_job(source_job, only_status), "FROZEN label with unresolved decisions"))
    core = load_core(HERE.parents[1] / "revision-publish-20260928")
    module, official_source = bridge.official_adapter()
    with np.load(HERE / "synthetic_pipeline/scores_mapping_predictions.npz", allow_pickle=False) as source:
        toy = {key: source[key].copy() for key in source.files}
    with tempfile.TemporaryDirectory(prefix="acceptance_fixture_", dir=HERE) as temporary:
        fixture = Path(temporary)
        approved = copy.deepcopy(protocol)
        approved["status"] = "FROZEN"
        for decision in approved["decisions"].values():
            decision["status"] = "APPROVED"
        settings = dict(global_delta=.02, local_delta=.02, post_rounds=3, local_steps=3, global_steps=3, post_lr=.005, beta=1000., calibration=True)
        chosen = next(row for row in approved["candidates"] if row["id"] == source_job["candidate"])
        chosen["settings"] = settings
        write_json(fixture / "protocol.json", approved)
        for name in bridge.LOCAL_FILES - {"protocol.json"}:
            (fixture / name).write_bytes((HERE / name).read_bytes())
        n_valid = 19867
        mapping = dict(root_image_id=toy["root_image_id"], root_client_id=toy["root_client_id"],
            valid_image_id=np.arange(100001, 100001 + n_valid, dtype=np.int64),
            valid_client_id=np.resize(toy["valid_client_id"], n_valid))
        accepted_cache = dict(root_margins=torch.logit(torch.from_numpy(toy["root_probability"])).numpy(),
            valid_margins=torch.logit(torch.from_numpy(np.resize(toy["valid_probability"], n_valid))).numpy(),
            root_y=toy["root_y"], valid_y=np.resize(toy["valid_y"], n_valid),
            root_sensitive=toy["root_sensitive"], valid_sensitive=np.resize(toy["valid_sensitive"], n_valid))
        probability = bridge.scores_from_accepted_margins(accepted_cache)
        metadata = dict(status="FROZEN", approved=True, semantics="declared_virtual_partition",
            fixture_note="TEMPORARY SYNTHETIC STRUCTURE, NOT REAL CLIENT IDENTITIES OR AN APPROVAL",
            root_image_ids_sha256=array_digest(mapping["root_image_id"]),
            valid_image_ids_sha256=array_digest(mapping["valid_image_id"]))
        output = fixture / "mock_output"
        output.mkdir()
        np.savez_compressed(output / "mapping.npz", **mapping)
        write_json(output / "mapping_metadata.json", metadata)
        (output / "post_state.pkl").write_bytes((HERE / "synthetic_pipeline/post_state.pkl").read_bytes())
        job = copy.deepcopy(source_job)
        job.update(settings=settings, mapping_sha256=digest(output / "mapping.npz"),
            mapping_metadata_sha256=digest(output / "mapping_metadata.json"),
            local_hashes={name: digest(fixture / name) for name in bridge.LOCAL_FILES})
        job_path = fixture / "job.json"
        write_json(job_path, job)
        post = load_post(output / "post_state.pkl", digest(output / "post_state.pkl"))
        prediction = post.predict(probability["valid_probability"], accepted_cache["valid_sensitive"], mapping["valid_client_id"])
        arrays = dict(**probability, **mapping, **{key: accepted_cache[key] for key in ("root_y", "valid_y", "root_sensitive", "valid_sensitive")},
            prediction=prediction, valid_native_prediction=(accepted_cache["valid_margins"] > 0).astype(int))
        entry = next(row for row in json.loads((HERE / "reuse_manifest.json").read_text(encoding="utf8"))["entries"] if row["id"] == job["baseline_id"])
        record = entry["accepted_record"]
        result = dict(status="complete", evidence_stage="validation_postprocessing_screen", job=job, baseline_id=entry["id"],
            checkpoint_sha256=record["checkpoint_sha256"], original_result_sha256=record["original_result_sha256"],
            accepted_margin_cache_sha256=record["cache_sha256"], mapping_sha256=job["mapping_sha256"],
            mapping_metadata_sha256=job["mapping_metadata_sha256"], root_image_ids_sha256=metadata["root_image_ids_sha256"],
            valid_image_ids_sha256=metadata["valid_image_ids_sha256"], settings=settings, fit_seed=1719,
            metrics=core.compute_metrics(accepted_cache["valid_y"], prediction, accepted_cache["valid_sensitive"]),
            history=post.history, thresholds={str(cid): value.tolist() for cid, value in post.thresholds.items()},
            component=dict(adapter_sha256=bridge.ADAPTER_SHA, official_lf_sha256=bridge.OFFICIAL_LF_SHA, core_sha256=bridge.CORE_SHA))
        def save(observed=result, cache=arrays):
            np.savez_compressed(output / "scores_predictions.npz", **cache)
            write_json(output / "result.json", observed)
            write_json(output / "acceptance.json", dict(status="PASS", job_sha256=digest(job_path), artifact_hashes={name: digest(output / name) for name in
                ("post_state.pkl", "scores_predictions.npz", "result.json", "mapping.npz", "mapping_metadata.json")}))
        original_here, original_external, original_adapter = bridge.HERE, bridge.verify_external, bridge.official_adapter
        bridge.HERE = fixture
        bridge.official_adapter = lambda: (module, official_source)
        # Only the structure fixture substitutes the external data identity.
        # The actual accepted external bundle was independently checked above.
        bridge.verify_external = lambda *args: (dict(data_contract=dict(image_data_contract=dict(
            root_image_ids_sha256=metadata["root_image_ids_sha256"], evaluation_image_ids_sha256=metadata["valid_image_ids_sha256"]))), accepted_cache, None)
        try:
            save()
            assert bridge.checked_output(job_path, output, core, Path("synthetic_reference_fixture"))["status"] == "complete"
            checks.append(dict(check="temporary synthetic full metadata/prediction structure", accepted=True, scientific_results=0))
            for label, mutate in {
                "wrong source checkpoint": lambda value: value.update(checkpoint_sha256="0" * 64),
                "mixed-predictor metric": lambda value: value["metrics"].update(accuracy=.123456789),
                "missing post round": lambda value: value["history"].pop(),
                "threshold state identity changed": lambda value: value["thresholds"]["0"].__setitem__(0, 99.),
                "method component changed": lambda value: value["component"].update(adapter_sha256="0" * 64),
            }.items():
                bad = copy.deepcopy(result)
                mutate(bad)
                save(bad)
                checks.append(refused(lambda: bridge.checked_output(job_path, output, core, Path("synthetic_reference_fixture")), label))
            for label, mutate in {
                "changed predictor client mapping": lambda value: value["valid_client_id"].__setitem__(0, 19),
                "changed model score cache": lambda value: value["valid_probability"].__setitem__(0, .123456789),
                "changed evaluation label cache": lambda value: value["valid_y"].__setitem__(0, 1 - value["valid_y"][0]),
                "changed saved prediction": lambda value: value["prediction"].__setitem__(0, 1 - value["prediction"][0]),
            }.items():
                bad = copy.deepcopy(arrays)
                mutate(bad)
                save(result, bad)
                checks.append(refused(lambda: bridge.checked_output(job_path, output, core, Path("synthetic_reference_fixture")), label))
            save()
            with (output / "post_state.pkl").open("ab") as artifact:
                artifact.write(b"corrupted")
            checks.append(refused(lambda: bridge.checked_output(job_path, output, core, Path("synthetic_reference_fixture")), "fitted classifier artifact SHA changed"))
            write_json(output / "failure.json", dict(error="preserved synthetic fixture failure"))
            checks.append(refused(lambda: bridge.checked_output(job_path, output, core, Path("synthetic_reference_fixture")), "preserved failure"))
        finally:
            bridge.HERE, bridge.verify_external, bridge.official_adapter = original_here, original_external, original_adapter
    assert digest(HERE / "protocol.json") == protocol_before and not list(HERE.glob("acceptance_fixture_*"))
    write_json(HERE / "acceptance_gate.json", dict(status="PASS", evidence_stage="structural_synthetic_fixture_only",
        scientific_results=0, temporary_mock_and_FROZEN_metadata_removed=True, actual_protocol_unchanged=True,
        checks=checks, bridge_sha256=digest(HERE / "bridge.py"), check_sha256=digest(HERE / "check_acceptance.py")))
    print(json.dumps(dict(status="PASS", checks=len(checks), scientific_results=0)))


if __name__ == "__main__":
    main()
