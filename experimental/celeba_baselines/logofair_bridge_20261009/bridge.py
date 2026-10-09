"""Pinned official LoGoFair-DP score/postprocessing bridge; no model training.

Production execution requires a reviewed frozen mapping and protocol. Missing
client identities are never invented. The existing calibrated official adapter
is reused verbatim, including explicit failures for exact threshold ties.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ADAPTER_SHA = "8ac65ac38812b90ab44c71284d19184fe55f120e9d44262841ca9d925c26a5a7"
OFFICIAL_LF_SHA = "244e8bd3f3a6f5b4945cd8cf93035d5996954acc7f5b61e41e89651d9f88fba2"
CORE_SHA = "cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed"
LOCAL_FILES = {"bridge.py", "prepare_reuse.py", "protocol.json", "reuse_manifest.json"}
METRICS = ("accuracy", "aeod", "aspd")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_core(repo):
    repo = Path(repo).resolve()
    if digest(repo / "scripts/reproduce_paper_tables.py") != CORE_SHA:
        raise ValueError("Frozen scoring/metrics core changed")
    sys.path.insert(0, str(repo))
    return load_module(repo / "scripts/reproduce_paper_tables.py", "guardfed_logofair_score_core")


def official_adapter():
    root = HERE.parent / "logofair"
    if digest(root / "adapter.py") != ADAPTER_SHA:
        raise ValueError("Audited LoGoFair adapter changed")
    source = root / "upstream/fedlearn/models/FedFairPostClient.py"
    if hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != OFFICIAL_LF_SHA:
        raise ValueError("Pinned official source changed")
    return load_module(root / "adapter.py", "guardfed_logofair_official_bridge"), source


def verify_external(reference, entry, core):
    """Authenticate already accepted FedAvg model, job, root and margin cache."""
    reference = Path(reference)
    record, job = entry["accepted_record"], entry["source_job"]
    for name, wanted in (("model.pt", record["checkpoint_sha256"]),
                         ("result.json", record["original_result_sha256"]),
                         ("margins.npz", record["cache_sha256"])):
        if digest(reference / name) != wanted:
            raise ValueError("External accepted FedAvg artifact changed: " + name)
    result = json.loads((reference / "result.json").read_text(encoding="utf8"))
    if (result["dataset"], result["method"], result["rounds"], result["seed"]) != (
            "celeba", "FedAvg", 70, job["config"]["seed"]):
        raise ValueError("Expected an accepted seventy-round FedAvg checkpoint")
    for key in ("config", "source_hashes", "distribution", "attack", "method", "dataset"):
        observed = result["revision_job"][key]
        if observed != job[key] or (key in result and result[key] != job[key]):
            raise ValueError("FedAvg external source/config/condition mismatch: " + key)
    if result["metrics"] != result["trajectory_metrics"][-1]["metrics"] or [row["round"] for row in result["trajectory_metrics"]] != list(range(1, 71)):
        raise ValueError("FedAvg terminal checkpoint/result horizon mismatch")
    contract = result["data_contract"]["image_data_contract"]
    if (contract["evaluation_split"], contract["actual_train_rows"], contract["actual_evaluation_rows"],
        contract["train_eval_disjoint"], contract["root_client_disjoint"], result["data_contract"]["root_synthetic_rows"]) != (
            "valid", 162770, 19867, True, True, 0):
        raise ValueError("FedAvg split/root identity mismatch")
    if job["config"]["root_label_noise"] or job["config"]["root_sensitive_noise"] or job["config"]["celeba_train_limit"] or job["config"]["celeba_eval_limit"]:
        raise ValueError("Expected full clean train-root and validation split")
    with np.load(reference / "margins.npz", allow_pickle=False) as source:
        cache = {key: source[key].copy() for key in source.files}
    if set(cache) != {"root_margins", "valid_margins", "root_y", "root_sensitive", "valid_y", "valid_sensitive"}:
        raise ValueError("Unexpected accepted margin cache layout")
    for split, expected in (("root", result["data_contract"]["root_clean_rows"]), ("valid", 19867)):
        if any(len(cache[split + suffix]) != expected for suffix in ("_margins", "_y", "_sensitive")) or not np.isfinite(cache[split + "_margins"]).all():
            raise ValueError("Incomplete/nonfinite accepted score cache")
    raw = core.compute_metrics(cache["valid_y"], (cache["valid_margins"] > 0).astype(int), cache["valid_sensitive"])
    if any(abs(raw[key] - record["raw"][key]) > 1e-12 or abs(result["metrics"][key] - record["native"][key]) > 1e-12 for key in METRICS):
        raise ValueError("External native/margin/checkpoint metrics mismatch")
    checkpoint = torch.load(reference / "model.pt", map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint or any(not torch.isfinite(value).all() for value in checkpoint.values()):
        raise ValueError("Invalid accepted CNN checkpoint")
    return result, cache, checkpoint


@torch.no_grad()
def model_scores(model, images, batch_size):
    """Actual frozen CNN positive-class softmax scores, no margin surrogate."""
    if not isinstance(batch_size, int) or batch_size < 1 or len(images) < 1:
        raise ValueError("Nonempty images and positive scoring batch required")
    original_mode = model.training
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    probability, native = [], []
    model.eval()
    try:
        for first in range(0, len(images), batch_size):
            logits = model(images[first:first + batch_size])
            if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
                raise ValueError("Finite binary CNN logits required")
            probability.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            native.append(logits.argmax(dim=1).cpu().numpy())
    finally:
        model.train(original_mode)
    if any(not torch.equal(original[key], value) for key, value in model.state_dict().items()):
        raise ValueError("Score extraction changed the accepted model")
    return np.concatenate(probability), np.concatenate(native)


def scores_from_accepted_margins(cache):
    # Binary p1 = sigmoid(logit1-logit0) exactly as a mathematical identity.
    # float32 sigmoid may differ by one ULP from a fresh softmax; retain raw
    # margins/native decisions and declare the actual cache conversion path.
    return {split + "_probability": torch.sigmoid(torch.from_numpy(cache[split + "_margins"])).numpy()
            for split in ("root", "valid")}


def validate_mapping(mapping, metadata, root_hash, valid_hash, require_frozen=True):
    if require_frozen and (metadata.get("status") != "FROZEN" or metadata.get("approved") is not True):
        raise ValueError("Explicit reviewed client mapping is not frozen")
    if metadata["semantics"] not in {"declared_virtual_partition", "synthetic_cohorts_only"}:
        raise ValueError("Central validation cannot be claimed to have true training-client identities")
    if require_frozen and metadata["semantics"] == "synthetic_cohorts_only":
        raise ValueError("Synthetic IDs cannot authorize a real-data evaluation")
    if set(mapping) != {"root_image_id", "root_client_id", "valid_image_id", "valid_client_id"}:
        raise ValueError("Incomplete explicit client mapping")
    for split, expected_hash in (("root", root_hash), ("valid", valid_hash)):
        image, client = mapping[split + "_image_id"], mapping[split + "_client_id"]
        if image.dtype != np.int64 or client.dtype != np.int64 or image.ndim != 1 or client.shape != image.shape or len(np.unique(image)) != len(image):
            raise ValueError("Stable unique int64 sample IDs and explicit equal-length int64 client IDs required")
        if array_digest(image) != expected_hash or metadata[split + "_image_ids_sha256"] != expected_hash:
            raise ValueError("Mapping row order/sample identity differs from accepted root/valid")
        if set(np.unique(client)) != set(range(20)):
            raise ValueError("Explicit mapping must preserve all twenty fitted local populations")
    if np.intersect1d(mapping["root_image_id"], mapping["valid_image_id"]).size:
        raise ValueError("Calibration/evaluation samples overlap")
    return True


def fit_predict(probability, labels, sensitive, mapping, settings, fit_seed):
    """Only root labels enter official local/global optimization and beta MLE."""
    if settings.get("calibration") is not True:
        raise ValueError("This interface is the calibrated official DP variant")
    import netcal
    if netcal.__version__ != "1.3.6":
        raise ValueError("Revalidate a changed netcal runtime before use")
    torch.manual_seed(fit_seed)
    np.random.seed(fit_seed)
    torch.use_deterministic_algorithms(True)
    module, source = official_adapter()
    post = module.OfficialLoGoFairDP(source=source, **settings)
    post.fit(probability["root_probability"], labels["root_y"], sensitive["root_sensitive"], mapping["root_client_id"])
    prediction = post.predict(probability["valid_probability"], sensitive["valid_sensitive"], mapping["valid_client_id"])
    for client in post.clients.values():
        for calibrator in (client.score_calibrated_0, client.score_calibrated_1):
            if calibrator.method != "mle" or any(not np.isfinite(site["values"]).all() for site in calibrator._sites.values()):
                raise FloatingPointError("Nonfinite or changed official BetaCalibration fit")
    return post, prediction


def save_post(post, path):
    # Serialize only fitted netcal calibrators and numerical state. Dynamic
    # official client classes / training datasets are never pickled.
    state = dict(format="logofair_dp_calibrated_v1", settings=post.settings, history=post.history,
        global_lambda=post.global_lambda,
        thresholds={cid: value.cpu().numpy() for cid, value in post.thresholds.items()},
        calibration={cid: (client.score_calibrated_0, client.score_calibrated_1) for cid, client in post.clients.items()},
        local_mu={cid: client.local_mu.detach().cpu().tolist() for cid, client in post.clients.items()})
    with Path(path).open("wb") as destination:
        pickle.dump(state, destination, protocol=4)
    return state


def load_post(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError("Fitted state SHA mismatch")
    with Path(path).open("rb") as source:
        state = pickle.load(source)
    if state["format"] != "logofair_dp_calibrated_v1" or state["settings"]["calibration"] is not True:
        raise ValueError("Unexpected fitted postprocessing state")
    module, source = official_adapter()
    post = module.OfficialLoGoFairDP(source=source, **state["settings"])
    post.clients = {cid: SimpleNamespace(calib=True, score_calibrated_0=pair[0], score_calibrated_1=pair[1]) for cid, pair in state["calibration"].items()}
    post.thresholds = {cid: torch.from_numpy(value) for cid, value in state["thresholds"].items()}
    post.history, post.global_lambda = state["history"], state["global_lambda"]
    return post


def validate_job(job, protocol, require_frozen=True):
    if require_frozen and (protocol["status"] != "FROZEN" or any(row["status"] != "APPROVED" for row in protocol["decisions"].values())):
        raise ValueError("LoGoFair protocol/mapping decisions are not frozen; PREPARED cannot execute")
    candidate = next((row for row in protocol["candidates"] if row["id"] == job["candidate"]), None)
    if candidate is None or job["settings"] != candidate["settings"] or job["method"] != "LoGoFair-DP-official-adapted":
        raise ValueError("Unknown calibrated DP candidate")
    if job["evaluation_split"] != "valid" or job["seed"] != 91001 or job["fit_seed"] != 1719:
        raise ValueError("Expected fixed single-seed valid-only postprocessing screen")
    if set(job["local_hashes"]) != LOCAL_FILES:
        raise ValueError("Missing postprocessing source identities")
    if require_frozen and any(not isinstance(job[key], str) or len(job[key]) != 64 for key in ("mapping_sha256", "mapping_metadata_sha256")):
        raise ValueError("Explicit mapping artifacts are not hash-bound")


def run(repo, reference, job_path, mapping_path, mapping_meta_path, output):
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    job = json.loads(Path(job_path).read_text(encoding="utf8"))
    validate_job(job, protocol)
    if digest(mapping_path) != job["mapping_sha256"] or digest(mapping_meta_path) != job["mapping_metadata_sha256"]:
        raise ValueError("Input client mapping changed")
    for name, wanted in job["local_hashes"].items():
        if digest(HERE / name) != wanted:
            raise ValueError("Postprocessing source/protocol changed: " + name)
    entries = json.loads((HERE / "reuse_manifest.json").read_text(encoding="utf8"))["entries"]
    entry = next(row for row in entries if row["id"] == job["baseline_id"])
    if entry["source_job"]["config"]["seed"] != job["seed"]:
        raise ValueError("FedAvg seed differs from postprocessing job")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "mapping.npz").write_bytes(Path(mapping_path).read_bytes())
        (output / "mapping_metadata.json").write_bytes(Path(mapping_meta_path).read_bytes())
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        core = load_core(repo)
        original, cache, _ = verify_external(reference, entry, core)
        with np.load(mapping_path, allow_pickle=False) as data:
            mapping = {key: data[key].copy() for key in data.files}
        metadata = json.loads(Path(mapping_meta_path).read_text(encoding="utf8"))
        contract = original["data_contract"]["image_data_contract"]
        validate_mapping(mapping, metadata, contract["root_image_ids_sha256"], contract["evaluation_image_ids_sha256"])
        probability = scores_from_accepted_margins(cache)
        labels = {"root_y": cache["root_y"]}  # evaluation labels not passed to fit
        sensitive = {split + "_sensitive": cache[split + "_sensitive"] for split in ("root", "valid")}
        post, prediction = fit_predict(probability, labels, sensitive, mapping, job["settings"], job["fit_seed"])
        save_post(post, output / "post_state.pkl")
        state_sha = digest(output / "post_state.pkl")
        restored = load_post(output / "post_state.pkl", state_sha)
        if not np.array_equal(prediction, restored.predict(probability["valid_probability"], sensitive["valid_sensitive"], mapping["valid_client_id"])):
            raise ValueError("Serialized calibrated classifier prediction differs")
        np.savez_compressed(output / "scores_predictions.npz", **probability, **mapping, **sensitive,
                            root_y=cache["root_y"], valid_y=cache["valid_y"], prediction=prediction,
                            valid_native_prediction=(cache["valid_margins"] > 0).astype(int))
        import netcal
        result = dict(status="complete", evidence_stage="validation_postprocessing_screen", job=job,
            baseline_id=entry["id"], checkpoint_sha256=entry["accepted_record"]["checkpoint_sha256"],
            original_result_sha256=entry["accepted_record"]["original_result_sha256"],
            accepted_margin_cache_sha256=entry["accepted_record"]["cache_sha256"],
            mapping_sha256=job["mapping_sha256"], mapping_metadata_sha256=job["mapping_metadata_sha256"],
            root_image_ids_sha256=contract["root_image_ids_sha256"], valid_image_ids_sha256=contract["evaluation_image_ids_sha256"],
            metrics=core.compute_metrics(cache["valid_y"], prediction, cache["valid_sensitive"]),
            settings=job["settings"], fit_seed=job["fit_seed"], history=post.history,
            thresholds={str(cid): value.tolist() for cid, value in post.thresholds.items()},
            score_path="sigmoid of accepted frozen-model binary logit margins; no new CNN inference",
            method_note="official calibrated DP local/global optimizer; calibration-only priors correction; explicit declared client-ID partition; not original EO or true training-client identity",
            environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, netcal=netcal.__version__),
            component=dict(adapter_sha256=ADAPTER_SHA, official_lf_sha256=OFFICIAL_LF_SHA, core_sha256=CORE_SHA))
        write_json(output / "result.json", result)
        write_json(output / "acceptance.json", dict(status="PASS", job_sha256=digest(job_path),
            artifact_hashes={name: digest(output / name) for name in ("post_state.pkl", "scores_predictions.npz", "result.json", "mapping.npz", "mapping_metadata.json")}))
        checked_output(job_path, output, core, reference)
    except BaseException as error:
        write_json(output / "failure.json", dict(error=repr(error), traceback=traceback.format_exc(), failed_unix=time.time()))
        raise


def checked_output(job_path, output, core, reference):
    output = Path(output)
    job = json.loads(Path(job_path).read_text(encoding="utf8"))
    validate_job(job, json.loads((HERE / "protocol.json").read_text(encoding="utf8")))
    for name, wanted in job["local_hashes"].items():
        if digest(HERE / name) != wanted:
            raise ValueError("Postprocessing source/protocol identity changed")
    if list(output.glob("failure*.json")):
        raise ValueError("Preserved postprocessing failure blocks acceptance")
    if not (output / "result.json").exists():
        return None
    receipt = json.loads((output / "acceptance.json").read_text(encoding="utf8"))
    if receipt["status"] != "PASS" or receipt["job_sha256"] != digest(job_path) or set(receipt["artifact_hashes"]) != {"post_state.pkl", "scores_predictions.npz", "result.json", "mapping.npz", "mapping_metadata.json"}:
        raise ValueError("Incomplete postprocessing artifact receipt")
    for name, wanted in receipt["artifact_hashes"].items():
        if digest(output / name) != wanted:
            raise ValueError("Postprocessing artifact hash changed")
    result = json.loads((output / "result.json").read_text(encoding="utf8"))
    if result["job"] != job or result["status"] != "complete" or result["settings"] != job["settings"] or result["fit_seed"] != job["fit_seed"]:
        raise ValueError("Postprocessing job/settings mismatch")
    if result["mapping_sha256"] != job["mapping_sha256"] or result["mapping_metadata_sha256"] != job["mapping_metadata_sha256"]:
        raise ValueError("Postprocessing mapping identity mismatch")
    if digest(output / "mapping.npz") != job["mapping_sha256"] or digest(output / "mapping_metadata.json") != job["mapping_metadata_sha256"]:
        raise ValueError("Saved mapping artifacts differ from reviewed inputs")
    entries = json.loads((HERE / "reuse_manifest.json").read_text(encoding="utf8"))["entries"]
    entry = next(row for row in entries if row["id"] == job["baseline_id"])
    original, accepted_cache, _ = verify_external(reference, entry, core)
    contract = original["data_contract"]["image_data_contract"]
    with np.load(output / "mapping.npz", allow_pickle=False) as data:
        original_mapping = {key: data[key].copy() for key in data.files}
    mapping_metadata = json.loads((output / "mapping_metadata.json").read_text(encoding="utf8"))
    validate_mapping(original_mapping, mapping_metadata, contract["root_image_ids_sha256"], contract["evaluation_image_ids_sha256"])
    if result["baseline_id"] != entry["id"] or result["evidence_stage"] != "validation_postprocessing_screen" or result["component"] != dict(adapter_sha256=ADAPTER_SHA, official_lf_sha256=OFFICIAL_LF_SHA, core_sha256=CORE_SHA):
        raise ValueError("Changed method/component/baseline identity")
    for output_key, record_key in (("checkpoint_sha256", "checkpoint_sha256"), ("original_result_sha256", "original_result_sha256"), ("accepted_margin_cache_sha256", "cache_sha256")):
        if result[output_key] != entry["accepted_record"][record_key]:
            raise ValueError("External checkpoint/cache identity mismatch")
    if [row["round"] for row in result["history"]] != list(range(1, job["settings"]["post_rounds"] + 1)) or any(not math.isfinite(row["global_lambda"]) for row in result["history"]):
        raise ValueError("Incomplete/nonfinite official postprocessing rounds")
    with np.load(output / "scores_predictions.npz", allow_pickle=False) as data:
        cache = {key: data[key].copy() for key in data.files}
    if len(cache["valid_y"]) != 19867 or any(not np.isfinite(cache[key]).all() or ((cache[key] < 0) | (cache[key] > 1)).any() for key in ("root_probability", "valid_probability")):
        raise ValueError("Wrong/nonfinite validation score cache")
    if array_digest(cache["root_image_id"]) != result["root_image_ids_sha256"] or array_digest(cache["valid_image_id"]) != result["valid_image_ids_sha256"]:
        raise ValueError("Postprocessing sample identity mismatch")
    if any(not np.array_equal(cache[key], original_mapping[key]) for key in original_mapping):
        raise ValueError("Prediction mapping differs from reviewed input mapping")
    if result["root_image_ids_sha256"] != contract["root_image_ids_sha256"] or result["valid_image_ids_sha256"] != contract["evaluation_image_ids_sha256"]:
        raise ValueError("Root/validation rows differ from accepted external checkpoint")
    expected_probability = scores_from_accepted_margins(accepted_cache)
    for key in ("root_y", "valid_y", "root_sensitive", "valid_sensitive"):
        if not np.array_equal(cache[key], accepted_cache[key]):
            raise ValueError("Postprocessing labels/attributes differ from accepted cache")
    for key in ("root_probability", "valid_probability"):
        if not np.array_equal(cache[key], expected_probability[key]):
            raise ValueError("Postprocessing score cache differs from accepted model")
    if not np.array_equal(cache["valid_native_prediction"], (accepted_cache["valid_margins"] > 0).astype(int)):
        raise ValueError("Native prediction cache changed")
    post = load_post(output / "post_state.pkl", receipt["artifact_hashes"]["post_state.pkl"])
    if post.history != result["history"] or {str(cid): value.tolist() for cid, value in post.thresholds.items()} != result["thresholds"]:
        raise ValueError("Recorded threshold/history differs from serialized official classifier")
    prediction = post.predict(cache["valid_probability"], cache["valid_sensitive"], cache["valid_client_id"])
    if not np.array_equal(prediction, cache["prediction"]) or any(not math.isfinite(result["metrics"][key]) or abs(core.compute_metrics(cache["valid_y"], prediction, cache["valid_sensitive"])[key] - result["metrics"][key]) > 1e-12 for key in METRICS):
        raise ValueError("Mixed-predictor or invalid metric result")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "reference", "job", "mapping", "mapping-meta", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.reference, args.job, args.mapping, args.mapping_meta, args.out)
