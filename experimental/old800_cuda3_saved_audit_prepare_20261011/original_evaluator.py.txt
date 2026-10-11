"""Terminal-model prediction views; labels are scored after predictions are fixed.

This component does not freeze a protocol, dispatch a job, train a model, or
load a dataset. The caller must supply the declared model/root/target identities.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import re
import types

import numpy as np
import torch

METHODS = {"FedAvg", "Median", "FLTrust", "FairFed", "FairGuard", "FLTrust+FairGuard",
           "GuardFed-AD2+", "FedAA", "LASA"}
VIEWS = {"native", "raw", "shared_calibration"}
CORE_SHA256 = "cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed"
SHARED_CALIBRATION = {
    "ad2_calibration_base_weight": 1.0, "ad2_calibration_budget": 0.06,
    "ad2_calibration_temperature": 0.03, "ad2_calibration_quantiles": 41,
    "ad2_calibration_max_acc_drop": 0.005, "ad2_calibration_objective": "acc_floor",
    "ad2_calibration_enabled": True,
}
DECISIONS = {
    "campaign_scope_and_wait_for_remaining_methods", "evaluation_endpoint_and_prior_exposure_claim_wording",
    "primary_view_and_complete_collected_views", "paired_inference_and_multiplicity_policy",
}
REQUIRED_IDENTITY_FILES = {
    "scripts/reproduce_paper_tables.py", "src/celeba_data.py", "src/data_loader.py",
    "data/celeba/derived/rgb64_v1/manifest.json", "data/celeba/derived/rgb64_v1/metadata.npz",
    "data/celeba/derived/rgb64_v1/images.npy", "data/celeba/derived/rgb64_v1/available.npy",
    "data/celeba/list_attr_celeba.txt", "data/celeba/list_eval_partition.txt",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def vector_identity(values):
    array = np.asarray(values)
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def weights_identity(model):
    return {name: vector_identity(tensor.detach().cpu().numpy()) for name, tensor in model.state_dict().items()}


def check_binary(values, name, n):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != n or not np.isin(values, [0, 1]).all():
        raise ValueError(name + " must be a complete binary vector")
    return values.astype(np.int64, copy=False)


def group_metrics(labels, predictions, sensitive):
    """Metrics and all denominators from one exact prediction vector."""
    pred = np.asarray(predictions)
    if pred.ndim != 1 or not len(pred):
        raise ValueError("Nonempty prediction vector required")
    n = len(pred)
    pred = check_binary(pred, "predictions", n)
    y = check_binary(labels, "labels", n)
    s = check_binary(sensitive, "sensitive", n)
    groups, warnings = {}, []
    for group in [0, 1]:
        mask = s == group
        tp = int(np.sum(mask & (y == 1) & (pred == 1)))
        fp = int(np.sum(mask & (y == 0) & (pred == 1)))
        tn = int(np.sum(mask & (y == 0) & (pred == 0)))
        fn = int(np.sum(mask & (y == 1) & (pred == 0)))
        total = tp + fp + tn + fn
        positives, negatives = tp + fn, tn + fp
        tpr = tp / positives if positives else None
        fpr = fp / negatives if negatives else None
        rate = (tp + fp) / total if total else None
        groups[str(group)] = {"n": total, "positives": positives, "negatives": negatives,
                              "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                              "tpr": tpr, "fpr": fpr, "positive_rate": rate}
        if not positives:
            warnings.append(f"AEOD denominator is zero for sensitive group {group}")
        if not total:
            warnings.append(f"ASPD denominator is zero for sensitive group {group}")
    def gap(field):
        a, b = groups["0"][field], groups["1"][field]
        return None if a is None or b is None else abs(a - b)
    return {"accuracy": int(np.sum(y == pred)) / n, "aeod": gap("tpr"), "aspd": gap("positive_rate"),
            "metric_definition": "aeod is absolute TPR gap, not full equalized odds",
            "positive_rate": int(pred.sum()) / n,
            "majority_accuracy": max(int(np.sum(y == 0)), int(np.sum(y == 1))) / n,
            "prediction_count": n, "constant_predictions": bool(np.all(pred == pred[0])),
            "group_confusion_counts": groups, "warnings": warnings,
            "fairness_status": "DEFINED" if gap("tpr") is not None and gap("positive_rate") is not None else "UNDEFINED_DENOMINATOR"}


def thresholds_from_root(core, root_margins, root_y, root_sensitive, config):
    """Execute unchanged frozen threshold code using root-only cached margins.

    A private function-global mapping avoids mutating core.model_margins or
    exposing any target labels, target scores, or target attributes to fitting.
    """
    if digest(core.__file__) != CORE_SHA256:
        raise ValueError("Frozen calibration core bytes changed")
    for field in ("batch_size", "ad2_calibration_quantiles"):
        value = getattr(config, field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Invalid calibration integer: " + field)
    for field in ("ad2_calibration_base_weight", "ad2_calibration_budget",
                  "ad2_calibration_temperature", "ad2_calibration_max_acc_drop"):
        value = getattr(config, field)
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError("Invalid finite calibration setting: " + field)
    if config.ad2_calibration_temperature <= 0 or config.ad2_calibration_objective not in {"acc_floor", "original"}:
        raise ValueError("Unsupported calibration temperature or objective")
    margins = np.asarray(root_margins)
    if margins.ndim != 1 or not len(margins) or not np.isfinite(margins).all():
        raise ValueError("Finite complete root margins required")
    y = check_binary(root_y, "root labels", len(margins))
    s = check_binary(root_sensitive, "root sensitive", len(margins))
    if any(not np.any((s == group) & (y == label)) for group in [0, 1] for label in [0, 1]):
        raise ValueError("Root calibration lacks a sensitive/label cell")
    sentinel = object()
    def cached_root(_model, X, batch_size):
        if X is not sentinel or batch_size != config.batch_size:
            raise ValueError("Threshold fit attempted a non-root margin access")
        return margins
    globals_copy = dict(core.fit_group_thresholds.__globals__)
    globals_copy["model_margins"] = cached_root
    fitter = types.FunctionType(core.fit_group_thresholds.__code__, globals_copy,
                               core.fit_group_thresholds.__name__, core.fit_group_thresholds.__defaults__)
    root_only = {"server_X": sentinel, "server_y": torch.from_numpy(y.copy()), "server_sensitive": s}
    thresholds, diagnostics = fitter(None, root_only, config)
    if set(thresholds) != {0, 1} or not all(np.isfinite(v) for v in thresholds.values()):
        raise ValueError("Invalid fitted thresholds")
    return thresholds, diagnostics


def _seal_fit(fit):
    # Detect a changed fit between the label-free fit and prediction stages.
    return {**fit, "fit_sha256": canonical_sha(fit)}


def fit_views(core, method, root_margins, root_y, root_sensitive, config, selected_views, shared_calibration):
    if method not in METHODS or not selected_views or len(set(selected_views)) != len(selected_views) or not set(selected_views) <= VIEWS:
        raise ValueError("Unknown method, repeated view, or incomplete declared view set")
    if "shared_calibration" in selected_views and shared_calibration != SHARED_CALIBRATION:
        raise ValueError("Shared calibration recipe differs from the prepared frozen-core recipe")
    if method == "GuardFed-AD2+" and "native" in selected_views and config.ad2_calibration_enabled is not True:
        raise ValueError("Current900 AD2+ native requires its original calibration")
    fits = {}
    for view in selected_views:
        calibrated = view == "shared_calibration" or (view == "native" and method == "GuardFed-AD2+" and config.ad2_calibration_enabled)
        if calibrated:
            selected_config = dataclasses.replace(config, **shared_calibration) if view == "shared_calibration" else config
            thresholds, diagnostics = thresholds_from_root(core, root_margins, root_y, root_sensitive, selected_config)
            fits[view] = {"rule": "group_margin_greater_equal", "thresholds": thresholds,
                          "fit_data": "clean_train_root_only", "fit_diagnostics": diagnostics}
        else:
            fits[view] = {"rule": "argmax_margin_strictly_positive", "thresholds": None, "fit_data": "none"}
        fits[view] = _seal_fit({"view": view, "method": method, **fits[view]})
    return fits


def predict_views(target_margins, target_sensitive, fits):
    if not fits or not set(fits) <= VIEWS:
        raise ValueError("Nonempty known prediction views required")
    margins = np.asarray(target_margins)
    if margins.ndim != 1 or not len(margins) or not np.isfinite(margins).all():
        raise ValueError("Finite complete target margins required")
    sensitive = check_binary(target_sensitive, "target sensitive", len(margins))
    predictions = {}
    for name, fit in fits.items():
        if not isinstance(fit, dict):
            raise ValueError("Invalid prediction fit")
        payload = {key: value for key, value in fit.items() if key != "fit_sha256"}
        if fit.get("fit_sha256") != canonical_sha(payload) or fit.get("view") != name or fit.get("method") not in METHODS:
            raise ValueError("Prediction fit identity changed")
        calibrated = name == "shared_calibration" or (name == "native" and fit["method"] == "GuardFed-AD2+")
        expected_rule = "group_margin_greater_equal" if calibrated else "argmax_margin_strictly_positive"
        if fit.get("rule") != expected_rule:
            raise ValueError("Prediction rule contradicts the declared view")
        if fit["rule"] == "argmax_margin_strictly_positive":
            if fit.get("thresholds") is not None or fit.get("fit_data") != "none":
                raise ValueError("Raw argmax must not carry fitted thresholds")
            predictions[name] = (margins > 0).astype(np.uint8)
        elif fit["rule"] == "group_margin_greater_equal":
            thresholds = fit["thresholds"]
            if (not isinstance(thresholds, dict) or set(thresholds) != {0, 1}
                    or any(isinstance(t, bool) or not isinstance(t, (int, float, np.number)) or not np.isfinite(t)
                           for t in thresholds.values()) or fit.get("fit_data") != "clean_train_root_only"):
                raise ValueError("Incomplete declared group thresholds")
            predictions[name] = np.where(sensitive == 0, margins >= thresholds[0], margins >= thresholds[1]).astype(np.uint8)
        else:
            raise ValueError("Unknown prediction rule")
    return predictions


def evaluate_frozen_predictions(predictions, labels, sensitive):
    # This is the first interface receiving target y; fitting/prediction cannot.
    if not predictions or not set(predictions) <= VIEWS:
        raise ValueError("Nonempty known prediction views required")
    return {name: group_metrics(labels, pred, sensitive) for name, pred in predictions.items()}


def extract_and_predict(core, model, root_bundle, target_X, target_sensitive, method, config,
                        selected_views, shared_calibration):
    before = weights_identity(model)
    root_margins = core.model_margins(model, root_bundle["server_X"], config.batch_size)
    fits = fit_views(core, method, root_margins, root_bundle["server_y"].cpu().numpy(),
                     root_bundle["server_sensitive"], config, selected_views, shared_calibration)
    target_margins = core.model_margins(model, target_X, config.batch_size)
    predictions = predict_views(target_margins, target_sensitive, fits)
    if weights_identity(model) != before:
        raise RuntimeError("Inference or calibration changed checkpoint weights")
    return root_margins, target_margins, fits, predictions


def require_dispatch(protocol, job, receipt, source_hashes, observed=None):
    """Validate an external trusted receipt and independently measured identities.

    This is a gate, not a dispatcher or attestation generator. The caller hashes
    files and non-label sample identities before loading target labels/weights.
    """
    def sha(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    if protocol.get("status") != "FROZEN" or job.get("status") != "FROZEN" or job.get("dispatchable") is not True:
        raise ValueError("Prepared evaluation is not dispatchable")
    if (set(protocol.get("decisions", {})) != DECISIONS
            or any(value is None or value == "" for value in protocol["decisions"].values())):
        raise ValueError("Final evaluation choices are unresolved")
    views = protocol.get("selected_views")
    if not views or len(set(views)) != len(views) or not set(views) <= VIEWS or job.get("selected_views") != views:
        raise ValueError("Complete predeclared views required")
    if (protocol.get("round") != 70 or protocol.get("original_split") != "valid"
            or protocol.get("calibration_core_sha256") != CORE_SHA256
            or ("shared_calibration" in views and protocol.get("available_views", {}).get("shared_calibration", {}).get("calibration") != SHARED_CALIBRATION)):
        raise ValueError("Protocol model round, native interface or calibration recipe drifted")
    if (not sha(protocol.get("target_image_ids_sha256")) or not sha(protocol.get("target_order_sha256"))
            or not sha(protocol.get("model_inventory_sha256")) or not protocol.get("target_split")
            or type(protocol.get("target_n")) is not int or protocol["target_n"] <= 0):
        raise ValueError("Target sample identity has not been attested")
    if receipt.get("status") != "DISPATCH_FROZEN" or receipt.get("protocol_canonical_sha256") != canonical_sha(protocol):
        raise ValueError("Missing or changed frozen dispatch receipt")
    if (not source_hashes or not {"evaluator.py", "scripts/reproduce_paper_tables.py", "src/celeba_data.py"} <= set(source_hashes)
            or not all(sha(value) for value in source_hashes.values())
            or source_hashes["scripts/reproduce_paper_tables.py"] != CORE_SHA256
            or source_hashes.get("evaluator.py") != digest(__file__)
            or receipt.get("evaluation_source_sha256") != source_hashes):
        raise ValueError("Evaluation source identity changed")
    if receipt.get("job_canonical_sha256") != canonical_sha(job):
        raise ValueError("Evaluation job identity changed")
    replay = receipt.get("native_valid_replay", {})
    if (replay.get("accepted") is not True or replay.get("model_count") != 900
            or not isinstance(replay.get("max_abs_metric_difference"), (int, float))
            or not np.isfinite(replay["max_abs_metric_difference"])
            or not 0 <= replay["max_abs_metric_difference"] <= 1e-12
            or not sha(replay.get("acceptance_sha256"))
            or replay.get("model_inventory_sha256") != protocol["model_inventory_sha256"]):
        raise ValueError("Native valid replay has not been accepted")
    if job.get("terminal_round") != 70 or job.get("original_split") != "valid":
        raise ValueError("Accepted terminal valid model required")
    if (job.get("method") not in METHODS or type(job.get("seed")) is not int or job["seed"] not in range(91001, 91011)
            or job.get("distribution") not in {"IID", "non-IID"}
            or job.get("actual_alpha") != {"IID": 5000.0, "non-IID": 5.0}.get(job.get("distribution"))
            or job.get("attack") not in {"Benign", "F Flip", "FedSA", "S-DFA", "Sp-DFA"}):
        raise ValueError("Evaluation is outside the current nine-method cohort")
    if not isinstance(job.get("model_id"), str) or not job["model_id"] or job.get("id") != "prepared_eval_" + job["model_id"]:
        raise ValueError("Model/job identity is inconsistent")
    expected_source_method = {"FedAA": "FedAA-DDPG-adapted-v1", "LASA": "LASA-official"}.get(job["method"], job["method"])
    if job.get("source_method") != expected_source_method:
        raise ValueError("Source method identity is inconsistent")
    expected = {key: job.get(key) for key in ("checkpoint_sha256", "original_result_sha256", "original_job_sha256",
                "config_canonical_sha256", "root_image_ids_sha256", "source_hashes", "model_id")}
    expected.update({key: protocol[key] for key in ("model_inventory_sha256", "target_image_ids_sha256",
                                                  "target_order_sha256", "target_n", "target_split")})
    if any(not sha(value) for key, value in expected.items() if key.endswith("sha256")):
        raise ValueError("Incomplete expected model/data identity")
    model_sources = expected["source_hashes"]
    if (not isinstance(model_sources, dict) or not REQUIRED_IDENTITY_FILES <= set(model_sources)
            or not all(sha(value) for value in model_sources.values())
            or model_sources["scripts/reproduce_paper_tables.py"] != CORE_SHA256
            or model_sources["src/celeba_data.py"] != source_hashes["src/celeba_data.py"]):
        raise ValueError("Incomplete model source/data identity")
    if (not isinstance(observed, dict) or any(observed.get(key) != value for key, value in expected.items())
            or receipt.get("model_inventory_sha256") != protocol["model_inventory_sha256"]):
        raise ValueError("Observed model, data or target identity drifted")
    record = observed.get("model_inventory_record", {})
    record_fields = ("method", "source_method", "distribution", "actual_alpha", "attack", "seed", "terminal_round",
                     "original_split", "config_canonical_sha256", "source_hashes")
    if (not isinstance(record, dict) or record.get("id") != job["model_id"]
            or any(record.get(key) != job[key] for key in record_fields)
            or canonical_sha(record.get("config")) != job["config_canonical_sha256"]
            or record.get("checkpoint", {}).get("sha256") != job["checkpoint_sha256"]
            or record.get("result", {}).get("sha256") != job["original_result_sha256"]
            or record.get("raw_job", {}).get("sha256") != job["original_job_sha256"]
            or record.get("data_contract", {}).get("root_image_ids_sha256") != job["root_image_ids_sha256"]):
        raise ValueError("Job does not match its measured inventory record")
    adapters = record.get("adapter_source_hashes")
    if (not isinstance(adapters, dict) or not all(sha(value) for value in adapters.values())
            or observed.get("adapter_source_hashes") != adapters):
        raise ValueError("Method adapter source identity is incomplete or changed")
    for key in ("target_image_ids_sha256", "target_order_sha256", "target_n", "target_split"):
        if job.get(key) != protocol[key]:
            raise ValueError("Job target identity is incomplete or inconsistent")
    backup = job.get("model_backup_ref", {})
    if (backup.get("sha256") != job["checkpoint_sha256"] or not sha(backup.get("archive_sha256"))
            or not isinstance(backup.get("member"), str) or not backup["member"].endswith("/model.pt")
            or type(backup.get("bytes")) is not int or backup["bytes"] <= 0):
        raise ValueError("Incomplete checkpoint backup identity")
