"""Local acceptance only: synthetic images and eight already accepted valid caches.

No server, dataset loader, test labels, checkpoint loading, training or dispatch.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tarfile
import traceback
from unittest import mock

import numpy as np
import torch

import evaluator as ev

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FROZEN = None
HISTORY = REPO / "docs/server_deployment_20260923/training_20260923/celeba_shared_calibration_v1"
PREPARED = REPO / "docs/server_deployment_20260923/training_20260923/final_evaluation_prepared_20261009"
ARCHIVE = HISTORY / "sharedcal700_and_baseline_gates_20260928.tar.gz"
ARCHIVE_SHA = "f7528394e8888163e157323654ad9cee31f4d32a1b65a0ef57cb6894382e352e"
PREFIX = "results/revision_20260928/celeba_shared_calibration_v1"
SELECTED = [
    "FedAvg_IID_Benign_seed91001", "FairFed_lr0.001_S-DFA_seed91002",
    "Median_IID_S-DFA_seed91003", "FLTrust_lr0.0005_Benign_seed91004",
    "FairGuard_IID_Benign_seed91005", "FLTrust+FairGuard_non-IID_S-DFA_seed91006",
    "GuardFed-AD2+_IID_S-DFA_seed91007", "GuardFed-AD2+_non-IID_Benign_seed91008",
]
CHECKS = []
INPUTS = {}


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def check(name, function):
    try:
        detail = function()
        CHECKS.append({"name": name, "status": "PASS", "detail": detail})
        print("PASS " + name, flush=True)
        return detail
    except Exception:
        CHECKS.append({"name": name, "status": "FAIL", "traceback": traceback.format_exc()})
        print("FAIL " + name, flush=True)
        raise


def rejected(function):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        return {"exception": type(error).__name__, "reason": str(error)}
    raise AssertionError("Unsafe input was accepted")


def resolve_frozen_root(repo):
    candidates = [repo, repo / "tmp/revision-publish-20260928"]
    inspected = []
    for candidate in candidates:
        source = candidate / "scripts/reproduce_paper_tables.py"
        actual = ev.digest(source) if source.is_file() else "MISSING"
        if actual == ev.CORE_SHA256:
            return candidate
        inspected.append(str(source) + "=" + actual)
    raise RuntimeError("Frozen calibration core unavailable; required SHA256 " + ev.CORE_SHA256
                       + "; checked " + "; ".join(inspected))


def frozen_root_resolution():
    primary = REPO / "scripts/reproduce_paper_tables.py"
    fallback_root = REPO / "tmp/revision-publish-20260928"
    fallback = fallback_root / "scripts/reproduce_paper_tables.py"
    with mock.patch.object(Path, "is_file", return_value=True):
        with mock.patch.object(ev, "digest", side_effect=lambda source: ev.CORE_SHA256):
            assert resolve_frozen_root(REPO) == REPO
        with mock.patch.object(ev, "digest", side_effect=lambda source: ev.CORE_SHA256 if source == fallback else "d" * 64):
            assert resolve_frozen_root(REPO) == fallback_root
        with mock.patch.object(ev, "digest", return_value="d" * 64):
            wrong_hashes = rejected(lambda: resolve_frozen_root(REPO))
    with mock.patch.object(Path, "is_file", return_value=False), mock.patch.object(ev, "digest") as hasher:
        missing = rejected(lambda: resolve_frozen_root(REPO))
        hasher.assert_not_called()
    return {"root_preferred_when_exact_hash": True, "fallback_requires_exact_hash": True,
            "both_hashes_wrong_rejected": wrong_hashes, "both_missing_rejected": missing,
            "candidate_paths": [str(primary), str(fallback)]}


def load_core():
    global FROZEN
    FROZEN = resolve_frozen_root(REPO)
    path = FROZEN / "scripts/reproduce_paper_tables.py"
    assert ev.digest(path) == ev.CORE_SHA256
    INPUTS["selected_frozen_root"] = str(FROZEN)
    INPUTS["frozen_sources"] = {
        str(path): ev.digest(path), str(FROZEN / "src/celeba_data.py"): ev.digest(FROZEN / "src/celeba_data.py"),
        str(FROZEN / "src/data_loader.py"): ev.digest(FROZEN / "src/data_loader.py"),
    }
    spec = importlib.util.spec_from_file_location("frozen_guardfed_core_acceptance", path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    return core


def hand_metrics():
    y = [1, 1, 0, 0, 1, 0, 0, 0]
    p = [1, 0, 1, 0, 1, 0, 0, 0]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    result = ev.group_metrics(y, p, s)
    assert result["accuracy"] == 6 / 8
    assert result["aeod"] == 1 / 2
    assert result["aspd"] == 1 / 4
    assert result["positive_rate"] == 3 / 8
    assert result["majority_accuracy"] == 5 / 8
    assert result["group_confusion_counts"] == {
        "0": {"n": 4, "positives": 2, "negatives": 2, "tp": 1, "fp": 1, "tn": 1, "fn": 1,
              "tpr": 0.5, "fpr": 0.5, "positive_rate": 0.5},
        "1": {"n": 4, "positives": 1, "negatives": 3, "tp": 1, "fp": 0, "tn": 3, "fn": 0,
              "tpr": 1.0, "fpr": 0.0, "positive_rate": 0.25},
    }
    return {"labels": y, "predictions": p, "sensitive": s, "expected_accuracy": 0.75,
            "expected_aeod": 0.5, "expected_aspd": 0.25, "actual": result}


def zero_and_constant():
    no_positive = ev.group_metrics([0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1])
    assert no_positive["aeod"] is None and no_positive["aspd"] == 0
    assert no_positive["fairness_status"] == "UNDEFINED_DENOMINATOR"
    absent_group = ev.group_metrics([0, 1], [0, 1], [0, 0])
    assert absent_group["aeod"] is None and absent_group["aspd"] is None
    assert absent_group["group_confusion_counts"]["1"]["n"] == 0
    no_negative = ev.group_metrics([1, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1])
    assert no_negative["group_confusion_counts"]["0"]["fpr"] is None
    constant = [ev.group_metrics([0, 1, 1, 0], [value] * 4, [0, 0, 1, 1]) for value in [0, 1]]
    for result in constant:
        assert result["constant_predictions"] and result["accuracy"] == 0.5
        assert result["aeod"] == 0 and result["aspd"] == 0
    return {"no_positive": no_positive, "absent_group": absent_group, "no_negative": no_negative,
            "constant_all_zero_and_all_one": constant}


def binary_rejections():
    cases = {
        "empty": ([], [], []), "short_labels": ([1], [1, 0], [0, 1]),
        "column_predictions": ([1, 0], [[1], [0]], [0, 1]),
        "unknown_prediction": ([1, 0], [2, 0], [0, 1]),
        "unknown_sensitive": ([1, 0], [1, 0], [0, 2]),
        "nan_label": ([float("nan"), 0], [1, 0], [0, 1]),
    }
    return {name: rejected(lambda args=args: ev.group_metrics(*args)) for name, args in cases.items()}


ROOT_M = np.array([-0.8, 0.2, -0.1, 0.9, -0.7, 0.3, -0.2, 0.8], dtype=np.float32)
ROOT_Y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
ROOT_S = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def prediction_boundaries(core):
    config = core.ExperimentConfig(batch_size=4, device="cpu")
    args = (core, "FedAvg", ROOT_M, ROOT_Y, ROOT_S, config)
    fits = ev.fit_views(*args, ["native", "raw", "shared_calibration"], ev.SHARED_CALIBRATION)
    # A zero group threshold fixture deliberately exercises the genuine >= rule.
    fits["shared_calibration"] = ev._seal_fit({
        "view": "shared_calibration", "method": "FedAvg", "rule": "group_margin_greater_equal",
        "thresholds": {0: 0.0, 1: 0.0}, "fit_data": "clean_train_root_only",
    })
    margins = np.array([-1, 0, 1, -1, 0, 1], dtype=float)
    sensitive = np.array([0, 0, 0, 1, 1, 1])
    predictions = ev.predict_views(margins, sensitive, fits)
    assert predictions["raw"].tolist() == [0, 0, 1, 0, 0, 1]
    assert np.array_equal(predictions["native"], predictions["raw"])
    assert predictions["shared_calibration"].tolist() == [0, 1, 1, 0, 1, 1]
    invalid = {}
    invalid["empty_fits"] = rejected(lambda: ev.predict_views(margins, sensitive, {}))
    invalid["empty_prediction_score"] = rejected(lambda: ev.evaluate_frozen_predictions({}, [0], [0]))
    invalid["nan_margins"] = rejected(lambda: ev.predict_views([np.nan], [0], fits))
    invalid["infinite_margins"] = rejected(lambda: ev.predict_views([np.inf], [0], fits))
    invalid["short_sensitive"] = rejected(lambda: ev.predict_views(margins, [0], fits))
    bad = copy.deepcopy(fits)
    bad["shared_calibration"]["thresholds"][0] = 0.5
    invalid["tampered_finite_threshold"] = rejected(lambda: ev.predict_views(margins, sensitive, bad))
    for name, thresholds in [("nan_threshold", {0: np.nan, 1: 0.0}), ("inf_threshold", {0: np.inf, 1: 0.0}),
                             ("missing_threshold", {0: 0.0}), ("string_threshold", {0: "0", 1: 0.0}),
                             ("bool_threshold", {0: True, 1: 0.0})]:
        bad = copy.deepcopy(fits)
        payload = {k: v for k, v in bad["shared_calibration"].items() if k != "fit_sha256"}
        payload["thresholds"] = thresholds
        def bad_predict(payload=payload, bad=bad):
            bad["shared_calibration"] = ev._seal_fit(payload)
            return ev.predict_views(margins, sensitive, bad)
        invalid[name] = rejected(bad_predict)
    bad = copy.deepcopy(fits)
    payload = {k: v for k, v in bad["raw"].items() if k != "fit_sha256"}
    payload["rule"] = "group_margin_greater_equal"
    bad["raw"] = ev._seal_fit(payload)
    invalid["signed_wrong_view_rule"] = rejected(lambda: ev.predict_views(margins, sensitive, bad))
    for views in [[], ["raw", "raw"], ["unknown"]]:
        invalid["views_" + repr(views)] = rejected(lambda views=views: ev.fit_views(*args, views, ev.SHARED_CALIBRATION))
    bad_shared = dict(ev.SHARED_CALIBRATION, ad2_calibration_max_acc_drop=0.01)
    invalid["shared_recipe_drift"] = rejected(lambda: ev.fit_views(*args, ["shared_calibration"], bad_shared))
    disabled = dataclasses.replace(config, ad2_calibration_enabled=False)
    invalid["ad2_native_disabled"] = rejected(lambda: ev.fit_views(core, "GuardFed-AD2+", ROOT_M, ROOT_Y, ROOT_S,
                                                                  disabled, ["native"], ev.SHARED_CALIBRATION))
    return {"margins": margins.tolist(), "sensitive": sensitive.tolist(),
            "predictions": {k: v.tolist() for k, v in predictions.items()}, "rejections": invalid}


def root_isolation(core):
    config = core.ExperimentConfig(batch_size=4, device="cpu")
    original_margin_function = core.model_margins
    with mock.patch.object(core, "model_margins", side_effect=AssertionError("Original model_margins must not run")):
        fits1 = ev.fit_views(core, "GuardFed-AD2+", ROOT_M, ROOT_Y, ROOT_S, config,
                             ["raw", "native", "shared_calibration"], ev.SHARED_CALIBRATION)
        assert core.model_margins is not original_margin_function
        margins = np.array([-0.3, 0, 0.5, -0.1], dtype=np.float32)
        sensitive = [0, 0, 1, 1]
        pred = ev.predict_views(margins, sensitive, fits1)
        pred_hashes = {k: ev.vector_identity(v) for k, v in pred.items()}
        score_a = ev.evaluate_frozen_predictions(pred, [0, 0, 1, 1], sensitive)
        score_b = ev.evaluate_frozen_predictions(pred, [1, 1, 0, 0], sensitive)
        fits2 = ev.fit_views(core, "GuardFed-AD2+", ROOT_M, ROOT_Y, ROOT_S, config,
                             ["raw", "native", "shared_calibration"], ev.SHARED_CALIBRATION)
        assert fits1 == fits2
        assert pred_hashes == {k: ev.vector_identity(v) for k, v in pred.items()}
        assert score_a["raw"]["accuracy"] != score_b["raw"]["accuracy"]
    assert core.model_margins is original_margin_function
    bad = {}
    for name, margins, y, s in [
        ("empty_root", [], [], []), ("nan_root", [0, np.nan], [0, 1], [0, 1]),
        ("missing_root_cell", [0, 1, 2, 3], [0, 0, 1, 1], [0, 0, 1, 1]),
        ("short_root_labels", ROOT_M, [0], ROOT_S),
    ]:
        bad[name] = rejected(lambda margins=margins, y=y, s=s: ev.thresholds_from_root(core, margins, y, s, config))
    for field, value in [("ad2_calibration_temperature", np.nan), ("ad2_calibration_quantiles", 1.5),
                         ("ad2_calibration_base_weight", -1.0), ("ad2_calibration_objective", "unknown")]:
        cfg = dataclasses.replace(config, **{field: value})
        bad[field] = rejected(lambda cfg=cfg: ev.thresholds_from_root(core, ROOT_M, ROOT_Y, ROOT_S, cfg))
    with mock.patch.object(core, "__file__", __file__):
        bad["frozen_core_bytes_drift"] = rejected(lambda: ev.thresholds_from_root(core, ROOT_M, ROOT_Y, ROOT_S, config))
    return {"fits_identical_after_scoring_opposite_target_labels": True,
            "predictions_unchanged_after_label_access": pred_hashes,
            "original_core_margin_global_restored": True, "rejections": bad}


def current_nine_native_interfaces(core):
    cfg = core.ExperimentConfig(batch_size=4, device="cpu")
    margins, sensitive = np.array([-1.0, 0.0, 1.0, 0.0]), [0, 0, 1, 1]
    result = {}
    for method in sorted(ev.METHODS):
        fits = ev.fit_views(core, method, ROOT_M, ROOT_Y, ROOT_S, cfg, ["native", "raw"], ev.SHARED_CALIBRATION)
        predictions = ev.predict_views(margins, sensitive, fits)
        assert predictions["raw"].tolist() == [0, 0, 1, 0]
        if method != "GuardFed-AD2+":
            assert np.array_equal(predictions["native"], predictions["raw"])
            assert fits["native"]["rule"] == "argmax_margin_strictly_positive"
        else:
            assert fits["native"]["rule"] == "group_margin_greater_equal"
            assert fits["native"]["fit_data"] == "clean_train_root_only"
        result[method] = {"native_rule": fits["native"]["rule"], "predictions": {k: v.tolist() for k, v in predictions.items()}}
    return result


def synthetic_cnn(core):
    torch.set_num_threads(1)
    torch.manual_seed(20261009)
    bundle = {"dataset": "celeba", "server_X": torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.uint8),
              "server_y": torch.tensor(ROOT_Y), "server_sensitive": ROOT_S.copy()}
    target = torch.randint(0, 256, (6, 3, 64, 64), dtype=torch.uint8)
    sensitive = np.array([0, 0, 0, 1, 1, 1])
    cfg = core.ExperimentConfig(batch_size=3, device="cpu", seed=20261009)
    model = core.make_model(bundle, cfg, torch.device("cpu"))
    before = ev.weights_identity(model)
    with mock.patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("Optimizer prohibited")):
        root, margins, fits, predictions = ev.extract_and_predict(core, model, bundle, target, sensitive,
                "GuardFed-AD2+", cfg, ["native", "raw", "shared_calibration"], ev.SHARED_CALIBRATION)
    assert before == ev.weights_identity(model)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert root.shape == (8,) and margins.shape == (6,)
    with torch.no_grad():
        direct = torch.argmax(model(target), dim=1).numpy()
    assert np.array_equal(predictions["raw"], direct)
    score = ev.evaluate_frozen_predictions(predictions, [0, 1, 0, 1, 0, 1], sensitive)
    broken = copy.deepcopy(model)
    real_margins = core.model_margins
    def mutating_inference(model, images, batch_size):
        with torch.no_grad():
            next(model.parameters()).add_(0.001)
        return real_margins(model, images, batch_size)
    with mock.patch.object(core, "model_margins", side_effect=mutating_inference):
        mutation = rejected(lambda: ev.extract_and_predict(core, broken, bundle, target, sensitive,
                    "FedAvg", cfg, ["raw"], ev.SHARED_CALIBRATION))
    return {"evidence_stage": "measured_synthetic_CPU_acceptance_only", "model_class": type(model).__name__,
            "root_shape": list(bundle["server_X"].shape), "target_shape": list(target.shape),
            "input_dtype": str(target.dtype), "optimizer_constructed": False,
            "weights_unchanged": True, "gradients_absent": True, "raw_matches_direct_argmax": True,
            "root_pixels": ev.vector_identity(bundle["server_X"].numpy()),
            "target_pixels": ev.vector_identity(target.numpy()), "weights_before_and_after": before,
            "root_margins": root.tolist(), "target_margins": margins.tolist(), "fits": fits,
            "scores": score, "changed_weights_rejected": mutation,
            "real_celeba_images_evaluated": 0, "gpu_equivalence_checked": False}


def freeze_gate(core):
    proto = json.loads((PREPARED / "protocol.json").read_text(encoding="utf-8-sig"))
    jobs = json.loads((PREPARED / "evaluation_jobs_draft.json").read_text(encoding="utf-8-sig"))
    inventory = json.loads((PREPARED / "model_inventory.json").read_text(encoding="utf-8-sig"))
    INPUTS["prepared_inputs"] = {str(PREPARED / name): ev.digest(PREPARED / name)
                                for name in ["protocol.json", "evaluation_jobs_draft.json", "model_inventory.json"]}
    original_bytes = {name: (PREPARED / name).read_bytes() for name in ["protocol.json", "evaluation_jobs_draft.json"]}
    draft = rejected(lambda: ev.require_dispatch(proto, jobs["jobs"][0], {}, {}))
    # This in-memory toy receipt is an acceptance fixture, never a saved dispatch receipt.
    p = copy.deepcopy(proto)
    p.update(status="FROZEN", selected_views=["native", "raw", "shared_calibration"],
             target_image_ids_sha256="a" * 64, target_order_sha256="b" * 64,
             target_split="fixture_only_not_a_dataset", target_n=6,
             model_inventory_sha256=ev.digest(PREPARED / "model_inventory.json"))
    p["decisions"] = {key: "SYNTHETIC_GATE_FIXTURE_ONLY" for key in ev.DECISIONS}
    j = copy.deepcopy(jobs["jobs"][0])
    j.update(status="FROZEN", dispatchable=True, selected_views=p["selected_views"])
    for key in ["target_image_ids_sha256", "target_order_sha256", "target_split", "target_n"]:
        j[key] = p[key]
    sources = {"evaluator.py": ev.digest(ev.__file__), "scripts/reproduce_paper_tables.py": ev.CORE_SHA256,
               "src/celeba_data.py": ev.digest(FROZEN / "src/celeba_data.py")}
    observed = {key: j[key] for key in ["checkpoint_sha256", "original_result_sha256", "original_job_sha256",
                 "config_canonical_sha256", "root_image_ids_sha256", "source_hashes", "model_id"]}
    observed.update({key: p[key] for key in ["model_inventory_sha256", "target_image_ids_sha256", "target_order_sha256",
                                           "target_n", "target_split"]})
    observed = copy.deepcopy(observed)
    records_by_id = {record["id"]: record for record in inventory["records"]}
    observed["model_inventory_record"] = copy.deepcopy(records_by_id[j["model_id"]])
    observed["adapter_source_hashes"] = copy.deepcopy(observed["model_inventory_record"]["adapter_source_hashes"])
    receipt = {"status": "DISPATCH_FROZEN", "protocol_canonical_sha256": ev.canonical_sha(p),
               "job_canonical_sha256": ev.canonical_sha(j), "evaluation_source_sha256": sources,
               "model_inventory_sha256": p["model_inventory_sha256"],
               "native_valid_replay": {"accepted": True, "model_count": 900,
                     "max_abs_metric_difference": 0.0, "acceptance_sha256": "c" * 64,
                     "model_inventory_sha256": p["model_inventory_sha256"]}}
    ev.require_dispatch(p, j, receipt, sources, observed)
    failures = {"real_prepared_protocol": draft}
    def gate_case(name, edit, refresh_receipt=False):
        values = [copy.deepcopy(value) for value in [p, j, receipt, sources, observed]]
        edit(*values)
        if refresh_receipt:
            values[2]["protocol_canonical_sha256"] = ev.canonical_sha(values[0])
            values[2]["job_canonical_sha256"] = ev.canonical_sha(values[1])
        failures[name] = rejected(lambda: ev.require_dispatch(*values))
    gate_case("missing_observed", lambda p, j, r, s, o: o.clear())
    gate_case("protocol_drift", lambda p, j, r, s, o: p.update(scope="drift"))
    gate_case("job_drift", lambda p, j, r, s, o: j.update(checkpoint_sha256="d" * 64))
    gate_case("source_drift", lambda p, j, r, s, o: s.update({"src/celeba_data.py": "d" * 64}))
    gate_case("empty_sources", lambda p, j, r, s, o: (s.clear(), r.update(evaluation_source_sha256={})))
    gate_case("unresolved_decision", lambda p, j, r, s, o: p["decisions"].update({next(iter(ev.DECISIONS)): None}), True)
    gate_case("missing_decision", lambda p, j, r, s, o: p["decisions"].pop(next(iter(ev.DECISIONS))), True)
    gate_case("empty_views", lambda p, j, r, s, o: (p.update(selected_views=[]), j.update(selected_views=[])), True)
    gate_case("duplicate_views", lambda p, j, r, s, o: (p.update(selected_views=["raw", "raw"]), j.update(selected_views=["raw", "raw"])), True)
    gate_case("unrecognised_view", lambda p, j, r, s, o: (p.update(selected_views=["unknown"]), j.update(selected_views=["unknown"])), True)
    gate_case("missing_target_order", lambda p, j, r, s, o: p.pop("target_order_sha256"), True)
    gate_case("not_terminal_round", lambda p, j, r, s, o: j.update(terminal_round=69), True)
    gate_case("protocol_round_drift", lambda p, j, r, s, o: p.update(round=69), True)
    gate_case("protocol_core_drift", lambda p, j, r, s, o: p.update(calibration_core_sha256="d" * 64), True)
    gate_case("protocol_calibration_drift", lambda p, j, r, s, o: p["available_views"]["shared_calibration"]["calibration"].update(ad2_calibration_max_acc_drop=0.01), True)
    gate_case("wrong_original_split", lambda p, j, r, s, o: j.update(original_split="test"), True)
    gate_case("wrong_method", lambda p, j, r, s, o: j.update(method="GuardFed"), True)
    gate_case("wrong_source_method", lambda p, j, r, s, o: j.update(source_method="different"), True)
    gate_case("wrong_seed", lambda p, j, r, s, o: j.update(seed=91011), True)
    gate_case("wrong_distribution_alpha", lambda p, j, r, s, o: j.update(actual_alpha=5), True)
    gate_case("wrong_attack", lambda p, j, r, s, o: j.update(attack="unknown"), True)
    gate_case("model_id_drift", lambda p, j, r, s, o: j.update(model_id="unknown"), True)
    gate_case("partial_replay", lambda p, j, r, s, o: r["native_valid_replay"].update(model_count=899))
    gate_case("replay_failure", lambda p, j, r, s, o: r["native_valid_replay"].update(accepted=False))
    gate_case("replay_nan_difference", lambda p, j, r, s, o: r["native_valid_replay"].update(max_abs_metric_difference=np.nan))
    gate_case("replay_excess_difference", lambda p, j, r, s, o: r["native_valid_replay"].update(max_abs_metric_difference=1e-11))
    gate_case("replay_wrong_inventory", lambda p, j, r, s, o: r["native_valid_replay"].update(model_inventory_sha256="d" * 64))
    for field in observed:
        if field == "source_hashes":
            gate_case("observed_data_source_drift", lambda p, j, r, s, o: o["source_hashes"].update({"data/celeba/derived/rgb64_v1/images.npy": "d" * 64}))
        else:
            gate_case("observed_" + field + "_drift", lambda p, j, r, s, o, field=field: o.pop(field))
    gate_case("partial_job_sources", lambda p, j, r, s, o: (j.update(source_hashes={}), o.update(source_hashes={})), True)
    gate_case("missing_job_target", lambda p, j, r, s, o: j.pop("target_image_ids_sha256"), True)
    gate_case("partial_backup", lambda p, j, r, s, o: j["model_backup_ref"].pop("archive_sha256"), True)
    # Exercise all declared metadata shapes (including FedAA/LASA source aliases).
    inventory_ids = {record["id"] for record in inventory["records"]}
    assert len(inventory_ids) == 900
    for real_job in jobs["jobs"]:
        fixture_job = copy.deepcopy(real_job)
        assert fixture_job["model_id"] in inventory_ids
        fixture_job.update(status="FROZEN", dispatchable=True, selected_views=p["selected_views"])
        for key in ["target_image_ids_sha256", "target_order_sha256", "target_split", "target_n"]:
            fixture_job[key] = p[key]
        fixture_observed = {key: fixture_job[key] for key in ["checkpoint_sha256", "original_result_sha256", "original_job_sha256",
                           "config_canonical_sha256", "root_image_ids_sha256", "source_hashes", "model_id"]}
        fixture_observed.update({key: p[key] for key in ["model_inventory_sha256", "target_image_ids_sha256", "target_order_sha256",
                                                        "target_n", "target_split"]})
        fixture_observed["model_inventory_record"] = records_by_id[fixture_job["model_id"]]
        fixture_observed["adapter_source_hashes"] = fixture_observed["model_inventory_record"]["adapter_source_hashes"]
        fixture_receipt = {**receipt, "job_canonical_sha256": ev.canonical_sha(fixture_job)}
        ev.require_dispatch(p, fixture_job, fixture_receipt, sources, fixture_observed)
    assert len(inventory["records"]) == len(jobs["jobs"]) == 900
    assert all(original_bytes[name] == (PREPARED / name).read_bytes() for name in original_bytes)
    assert proto["status"] == jobs["status"] == "PREPARED_NOT_FROZEN"
    assert all(value is None for value in proto["decisions"].values())
    return {"fixture_positive_gate": "PASS_IN_MEMORY_ONLY_NO_DISPATCH", "rejection_count": len(failures),
            "rejections": failures, "real_prepared_protocol_unchanged": True,
            "real_protocol_status": proto["status"], "real_decisions_all_null": True,
            "real_inventory_records": 900, "synthetic_positive_gates_from_declared_job_metadata": 900,
            "real_dispatch_calls": 0}


def flatten_comparison(expected, actual, prefix=""):
    if isinstance(expected, dict):
        assert set(expected) <= set(actual), (prefix, set(expected) - set(actual))
        return [item for key in sorted(expected) for item in flatten_comparison(expected[key], actual[key], prefix + "." + key)]
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        difference = float(actual) - float(expected)
        assert abs(difference) <= 1e-12, (prefix, expected, actual, difference)
        return [{"field": prefix.lstrip("."), "expected": expected, "actual": actual, "difference": difference}]
    assert expected == actual, (prefix, expected, actual)
    return [{"field": prefix.lstrip("."), "expected": expected, "actual": actual, "equal": True}]


def accepted_cache_regression(core):
    assert ev.digest(ARCHIVE) == ARCHIVE_SHA
    inventory = json.loads((HISTORY / "backup_inventory.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((HISTORY / "manifest.json").read_text(encoding="utf-8-sig"))
    assert manifest["calibration"] == ev.SHARED_CALIBRATION
    by_id = {job["id"]: job for job in manifest["jobs"]}
    extraction = []
    inputs_dir = HERE / "accepted_valid_inputs"
    inputs_dir.mkdir(exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for run_id in SELECTED:
            directory = inputs_dir / run_id
            directory.mkdir(exist_ok=True)
            for filename in ["margins.npz", "evaluation.json"]:
                member = f"{PREFIX}/runs/{run_id}/{filename}"
                data = archive.extractfile(member).read()
                actual_sha = hashlib.sha256(data).hexdigest()
                assert actual_sha == inventory[member]["sha256"]
                assert len(data) == inventory[member]["bytes"]
                (directory / filename).write_bytes(data)
                extraction.append({"archive_member": member, "archive_sha256": ARCHIVE_SHA,
                                   "sha256": actual_sha, "bytes": len(data),
                                   "local_path": str(directory / filename)})
    INPUTS["historical_regression"] = {
        "archive": str(ARCHIVE), "archive_sha256": ARCHIVE_SHA,
        "backup_inventory_sha256": ev.digest(HISTORY / "backup_inventory.json"),
        "manifest_sha256": ev.digest(HISTORY / "manifest.json"), "extracted_members": extraction,
        "selection_rule": "Predeclared eight IDs in acceptance.py; seven StageA methods, two distributions, Benign/S-DFA; no outcome selection",
    }
    comparisons = []
    for run_id in SELECTED:
        job = by_id[run_id]
        directory = inputs_dir / run_id
        accepted = json.loads((directory / "evaluation.json").read_text(encoding="utf-8-sig"))
        assert accepted["id"] == run_id and accepted["evaluation_split"] == "valid" and accepted["round"] == 70
        assert accepted["checkpoint_sha256"] == job["checkpoint_sha256"]
        assert accepted["manifest_sha256"] == ev.digest(HISTORY / "manifest.json")
        assert accepted["cache_sha256"] == ev.digest(directory / "margins.npz")
        cfg = core.ExperimentConfig(**job["config"])
        with np.load(directory / "margins.npz", allow_pickle=False) as cache:
            assert set(cache.files) == {"root_margins", "valid_margins", "root_y", "root_sensitive", "valid_y", "valid_sensitive"}
            assert len(cache["valid_margins"]) == 19867
            fits = ev.fit_views(core, job["method"], cache["root_margins"], cache["root_y"], cache["root_sensitive"],
                                cfg, ["native", "raw", "shared_calibration"], ev.SHARED_CALIBRATION)
            predictions = ev.predict_views(cache["valid_margins"], cache["valid_sensitive"], fits)
            actual = ev.evaluate_frozen_predictions(predictions, cache["valid_y"], cache["valid_sensitive"])
            fields = []
            for view in ["native", "raw", "shared_calibration"]:
                fields.extend(flatten_comparison({key: accepted[view][key] for key in ["accuracy", "aeod", "aspd", "positive_rate", "majority_accuracy", "prediction_count"]},
                                                 actual[view], view))
            shared_fit = fits["shared_calibration"]
            fields.extend(flatten_comparison(accepted["thresholds"], {str(k): v for k, v in shared_fit["thresholds"].items()}, "thresholds"))
            fields.extend(flatten_comparison(accepted["calibration_info"], shared_fit["fit_diagnostics"], "calibration_info"))
            comparisons.append({"id": run_id, "method": job["method"], "distribution": job["distribution"], "attack": job["attack"],
                 "seed": cfg.seed, "config": job["config"], "config_canonical_sha256": ev.canonical_sha(job["config"]),
                 "checkpoint_sha256": job["checkpoint_sha256"], "original_result_sha256": accepted["original_result_sha256"],
                 "root_contract": accepted["root_contract"], "input_arrays": {k: ev.vector_identity(cache[k]) for k in cache.files},
                 "root_n": len(cache["root_margins"]), "valid_n": len(cache["valid_margins"]),
                 "zero_root_margins": int(np.sum(cache["root_margins"] == 0)), "zero_valid_margins": int(np.sum(cache["valid_margins"] == 0)),
                 "accepted_metrics": {view: accepted[view] for view in ["native", "raw", "shared_calibration"]},
                 "actual_metrics": actual, "fits": fits,
                 "prediction_identities": {k: ev.vector_identity(v) for k, v in predictions.items()},
                 "numeric_comparisons": fields, "max_abs_difference": max(abs(row.get("difference", 0)) for row in fields)})
        print("REGRESSION " + run_id, flush=True)
    assert len({row["method"] for row in comparisons}) == 7
    assert {row["distribution"] for row in comparisons} == {"IID", "non-IID"}
    assert {row["attack"] for row in comparisons} == {"Benign", "S-DFA"}
    report = {"status": "PASS", "evidence_stage": "local_software_regression_against_previously_accepted_valid_caches",
              "not_new_scientific_results": True, "selected_count": 8, "historical_cohort_size": 700,
              "real_image_inference_count": 0, "test_labels_read": False, "tolerance": 1e-12,
              "max_abs_difference": max(row["max_abs_difference"] for row in comparisons), "comparisons": comparisons}
    save("regression.json", report)
    return {key: value for key, value in report.items() if key != "comparisons"}


def files_manifest():
    entries = []
    for path in sorted(HERE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name != "FILES_SHA256":
            entries.append(ev.digest(path) + "  " + path.relative_to(HERE).as_posix())
    (HERE / "FILES_SHA256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def main():
    success = False
    try:
        check("frozen_core_root_resolution", frozen_root_resolution)
        core = check("frozen_core_source_identity", load_core)
        CHECKS[-1]["detail"] = {"sha256": ev.CORE_SHA256, "dataset_loader_called": False}
        check("hand_computed_confusion_matrices", hand_metrics)
        check("undefined_denominators_and_constant_predictions", zero_and_constant)
        check("malformed_binary_vectors_rejected", binary_rejections)
        check("ties_views_and_bad_fits", lambda: prediction_boundaries(core))
        check("root_only_fit_and_target_label_isolation", lambda: root_isolation(core))
        check("current_nine_method_native_interfaces", lambda: current_nine_native_interfaces(core))
        check("real_RGB64_CNN_synthetic_CPU_inference", lambda: synthetic_cnn(core))
        check("freeze_gate_and_identity_drift_rejections", lambda: freeze_gate(core))
        check("eight_accepted_validation_cache_regressions", lambda: accepted_cache_regression(core))
        success = True
    finally:
        save("checks.json", CHECKS)
        INPUTS["local_sources"] = {path.name: ev.digest(path) for path in [HERE / "evaluator.py", HERE / "acceptance.py"]}
        save("input_identity.json", INPUTS)
        report = {"status": "PASS" if success else "FAIL", "protocol_status": "PREPARED_NOT_FROZEN",
                  "checks_passed": sum(row["status"] == "PASS" for row in CHECKS), "checks_total": len(CHECKS),
                  "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                  "runtime_device": "CPU", "real_celeba_image_inference_count": 0, "test_labels_read": False,
                  "new_training_count": 0, "real_dispatch_count": 0, "all900_native_replay_performed": False,
                  "gpu_equivalence_claimed": False, "protocol_decisions_frozen": False,
                  "limits": ["Synthetic CPU images establish software behavior, not GPU equivalence or checkpoint replay.",
                             "Eight historical caches establish numerical regression only; native valid image replay of all900 remains required.",
                             "Trusted dispatch receipt and independently measured observed identities are caller responsibilities; no real dispatcher is provided."]}
        save("acceptance.json", report)
        (HERE / "checks.log").write_text("\n".join(row["status"] + " " + row["name"] for row in CHECKS) + "\n", encoding="utf-8")
        files_manifest()
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
