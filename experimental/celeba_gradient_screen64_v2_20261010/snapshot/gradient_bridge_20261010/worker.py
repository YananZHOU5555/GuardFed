"""Draft same-point-gradient CelebA worker. PREPARED never authorizes execution.

Only data/runtime attacks, CNN and reporting are reused from the pinned core.
Client gradients are empirical gradients, never Adam state differences. Clean
root access is an explicitly chosen adversary reference, not a defense input.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
METHODS = {"Fed-NGA-gradient", "Huber-BRFL-gradient"}
LOCAL_SOURCES = {"worker.py", "accept_result.py", "prepare.py", "protocol.json"}
COMPONENTS = {
    "remaining_20261009/group_a/adapters.py": "55ca0f22836e2e6b86ccb4798bb13948a211db4cc09e4c076a45e1411c3125c0",
    "fednga/fednga.py": "9784ece5bd88aa4813693ed69aaf7f7505b5d16d2a6eb18c23700a1bab4de6ee",
}
OBJECTIVES = {"original_unweighted_ce", "shared_reweighted_ce"}
ROOT_REFERENCES = {"same_point_unweighted_gradient", "frozen_root_localadam_delta"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def vector_digest(vector):
    return hashlib.sha256(vector.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, obj):
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_components():
    # Keep the already accepted snapshot and Fed-NGA component read-only.
    for name, expected in COMPONENTS.items():
        if digest(HERE.parent / name) != expected:
            raise ValueError("Pinned gradient component changed: " + name)
    return load_module(HERE.parent / "remaining_20261009/group_a/adapters.py", "guardfed_gradient_components")


def load_core(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    return load_module(repo / "scripts/reproduce_paper_tables.py", "guardfed_same_point_core")


def parameters_vector(model):
    return torch.cat([value.detach().reshape(-1) for value in model.parameters() if value.requires_grad])


def require_parameter_only_model(model):
    if list(model.state_dict()) != [name for name, value in model.named_parameters() if value.requires_grad]:
        raise ValueError("Gradient bridge needs a frozen buffer policy for this model")


def validate_adapter(method, adapter):
    common = {"server_eta", "objective", "root_reference", "projection"}
    extra = set() if method == "Fed-NGA-gradient" else {"t0", "m", "max_iter", "tolerance"}
    if method not in METHODS or set(adapter) != common | extra:
        raise ValueError("Unknown gradient method or recipe fields")
    if adapter["objective"] not in OBJECTIVES or adapter["root_reference"] not in ROOT_REFERENCES:
        raise ValueError("Unspecified gradient objective or adversary reference")
    if adapter["projection"] != "identity_Rp":
        raise ValueError("Only explicitly declared unconstrained parameter space is implemented")
    if not math.isfinite(adapter["server_eta"]) or adapter["server_eta"] <= 0:
        raise ValueError("Positive finite independent server step required")
    if extra:
        if any(not math.isfinite(adapter[name]) or adapter[name] < 0 for name in ("t0", "m")) or adapter["t0"] + adapter["m"] <= 0:
            raise ValueError("Invalid fixed Huber threshold family")
        if not isinstance(adapter["max_iter"], int) or isinstance(adapter["max_iter"], bool) or adapter["max_iter"] <= 0:
            raise ValueError("Invalid Huber iteration bound")
        if not math.isfinite(adapter["tolerance"]) or adapter["tolerance"] <= 0:
            raise ValueError("Invalid Huber stationarity tolerance")


@torch.no_grad()
def attack_gradient(gradient, client, root_descent, config, audit):
    """Sign-conjugate the frozen upload attack: g_attack = -A(-g, r).

    r is a declared clean-root descent reference. No eta is applied until the
    server aggregates. The old state_dict FOE has no unique gradient meaning
    and is rejected. No fake local model is constructed by this worker.
    """
    if gradient.ndim != 1 or root_descent.shape != gradient.shape or not torch.isfinite(gradient).all() or not torch.isfinite(root_descent).all():
        raise ValueError("Finite same-layout gradient and root reference required")
    audit = copy.deepcopy(audit)
    if "foe" not in client["attack_types"]:
        return gradient.clone(), audit
    mode = client["foe_mode"]
    descent = -gradient
    if mode == "state":
        raise ValueError("state_dict FOE is ambiguous for a gradient upload; freeze a separate threat protocol")
    if mode == "delta":
        post = -.5 * descent
    elif mode == "zero":
        post = torch.zeros_like(descent)
    elif mode == "fedsa":
        norm = float(torch.linalg.vector_norm(descent))
        root_norm = float(torch.linalg.vector_norm(root_descent))
        if norm <= 0 or root_norm <= 0:
            post = -config.fedsa_gain * descent
        else:
            clean_direction = root_descent / torch.clamp(torch.linalg.vector_norm(root_descent), min=1e-12)
            post = descent - float(config.fedsa_gain) * norm * clean_direction
            cap = max(norm, float(config.fedsa_norm_ratio) * norm)
            if float(torch.linalg.vector_norm(post)) > cap:
                post = post * (cap / torch.clamp(torch.linalg.vector_norm(post), min=1e-12))
    else:
        raise ValueError("Unknown gradient upload attack: " + str(mode))
    uploaded = -post
    if not torch.isfinite(uploaded).all():
        raise FloatingPointError("Gradient upload attack overflow")
    audit.update(foe_impl="gradient sign-conjugacy g_attack=-A(-g,root_descent); frozen " + mode + " algebra",
                 upload_semantics="positive empirical gradient before server minus-eta step",
                 foe_pre_update_norm=float(torch.linalg.vector_norm(gradient)),
                 foe_post_update_norm=float(torch.linalg.vector_norm(uploaded)),
                 foe_norm_ratio=float(torch.linalg.vector_norm(uploaded)) / (float(torch.linalg.vector_norm(gradient)) + 1e-12),
                 foe_pre_post_cosine=float(F.cosine_similarity(gradient[None], uploaded[None]).item()))
    return uploaded, audit


def aggregate_gradients(method, uploaded, counts, adapter, components):
    validate_adapter(method, adapter)
    if len(counts) != len(uploaded) or any(n <= 0 for n in counts):
        raise ValueError("Every client needs positive exact sample counts")
    if method == "Fed-NGA-gradient":
        step = components.fednga_step(uploaded, counts, adapter["server_eta"])
        info = dict(mechanism="Eq9 sample-weighted mean of unit uploaded gradients",
                    zero_gradient_extension="zero direction retains raw sample count in denominator")
    else:
        thresholds = components.huber_thresholds(counts, adapter["t0"], adapter["m"])
        center, info = components.huber_center(uploaded, counts, thresholds,
            max_iter=adapter["max_iter"], tolerance=adapter["tolerance"])
        if not info["converged"]:
            error = RuntimeError("Fixed-Ti Huber solver did not converge; preserve failure, no model step")
            error.solver_diagnostics = info
            raise error
        step = -adapter["server_eta"] * center
    info.update(actual_parameters=copy.deepcopy(adapter), aggregation_weighting="raw_client_sample_count",
                counts=list(counts), client_order=list(range(len(counts))),
                uploaded_norms=torch.linalg.vector_norm(uploaded.double(), dim=1).cpu().tolist(),
                uploaded_sha256=vector_digest(uploaded), step_sha256=vector_digest(step),
                server_step_norm=float(torch.linalg.vector_norm(step.double())),
                defense_root_fairness_used=False, defense_root_reference_used=False)
    return step, info


def root_reference(core, components, model, bundle, config, adapter, needed):
    if not needed:
        return torch.zeros_like(parameters_vector(model)), dict(used=False, reason="No upload foe in this condition")
    mode = adapter["root_reference"]
    if mode == "same_point_unweighted_gradient":
        y = bundle["server_y"]
        root = dict(X=bundle["server_X"], y=y, weights=torch.ones(len(y)), n=len(y))
        descent = -components.empirical_gradient(model, root, config.batch_size, use_reweighting=False)
        identity = "negative clean full-root unweighted empirical gradient at current global point"
    elif mode == "frozen_root_localadam_delta":
        delta = core.train_server_update(model, bundle, config)
        descent = core.vectorize(delta)
        identity = "unchanged frozen clean-root localAdam delta; adversary reference only"
    else:
        raise ValueError("Unknown root threat reference")
    return descent, dict(used=True, mode=mode, semantics=identity, norm=float(torch.linalg.vector_norm(descent)),
                         sha256=vector_digest(descent), root_rows=len(bundle["server_y"]),
                         evaluation_labels_used=False, defense_input=False)


def train_pipeline(core, components, method, distribution, attack, config, adapter,
                   device, progress_callback=None, checkpoint_path=None, diagnostics_callback=None):
    """Independent gradient loop; no client optimizer or model-delta roundtrip."""
    validate_adapter(method, adapter)
    if (config.optimizer, config.local_epochs, config.use_reweighting) != (
            "adam", 1, adapter["objective"] == "shared_reweighted_ce"):
        raise ValueError("Declare objective and root-only Adam/unused local-epoch metadata consistently")
    if config.ad2_calibration_enabled or config.celeba_evaluation_split != "valid":
        raise ValueError("Gradient screens require native valid-only reporting")
    if config.root_label_noise or config.root_sensitive_noise or config.synthetic_ratio:
        raise ValueError("Only the declared clean real train-root threat reference is implemented")
    alpha = config.client_alpha if config.client_alpha is not None else core.DISTRIBUTIONS[distribution]
    started = time.time()
    bundle = core.load_bundle("celeba", alpha, config, device)
    if bundle["feature_includes_label"] or bundle["feature_includes_sensitive"]:
        raise ValueError("Image gradient method requires metadata-only sensitive attribute and no label leakage")
    core.set_seed(config.seed, deterministic_image=True)
    model = core.make_model(bundle, config, device)
    require_parameter_only_model(model)
    malicious_ids = list(range(config.num_malicious))
    clients, audits = [], []
    for cid in range(config.num_clients):
        client, audit = core.client_runtime_data(bundle["clients"][cid], cid, attack, malicious_ids,
            bundle["rw_weights"], device, config.fflip_mode, config.foe_mode, config.sdfa_foe_mode, config.spdfa_foe_mode)
        if client["n"] <= 0:
            raise ValueError("Empty client requires a separately frozen participation rule")
        clients.append(client)
        audits.append(audit)
    counts = [client["n"] for client in clients]
    trajectory, summaries, warnings = [], [], []
    for round_index in range(config.rounds):
        point = parameters_vector(model).clone()
        root_descent, root_info = root_reference(core, components, model, bundle, config, adapter,
                                               any("foe" in client["attack_types"] for client in clients))
        uploads, raw_norms = [], []
        for client in clients:
            gradient = components.empirical_gradient(model, client, config.batch_size,
                use_reweighting=adapter["objective"] == "shared_reweighted_ce")
            raw_norms.append(float(torch.linalg.vector_norm(gradient.double())))
            upload, audits[client["cid"]] = attack_gradient(gradient, client, root_descent, config, audits[client["cid"]])
            uploads.append(upload)
        if not torch.equal(point, parameters_vector(model)):
            raise ValueError("Client/root gradient evaluation changed the shared global model")
        if round_index == 0:
            warnings.extend(core.validate_attack_audit(attack, malicious_ids, audits))
            if warnings:
                raise RuntimeError("Attack self-check failed: " + "; ".join(warnings))
        step, info = aggregate_gradients(method, torch.stack(uploads), counts, adapter, components)
        info.update(global_point_sha256=vector_digest(point), raw_gradient_norms=raw_norms,
                    root_threat_reference=root_info,
                    client_gradient_semantics="positive full empirical gradient at identical unchanged global point",
                    signed_server_update="w_next=w-server_eta*aggregated_gradient; Fed-NGA normalizes each upload",
                    local_optimizer_steps=0, projection="identity_Rp",
                    objective_denominator="sum of all client sample weights" if config.use_reweighting else "raw client sample count")
        components.apply_parameter_step(model, step)
        # Neither method uses fairness inference at client candidate models.
        # Evaluation labels are only used after the step, by unchanged reporting.
        metrics = core.evaluate_for_reporting(method, model, bundle, config)
        item = dict(round=round_index + 1, metrics={key: metrics[key] for key in core.METRICS})
        trajectory.append(item)
        summaries.append(dict(round=round_index + 1, aggregate=info, client_ids=[client["cid"] for client in clients],
                              malicious_mask=[client["cid"] in malicious_ids for client in clients]))
        if diagnostics_callback:
            diagnostics_callback(summaries)
        if progress_callback:
            progress_callback(copy.deepcopy(item))
    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
    contract = {key: bundle.get(key) for key in ("label_col", "sensitive_col", "feature_includes_label",
        "feature_includes_sensitive", "num_features", "train_rows", "test_rows", "root_clean_rows",
        "root_synthetic_rows", "image_data_contract")}
    return dict(dataset="celeba", method=method, distribution=distribution, alpha=alpha, attack=attack,
        seed=config.seed, rounds=config.rounds, num_clients=config.num_clients, num_malicious=config.num_malicious,
        config=asdict(config), gradient_recipe=copy.deepcopy(adapter), metrics=trajectory[-1]["metrics"],
        evaluation_stats={key: metrics.get(key) for key in ("positive_rate", "majority_accuracy", "prediction_count")},
        trajectory_metrics=trajectory, round_summaries=summaries, attack_audit=audits, warnings=warnings,
        data_contract=contract, duration_sec=time.time() - started,
        method_impl_note="Genuine same-point empirical-gradient algorithm; objective, threat reference and identity projection explicitly adapted; not a localAdam-delta branch")


def validate_job(job, protocol, require_frozen=True):
    if require_frozen and (protocol.get("status") != "FROZEN" or any(
            row.get("status") != "APPROVED" for row in protocol["protocol_decisions"].values())):
        raise ValueError("Gradient protocol not frozen and fully approved; PREPARED is non-executable")
    candidate = next((row for row in protocol["candidates"] if row["id"] == job.get("tuning_candidate")), None)
    if candidate is None or candidate["method"] != job.get("method") or candidate["adapter"] != job.get("adapter"):
        raise ValueError("Unknown gradient candidate")
    validate_adapter(job["method"], job["adapter"])
    distribution, attack = job["distribution"], job["attack"]
    if distribution not in protocol["distributions"] or attack not in protocol["attacks"]:
        raise ValueError("Unknown screen condition")
    identity = candidate["id"] + "_" + distribution + "_" + attack + "_seed91001_screen"
    expected = dict(protocol["base_config"], seed=91001, rounds=70,
        client_alpha=protocol["distributions"][distribution],
        use_reweighting=candidate["adapter"]["objective"] == "shared_reweighted_ce",
        experiment_suite=protocol["version"], experiment_tag=identity)
    if (job.get("id"), job.get("dataset"), job.get("evidence_stage"), job.get("config")) != (
            identity, "celeba", "validation_screen", expected):
        raise ValueError("Gradient config/identity mismatch")
    if job.get("source_hashes") != protocol["source_hashes"] or job.get("component_hashes") != COMPONENTS:
        raise ValueError("Frozen core/data/component identity mismatch")
    if set(job.get("local_hashes", {})) != LOCAL_SOURCES:
        raise ValueError("Incomplete worker/protocol identities")
    return candidate


def verify_hashes(root, hashes):
    root = Path(root).resolve()
    for relative, expected in hashes.items():
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(root):
            raise ValueError("Hash path escapes declared root")
        if digest(path) != expected:
            raise ValueError("Source/data identity changed: " + relative)


def run(repo, job_path, output):
    repo, job_path, output = Path(repo).resolve(), Path(job_path).resolve(), Path(output).resolve()
    job = json.loads(job_path.read_text(encoding="utf8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    validate_job(job, protocol)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        write_json(output / "job.json", job)
        verify_hashes(repo, job["source_hashes"])
        verify_hashes(HERE, job["local_hashes"])
        verify_hashes(HERE.parent, job["component_hashes"])
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(int(os.environ.get("GUARDFED_CPU_THREADS", "1")))
        torch.set_num_interop_threads(1)
        core, components = load_core(repo), load_components()
        config = core.ExperimentConfig(**job["config"])
        device = core.choose_device(config.device)
        provenance = dict(job_sha256=digest(job_path), source_hashes=job["source_hashes"],
            local_hashes=job["local_hashes"], component_hashes=job["component_hashes"], started_unix=started,
            resume_supported=False, environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
        write_json(output / "provenance.json", provenance)
        def progress(item):
            write_json(output / "progress.json", dict(item, job_id=job["id"], pid=os.getpid(), updated_unix=time.time()))
        result = train_pipeline(core, components, job["method"], job["distribution"], job["attack"], config,
            job["adapter"], device, progress_callback=progress, checkpoint_path=output / "model.pt",
            diagnostics_callback=lambda rows: write_json(output / "diagnostics.json", rows))
        result.update(status="screen_complete", evidence_stage="validation_screen", revision_job=job,
                      tuning_candidate=job["tuning_candidate"], provenance=provenance)
        write_json(output / "result.json", result)
        write_json(output / "acceptance.json", dict(status="PASS", artifact_hashes={name: digest(output / name) for name in
            ("model.pt", "diagnostics.json", "result.json", "provenance.json")}))
        from accept_result import checked_result
        checked_result(job_path, output)
        print(json.dumps(dict(complete=job["id"], metrics=result["metrics"])), flush=True)
    except BaseException as error:
        write_json(output / "failure.json", dict(job_id=job["id"], error=repr(error), traceback=traceback.format_exc(),
            solver_diagnostics=getattr(error, "solver_diagnostics", None), failed_unix=time.time()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.job, args.out)
