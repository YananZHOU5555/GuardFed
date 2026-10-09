"""FLGMM author-code screen on the unchanged GuardFed image-training core.

Draft templates cannot run. A parent-frozen protocol and matching source hashes
are mandatory. Partial output is preserved, never implicitly resumed/overwritten.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
METHOD = "FLGMM-author-code"
LABEL = "FLGMM author-code full-local-model aggregation adapted to common local Adam"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        return None if isinstance(item, float) and not math.isfinite(item) else item
    temp.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def aggregation_wrapper(original, adapter_parameters, output=None):
    """Create one controller per run; bridge FULL states to the core delta API."""
    from flgmm_adapter import FLGMMAdapter, SOURCE_COMMIT
    import torch
    controller = FLGMMAdapter(range(20), **adapter_parameters)
    diagnostics = []

    def wrapped(method, updates, counts, fairness, server_update, config, **kwargs):
        if method != METHOD:
            return original(method, updates, counts, fairness, server_update, config, **kwargs)
        if config.num_clients != 20 or len(updates) != 20 or len(counts) != 20:
            raise ValueError("FLGMM requires all 20 clients in frozen cid=0..19 order")
        global_state = kwargs.get("global_state")
        if global_state is None:
            raise ValueError("Full-model FLGMM requires the current global state")
        if any(tuple(update) != tuple(global_state) for update in updates):
            raise ValueError("Attacked delta/global tensor layout mismatch")
        received = [{key: global_state[key] + update[key] for key in global_state}
                    for update in updates]
        full_state, info = controller.step(received, tuple(range(20)))
        delta = ({key: torch.zeros_like(value) for key, value in global_state.items()}
                 if full_state is None else
                 {key: full_state[key] - global_state[key] for key in global_state})
        if any(not torch.isfinite(value).all() for value in delta.values()):
            raise FloatingPointError("Nonfinite FLGMM model-to-delta bridge")
        error = (0.0 if full_state is None else max(
            float(torch.max(torch.abs(global_state[key] + delta[key] - full_state[key])).item())
            for key in global_state))
        info.update(report_label=LABEL, source_commit=SOURCE_COMMIT,
                    message_semantics="attacked delta -> global+delta full local model -> aggregate-global delta",
                    client_order=list(range(20)), client_counts_for_identity_only=list(counts),
                    model_delta_reconstruction_max_abs=error,
                    empty_selection_behavior="zero delta preserves global model")
        diagnostics.append(dict(round=controller.round_index, aggregate=info))
        if output is not None:
            write_json(Path(output) / "state.json", controller.state_dict())
            write_json(Path(output) / "diagnostics.json", diagnostics)
        return delta, info

    wrapped.controller = controller
    wrapped.diagnostics = diagnostics
    return wrapped


def load_core(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("guardfed_flgmm_isolated_core", repo / "scripts/reproduce_paper_tables.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    return core


def validate_job(job, protocol, require_frozen=True):
    if require_frozen and protocol.get("status") != "FROZEN":
        raise ValueError("Prepared protocol cannot execute")
    if protocol.get("scope") != "flgmm_selected_100_valid_only":
        raise ValueError("Wrong coverage scope")
    if (protocol["distributions"], protocol["attacks"], protocol["seeds"]) != (
            {"IID": 5000.0, "non-IID": 5.0},
            ["Benign", "F Flip", "FedSA", "S-DFA", "Sp-DFA"], list(range(91001, 91011))):
        raise ValueError("Wrong exact coverage grid")
    candidate = protocol.get("selected_recipe")
    if not isinstance(candidate, dict) or protocol["candidates"] != [candidate]:
        raise ValueError("One externally adopted recipe is required")
    if job.get("tuning_candidate") != candidate["id"] or job.get("adapter") != candidate["adapter"]:
        raise ValueError("Changed recipe")
    phase = job.get("phase")
    if phase not in {"fullcoverage", "preflight"}:
        raise ValueError("Screen/reference jobs cannot enter the new worker")
    seed = job["config"]["seed"]
    distribution, attack = job["distribution"], job["attack"]
    if type(seed) is not int or seed not in protocol["seeds"] or distribution not in protocol["distributions"] or attack not in protocol["attacks"]:
        raise ValueError("Unknown seed/condition")
    stage = "multi_seed_validation_coverage" if phase == "fullcoverage" else "pipeline_canary_only"
    if (job.get("dataset"), job.get("method"), job.get("evidence_stage")) != ("celeba", METHOD, stage):
        raise ValueError("Wrong method/evidence stage")
    if phase == "preflight" and (seed != 91001 or distribution != "non-IID"):
        raise ValueError("Canary scope is non-IID seed91001 only")
    if phase == "fullcoverage" and seed == 91001 and attack in {"Benign", "S-DFA"}:
        raise ValueError("Original four screen results must be reused, not retrained")
    identity = f"{candidate['id']}_{distribution}_{attack}_seed{seed}_{phase}"
    expected = dict(protocol["base_config"], rounds=70 if phase == "fullcoverage" else 3, seed=seed,
                    client_alpha=protocol["distributions"][distribution], learning_rate=candidate["learning_rate"],
                    experiment_suite=protocol["version"], experiment_tag=identity)
    if job.get("id") != identity or job.get("config") != expected or job.get("source_hashes") != protocol["source_hashes"]:
        raise ValueError("Job/config/source identity mismatch")
    if set(job.get("adapter_source_hashes", {})) != {
            "flgmm_adapter.py", "worker.py", "prepare_jobs.py", "protocol.json",
            "sources/flgmm_pinned.py", "sources/flgmm_fedavg.py", "sources/flgmm_license.txt"}:
        raise ValueError("Incomplete original adapter provenance fields")
    return candidate


def run(repo, job_path, output):
    repo, job_path, output = repo.resolve(), job_path.resolve(), output.resolve()
    job = json.loads(job_path.read_text(encoding="utf8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    validate_job(job, protocol)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        write_json(output / "job.json", job)
        for prefix, hashes in ((repo, job["source_hashes"]), (HERE, job["adapter_source_hashes"])):
            for relative, expected in hashes.items():
                name = Path(relative)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError("Source path escapes its declared root")
                if digest(prefix / name) != expected:
                    raise ValueError("Frozen source/data/adapter changed: " + relative)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.set_num_threads(int(os.environ.get("GUARDFED_CPU_THREADS", "1")))
        torch.set_num_interop_threads(1)
        core = load_core(repo)
        cfg = core.ExperimentConfig(**job["config"])
        device = core.choose_device(cfg.device)
        provenance = dict(job_sha256=digest(job_path), source_hashes=job["source_hashes"],
                          adapter_source_hashes=job["adapter_source_hashes"],
                          report_label=LABEL, started_unix=started,
                          environment=dict(python=sys.version, torch=torch.__version__,
                              cuda=torch.version.cuda, device=str(device), cpu_threads=torch.get_num_threads(),
                              gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None),
                          resume_supported=False)
        write_json(output / "provenance.json", provenance)
        original = core.aggregate_round
        wrapped = aggregation_wrapper(original, job["adapter"], output)
        core.aggregate_round = wrapped

        def progress(item):
            write_json(output / "progress.json", dict(item, job_id=job["id"], pid=os.getpid(),
                updated_unix=time.time(), elapsed_sec=time.time() - started))

        try:
            result = core.run_experiment("celeba", job["distribution"], METHOD, job["attack"], cfg,
                "revision_flgmm_validation_screen", device, progress_callback=progress,
                checkpoint_path=output / "model.pt")
        finally:
            core.aggregate_round = original
        for name in ("trajectory_metrics", "round_summaries"):
            if [row["round"] for row in result[name]] != list(range(1, cfg.rounds + 1)):
                raise ValueError("Incomplete FLGMM terminal-round trajectory or diagnostics")
        contract = result["data_contract"]["image_data_contract"]
        if (contract["evaluation_split"], contract["actual_train_rows"], contract["actual_evaluation_rows"],
                contract["train_eval_disjoint"], contract["root_client_disjoint"]) != ("valid", 162770, 19867, True, True):
            raise ValueError("Unexpected FLGMM image split/data contract")
        if result["config"] != job["config"] or result["metrics"] != result["trajectory_metrics"][-1]["metrics"]:
            raise ValueError("Config or same-terminal-checkpoint metrics mismatch")
        if not all(math.isfinite(value) and 0 <= value <= 1 for item in result["trajectory_metrics"]
                   for value in item["metrics"].values()):
            raise ValueError("Nonfinite/out-of-range metrics")
        if wrapped.controller.round_index != cfg.rounds or not torch.are_deterministic_algorithms_enabled():
            raise ValueError("FLGMM state count or deterministic image execution mismatch")
        result.update(status="coverage_complete" if job["phase"] == "fullcoverage" else "canary_complete", evidence_stage=job["evidence_stage"], method_impl_note=LABEL,
                      revision_job=job, provenance=provenance, tuning_candidate=job["tuning_candidate"])
        write_json(output / "result.json", result)
        hashes = {name: digest(output / name) for name in
                  ("model.pt", "state.json", "diagnostics.json", "result.json", "provenance.json")}
        write_json(output / "acceptance.json", dict(status="PASS", rounds=cfg.rounds,
            evaluation_split="valid", train_rows=162770, evaluation_rows=19867,
            tuning_candidate=job["tuning_candidate"], artifact_hashes=hashes,
            limitation="fixed recipe valid-only coverage or CANARY; no test or mid-round resume guarantee"))
        print(json.dumps(dict(complete=job["id"], metrics=result["metrics"])), flush=True)
    except BaseException as error:
        write_json(output / "failure.json", dict(job_id=job["id"], error=repr(error),
            traceback=traceback.format_exc(), failed_unix=time.time()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.job, args.out)
