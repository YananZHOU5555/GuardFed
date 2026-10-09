"""CosineFairnessHybrid on the unchanged image core; prepared drafts cannot run.

Only the custom aggregation branch is replaced. Local Adam, attacks, post-attack
root AEOD, train/evaluation splits and native reporting remain in the frozen core.
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
METHOD = "CosineFairnessHybrid"
LABEL = "Custom cosine/fairness control: post-attack clean-root AEOD trust threshold, equal selected mean; legacy GuardFed mechanism"
ADAPTER_SOURCES = {"adapters.py", "worker.py", "prepare_jobs.py", "protocol.json", "accept_result.py"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        return None if isinstance(item, float) and not math.isfinite(item) else item
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def aggregation_wrapper(original, adapter_parameters, output=None):
    """Receive the frozen core's already attacked deltas and root AEOD values."""
    from adapters import cosine_fairness_hybrid
    import torch
    if set(adapter_parameters) != {"fairness_lambda", "threshold"}:
        raise ValueError("Unexpected hybrid adapter fields")
    diagnostics = []

    def wrapped(method, updates, counts, fairness, server_update, config, **kwargs):
        if method != METHOD:
            return original(method, updates, counts, fairness, server_update, config, **kwargs)
        if config.num_clients != 20 or len(updates) != 20 or len(counts) != 20:
            raise ValueError("Expected frozen twenty-client order")
        if (config.guardfed_fairness_lambda, config.trust_threshold) != (
                adapter_parameters["fairness_lambda"], adapter_parameters["threshold"]):
            raise ValueError("Reported config and actual hybrid parameters disagree")
        details = kwargs.get("fairness_details")
        global_state = kwargs.get("global_state")
        if details is None or len(details) != len(updates) or global_state is None:
            raise ValueError("Attacked model root metrics and current global state are required")
        if any(tuple(update) != tuple(global_state) for update in updates):
            raise ValueError("Attacked delta/global tensor layout mismatch")
        root_fairness = []
        for detail, reported in zip(details, fairness):
            value = float(detail["aeod"])
            value = 1.0 if math.isnan(value) else value
            if value != float(reported):
                raise ValueError("Hybrid input AEOD differs from frozen-core post-attack root evaluation")
            root_fairness.append(value)
        delta, info = cosine_fairness_hybrid(updates, root_fairness, server_update, **adapter_parameters)
        if any(not torch.isfinite(value).all() for value in delta.values()):
            raise FloatingPointError("Nonfinite hybrid aggregate")
        info.update(report_label=LABEL, client_order=list(range(20)),
                    client_counts_for_identity_only=list(counts),
                    root_aeod=root_fairness,
                    fairness_semantics="implemented absolute TPR gap, proportion units; NaN assigned 1",
                    fairness_source="frozen core evaluate_state_on_server(global+attacked_delta), clean train-root",
                    reference_semantics="frozen core clean-root local Adam model delta",
                    message_semantics="attacked local model deltas; return selected equal-mean delta",
                    actual_parameters=dict(adapter_parameters),
                    reporting="native logits argmax; no AD2+ group-threshold calibration")
        diagnostics.append(dict(round=len(diagnostics) + 1, aggregate=info))
        if output is not None:
            write_json(Path(output) / "diagnostics.json", diagnostics)
        return delta, info

    wrapped.diagnostics = diagnostics
    return wrapped


def load_core(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("guardfed_hybrid_isolated_core", repo / "scripts/reproduce_paper_tables.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    return core


def validate_job(job, protocol, require_frozen=True):
    if require_frozen and protocol.get("status") != "FROZEN":
        raise ValueError("Prepared protocol is not frozen; this template cannot authorize training")
    candidates = {row["id"]: row for row in protocol["candidates"]}
    candidate = candidates.get(job.get("tuning_candidate"))
    if candidate is None or job.get("adapter") != candidate["adapter"]:
        raise ValueError("Unknown hybrid candidate or adapter recipe")
    if (job.get("dataset"), job.get("method"), job.get("phase"), job.get("evidence_stage")) != (
            "celeba", METHOD, "screen", "validation_screen"):
        raise ValueError("Expected an individual valid-only hybrid screen job")
    distribution, attack = job["distribution"], job["attack"]
    if distribution not in protocol["distributions"] or attack not in protocol["attacks"]:
        raise ValueError("Unexpected hybrid screen condition")
    identity = f"{candidate['id']}_{distribution}_{attack}_seed91001_screen"
    expected = dict(protocol["base_config"], rounds=70, seed=91001,
                    client_alpha=protocol["distributions"][distribution],
                    learning_rate=candidate["learning_rate"],
                    guardfed_fairness_lambda=candidate["adapter"]["fairness_lambda"],
                    trust_threshold=candidate["adapter"]["threshold"],
                    experiment_suite=protocol["version"], experiment_tag=identity)
    if job.get("id") != identity or job.get("config") != expected:
        raise ValueError("Hybrid job identity/config differs from protocol")
    if job.get("source_hashes") != protocol["source_hashes"]:
        raise ValueError("Hybrid source/data identity differs from protocol")
    if set(job.get("adapter_source_hashes", {})) != ADAPTER_SOURCES:
        raise ValueError("Incomplete hybrid adapter provenance")
    return candidate


def run(repo, job_path, output):
    repo, job_path, output = repo.resolve(), job_path.resolve(), output.resolve()
    job = json.loads(job_path.read_text(encoding="utf8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    validate_job(job, protocol)
    # Never overwrite a partial directory, saved failures or completed results.
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
        config = core.ExperimentConfig(**job["config"])
        device = core.choose_device(config.device)
        provenance = dict(job_sha256=digest(job_path), source_hashes=job["source_hashes"],
                          adapter_source_hashes=job["adapter_source_hashes"],
                          report_label=LABEL, started_unix=started, resume_supported=False,
                          environment=dict(python=sys.version, torch=torch.__version__,
                              cuda=torch.version.cuda, device=str(device), cpu_threads=torch.get_num_threads(),
                              gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
        write_json(output / "provenance.json", provenance)
        original = core.aggregate_round
        wrapped = aggregation_wrapper(original, job["adapter"], output)
        core.aggregate_round = wrapped

        def progress(item):
            write_json(output / "progress.json", dict(item, job_id=job["id"], pid=os.getpid(),
                updated_unix=time.time(), elapsed_sec=time.time() - started))

        try:
            result = core.run_experiment("celeba", job["distribution"], METHOD, job["attack"], config,
                "revision_hybrid_validation_screen", device, progress_callback=progress,
                checkpoint_path=output / "model.pt")
        finally:
            core.aggregate_round = original
        if len(wrapped.diagnostics) != 70 or not torch.are_deterministic_algorithms_enabled():
            raise ValueError("Hybrid diagnostic horizon or deterministic execution mismatch")
        result.update(status="screen_complete", evidence_stage="validation_screen", method_impl_note=LABEL,
                      revision_job=job, provenance=provenance, tuning_candidate=job["tuning_candidate"])
        write_json(output / "result.json", result)
        hashes = {name: digest(output / name) for name in
                  ("model.pt", "diagnostics.json", "result.json", "provenance.json")}
        write_json(output / "acceptance.json", dict(status="PASS", rounds=70,
            evaluation_split="valid", train_rows=162770, evaluation_rows=19867,
            tuning_candidate=job["tuning_candidate"], artifact_hashes=hashes,
            limitation="single-seed valid-only screen; no independent test or training-resume guarantee"))
        from accept_result import checked_result
        checked_result(job_path, output)
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
