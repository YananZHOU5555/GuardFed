#!/usr/bin/env python3
"""Isolated 70-round validation screen or separate three-round preflight.

Only aggregate_round is wrapped. LASA consumes local model deltas from that core;
this is explicitly a local-delta protocol adaptation, not a new original method.
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
METHOD = "LASA-official"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def validate_job(job):
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    candidates = {v["id"]: v for v in protocol["candidates"]}
    if job.get("phase") not in {"screen", "preflight"}:
        raise ValueError("job must identify screen or independent preflight phase")
    if job.get("dataset") != "celeba" or job.get("method") != METHOD:
        raise ValueError("only frozen CelebA LASA screen jobs are accepted")
    if job.get("distribution") not in protocol["distributions"] or job.get("attack") not in protocol["attacks"]:
        raise ValueError("distribution/attack is outside frozen screen")
    candidate = candidates.get(job.get("tuning_candidate"))
    if candidate is None or job.get("adapter") != candidate["adapter"]:
        raise ValueError("unknown candidate or mismatched adapter parameters")
    rounds = protocol["screen_rounds"] if job["phase"] == "screen" else protocol["preflight_rounds"]
    if job["phase"] == "preflight" and (candidate["id"] != protocol["preflight_candidate"] or job["distribution"] != protocol["preflight_distribution"]):
        raise ValueError("preflight is limited to the previously passed pilot recipe")
    expected_id = f"{candidate['id']}_{job['distribution']}_{job['attack']}_seed91001_{job['phase']}"
    if job.get("id") != expected_id:
        raise ValueError("job identity mismatch")
    expected = dict(protocol["base_config"], rounds=rounds, learning_rate=candidate["learning_rate"],
                    client_alpha=protocol["distributions"][job["distribution"]],
                    experiment_suite=protocol["version"] + "_" + job["phase"], experiment_tag=expected_id)
    if job.get("config") != expected:
        raise ValueError("configuration differs from frozen recipe")
    if job.get("source_hashes") != protocol["source_hashes"]:
        raise ValueError("source/data identity differs from frozen protocol")
    if set(job.get("adapter_source_hashes", {})) != {"adapter.py", "worker.py", "protocol.json", "prepare_jobs.py"}:
        raise ValueError("screen code/protocol hash set is incomplete")


def aggregation_wrapper(original, adapter_parameters):
    from adapter import lasa_aggregate

    def wrapped(method, updates, counts, fairness, server_update, config, **kwargs):
        if method != METHOD:
            return original(method, updates, counts, fairness, server_update, config, **kwargs)
        delta, info = lasa_aggregate(updates, **adapter_parameters)
        info["message_semantics"] = "frozen-core local Adam model deltas, after attack injection"
        info["report_label"] = "LASA official aggregation adapted to local-model deltas"
        info["client_counts_for_identity_only"] = list(counts)
        return delta, info
    return wrapped


def run(repo, job_path, out):
    repo, job_path, out = repo.resolve(), job_path.resolve(), out.resolve()
    job = json.loads(job_path.read_text(encoding="utf-8-sig"))
    validate_job(job)
    # Never overwrite a previous record or restart it implicitly.
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        write_json(out / "job.json", job)
        actual_hashes = {}
        for name, expected in job["source_hashes"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("source path escapes repository: " + name)
            # The existing verified CelebA cache may be a repo-owned symlink.
            path = repo / relative
            actual_hashes[name] = digest(path)
            if actual_hashes[name] != expected:
                raise ValueError("frozen source/data changed: " + name)
        actual_adapter_hashes = {name: digest(HERE / name) for name in job["adapter_source_hashes"]}
        if actual_adapter_hashes != job["adapter_source_hashes"]:
            raise ValueError("frozen adapter source changed")
        write_json(out / "provenance.json", dict(
            job_id=job["id"], job_sha256=digest(job_path), source_hashes=actual_hashes,
            adapter_source_hashes=actual_adapter_hashes, repo=str(repo), started_unix=started,
            evidence_stage=job["evidence_stage"], phase=job["phase"], tuning_candidate=job["tuning_candidate"]))

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.set_num_threads(int(os.environ.get("GUARDFED_CPU_THREADS", "1")))
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(repo))
        spec = importlib.util.spec_from_file_location("guardfed_lasa_screen_core", repo / "scripts/reproduce_paper_tables.py")
        core = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = core
        spec.loader.exec_module(core)
        cfg = core.ExperimentConfig(**job["config"])
        device = core.choose_device(cfg.device)
        original = core.aggregate_round
        core.aggregate_round = aggregation_wrapper(original, job["adapter"])
        def progress(item):
            write_json(out / "progress.json", dict(item, job_id=job["id"], pid=os.getpid(),
                       updated_unix=time.time(), elapsed_sec=time.time() - started))
        try:
            result = core.run_experiment(job["dataset"], job["distribution"], METHOD, job["attack"],
                                        cfg, "revision_lasa_screen", device, progress_callback=progress,
                                        checkpoint_path=out / "model.pt")
        finally:
            core.aggregate_round = original
        if [v["round"] for v in result["trajectory_metrics"]] != list(range(1, cfg.rounds + 1)):
            raise RuntimeError("incomplete fixed-final-round trajectory")
        if len(result["round_summaries"]) != cfg.rounds:
            raise RuntimeError("incomplete aggregation diagnostics")
        if [v["round"] for v in result["round_summaries"]] != list(range(1, cfg.rounds + 1)):
            raise RuntimeError("aggregation rounds are missing or repeated")
        if result["config"] != job["config"] or result["metrics"] != result["trajectory_metrics"][-1]["metrics"]:
            raise RuntimeError("config or fixed-final-round metrics mismatch")
        contract = result["data_contract"]["image_data_contract"]
        if (contract["evaluation_split"], contract["actual_train_rows"], contract["actual_evaluation_rows"]) != ("valid", 162770, 19867):
            raise RuntimeError("record did not use frozen full train / official valid")
        if not all(math.isfinite(v) for item in result["trajectory_metrics"] for v in item["metrics"].values()):
            raise RuntimeError("non-finite evaluation metrics")
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("strict deterministic image path was not enabled")
        result["method_impl_note"] = "Pinned official LASA aggregation adapted to frozen local Adam model deltas; raw native reporting; single-seed validation search / separately labelled preflight only."
        result["revision_job"] = dict(job, output=str(out), source_hashes=actual_hashes,
            adapter_source_hashes=actual_adapter_hashes, checkpoint_sha256=digest(out / "model.pt"),
            finished_unix=time.time(), visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch_version=torch.__version__, python_version=sys.version, cpu_threads=torch.get_num_threads())
        write_json(out / "diagnostics.json", result["round_summaries"])
        write_json(out / "result.json", result)
        write_json(out / "acceptance.json", dict(status="PASS", rounds=cfg.rounds, phase=job["phase"], tuning_candidate=job["tuning_candidate"], evaluation_split="valid",
            full_train_rows=162770, evaluation_rows=19867, checkpoint_sha256=digest(out / "model.pt"),
            result_sha256=digest(out / "result.json"), diagnostics_sha256=digest(out / "diagnostics.json"),
            source_hashes=actual_hashes, adapter_source_hashes=actual_adapter_hashes,
            limitations="single-seed valid-only fixed-final-round record; not an independent-test or significance claim"))
        print(json.dumps({"complete": job["id"], "metrics": result["metrics"], "rounds": cfg.rounds}), flush=True)
    except Exception as error:
        write_json(out / "failure.json", dict(job_id=job["id"], error=repr(error),
            traceback=traceback.format_exc(), failed_unix=time.time()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.job, args.out)
