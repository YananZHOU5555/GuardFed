#!/usr/bin/env python3
"""Isolated three-round real CelebA pilot using the frozen core training path.

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
    if job["dataset"] != "celeba" or job["distribution"] != "non-IID" or job["method"] != METHOD:
        raise ValueError("pilot is frozen to non-IID CelebA / LASA-official")
    if job["attack"] not in ("Benign", "S-DFA"):
        raise ValueError("pilot only supports the two frozen attack conditions")
    required = dict(seed=91001, rounds=3, batch_size=64, learning_rate=.001,
                    num_clients=20, num_malicious=4, client_alpha=5., local_epochs=1,
                    optimizer="adam", server_ratio=.1, synthetic_ratio=0.,
                    celeba_train_limit=0, celeba_eval_limit=0, celeba_evaluation_split="valid",
                    full_round_diagnostics=True, device="cuda")
    for key, value in required.items():
        if job["config"].get(key) != value:
            raise ValueError("unexpected frozen pilot config: " + key)
    if job["config"].get("celeba_cache_dir", ""):
        raise ValueError("pilot requires repo-relative frozen image cache")
    if job["adapter"] != {"sparsity": .3, "lambda_n": 1., "lambda_s": 1.}:
        raise ValueError("pilot adapter parameters must use pinned official defaults")
    required_sources = {
        "scripts/reproduce_paper_tables.py", "src/data_loader.py", "src/celeba_data.py",
        "data/celeba/derived/rgb64_v1/manifest.json", "data/celeba/derived/rgb64_v1/metadata.npz",
        "data/celeba/derived/rgb64_v1/images.npy", "data/celeba/derived/rgb64_v1/available.npy",
        "data/celeba/list_attr_celeba.txt", "data/celeba/list_eval_partition.txt",
    }
    if not required_sources.issubset(job["source_hashes"]):
        raise ValueError("source/data identity is incomplete")
    if set(job["adapter_source_hashes"]) != {"adapter.py", "run_pilot.py"}:
        raise ValueError("both pilot and adapter code hashes must be frozen")


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
    # Never overwrite a previous pilot or restart it implicitly.
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
            evidence_stage="real_image_integration_pilot_not_formal_result"))

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.set_num_threads(int(os.environ.get("GUARDFED_CPU_THREADS", "1")))
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(repo))
        spec = importlib.util.spec_from_file_location("guardfed_lasa_pilot_core", repo / "scripts/reproduce_paper_tables.py")
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
                                        cfg, "revision_lasa_pilot", device, progress_callback=progress,
                                        checkpoint_path=out / "model.pt")
        finally:
            core.aggregate_round = original
        if [v["round"] for v in result["trajectory_metrics"]] != [1, 2, 3]:
            raise RuntimeError("incomplete pilot trajectory")
        if len(result["round_summaries"]) != 3:
            raise RuntimeError("incomplete aggregation diagnostics")
        contract = result["data_contract"]["image_data_contract"]
        if (contract["evaluation_split"], contract["actual_train_rows"], contract["actual_evaluation_rows"]) != ("valid", 162770, 19867):
            raise RuntimeError("pilot did not use frozen full train / official valid")
        if not all(math.isfinite(v) for item in result["trajectory_metrics"] for v in item["metrics"].values()):
            raise RuntimeError("non-finite evaluation metrics")
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("strict deterministic image path was not enabled")
        result["method_impl_note"] = "Pinned official LASA aggregation adapted to frozen local Adam model deltas; raw native reporting; three-round integration pilot only."
        result["revision_job"] = dict(job, output=str(out), source_hashes=actual_hashes,
            adapter_source_hashes=actual_adapter_hashes, checkpoint_sha256=digest(out / "model.pt"),
            finished_unix=time.time(), visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch_version=torch.__version__, python_version=sys.version, cpu_threads=torch.get_num_threads())
        write_json(out / "diagnostics.json", result["round_summaries"])
        write_json(out / "result.json", result)
        write_json(out / "acceptance.json", dict(status="PASS", rounds=3, evaluation_split="valid",
            full_train_rows=162770, evaluation_rows=19867, checkpoint_sha256=digest(out / "model.pt"),
            result_sha256=digest(out / "result.json"), diagnostics_sha256=digest(out / "diagnostics.json"),
            source_hashes=actual_hashes, adapter_source_hashes=actual_adapter_hashes,
            limitations="single-seed three-round interface/finite-value pilot; no comparative performance claim"))
        print(json.dumps({"complete": job["id"], "metrics": result["metrics"], "rounds": 3}), flush=True)
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
