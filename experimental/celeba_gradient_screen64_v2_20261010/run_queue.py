"""One sequential gradient64 queue. Preparation never dispatches it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
STAGE = HERE / "snapshot/gradient_bridge_20261010"
GUIDE_SHA = "42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_resources(proof, now):
    require(proof["status"] == "ROOT_GRADIENT64_V2_RESOURCE_PREFLIGHT_PASS", "Actual root preflight required")
    require(0 <= now - proof["at_unix"] <= 90, "Stale resource preflight")
    require(proof["cpu_ids"] == [105] and proof["cpu_threads"] == proof["max_workers"] == 1
            and proof["cuda_visible_device"] == "1" and proof["nice"] >= 10 and proof["idle_io"] is True,
            "Only CPU105/thread1/GPU1/nice10/idleIO allocation")
    require(proof["no_restricted_cpu105_owner"] is True and proof["no_duplicate_worker"] is True
            and proof["no_restricted_CPU_overlap"] is True and proof["source_data_hashes_verified"] is True,
            "Actual ownership/source checks required")
    require(proof["existing_nominal_compute_threads"] + 1 <= proof["actual_quota_cores"]
            and proof["gpu_free_memory_mib"] >= 4096 and len(proof["gpu_uuid"]) > 8, "Insufficient actual capacity")
    require(proof["guide_sha256"] == GUIDE_SHA and proof["package_seal_sha256"] == digest(HERE / "FILES_SHA256.json")
            and proof["author_decisions_sha256"] == digest(HERE / "AUTHOR_DECISIONS.json"), "Preflight source/guide binding changed")
    require(proof["test_authorized"] is False and proof["automatic_retry_authorized"] is False, "Test/retry forbidden")


def bind_shared_inputs(worker, repo):
    """Same six registered physical targets as the accepted real-image gate."""
    binding = read(HERE / "shared_cache_bindings.json")
    names = {"data/celeba/derived/rgb64_v1/" + n for n in ("manifest.json", "metadata.npz", "images.npy", "available.npy")}
    names |= {"data/celeba/list_attr_celeba.txt", "data/celeba/list_eval_partition.txt"}
    require(binding["status"] == "APPROVED_EXACT_SHARED_INPUT_TARGETS" and set(binding["bindings"]) == names,
            "Only original six registered shared inputs")
    original = worker.verify_hashes
    protocol = read(STAGE / "protocol.json")
    for name, row in binding["bindings"].items():
        require(row["resolved_absolute"] == "/workspace/GuardFed-revision/" + name
                and row["expected_sha256"] == protocol["source_hashes"][name], "Registration differs from frozen source")

    def bound(root, hashes):
        root = Path(root).resolve()
        if root != repo:
            return original(root, hashes)
        for name, expected in hashes.items():
            if name not in binding["bindings"]:
                original(root, {name: expected})
                continue
            row = binding["bindings"][name]
            path = root / name
            require(not Path(name).is_absolute() and ".." not in Path(name).parts
                    and expected == row["expected_sha256"] and path.is_file()
                    and str(path.resolve()) == row["resolved_absolute"]
                    and path.stat().st_size == row["size_bytes"] and digest(path) == expected,
                    "Exact registered shared-input identity changed: " + name)
    worker.verify_hashes = bound


def run(args):
    require(sys.platform == "linux" and not sys.flags.optimize, "Linux without -O required; no local model outputs")
    repo, output = args.repo.resolve(), args.out.resolve()
    require(output.is_relative_to(Path("/workspace")) and output != Path("/workspace"), "Outputs must stay on server /workspace")
    require(not output.exists(), "Preserve existing output; no retry or resume")
    for name, expected in read(HERE / "FILES_SHA256.json")["files"].items():
        require(digest(HERE / name) == expected, "Frozen delivery changed: " + name)
    manifest = read(HERE / "jobs/manifest.json")
    require(len(manifest["jobs"]) == len({r["id"] for r in manifest["jobs"]}) == 64, "Exact64 manifest required")
    if args.single_job:
        entry = next((r for r in manifest["jobs"] if r["id"] == args.single_job), None)
        require(entry is not None and digest(HERE / "jobs" / entry["job"]) == entry["job_sha256"], "Foreign/mutated job")
        sys.path.insert(0, str(STAGE))
        import worker
        bind_shared_inputs(worker, repo)
        worker.run(repo, HERE / "jobs" / entry["job"], output)
        return
    require(args.resource_preflight is not None and args.resource_sha256 is not None, "Root resource proof path/SHA required")
    require(digest(args.resource_preflight) == args.resource_sha256, "Resource proof bytes changed")
    proof = read(args.resource_preflight)
    validate_resources(proof, time.time())
    require(digest("/etc/vast-agents-guide.md") == GUIDE_SHA and os.sched_getaffinity(0) == {105}
            and os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Guide/actual affinity/nice changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "Physical GPU1 must be explicitly isolated")
    uuid = subprocess.check_output(["nvidia-smi", "--id=1", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    require(uuid == proof["gpu_uuid"], "Physical GPU UUID changed")
    import fcntl
    lock = (HERE / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", GUARDFED_CPU_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    output.mkdir()
    completed = []
    try:
        for entry in manifest["jobs"]:
            subprocess.run(["ionice", "-c", "3", sys.executable, "-B", str(HERE / "run_queue.py"),
                            "--repo", str(repo), "--out", str(output / entry["id"]), "--single-job", entry["id"]],
                           env=env, check=True)
            completed.append(entry["id"])
            (output / "QUEUE_PROGRESS.json").write_text(json.dumps({"strict_server_completed_ids": completed,
                "offserver_accepted": 0, "test": False}) + "\n", encoding="utf-8")
    except BaseException as error:
        (output / "QUEUE_FAILURE.json").write_text(json.dumps({"error": repr(error), "traceback": traceback.format_exc(),
            "strict_server_completed_ids": completed, "automatic_retry": False}) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resource-preflight", type=Path)
    p.add_argument("--resource-sha256")
    p.add_argument("--single-job", help=argparse.SUPPRESS)
    run(p.parse_args())
