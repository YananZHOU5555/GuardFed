"""One CPU score-only worker, original32 LoGoFair fits; F-only, fail-stop."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback
from stage_inputs import HERE, bulk_path, digest, read, require, verify_delivery, write


def run(repo, inputs, output):
    verify_delivery()
    inputs, _ = bulk_path(inputs, 0)
    output, storage = bulk_path(output, 128 * 1024 * 1024)
    require(not output.exists() and inputs != output and not output.is_relative_to(inputs), "Preserve inputs/existing attempts; no retry")
    require(not (inputs / "STAGING_FAILURE.json").exists(), "Preserved staging failure blocks reuse")
    receipt = read(inputs / "INPUT_RECEIPT.json")
    require(receipt["status"] == "EXACT4_ACCEPTED_REFERENCES_AND_APPROVED_VIRTUAL_MAPPING_STAGED"
            and receipt["source_seal_sha256"] == digest(HERE / "FILES_SHA256.json"), "Frozen input stage differs")
    for name, expected in receipt["files"].items():
        require(digest(inputs / name) == expected, "Staged accepted input changed: " + name)
    manifest = read(HERE / "jobs/manifest.json")
    require(len(manifest["jobs"]) == len({r["id"] for r in manifest["jobs"]}) == 32, "Exact32 manifest required")
    require(digest(inputs / "mapping.npz") == read(HERE / "REFERENCE_INPUTS.json")["mapping_sha256"], "Frozen virtual mapping differs")
    output.mkdir(parents=True)
    with (inputs / "SCREEN32_STARTED.json").open("x", encoding="utf-8") as marker:
        import json
        json.dump(dict(pid=os.getpid(), output=output.as_posix(), automatic_retry=False), marker)
    write(output / "DISPATCH.json", dict(source_seal_sha256=digest(HERE / "FILES_SHA256.json"),
        input_receipt_sha256=digest(inputs / "INPUT_RECEIPT.json"), storage_preflight=storage,
        pid=os.getpid(), max_workers=1, CPU_threads=1, new_CNN_calls=0, test=False, automatic_retry=False))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    accepted = []
    try:
        for entry in manifest["jobs"]:
            bulk_path(output, 8 * 1024 * 1024)
            job_path = HERE / "jobs" / entry["job"]
            require(digest(job_path) == entry["job_sha256"], "Frozen job changed")
            job = read(job_path)
            reference = inputs / job["baseline_id"]
            with (output / (entry["id"] + ".log")).open("x", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-B", str(HERE / "snapshot/logofair_bridge_20261010/bridge.py"),
                    "--repo", str(repo), "--reference", str(reference), "--job", str(job_path),
                    "--mapping", str(inputs / "mapping.npz"), "--mapping-meta", str(inputs / "mapping_metadata.json"),
                    "--out", str(output / entry["id"])], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result_path = output / entry["id"] / "result.json"
            proof_path = output / entry["id"] / "acceptance.json"
            proof = read(proof_path)
            require(proof["status"] == "PASS" and proof["job_sha256"] == entry["job_sha256"], "Original strict bridge must finish")
            accepted.append(dict(id=entry["id"], result=result_path.as_posix(), result_sha256=digest(result_path),
                                 acceptance=proof_path.as_posix(), acceptance_sha256=digest(proof_path)))
            write(output / "PROGRESS.json", dict(status="LOCAL_ORIGINAL_STRICT_PROGRESS", records=accepted,
                  offserver_accepted=0, root_adopted=0, final_test=False))
        write(output / "STRICT32_INDEX.json", dict(status="LOCAL_ORIGINAL_STRICT32_COMPLETE_ROOT_REVIEW_PENDING",
            source_seal_sha256=digest(HERE / "FILES_SHA256.json"), records=accepted, seed_n=1,
            post_rounds=30, fit_seed=1719, new_CNN_calls=0, test=False, recipe_adopted=False))
    except BaseException as error:
        write(output / "QUEUE_FAILURE.json", dict(error=repr(error), traceback=traceback.format_exc(),
              strict_completed_ids=[r["id"] for r in accepted], automatic_retry=False))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.repo.resolve(), a.inputs, a.out)
