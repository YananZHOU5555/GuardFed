"""Copy only four frozen accepted references and the mapping to guarded F storage."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import traceback
from guardfed_local_storage import STORAGE_ROOT, check_bulk_storage

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bulk_path(path, required_bytes):
    require(not sys.flags.optimize, "-O is forbidden; storage/source guards must remain active")
    proof = check_bulk_storage(required_bytes)
    path = Path(path).resolve()
    require(path.drive.upper() == "F:" and path.is_relative_to(STORAGE_ROOT.resolve())
            and path != STORAGE_ROOT.resolve(), "New artifacts must stay on guarded F storage")
    return path, proof


def verify_delivery():
    seal = read(HERE / "FILES_SHA256.json")
    for name, expected in seal["files"].items():
        require(digest(HERE / name) == expected, "Sealed source changed: " + name)


def stage(output):
    verify_delivery()
    output, storage = bulk_path(output, 32 * 1024 * 1024)
    require(not output.exists(), "Preserve existing/partial input stage; no automatic retry")
    inputs = read(HERE / "REFERENCE_INPUTS.json")
    require(len(inputs["references"]) == len({r["id"] for r in inputs["references"]}) == 4, "Exact four accepted references required")
    grouped = {}
    for entry in inputs["references"]:
        for name, row in entry["files"].items():
            require(name in {"model.pt", "result.json", "source_job.json", "margins.npz"}, "Foreign reference artifact")
            grouped.setdefault((row["archive"], row["archive_sha256"]), []).append((entry["id"], name, row))
    output.mkdir(parents=True)
    files = {}
    try:
        for (archive, expected), members in grouped.items():
            require(digest(archive) == expected, "Original accepted archive changed: " + archive)
            with tarfile.open(archive, "r:gz") as source:
                for identity, name, row in members:
                    member = source.getmember(row["member"])
                    require(member.isfile(), "Expected regular frozen archive member")
                    data = source.extractfile(member).read()
                    require(hashlib.sha256(data).hexdigest() == row["sha256"]
                            and ("bytes" not in row or len(data) == row["bytes"]), "Frozen member identity changed")
                    bulk_path(output, len(data))
                    target = output / identity / name
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(data)
                    files[target.relative_to(output).as_posix()] = digest(target)
        mapping = Path(inputs["mapping_source"])
        require(digest(mapping) == inputs["mapping_sha256"]
                and digest(HERE / "mapping_metadata.json") == inputs["mapping_metadata_sha256"], "Approved mapping identity changed")
        for source, name in ((mapping, "mapping.npz"), (HERE / "mapping_metadata.json", "mapping_metadata.json")):
            bulk_path(output, source.stat().st_size)
            (output / name).write_bytes(source.read_bytes())
            files[name] = digest(output / name)
        write(output / "INPUT_RECEIPT.json", dict(status="EXACT4_ACCEPTED_REFERENCES_AND_APPROVED_VIRTUAL_MAPPING_STAGED",
            source_seal_sha256=digest(HERE / "FILES_SHA256.json"), files=files,
            reference_ids=[r["id"] for r in inputs["references"]], storage_preflight=storage,
            new_CNN_calls=0, new_training=0, new_postprocessing_fits=0))
    except BaseException as error:
        write(output / "STAGING_FAILURE.json", dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    stage(p.parse_args().out)
