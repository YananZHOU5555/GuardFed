"""Verify the fixed shared-calibration archive and extract only small reports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

STAGE = Path(__file__).resolve().parent
ARCHIVE = STAGE / "sharedcal700_and_baseline_gates_20260928.tar.gz"
SHA = "f7528394e8888163e157323654ad9cee31f4d32a1b65a0ef57cb6894382e352e"
INVENTORY = "deployment/sharedcal700_and_baseline_gates_20260928_inventory.json"
PREFIX = "results/revision_20260928/"


def safe_name(name):
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or ".." in parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")


def destination(name):
    shared = PREFIX + "celeba_shared_calibration_v1/"
    pilots = PREFIX + "baseline_image_gates_v1/"
    if name.startswith(shared):
        relative = name[len(shared):]
        if relative.startswith("final/") or relative in {
            "PROTOCOL.md", "manifest.json", "preflight.json", "canary_acceptance.json", "queue_exit.json"
        }:
            return STAGE / relative
    if name.startswith(pilots) and name.endswith((".json", ".md", ".txt")):
        return STAGE.parent / "baseline_image_gates_v1" / name[len(pilots):]
    return None


def main():
    if ARCHIVE.stat().st_size != 128662209:
        raise ValueError("Archive size mismatch or incomplete transfer")
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as handle:
        while data := handle.read(1024 * 1024):
            digest.update(data)
    archive_sha = digest.hexdigest()
    if archive_sha != SHA:
        raise ValueError("Archive SHA256 mismatch")
    with tarfile.open(ARCHIVE, "r|gz") as handle:
        for member in handle:
            if member.name == INVENTORY:
                if not member.isfile():
                    raise ValueError("Inventory is not a regular file")
                inventory_bytes = handle.extractfile(member).read()
                inventory = json.loads(inventory_bytes)
                break
        else:
            raise ValueError("Inventory absent")
    if len(inventory) != 2904:
        raise ValueError("Unexpected inventory count")
    seen, reports = set(), {}
    with tarfile.open(ARCHIVE, "r|gz") as handle:
        for member in handle:
            safe_name(member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError(f"Non-file or duplicate member: {member.name}")
            seen.add(member.name)
            source = handle.extractfile(member)
            if member.name == INVENTORY:
                if source.read() != inventory_bytes:
                    raise ValueError("Inventory changed between passes")
                continue
            expected = inventory[member.name]
            target = destination(member.name)
            if target is not None:
                target = target.resolve()
                target.relative_to(STAGE.parent.resolve())
                if target in reports or member.size > 20_000_000:
                    raise ValueError(f"Unexpected selected report: {member.name}")
            digest, count, chunks = hashlib.sha256(), 0, []
            while data := source.read(1024 * 1024):
                digest.update(data)
                count += len(data)
                if target is not None:
                    chunks.append(data)
            if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Member mismatch: {member.name}")
            if target is not None:
                reports[target] = b"".join(chunks)
    if seen != set(inventory) | {INVENTORY}:
        raise ValueError("Archive and inventory member sets differ")
    for target, data in reports.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if target.read_bytes() != data:
            raise ValueError(f"Extracted report mismatch: {target}")
    (STAGE / "backup_inventory.json").write_bytes(inventory_bytes)
    receipt_path = STAGE / "sharedcal_backup.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(off_server_verified=True, local_archive=str(ARCHIVE),
                   local_verified_utc=datetime.now(timezone.utc).isoformat(),
                   member_hashes_verified=len(inventory), archive_members_verified=len(seen),
                   extracted_reports=[str(path) for path in sorted(reports)],
                   local_archive_sha256=archive_sha)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "sha256": archive_sha, "content_members": len(inventory),
                      "archive_members": len(seen), "reports_extracted": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
