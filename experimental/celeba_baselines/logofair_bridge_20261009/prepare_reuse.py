"""Prepare existing accepted FedAvg references/drafts; no mapping or training.

Optional canary extraction copies only existing accepted artifacts into this
new directory and verifies archive/member hashes, never fits real LoGoFair.
"""
import argparse
import copy
import json
from pathlib import Path
import tarfile

from bridge import HERE, LOCAL_FILES, digest, validate_job, write_json

WORKSPACE = HERE.parents[2]
TRAINING = WORKSPACE / "docs/server_deployment_20260923/training_20260923"


def prepare_metadata():
    if (HERE / "protocol.json").exists() or (HERE / "reuse_manifest.json").exists():
        raise ValueError("Preserve existing prepared/frozen metadata; use --jobs-only for a new job directory")
    shared = TRAINING / "celeba_shared_calibration_v1"
    manifest_path, accepted_path = shared / "manifest.json", shared / "final/accepted.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    accepted = json.loads(accepted_path.read_text(encoding="utf8"))
    if accepted["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Existing accepted score-cache manifest changed")
    records = {row["id"]: row for row in accepted["records"]}
    entries = []
    for job in manifest["jobs"]:
        if job["method"] != "FedAvg":
            continue
        record = records[job["id"]]
        if (record["checkpoint_sha256"], record["evaluation_split"], record["round"], record["seed"]) != (
                job["checkpoint_sha256"], "valid", 70, job["config"]["seed"]):
            raise ValueError("Existing FedAvg accepted identity differs")
        entries.append(dict(id=job["id"], cell_id=f"FedAvg_{job['distribution']}_{job['attack']}_seed{job['config']['seed']}",
            source_job=job, accepted_record=record,
            existing_checkpoint_path=job["output"] + "/model.pt",
            existing_margin_cache_path=f"/workspace/GuardFed-celeba-expanded/results/revision_20260928/celeba_shared_calibration_v1/runs/{job['id']}/margins.npz",
            reuse_status="Previously accepted artifacts; real LoGoFair mapping not yet supplied, current bridge must recheck selected artifacts"))
    cells = {(row["source_job"]["distribution"], row["source_job"]["attack"], row["source_job"]["config"]["seed"]) for row in entries}
    assert len(entries) == len(cells) == 100
    write_json(HERE / "reuse_manifest.json", dict(status="PREPARED_NOT_FROZEN", execution_started=False,
        source_manifest_sha256=digest(manifest_path), source_accepted_sha256=digest(accepted_path), entries=entries,
        limitation="100 accepted FedAvg checkpoints/caches referenced, not 100 newly audited/fitted LoGoFair runs"))
    source = json.loads((HERE.parent / "remaining_20261009/group_a/candidate_configurations.json").read_text(encoding="utf8"))
    candidates = [dict(id=row["candidate"], settings={key: value for key, value in row.items() if key != "candidate"})
                  for row in source["methods"]["LoGoFair-DP"]]
    protocol = dict(status="PREPARED_NOT_FROZEN", execution_started=False, version="celeba_logofair_dp_postprocessing_20261009_v1",
        candidates=candidates, seed=91001, fit_seed=1719, calibration="clean accepted training-root only; validation labels never enter fitting",
        mapping_input="Explicit int64 root/valid image-ID -> client-ID NPZ plus reviewed metadata JSON, both SHA-bound per job",
        decisions={
            "mapping_population": dict(status="UNRESOLVED", options=[
                "Declared virtual 20-cohort partition of existing root/central validation via a label/score-independent rule; preserves existing CNN/cache and global validation, changes local population interpretation",
                "True per-training-client holdout and corresponding evaluation cohorts; needs new sample/protocol identity and possibly new FedAvg runs, unsupported by this central-cache worker"],
                constraint="No true training-client claim for central CelebA IDs; no dummy single client or silent remapping"),
            "postprocessing_search": dict(status="UNRESOLVED", proposed="8 existing calibrated DP candidates, same four valid-only conditions, root-only fit, shared candidate score rule"),
            "tie_and_missing_population": dict(status="UNRESOLVED", proposed="Exact threshold ties, missing client/group/binary-label support and numerical divergence fail explicitly; no pooling, randomness or beta repair"),
            "real_mapping_gate": dict(status="UNRESOLVED", proposed="After selected mapping is approved, verify root/valid identity and one real score-only calibrated fit; current gate is synthetic RGB64 only")},
        method_identity="Pinned official local/global DP plus official BetaCalibration MLE; calibration-only corrected group priors; explicit reviewed virtual local populations; no EO claim",
        selection_rule="Mean prior frozen score over four conditions for each candidate; exact tie candidate lexical. Keep all candidates/accuracy champions/Pareto; n=1, four conditions not independent seeds; no test selection.",
        score="accuracy-.35*(.45*aeod+.45*aspd+.10*max(aeod,aspd))-.10*max(0,max(aeod,aspd)-.06)",
        limitations=["No client mapping generated; mapping hashes are null in draft jobs", "No real LoGoFair or GPU/test evaluation launched",
            "Model/root/cache/config/source identities must all match accepted external reference", "Score-only CPU reuse can avoid CNN retraining; fresh actual CNN softmax extractor also provided"])
    write_json(HERE / "protocol.json", protocol)
    return entries, protocol


def prepare_jobs(entries, protocol, output, mapping_path=None, mapping_meta_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {name: digest(HERE / name) for name in LOCAL_FILES}
    rows = []
    eligible = [entry for entry in entries if entry["source_job"]["config"]["seed"] == 91001 and entry["source_job"]["attack"] in {"Benign", "S-DFA"}]
    assert len(eligible) == 4
    for candidate in protocol["candidates"]:
        for entry in eligible:
            identity = candidate["id"] + "_" + entry["cell_id"]
            job = dict(id=identity, method="LoGoFair-DP-official-adapted", baseline_id=entry["id"],
                candidate=candidate["id"], settings=candidate["settings"], seed=91001, fit_seed=1719,
                evaluation_split="valid", mapping_sha256=digest(mapping_path) if mapping_path else None,
                mapping_metadata_sha256=digest(mapping_meta_path) if mapping_meta_path else None, local_hashes=hashes)
            validate_job(job, protocol, require_frozen=protocol["status"] == "FROZEN")
            path = output / (identity + ".json")
            write_json(path, job)
            rows.append(dict(id=identity, job=path.name, job_sha256=digest(path)))
    assert len(rows) == 32
    write_json(output / "manifest.json", dict(status=protocol["status"], execution_started=False, jobs=rows,
        protocol_sha256=digest(HERE / "protocol.json"), limitation="Mapping unresolved and null-hash-bound draft, not executable"))


def extract_canary(entries):
    stage = TRAINING / "celeba_fullcoverage_v1"
    reference = HERE / "accepted_fedavg_reference"
    reference.mkdir(parents=True, exist_ok=False)
    identity = "FedAvg_IID_Benign_seed91001"
    entry = next(row for row in entries if row["id"] == identity)
    chain = json.loads((stage / "restore_chain_final.json").read_text(encoding="utf8"))
    archive_record = next(row for row in chain["archives"] if "incremental409" in row["archive"])
    archive = stage / archive_record["archive"]
    if digest(archive) != archive_record["sha256"]:
        raise ValueError("Existing FedAvg backup archive hash changed")
    prefix = "results/revision_20260926/celeba_fullcoverage_v1/"
    members = {"model.pt": prefix + "runs/" + identity + "/model.pt", "result.json": prefix + "runs/" + identity + "/result.json",
               "source_job.json": prefix + "jobs/" + identity + ".json"}
    evidence = {}
    with tarfile.open(archive, "r:gz") as source:
        inventory = json.load(source.extractfile(next(name for name in source.getnames() if name.endswith("_inventory.json"))))
        for name, member in members.items():
            data = source.extractfile(member).read()
            (reference / name).write_bytes(data)
            if digest(reference / name) != inventory[member]["sha256"]:
                raise ValueError("Existing FedAvg backup member hash changed")
            evidence[name] = dict(archive_sha256=archive_record["sha256"], member=member, sha256=digest(reference / name))
    shared = TRAINING / "celeba_shared_calibration_v1"
    backup = json.loads((shared / "sharedcal_backup.json").read_text(encoding="utf8"))
    shared_archive = shared / "sharedcal700_and_baseline_gates_20260928.tar.gz"
    expected_archive_sha = backup.get("sha256", backup.get("archive_sha256"))
    if expected_archive_sha and digest(shared_archive) != expected_archive_sha:
        raise ValueError("Existing accepted shared-calibration archive changed")
    member = "results/revision_20260928/celeba_shared_calibration_v1/runs/" + identity + "/margins.npz"
    with tarfile.open(shared_archive, "r:gz") as source:
        data = source.extractfile(member).read()
        (reference / "margins.npz").write_bytes(data)
    if digest(reference / "margins.npz") != entry["accepted_record"]["cache_sha256"]:
        raise ValueError("Existing accepted margin cache hash changed")
    evidence["margins.npz"] = dict(archive_sha256=digest(shared_archive), member=member, sha256=digest(reference / "margins.npz"))
    write_json(reference / "reference_receipt.json", dict(status="Existing accepted FedAvg artifacts copied and SHA-verified; not new training",
        id=identity, evidence=evidence, entry=entry))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-canary", action="store_true")
    parser.add_argument("--jobs-only", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "screen_jobs_draft")
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--mapping-meta", type=Path)
    args = parser.parse_args()
    if args.jobs_only:
        entries = json.loads((HERE / "reuse_manifest.json").read_text(encoding="utf8"))["entries"]
        protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    else:
        entries, protocol = prepare_metadata()
    prepare_jobs(entries, protocol, args.out, args.mapping, args.mapping_meta)
    if args.extract_canary:
        extract_canary(entries)
    print(json.dumps(dict(status="PREPARED_NOT_FROZEN", referenced_fedavg_checkpoints=len(entries), draft_postprocessing_jobs=32)))
