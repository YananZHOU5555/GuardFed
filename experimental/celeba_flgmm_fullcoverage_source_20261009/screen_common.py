"""Identity checks around the unchanged FLGMM worker and terminal checker."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))
from worker import digest, validate_job, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def under(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes root: ' + name)
    return path


def local_identity(require_frozen=True):
    seal = read(HERE / "PACKAGE_SHA256.json")
    if seal["status"] != "FROZEN_PENDING_EXECUTION": raise ValueError("Unbound stage")
    for name, expected in seal["files"].items():
        if digest(under(HERE, name)) != expected: raise ValueError("Package changed: " + name)
    protocol, manifest = read(HERE / "source/protocol.json"), read(HERE / "manifest.json")
    if protocol.get("status") != "FROZEN" or digest(HERE / "source/protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Protocol identity changed")
    if manifest["scope"] != "flgmm_selected_100_valid_only" or (manifest["planned_new"], manifest["planned_total"]) != (96, 100):
        raise ValueError("Wrong coverage scope")
    observed = set()
    for item in manifest["jobs"] + manifest["preflight_jobs"]:
        path = under(HERE / "jobs", item["job"])
        if digest(path) != item["job_sha256"]: raise ValueError("Job SHA changed")
        job = read(path); validate_job(job, protocol)
        if any(item[key] != job[key] for key in ("id", "distribution", "attack", "tuning_candidate")) or item["seed"] != job["config"]["seed"]:
            raise ValueError("Manifest/job mismatch")
        for name, expected in job["adapter_source_hashes"].items():
            if digest(under(HERE / "source", name)) != expected: raise ValueError("Adapter changed")
        if item in manifest["jobs"]:
            key = (job["distribution"], job["attack"], job["config"]["seed"])
            if key in observed: raise ValueError("Duplicate new cell")
            observed.add(key)
    if len(manifest["jobs"]) != 96 or len(manifest["preflight_jobs"]) != 5 or len(manifest["reused_jobs"]) != 4:
        raise ValueError("Coverage count changed")
    if {(r["distribution"], r["attack"], r["seed"]) for r in manifest["preflight_jobs"]} != {("non-IID", a, 91001) for a in protocol["attacks"]}:
        raise ValueError("Wrong gate grid")
    for item in manifest["reused_jobs"]:
        row = item["accepted_record"]
        if row["candidate"] != protocol["selected_recipe"]["id"] or row["rounds"] != 70 or row["seed"] != 91001 or row["attack"] not in ("Benign", "S-DFA"):
            raise ValueError("Wrong reused identity")
        key = (row["distribution"], row["attack"], row["seed"])
        if key in observed: raise ValueError("Duplicate reused/new cell")
        observed.add(key)
    if observed != {(d,a,s) for d in protocol["distributions"] for a in protocol["attacks"] for s in protocol["seeds"]}:
        raise ValueError("Expected exact100 union")
    return protocol, manifest


def checked_repo_path(repo, name, expected, mapping):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid frozen repository-relative path: ' + name)
    if name not in mapping:
        return under(repo, name)
    rule = mapping[name]
    actual = (repo / relative).resolve()
    if actual != Path(rule['resolved_target']) or expected != rule['sha256']:
        raise ValueError('Frozen data symlink target/SHA declaration changed: ' + name)
    return actual


def repo_identity(repo, protocol):
    mapping = read(HERE / 'REPO_SYMLINK_TARGETS.json')['entries']
    permitted = {'data/celeba/' + name for name in [
        'derived/rgb64_v1/manifest.json', 'derived/rgb64_v1/metadata.npz',
        'derived/rgb64_v1/images.npy', 'derived/rgb64_v1/available.npy',
        'list_attr_celeba.txt', 'list_eval_partition.txt']}
    if set(mapping) != permitted:
        raise ValueError('Only the six explicitly reviewed CelebA data targets are permitted')
    for name, expected in protocol['source_hashes'].items():
        if digest(checked_repo_path(repo, name, expected, mapping)) != expected:
            raise ValueError('Repository source/data changed: ' + name)
    return dict(protocol['source_hashes'])


def authorized(scope, fresh=False):
    if digest('/etc/vast-agents-guide.md') != '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa':
        raise ValueError('Server guide changed; read and review before execution')
    receipt = read(HERE / "EXECUTION_AUTHORIZATION.json")
    if (receipt.get("status"), receipt.get("scope"), receipt.get("package_sha256")) != (
            "AUTHORIZED", scope, digest(HERE / "PACKAGE_SHA256.json")):
        raise ValueError("Missing external exact-scope authorization")
    if not receipt.get("resource_review_utc") or receipt.get("no_duplicate_workers_verified") is not True:
        raise ValueError("Live resource/identity review required")
    if receipt.get("max_workers") != 2 or receipt.get("cpu_threads_per_worker") != 1 or receipt.get("final_test") is not False:
        raise ValueError("Wrong resource/test scope")
    resource_path = Path(receipt["resource_receipt_path"])
    if digest(resource_path) != receipt["resource_receipt_sha256"]: raise ValueError("Resource receipt drift")
    resource = read(resource_path)
    import time
    if fresh and not 0 <= time.time() - resource["observed_unix"] <= 120: raise ValueError("Stale initial resource receipt")
    if resource["protected_main_max_workers"] != 8 or resource["protected_main_healthy"] is not True:
        raise ValueError("Main queue protection not established")
    if resource["planned_total_cpu_threads"] > resource["cpu_quota_cores"] or resource["free_memory_bytes"] < 8 * 1024**3:
        raise ValueError("Insufficient shared resources")
    if resource["gpu_recovery_actions"] != ["None", "None"] or min(resource["gpu_free_memory_mib"]) < 2048:
        raise ValueError("GPU health/resource failure")
    if scope == "96_new_70round_valid_only":
        if digest(HERE / "GATE_ACCEPTANCE.json") != receipt["gate_acceptance_sha256"]:
            raise ValueError("Missing bound gate")
        gate = read(HERE / "GATE_ACCEPTANCE.json")
        if gate["status"] != "PASS" or gate["package_sha256"] != receipt["package_sha256"] or gate["accepted_new_canaries"] != 5 or gate["same_horizon_pairs"] != 2:
            raise ValueError("Incomplete interface gates")
        for name, expected in gate["artifact_hashes"].items():
            if digest(under(HERE, name)) != expected: raise ValueError("Gate evidence changed")
    return receipt


def accepted(item, out):
    from accept_result import checked_result
    result = checked_result(HERE / 'jobs' / item['job'], out)
    if result is None:
        return None
    proof = read(out / 'screen_identity.json')
    if proof['package_sha256'] != digest(HERE / 'PACKAGE_SHA256.json'):
        raise ValueError('Result package identity mismatch')
    if proof['before'] != proof['after'] or proof['before'] != result['revision_job']['source_hashes']:
        raise ValueError('Missing matching before/after source and data checks')
    if proof['acceptance_sha256'] != digest(out / 'acceptance.json'):
        raise ValueError('Acceptance receipt changed')
    if result['data_contract']['root_clean_rows'] != 16277:
        raise ValueError('Root row count mismatch')
    return result
