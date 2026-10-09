"""Authorized read-only identity/resource checks, then exact external approval creation."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
REPAIR = STAGE / 'execution_repair_v1'
CPUS = list(range(8, 16))
REPAIR_SEAL = '54c67847b44a26a654b380c6c4d03863279261126a61de29489bfa78572e332b'
REPAIR_SCOPE = '343993408bfb3fbefc8bc157f95034a7f45a43456f6b8145242d58c5860a773c'
GUIDE_SHA = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
VERIFIER_SHA = '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
RESOURCE_SOURCE_SHA = '5d537f129c6d96f370bbd93b321e86be1e1fef628ba9589485ac9a14486c00c0'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    assert not path.exists(), 'Never overwrite a preflight/approval/launch record'
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def load(name, path, sha):
    assert digest(path) == sha, 'Sealed dependency source changed'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def resources():
    dependency = STAGE.parent / 'celeba_mechanism_valid_replay_20261009/execution_attachments_v2/execute_one_v2.py'
    module = load('sealed_v2_hybrid_resource_only', dependency, RESOURCE_SOURCE_SHA)
    proxy = types.SimpleNamespace(CPU_IDS=CPUS, check_budget=module.v1.check_budget)
    reservation = types.FunctionType(module.reservations.__code__, dict(module.reservations.__globals__, v1=proxy))
    resource_snapshot = types.FunctionType(module.resource_snapshot.__code__,
        dict(module.resource_snapshot.__globals__, reservations=reservation, v1=proxy))
    report = resource_snapshot()
    assert 'RUNNING' in report['formal_service']
    # Catch any other tightly pinned Python main process sharing the reserved set,
    # including a duplicate repaired task not present in historical classifiers.
    unexpected = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [s.decode(errors='replace') for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
            entry = module.python_entry(argv)
            if entry is None or (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                continue
            cpus = sorted(os.sched_getaffinity(int(proc.name)))
            role = module.classify(argv, str((proc / 'cwd').resolve()))
            if set(cpus).intersection(CPUS) and len(cpus) <= 16 and role is None:
                unexpected.append(dict(pid=int(proc.name), argv=argv, cpus=cpus))
            assert not any('repair_execute.py' in arg for arg in argv), 'Duplicate repaired main process'
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            continue
    assert not unexpected, ('Unknown restricted Python shares reserved CPUs', unexpected)
    report.update(exclusive_cpu_ids=CPUS, other_restricted_compute_overlap=[],
        classifier_source_sha256=RESOURCE_SOURCE_SHA, source_classifier_body_unchanged=True)
    return report


def main():
    assert digest('/etc/vast-agents-guide.md') == GUIDE_SHA
    assert digest(REPAIR / 'FILES_SHA256.json') == REPAIR_SEAL and digest(REPAIR / 'REPAIR_SCOPE.json') == REPAIR_SCOPE
    assert not (REPAIR / 'runtime_overlay').exists() and not (HERE / 'APPROVED.json').exists()
    import sys
    sys.path.insert(0, str(REPAIR))
    import repair_execute
    gate, scope, original = repair_execute.inspect()
    gate.verify_scope(original)  # Entire frozen protected source/data and scientific job identity.
    references = gate.read(REPAIR / 'reused_IID_references.json')['records']
    for row in references:
        output = STAGE / row['output']
        assert digest(output / 'acceptance.json') == row['acceptance_sha256']
        for name, sha in row['artifact_hashes'].items():
            assert digest(output / name) == sha
    backup = STAGE / 'terminal_failure_backup_20261009'
    verifier = load('sealed_v4_failure_chain_verifier', HERE / 'sealed_archive_verifier_v4.py', VERIFIER_SHA)
    failure = verifier.verify_archive(backup / 'hybrid_terminal_failure_evidence.tar.gz', read(backup / 'backup_receipt.json'))
    offserver = read(HERE / 'original_failure_offserver_verification.json')
    assert offserver['pass'] and offserver['different_host_observed'] and offserver['members_verified'] == 46
    assert offserver['archive_sha256'] == failure['archive_sha256'] == scope['original_failed_attempt_archive_sha256']
    fresh = resources()
    assert set(CPUS) <= os.sched_getaffinity(0)
    snapshot = gate.snapshot()
    assert not snapshot['failed'] and len(snapshot['active']) == 8
    evidence = dict(status='PRELAUNCH_IDENTITY_RESOURCE_FAILURE_CHAIN_PASS', at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protected_source_data_hashes=original['protected_source_hashes'], reused_IID=references,
        original_failure_archive_members_verified=failure['members_verified'], original_failure_offserver=offserver,
        fresh_resources=fresh, formal_progress=snapshot, no_training_or_inference_by_preflight=True)
    save(HERE / 'preflight.json', evidence)
    launch_files = {path.name: digest(path) for path in HERE.iterdir() if path.is_file() and path.name not in {'APPROVED.json', 'APPROVED.sha256'}}
    approved = dict(status='APPROVED_TWO_UNACCEPTED_HYBRID_CANARIES_DIAGNOSTIC_WRITER_ONLY',
        authority='Root explicitly reviewed sealed9 writer/scope and authorized engineering recovery, 2026-10-09 in current agent task',
        repair_scope_sha256=REPAIR_SCOPE, repair_seal_sha256=REPAIR_SEAL,
        jobs={entry['id']: entry['job_sha256'] for entry in scope['new_jobs']}, exclusive_cpu_ids=CPUS, cpu_threads=8,
        no_live_compute_overlap_verified=True, formal32_authorized=False, test_authorized=False,
        automatic_retry_authorized=False, source_data_hashes=original['protected_source_hashes'],
        launch_attachment_hashes=launch_files, preflight_sha256=digest(HERE / 'preflight.json'),
        original_failed_attempt_status='TERMINAL_FAILURE', reused_IID_repeated_training_or_inference=False)
    save(HERE / 'APPROVED.json', approved)
    (HERE / 'APPROVED.sha256').write_bytes((digest(HERE / 'APPROVED.json') + '\n').encode('ascii'))
    repair_execute.validate_approval(gate, scope, HERE / 'APPROVED.json')
    print(json.dumps(dict(status='APPROVED_AND_PREFLIGHT_PASS_NOT_STARTED', approval_sha256=digest(HERE / 'APPROVED.json'),
        nominal_cpu_threads_including_repair=fresh['nominal_compute_threads_including_this8'],
        quota=fresh['actual_quota_cores'], formal_completed=snapshot['completed'])), flush=True)


if __name__ == '__main__':
    main()
