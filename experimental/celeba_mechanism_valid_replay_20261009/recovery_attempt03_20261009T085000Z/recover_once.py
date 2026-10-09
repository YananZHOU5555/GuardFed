"""Exact root-approved runtime-history move and one original supervisor start."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess

BASE = Path('/workspace/guardfed_checks/celeba_mechanism_valid_replay_20261009')
WORK = BASE / 'execution_attachments_v2'
PROV = BASE / 'recovery_attempt03_20261009T085000Z'


def save(name, value):
    with (PROV / name).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    os.chdir(WORK)  # original bridge identities include the relative approval path
    spec = importlib.util.spec_from_file_location('sealed_execute_v2', WORK / 'execute_one_v2.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    approval = m.read(WORK / 'dispatch_receipt.APPROVED.json')
    m.source_checks(approval)
    assert m.digest(BASE / 'engineering_recovery_approved.json') == '9a124c3c950e9f7d258e7b303002f95be8db99621033d475b4612dc03ff4efb9'
    planpath = BASE / 'runtime_recovery_plan.APPROVED_SCOPE.json'
    assert m.digest(planpath) == 'afd20ebf4ca32b91ef1a35c2d1500371100860bb1c428c8b55c6429a295fb727'
    approval_sha = m.digest(WORK / 'dispatch_receipt.APPROVED.json')
    assert approval_sha == '34fc8399e0cc8271c95001289e9b05a1412bbdcb9ed3894bc43fa38e27969162'
    assert (WORK / 'dispatch_receipt.APPROVED.sha256').read_bytes() == approval_sha.encode() + b'\n'
    phase5 = subprocess.run(['supervisorctl', 'status', 'guardfed_celeba_valid_v4_phase5_attempt1_20261009'], capture_output=True, text=True).stdout
    assert 'EXITED' in phase5
    resources = m.resource_snapshot()
    assert not any(row['role'] == 'baseline_valid_worker' for row in resources['tracked_compute'])
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [b.decode(errors='replace') for b in (proc / 'cmdline').read_bytes().split(b'\0') if b]
            if argv and 'python' in Path(argv[0]).name:
                cpus = set(os.sched_getaffinity(int(proc.name)))
                assert not (len(cpus) == 8 and cpus.intersection(m.v1.CPU_IDS)), 'CPU112..119 occupied'
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    failure = m.read(WORK / 'runs/minus_U_IID_Benign_seed91002.bridge_failure.json')

    def identities(pins):
        actual = {}
        for name in pins:
            p = Path(name)
            stat = p.stat()
            resolved = str(p.resolve())
            actual[name] = {'sha256': m.digest(p), 'resolved_path': resolved, 'bytes': stat.st_size,
                            'stat': {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                     'inode': stat.st_ino, 'device': stat.st_dev, 'resolved_path': resolved}}
        return actual

    sources, artifacts = identities(failure['source_before']), identities(failure['artifact_before'])
    assert sources == failure['source_before'] == failure['source_after']
    assert artifacts == failure['artifact_before'] == failure['artifact_after']
    plan = m.read(planpath)
    history = Path(plan['planned_history'])
    assert history == WORK / 'history/failed_attempt_02_20261009' and not history.exists()
    assert len(plan['planned_files']) == 4
    for row in plan['planned_files']:
        source, target = Path(row['source']), Path(row['destination'])
        assert source.resolve().is_relative_to(WORK.resolve()) and target.is_relative_to(history)
        assert m.digest(source) == row['sha256'] and source.stat().st_size == row['bytes'] and not target.exists()
    for relative in ['strict_acceptance.json', 'resource_after_approved_run.json', 'runs/minus_U_IID_Benign_seed91002']:
        assert not (WORK / relative).exists()
    save('pre_move_identity_resource.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'phase5_service': phase5, 'all_phase5_workers_absent': True, 'CPU112_119_free': True,
         'resources': resources, 'source_hashes': sources, 'artifact_hashes': artifacts,
         'all_three_seals_valid': True, 'approved_receipt_sha256': approval_sha,
         'preflight_helper_corrections': 'Earlier checks stopped without mutation: relative approval path required original cwd; full_hashes records are nested SHA/stat objects, not strings.'})
    for row in plan['planned_files']:
        source, target = Path(row['source']), Path(row['destination'])
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        assert not source.exists() and m.digest(target) == row['sha256'] and target.stat().st_size == row['bytes']
    assert (WORK / 'runs').is_dir() and not any((WORK / 'runs').iterdir())
    for relative in ['resource_before_approved_run.json', 'strict_acceptance.json', 'resource_after_approved_run.json',
                     'execution_failure.json', 'run.log', 'runs/minus_U_IID_Benign_seed91002']:
        assert not (WORK / relative).exists()
    m.source_checks(approval)
    assert m.digest(WORK / 'dispatch_receipt.APPROVED.json') == approval_sha
    save('history_recovery_chain.json', {'status': 'EXACT4_RUNTIME_HISTORY_MOVED_ORIGINAL_BYTES_PRESERVED',
         'mapping': plan['planned_files'], 'original_archive_sha256': plan['fault_backup_archive_SHA'],
         'runs_empty': True, 'target_output_absent': True, 'all_three_seals_and_APPROVED_unchanged': True,
         'no_scientific_source_changes': True, 'no_failure_bytes_deleted': True})
    resources = m.resource_snapshot()
    result = subprocess.run(['supervisorctl', 'start', 'guardfed_celeba_mechanism_valid_one'], capture_output=True, text=True)
    status = subprocess.run(['supervisorctl', 'status', 'guardfed_celeba_mechanism_valid_one'], capture_output=True, text=True).stdout
    launch = {'status': 'EXACT_ENGINEERING_RECOVERY_THIRD_START_ONCE',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'returncode': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr, 'service': status, 'resources': resources,
              'no_additional_retry': True, 'scientific_acceptance_not_yet_observed': True}
    save('launch_proof.json', launch)
    print(json.dumps({'status': launch['status'], 'service': status, 'returncode': result.returncode,
          'source_pins': len(sources), 'artifact_pins': len(artifacts),
          'nominal_compute_threads_with_this8': resources['nominal_compute_threads_including_this8']}))


if __name__ == '__main__':
    main()
