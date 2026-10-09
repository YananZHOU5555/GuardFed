"""Acceptance-tool boundary tests; no scientific results or training."""
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('mechanism_v3_tested', HERE / 'evidence_v4.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reject(call):
    try:
        call()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Expected rejection')


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    repo, adapter, output = root / 'repo', root / 'adapter', root / 'output'
    output.mkdir()
    job_file = root / 'job.json'; job_file.write_text('{}')
    entry = {'id': 'minus_U_IID_Benign_seed91001', 'job': str(job_file), 'output': str(output)}
    job = {'output': str(output)}
    proc = root / 'proc' / '123'; proc.mkdir(parents=True)
    argv = ['/python', str(adapter / 'worker.py'), '--repo', str(repo), '--job', str(job_file)]
    (proc / 'cmdline').write_bytes(b'\0'.join(x.encode() for x in argv) + b'\0')
    (proc / 'stat').write_text('123 (python worker) ' + ' '.join(['S'] + ['0'] * 18 + ['98765']))
    owner = module.live_worker(entry, repo, adapter, root / 'proc')
    assert owner['pid'] == 123 and owner['start_ticks'] == 98765
    (output / 'progress.json').write_text(json.dumps({'job_id': entry['id'], 'pid': 123, 'round': 8}))
    with patch.object(module, 'live_worker', return_value=owner):
        assert module.active_pending(entry, repo, adapter)['observed_round'] == 8
        (output / 'failure.json').write_text('{"error":"numeric failure"}')
        reject(lambda: module.active_pending(entry, repo, adapter))
        (output / 'failure.json').unlink()
        (output / 'progress.json').write_text(json.dumps({'job_id': entry['id'], 'pid': 124, 'round': 8}))
        reject(lambda: module.active_pending(entry, repo, adapter))
        pending, error = module.recheck_pending(entry, repo, adapter, ValueError('first wrong PID'))
        assert pending is None and 'first wrong PID' in str(error) and 'Live progress identity differs' in str(error)
        snapshot, _, _ = module.preserve_invalid_snapshot(root / 'recheck_inspection', entry, {str(output / 'progress.json'): module.digest(output / 'progress.json')})
        assert len(snapshot) == 1
    class PendingWorker:
        @staticmethod
        def checked(*unused):
            return None
    reject(lambda: module.accept_new(None, PendingWorker(), job,
                                    dict(entry, job_sha256=module.digest(job_file)), {}, None))
    (proc / 'cmdline').write_bytes(b'\0'.join(x.encode() for x in argv[:-1] + ['/other.json']) + b'\0')
    assert module.live_worker(entry, repo, adapter, root / 'proc') is None
    log = root / 'invalid.log'; log.write_text('failed\n')
    initial = module.digest(log)
    snapshots, identities, observations = module.preserve_invalid_snapshot(root / 'inspection', entry, {str(log): initial})
    log.write_text('failed\nmore diagnostic\n')
    assert all(module.digest(path) == expected for path, expected in snapshots.items())
    assert list(identities.values()) == [str(log) + '=' + initial]
    assert not observations[0]['changed_during_snapshot']

print(json.dumps({'status': 'PASS_TOOL_BOUNDARIES_ONLY', 'groups': 6,
                  'checks': ['exact_live_owner', 'live_partial_pending', 'live_failure_rejected',
                             'wrong_progress_pid_rejected', 'orphan_partial_rejected',
                             'wrong_job_process_rejected', 'diagnostic_snapshot_immutable', 'wrong_pid_recheck_preserved_invalid'],
                  'scientific_runs': 0, 'v2_complete_result_and_backup_guards': 'unchanged'}))
