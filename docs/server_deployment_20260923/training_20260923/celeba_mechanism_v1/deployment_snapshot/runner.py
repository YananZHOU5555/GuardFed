"""Bounded fail-stop execution, gated by actual image regressions and frozen identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def live_identity(repo, stage, manifest):
    assert sys.platform == 'linux', 'Real dispatch requires the authorized Linux GPU host'
    assert str(repo) == '/workspace/GuardFed-celeba-expanded'
    for rel, expected in manifest['source_hashes'].items():
        assert digest(repo / rel) == expected, ('source/data drift', rel)
    for path, expected in manifest['adapter_hashes'].items():
        assert digest(path) == expected, ('adapter drift', path)
    assert digest(stage / 'PROTOCOL.md') == manifest['protocol_sha256']
    assert len(manifest['jobs']) == 800 and len(manifest['reused_full']) == 100
    all_entries = manifest['jobs'] + manifest['preflight_jobs'] + manifest['reference_jobs']
    assert len({entry['id'] for entry in all_entries}) == 820
    for entry in all_entries:
        assert digest(entry['job']) == entry['job_sha256'], ('job drift', entry['id'])
        job = read(entry['job'])
        assert job['id'] == entry['id'] and job['output'] == entry['output']
        assert job['source_hashes'] == manifest['source_hashes']
        assert job['adapter_hashes'] == manifest['adapter_hashes']
    # Never overlap another project queue. A concurrent copy of this runner is
    # also prevented by the OS lock held throughout the queue.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(repo).encode() in cmd and any(x in cmd for x in (b'worker.py', b'run_revision_ablation.py')):
            raise RuntimeError('Existing training worker: ' + proc.name)


def modules(repo):
    from worker import load, checked
    sys.path.insert(0, str(repo / 'scripts')); sys.path.insert(0, str(repo))
    return load('mechanism_queue_original', repo / 'scripts/run_revision_ablation.py'), checked


def freeze(repo, stage, manifest):
    import torch
    original, checked = modules(repo)
    gates = {}; gate_identities = []
    for entry in manifest['preflight_jobs']:
        job = read(entry['job']); result = checked(original, job, Path(entry['job']))
        assert result is not None, ('missing image gate', entry['id'])
        gates[entry['id']] = result
        gate_identities.append({'id': entry['id'], 'output': entry['output'],
                                'result_sha256': digest(Path(entry['output']) / 'result.json'),
                                'checkpoint_sha256': result['revision_job']['checkpoint_sha256'],
                                'acceptance_sha256': digest(Path(entry['output']) / 'mechanism_acceptance.json')})
    comparisons = []
    for entry in manifest['reference_jobs']:
        job = read(entry['job']); reference = original.checked_result(job)
        assert reference is not None and not list(Path(job['output']).glob('failure*.json'))
        gate = gates[entry['reference_to']]
        assert reference['config'] == gate['config']
        for key in ('metrics', 'trajectory_metrics', 'round_summaries', 'data_contract'):
            assert reference[key] == gate[key], ('Full same-horizon regression', key)
        left = torch.load(Path(job['output']) / 'model.pt', map_location='cpu', weights_only=True)
        right = torch.load(Path(gate['revision_job']['output']) / 'model.pt', map_location='cpu', weights_only=True)
        assert left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)
        comparisons.append({'reference': entry['id'], 'gate': entry['reference_to'], 'tensor_and_diagnostics_exact': True})
    reused = []
    template = read(manifest['jobs'][0]['job'])['config']
    ignore = {'seed', 'client_alpha', 'ablation_component', 'experiment_suite', 'experiment_tag', 'full_round_diagnostics'}
    for entry in manifest['reused_full']:
        out = Path(entry['output']); result = read(out / 'result.json')
        assert digest(out / 'model.pt') == entry['checkpoint_sha256']
        assert result['revision_job']['checkpoint_sha256'] == entry['checkpoint_sha256']
        assert result['method'] == entry['method'] and result['distribution'] == entry['distribution']
        assert result['attack'] == entry['attack'] and result['seed'] == entry['seed']
        assert result['alpha'] == manifest['distributions'][entry['distribution']]
        assert [r['round'] for r in result['trajectory_metrics']] == list(range(1, 71))
        assert result['metrics'] == result['trajectory_metrics'][-1]['metrics']
        assert all(result['metrics'][k] == entry[k] for k in ('accuracy', 'aeod', 'aspd'))
        assert {k:v for k,v in result['config'].items() if k not in ignore} == {k:v for k,v in template.items() if k not in ignore}
        for rel in ('scripts/reproduce_paper_tables.py', 'src/celeba_data.py', 'data/celeba/derived/rgb64_v1/images.npy'):
            assert result['revision_job']['source_hashes'][rel] == manifest['source_hashes'][rel]
        reused.append({'id': entry['id'], 'output': str(out), 'result_sha256': digest(out / 'result.json'), 'checkpoint_sha256': entry['checkpoint_sha256']})
    guide = Path('/etc/vast-agents-guide.md')
    receipt = {'pass': True, 'checked_at_unix': time.time(), 'manifest_sha256': digest(stage / 'manifest.json'),
               'image_gates': len(gates), 'gate_identities': gate_identities, 'full_regressions': comparisons, 'reused_full': reused,
               'guide_sha256': digest(guide), 'torch_version': torch.__version__, 'cuda_build': torch.version.cuda,
               'gpu_snapshot': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True),
               'new_training_started': False}
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    assert not (stage / 'dispatch_receipt.json').exists(), 'Preserve existing freeze receipt'
    save(stage / 'dispatch_receipt.json', receipt)
    return receipt


def queue(repo, stage, manifest, phase):
    import fcntl
    original, checked = modules(repo)
    lock = (stage / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if phase == 'formal':
        import torch
        receipt = read(stage / 'dispatch_receipt.json')
        assert receipt['pass'] and receipt['manifest_sha256'] == digest(stage / 'manifest.json')
        assert receipt['image_gates'] == 18 and len(receipt['reused_full']) == 100
        assert receipt['torch_version'] == torch.__version__ and receipt['cuda_build'] == torch.version.cuda
        assert receipt['guide_sha256'] == digest('/etc/vast-agents-guide.md')
        for entry in receipt['gate_identities'] + receipt['reused_full']:
            out = Path(entry['output'])
            assert digest(out / 'result.json') == entry['result_sha256']
            assert digest(out / 'model.pt') == entry['checkpoint_sha256']
            if 'acceptance_sha256' in entry:
                assert digest(out / 'mechanism_acceptance.json') == entry['acceptance_sha256']
        entries = manifest['jobs']
    else:
        entries = manifest['reference_jobs'] + manifest['preflight_jobs']
    pending = []; complete = []
    for entry in entries:
        job = read(entry['job']); out = Path(job['output'])
        if 'reference_to' in job:
            assert not list(out.glob('failure*.json'))
            accepted = original.checked_result(job)
        else:
            accepted = checked(original, job, Path(entry['job']))
        if accepted is not None:
            complete.append(entry['id'])
        else:
            assert not out.exists(), ('Partial output requires review', str(out))
            pending.append((entry, job))
    logs = stage / ('logs' if phase == 'formal' else 'preflight/logs'); logs.mkdir(parents=True, exist_ok=True)
    slots = list(range(8)); active = {}; failures = []
    while pending or active:
        while pending and slots and not failures:
            slot = slots.pop(0); entry, job = pending.pop(0)
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       GUARDFED_CPU_THREADS='1', CUDA_VISIBLE_DEVICES=str(slot % 2), PYTHONUNBUFFERED='1')
            log = (logs / (entry['id'] + '.log')).open('ab')
            cmd = [sys.executable, str(repo / 'scripts/run_revision_ablation.py'), 'worker', '--job', entry['job']] if 'reference_to' in job else [sys.executable, str(Path(__file__).with_name('worker.py')), '--repo', str(repo), '--job', entry['job']]
            proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[slot] = (proc, log, entry, job)
        for slot, (proc, log, entry, job) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close(); del active[slot]; slots.append(slot)
            try:
                assert code == 0, ('worker exit', code)
                result = original.checked_result(job) if 'reference_to' in job else checked(original, job, Path(entry['job']))
                assert result is not None
                complete.append(entry['id'])
            except Exception as error:
                failure = {'id': entry['id'], 'exit_code': code, 'error': repr(error), 'at_unix': time.time()}
                failures.append(failure); save(Path(job['output']) / 'failure.json', failure)
        save(stage / (phase + '_queue_progress.json'), {'phase': phase, 'completed': complete, 'failed': failures,
             'active': [{'id': item[2]['id'], 'pid': item[0].pid} for item in active.values()],
             'pending': len(pending), 'updated_unix': time.time()})
        if failures and not active:
            raise RuntimeError('Fail-stop: preserve failed/partial outputs and review before recovery')
        if pending or active:
            time.sleep(5)
    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['preflight', 'freeze', 'run'])
    parser.add_argument('--repo', type=Path, required=True); parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args(); repo = args.repo.resolve(); stage = args.stage.resolve()
    assert not sys.flags.optimize, 'Assertions must remain enabled'
    manifest = read(stage / 'manifest.json'); live_identity(repo, stage, manifest)
    if args.action == 'freeze':
        print(json.dumps(freeze(repo, stage, manifest)))
    else:
        queue(repo, stage, manifest, 'preflight' if args.action == 'preflight' else 'formal')
