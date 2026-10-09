"""Strictly accept and archive one already-ended authorized v4 useful batch."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

BASE = Path('/workspace/guardfed_checks/celeba_final_valid_replay_20261009')
V4 = BASE / 'v4'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text())

def save(p, value):
    assert not p.exists(), str(p)
    p.write_text(json.dumps(value, indent=2) + '\n')

phase = int(sys.argv[1])
assert phase in (4, 5)
prefix = 'phase' + str(phase) + '_attempt1'
stage, batch = V4 / (prefix + '_execution_20261009'), V4 / (prefix + '_useful')
program = 'guardfed_celeba_valid_v4_phase' + str(phase) + '_attempt1_20261009'
execution = read(batch / 'batch_execution.json')
assert not execution['failures'] and not execution['stopped_after_failure'] and execution['source_unchanged'] and execution['identity_error'] is None
plan = read(BASE / 'v3/throughput_plan.json')['phases'][phase - 1]
ids = [r['id'] for r in plan['models']]
assert execution['requested_ids'] == ids and set(execution['finished_zero_exit_ids']) == set(ids) and len(ids) == plan['workers']
assert sha(V4 / 'replay_v4.py') == '43b16d20d2497b7762cd0f6039f7f4bfc8fddd4e4c32979a16291b26e394ae5e'
assert sha(V4 / 'semantic900_inspection.json') == 'cc93d94d69478a4ee190abff76da4fb185791ece7e4e1d2ee75360482ead80b4'
os.setpriority(os.PRIO_PROCESS, 0, 10)
subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
root = Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
queue = read(root / 'formal_queue_progress.json')
impact = {'unix': time.time(), 'training_queue': queue, 'active_rounds': [], 'replay_processes': []}
for item in queue.get('active', []):
    p = root / 'runs' / item['id'] / 'progress.json'
    impact['active_rounds'].append(dict(item, progress=read(p) if p.exists() else None))
for key, command in [('service', ['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal', program]), ('gpu', ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,temperature.gpu', '--format=csv,noheader'])]:
    p = subprocess.run(command, capture_output=True, text=True, timeout=15)
    impact[key] = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
assert not queue['failed'] and 'RUNNING' in impact['service']['stdout']
save(stage / 'impact_after.json', impact)
a = shlex.split((stage / ('run_phase' + str(phase) + '.sh')).read_text().splitlines()[-1])[1:]
command = a[:a.index('--ids')]
command[2] = 'accept'
command += ['--semantic-inspection', str(V4 / 'semantic900_inspection.json'), '--semantic-inspection-sha256', 'cc93d94d69478a4ee190abff76da4fb185791ece7e4e1d2ee75360482ead80b4', '--batch', str(batch), '--output', str(stage / 'strict_acceptance.json')]
with (stage / 'strict_acceptance.log').open('x') as f:
    p = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
assert p.returncode == 0, 'Strict acceptance failed; preserve files and stop'
accepted = read(stage / 'strict_acceptance.json')
assert accepted['scope'] == 'VALID_ONLY_IMPLEMENTATION_REPLAY_V4' and accepted['accepted_ids'] == ids and accepted['accepted_n'] == len(ids) and not accepted['invalid']
assert accepted['max_abs_native_metric_difference'] <= 1e-12 and not accepted['all900_native_valid_replayed']
diagnostic = read(stage / 'resource_diagnostic.json')
assert diagnostic['status'] == 'FINITE_BOUND_ALLOCATED_WORKERS_SAMPLE_COMPLETE'
assert sha(Path('/etc/supervisor/conf.d') / (program + '.conf')) == sha(stage / (program + '.conf'))
files = {prefix + '_useful/' + n: batch / n for n in ('batch_inputs.json', 'batch_execution.json')}
for i in ids:
    for suffix in (i + '.log', i + '.worker.json', i + '/receipt.json', i + '/validation_predictions.npz'):
        files[prefix + '_useful/runs/' + suffix] = batch / 'runs' / suffix
stage_names = ['run_phase' + str(phase) + '.sh', program + '.conf', 'execution_contract.json', 'impact_during.json', 'impact_after.json', 'supervisor.log', 'strict_acceptance.json', 'strict_acceptance.log', 'resource_diagnostic.json', 'capture_during.py']
files.update({prefix + '_execution_20261009/' + n: stage / n for n in stage_names})
for n, p in {'replay_v4.py': V4 / 'replay_v4.py', 'replay_v3.py': BASE / 'v3/replay_v3.py', 'replay.py': BASE / 'replay.py', 'evaluator.py': BASE / 'inputs/evaluator.py', 'reproduce_paper_tables.py': Path('/workspace/GuardFed-celeba-expanded/scripts/reproduce_paper_tables.py'), 'run_revision_ablation.py': Path('/workspace/GuardFed-celeba-expanded/scripts/run_revision_ablation.py'), 'celeba_data.py': Path('/workspace/GuardFed-celeba-expanded/src/celeba_data.py')}.items():
    files['sealed_sources/' + n] = p
files['frozen_inputs/semantic900_inspection.json'] = V4 / 'semantic900_inspection.json'
for n in ('authorization.json', 'sample_bound_workers.py', 'close_stage.py'):
    files['execution_sources/' + n] = V4 / 'execution_20261009' / n
files['execution_sources/original_sample_worker_resources.py'] = BASE / 'v3/throughput_execution_20261009/sample_worker_resources.py'
identities = {n: {'sha256': sha(p), 'bytes': p.stat().st_size} for n, p in files.items()}
assert len(identities) == 24 + 4 * len(ids)
archive = stage / (prefix + '_receipts_20261009.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tf:
    for n, p in sorted(files.items()):
        tf.add(p, arcname=n, recursive=False)
with tarfile.open(archive, 'r:gz') as tf:
    entries = tf.getmembers()
    assert len(entries) == len({e.name for e in entries}) == len(identities)
    for entry in entries:
        assert entry.isfile() and entry.size == identities[entry.name]['bytes']
        assert hashlib.sha256(tf.extractfile(entry).read()).hexdigest() == identities[entry.name]['sha256']
assert all(sha(p) == identities[n]['sha256'] for n, p in files.items())
receipt = {'status': 'ARCHIVE_CREATED_AND_ALL_MEMBER_SHA_VERIFIED', 'phase': phase, 'archive': archive.name, 'sha256': sha(archive), 'bytes': archive.stat().st_size, 'member_n': len(identities), 'members': identities, prefix + '_strict_acceptance_sha256': sha(stage / 'strict_acceptance.json'), 'new_image_replays': len(ids), 'test_images_inferred': 0, 'new_training': 0, 'original_source_or_models_modified': 0, 'ownership_rule': 'Explicit own stage/batch/input/source whitelist; no parent-owned new proof files'}
save(stage / 'remote_archive_inventory.json', receipt)
print(json.dumps({k: receipt[k] for k in ('status', 'phase', 'sha256', 'bytes', 'member_n')}, indent=2))
