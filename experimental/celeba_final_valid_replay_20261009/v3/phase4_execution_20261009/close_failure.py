"""Preserve the failed phase4 attempt; never dispatch inference or change source."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

BASE = Path('/workspace/guardfed_checks/celeba_final_valid_replay_20261009')
V3 = BASE / 'v3'
STAGE = V3 / 'phase4_execution_20261009'
BATCH = V3 / 'phase4_useful'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(p.read_text())

def save(p, data):
    assert not p.exists(), str(p)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[16:24])
os.nice(10)
subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
assert sha(V3 / 'replay_v3.py') == 'abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e'
assert sha(BASE / 'replay.py') == '8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803'
execution, inputs = read(BATCH / 'batch_execution.json'), read(BATCH / 'batch_inputs.json')
assert execution['source_unchanged'] and execution['stopped_after_failure']
assert execution['finished_zero_exit_ids'] == [] and len(execution['failures']) == 8
assert execution['requested_ids'] == inputs['selected_ids']
assert not any((BATCH / 'runs').glob('*/receipt.json'))
args = shlex.split((STAGE / 'run_phase4.sh').read_text().splitlines()[-1])[1:]
args[2] = 'accept'
args = args[:args.index('--ids')] + ['--batch', str(BATCH), '--output', str(STAGE / 'strict_acceptance.json')]
assert not (STAGE / 'strict_acceptance.json').exists()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
with (STAGE / 'strict_acceptance.log').open('x') as f:
    result = subprocess.run(args, env=env, stdout=f, stderr=subprocess.STDOUT)
strict = read(STAGE / 'strict_acceptance.json')
assert strict['status'] == 'PARTIAL_OR_INVALID_VALID_REPLAY' and strict['accepted_n'] == 0 and len(strict['invalid']) == 8
assert strict['max_abs_native_metric_difference'] is None and not strict['all900_native_valid_replayed']
workers = {i: read(BATCH / 'runs' / (i + '.worker.json')) for i in inputs['selected_ids']}
assert all(w['status'] == 'FAILED' and w['artifacts_unchanged'] and w['artifact_before'] == w['artifact_after'] for w in workers.values())
fedaa = next(i for i in workers if i.startswith('FedAA-'))
assert workers[fedaa]['error_type'] == 'KeyError' and workers[fedaa]['error'] == "'output'"
summary = {'status': 'FAILED_ATTEMPT_PRESERVED_NOT_ACCEPTED', 'phase': 4, 'accepted_n': 0, 'root_cause_id': fedaa, 'root_cause': workers[fedaa]['error'], 'root_cause_traceback': workers[fedaa]['traceback'], 'other_workers_interrupted': 7, 'source_unchanged': True, 'artifact_files_unchanged': 24, 'strict_returncode': result.returncode, 'phase5_started': False, 'automatic_retry': False, 'prior_accepted_replays_still_valid': 9, 'failure_is_not_a_native_metric_mismatch': True}
save(STAGE / 'failure_summary.json', summary)
save(STAGE / 'failure_measurement.json', {'status': 'REJECTED_INCOMPLETE_NOT_A_VALID_THROUGHPUT_POINT', 'workers': 8, 'wall_seconds': execution['wall_seconds'], 'accepted_n': 0, 'throughput_curve_inclusion': False, 'root_cause': 'Sealed validator assumed rawjob.output for FedAA', 'resource_diagnostic_scope': '15 second startup/import sample; does not measure steady-state CNN utilization', 'no900_bulk': True, 'phase5_not_started': True})
stage_names = ['execution_contract.json', 'run_phase4.sh', 'guardfed_celeba_valid_phase4_20261009.conf', 'capture_after.py', 'close_failure.py', 'supervisor.log', 'impact_during.json', 'impact_after.json', 'resource_diagnostic.json', 'resource_diagnostic.log', 'strict_acceptance.json', 'strict_acceptance.log', 'failure_summary.json', 'failure_measurement.json']
files = {'phase4_execution_20261009/' + n: STAGE / n for n in stage_names}
files.update({'phase4_useful/' + p.relative_to(BATCH).as_posix(): p for p in BATCH.rglob('*') if p.is_file()})
for n, p in {'replay_v3.py': V3 / 'replay_v3.py', 'replay.py': BASE / 'replay.py', 'evaluator.py': BASE / 'inputs/evaluator.py', 'reproduce_paper_tables.py': Path('/workspace/GuardFed-celeba-expanded/scripts/reproduce_paper_tables.py'), 'run_revision_ablation.py': Path('/workspace/GuardFed-celeba-expanded/scripts/run_revision_ablation.py'), 'celeba_data.py': Path('/workspace/GuardFed-celeba-expanded/src/celeba_data.py')}.items():
    files['sealed_sources/' + n] = p
identities = {n: {'sha256': sha(p), 'bytes': p.stat().st_size} for n, p in files.items()}
archive = STAGE / 'phase4_failed_attempt_evidence_20261009.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tf:
    for n, p in sorted(files.items()):
        tf.add(p, arcname=n, recursive=False)
with tarfile.open(archive, 'r:gz') as tf:
    members = tf.getmembers()
    assert len(members) == len({m.name for m in members}) == len(identities)
    for m in members:
        assert m.isfile() and m.name in identities and m.size == identities[m.name]['bytes']
        assert hashlib.sha256(tf.extractfile(m).read()).hexdigest() == identities[m.name]['sha256']
assert all(sha(p) == identities[n]['sha256'] for n, p in files.items())
receipt = {'status': 'FAILED_ATTEMPT_ARCHIVE_ALL_MEMBERS_VERIFIED', 'archive': archive.name, 'sha256': sha(archive), 'bytes': archive.stat().st_size, 'member_n': len(identities), 'members': identities, 'accepted_n': 0, 'is_successful_replay_archive': False, 'strict_acceptance_sha256': sha(STAGE / 'strict_acceptance.json'), 'ownership': 'Only explicitly owned stage files, failed batch files and read-only sealed source copies'}
save(STAGE / 'failure_archive_inventory.json', receipt)
print(json.dumps({k: receipt[k] for k in ('status', 'sha256', 'bytes', 'member_n')}, indent=2))
