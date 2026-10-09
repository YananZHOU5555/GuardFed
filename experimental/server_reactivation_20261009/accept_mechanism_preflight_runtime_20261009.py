"""Accept and preserve the completed pipeline gate; never launches training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

REPO = Path('/workspace/GuardFed-celeba-expanded')
STAGE = REPO / 'results/revision_20261009/celeba_mechanism_v1'
CHECKS = Path('/workspace/guardfed_checks/server_reactivation_20261009')
sys.path.insert(0, str(REPO / 'deployment/celeba_mechanism_20261009'))
import runner
import torch

assert not sys.flags.optimize
tag = sys.argv[1]
assert tag in ('cu130', 'cu128')
assert torch.__version__ == '2.11.0+' + tag
manifest = runner.read(STAGE / 'manifest.json')
runner.live_identity(REPO, STAGE, manifest)
progress = runner.read(STAGE / 'preflight_queue_progress.json')
entries = manifest['preflight_jobs'] + manifest['reference_jobs']
assert len(progress['completed']) == 20 and len(set(progress['completed'])) == 20
assert set(progress['completed']) == {e['id'] for e in entries}
assert not progress['failed'] and not progress['active'] and progress['pending'] == 0
original, checked = runner.modules(REPO)
results = {}
for entry in entries:
    job = runner.read(entry['job'])
    assert job['config']['rounds'] == 3
    result = original.checked_result(job) if 'reference_to' in job else checked(original, job, Path(entry['job']))
    assert result is not None
    assert result['revision_job']['torch_version'] == torch.__version__, ('gate runtime mismatch', entry['id'])
    assert [r['round'] for r in result['trajectory_metrics']] == [1, 2, 3]
    assert result['metrics'] == result['trajectory_metrics'][-1]['metrics']
    assert not list(Path(entry['output']).glob('failure*.json'))
    results[entry['id']] = result
comparisons = []
for entry in manifest['reference_jobs']:
    ref = results[entry['id']]
    gate = results[entry['reference_to']]
    for key in ('config', 'metrics', 'trajectory_metrics', 'round_summaries', 'data_contract'):
        assert ref[key] == gate[key], (entry['id'], key)
    a = torch.load(Path(entry['output']) / 'model.pt', map_location='cpu', weights_only=True)
    gate_entry = next(e for e in entries if e['id'] == entry['reference_to'])
    b = torch.load(Path(gate_entry['output']) / 'model.pt', map_location='cpu', weights_only=True)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    comparisons.append({'reference': entry['id'], 'gate': entry['reference_to'], 'all_tensors_metrics_diagnostics_exact': True})
cross_runtime = []
if tag == 'cu128':
    history = STAGE / 'preflight_history/cu130_20261009'
    for entry in entries:
        out = Path(entry['output'])
        relative = out.relative_to(STAGE / 'preflight')
        previous = history / relative
        old = runner.read(previous / 'result.json')
        current = results[entry['id']]
        metric_exact = all(old[key] == current[key] for key in ('config', 'metrics', 'trajectory_metrics', 'round_summaries', 'data_contract'))
        a = torch.load(previous / 'model.pt', map_location='cpu', weights_only=True)
        b = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
        tensor_exact = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        row = {'id': entry['id'], 'metrics_and_diagnostics_exact': metric_exact, 'all_tensors_exact': tensor_exact}
        cross_runtime.append(row)
        # Report runtime drift honestly; only Full same-runtime regressions are
        # formal dispatch gates. Cross-runtime equality is not assumed.
receipt = {'pass': True, 'checked_at_unix': time.time(), 'runtime': torch.__version__,
           'cuda_build': torch.version.cuda, 'server': '89.22.197.55:60350', 'instance': 52183675,
           'manifest_sha256': runner.digest(STAGE / 'manifest.json'), 'accepted_pipeline_jobs': 20,
           'image_gates': 18, 'same_runtime_full_regressions': comparisons,
           'cross_runtime_comparisons': cross_runtime,
           'scientific_formal_runs_started': 0, 'old_queues_restarted': False,
           'scope': 'Three-round real-image pipeline gates only; no70-round runtime equivalence claim',
           'guide_sha256': runner.digest('/etc/vast-agents-guide.md'),
           'gpu_snapshot': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True)}
receipt_path = CHECKS / ('preflight_acceptance_' + tag + '.json')
assert not receipt_path.exists()
runner.save(receipt_path, receipt)
archive = CHECKS / ('preflight_' + tag + '_accepted20_20261009.tar.gz')
assert not archive.exists()
files = sorted(p for p in (STAGE / 'preflight').rglob('*') if p.is_file())
files += [Path(e['job']) for e in entries]
files += [STAGE / name for name in ('manifest.json', 'PROTOCOL.md', 'preflight_queue_progress.json')]
files += list((REPO / 'deployment/celeba_mechanism_20261009').glob('*.py'))
files += [receipt_path, Path('/etc/supervisor/conf.d/guardfed_celeba_mechanism_preflight.conf'),
          Path('/opt/supervisor-scripts/guardfed_celeba_mechanism_preflight.sh')]
files = sorted(set(files))
members = []
with tarfile.open(archive, 'w:gz', compresslevel=3) as tf:
    for path in files:
        name = str(path).lstrip('/')
        members.append({'name': name, 'bytes': path.stat().st_size, 'sha256': runner.digest(path)})
        tf.add(path, arcname=name, recursive=False)
chain = {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': runner.digest(archive),
         'members': members, 'accepted_ids': sorted(results), 'runtime': torch.__version__,
         'scientific_results': False, 'off_server_verified': False}
runner.save(CHECKS / ('preflight_backup_' + tag + '.json'), chain)
print(json.dumps({'acceptance': str(receipt_path), 'archive': str(archive), 'sha256': chain['sha256'],
                  'bytes': chain['bytes'], 'members': len(members), 'accepted': 20,
                  'cross_runtime_exact': sum(r['all_tensors_exact'] and r['metrics_and_diagnostics_exact'] for r in cross_runtime)}))
