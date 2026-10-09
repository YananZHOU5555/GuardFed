"""Audit the complete prepared grid and execute CPU component gates; never dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(project, stage):
    manifest = json.loads((stage / 'manifest.json').read_text())
    expected = {(v, d, a, s) for v in manifest['variants'] if v != 'Full'
                for d in manifest['distributions'] for a in manifest['attacks'] for s in manifest['seeds']}
    actual = set(); ids = set(); outputs = set()
    source = project / 'tmp/celeba_mechanism_20261009'
    assert manifest['status'] == 'PREPARED_PENDING_SERVER_AND_REAL_IMAGE_GATES'
    assert not manifest['new_training_started'] and not manifest['real_image_gates_passed']
    assert not (stage / 'dispatch_receipt.json').exists()
    assert digest(stage / 'PROTOCOL.md') == manifest['protocol_sha256']
    for remote, sha in manifest['adapter_hashes'].items():
        assert digest(source / Path(remote).name) == sha
    for group, relative in [('jobs', 'jobs'), ('preflight_jobs', 'preflight/jobs'), ('reference_jobs', 'preflight/jobs')]:
        for entry in manifest[group]:
            local = stage / relative / Path(entry['job']).name
            assert digest(local) == entry['job_sha256']
            job = json.loads(local.read_text())
            assert job['id'] == entry['id'] and job['output'] == entry['output']
            assert job['id'] not in ids and job['output'] not in outputs
            ids.add(job['id']); outputs.add(job['output'])
            assert '\\' not in entry['job'] and '\\' not in job['output']
            assert job['source_hashes'] == manifest['source_hashes']
            assert job['adapter_hashes'] == manifest['adapter_hashes']
            assert job['protocol_sha256'] == manifest['protocol_sha256']
            cfg = job['config']
            assert cfg['celeba_evaluation_split'] == 'valid'
            assert cfg['learning_rate'] == .0005 and cfg['ad2_calibration_max_acc_drop'] == .005
            assert cfg['client_alpha'] == manifest['distributions'][job['distribution']]
            assert cfg['ablation_component'] == (job['variant'][-1] if job['variant'].startswith('minus_') else 'none')
            if group == 'jobs':
                assert cfg['rounds'] == 70 and job['variant'] != 'Full'
                actual.add((job['variant'], job['distribution'], job['attack'], cfg['seed']))
            else:
                assert cfg['rounds'] == 3 and cfg['seed'] == 91001 and job['attack'] == 'S-DFA'
            if group == 'reference_jobs':
                gate = json.loads((stage / 'preflight/jobs' / (entry['reference_to'] + '.json')).read_text())
                assert cfg == gate['config'] and gate['variant'] == 'Full'
    assert actual == expected and len(actual) == 800
    assert len(ids) == 820
    reused = manifest['reused_full']
    assert len(reused) == 100 and len({(r['distribution'], r['attack'], r['seed']) for r in reused}) == 100
    core = project / 'tmp/revision-publish-20260928'
    for rel in ['scripts/reproduce_paper_tables.py', 'scripts/run_revision_ablation.py', 'src/celeba_data.py', 'src/data_loader.py', 'scripts/build_celeba_cache.py']:
        assert digest(core / rel) == manifest['source_hashes'][rel]
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(source), '-p', 'test_*.py', '-v'],
                          capture_output=True, text=True, cwd=project)
    (stage / 'cpu_component_checks.log').write_text(test.stdout + test.stderr)
    assert test.returncode == 0, test.stderr
    result = {'pass': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': digest(stage / 'manifest.json'), 'new_jobs': 800, 'reused_full_records': 100,
              'image_gates_prepared': 18, 'unchanged_same_horizon_references_prepared': 2,
              'unique_ids_and_outputs': 820, 'component_test_exit_code': test.returncode,
              'component_test_count': 6, 'component_test_log_sha256': digest(stage / 'cpu_component_checks.log'),
              'scope': 'Prepared files and CPU/synthetic component gates only; no new CelebA/GPU execution',
              'server_data_checked_now': False, 'real_image_gates_run': False, 'new_training_started': False}
    (stage / 'prepared_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--project', type=Path, required=True); parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.project.resolve(), args.stage.resolve())))
