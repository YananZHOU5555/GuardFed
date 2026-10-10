"""One-time local metadata freeze. No Torch, CNN, fit, SSH or bulk copy."""
from pathlib import Path
import datetime
import importlib.util
import json
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BRIDGE = ROOT / 'tmp/celeba_added_cnn_three_view_bridge_20261010'
ROOT_REVIEW = 'be5e6c949254960f55b765ec976f1e681a26450416f156ca237965dedc87fe69'
sys.path.insert(0, str(HERE))
import candidate as c


def save(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    assert not (HERE / 'MANIFEST.json').exists(), 'Freeze once; preserve prior attempt'
    volume = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command',
        'Get-Volume -DriveLetter F | Select-Object DriveLetter,FileSystemLabel,HealthStatus,SizeRemaining,Size | ConvertTo-Json -Compress'], text=True))
    assert volume['FileSystemLabel'] == 'Yanan 2TB' and volume['HealthStatus'] == 'Healthy'
    assert volume['SizeRemaining'] >= 1024**3
    assert c.sha(BRIDGE / 'ROOT_SOURCE_REVIEW.json') == ROOT_REVIEW
    review = c.read(BRIDGE / 'ROOT_SOURCE_REVIEW.json')
    assert review['source_adopted'] is True and review['science_dispatch_authorized'] is False
    assert c.sha(BRIDGE / 'bridge.py') == review['bridge_sha256']
    spec = importlib.util.spec_from_file_location('original_added_bridge', BRIDGE / 'bridge.py')
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    proofs, reuse = c.read(BRIDGE / 'PROOF_PINS.json'), c.read(BRIDGE / 'SOURCE_REUSE.json')
    assert c.sha(BRIDGE / 'PROOF_PINS.json') == review['proof_pins_sha256']
    assert c.sha(BRIDGE / 'SOURCE_REUSE.json') == review['source_reuse_sha256']
    (HERE / 'originals').mkdir()
    (HERE / 'proofs').mkdir()
    mapping = {}

    def copy_pin(pin, relative):
        src, dest = Path(pin['path']), HERE / relative
        assert c.sha(src) == pin['sha256']
        if 'bytes' in pin:
            assert src.stat().st_size == pin['bytes']
        assert src.stat().st_size < 1024**2, 'Only scoped compact source/proof'
        if not dest.exists():
            shutil.copyfile(src, dest)
        assert c.sha(dest) == pin['sha256']
        mapping[str(src).replace('\\', '/')] = {'kind': 'package', 'relative': relative, 'sha256': pin['sha256'], 'bytes': src.stat().st_size}

    for name in ('bridge.py', 'PROOF_PINS.json', 'SOURCE_REUSE.json', 'ROOT_SOURCE_REVIEW.json'):
        p = BRIDGE / name
        copy_pin({'path': str(p), 'sha256': c.sha(p)}, 'originals/' + name)
    for entry, name in zip(reuse['function_sources'], ('evaluator.py', 'replay.py')):
        copy_pin(entry['file'], 'originals/' + name)
    copy_pin(reuse['original_core'], 'originals/core.py')
    cnn = ROOT / 'tmp/revision-publish-20260928/src/celeba_data.py'
    copy_pin({'path': str(cnn), 'sha256': '0f48fbc6d241e7d0cde692839659e817feab407991795a6f92a33052a9cc07ce'}, 'originals/celeba_data.py')
    records, common = [], {}
    methods = ['FLGMM', 'CosineFairnessHybrid', 'Fed-NGA-gradient']
    expected_ids = [
        'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage',
        'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage',
        'FedNGA_eta0.01_non-IID_Benign_seed91001_screen']
    for i, (method, rid) in enumerate(zip(methods, expected_ids)):
        chain = proofs['chains'][method]
        identity = bridge.identity_record(method, rid)  # original metadata/hash bridge only
        ev = chain['records'][rid]
        job, result, prov = (c.read(ev[k]['path']) for k in ('job', 'result', 'provenance'))
        for key in ('root', 'offserver', 'strict', 'raw_index', 'index_binding', 'checker'):
            copy_pin(chain[key], f'proofs/{i}_{key}' + Path(chain[key]['path']).suffix)
        if method == 'FLGMM':
            stage = '/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage'
            out = stage + '/runs/' + rid
            remote_job = out + '/job.json'
        elif method == 'CosineFairnessHybrid':
            stage = '/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/stage'
            out = stage + '/runs/' + rid
            remote_job = stage + '/jobs/' + rid + '.json'
        else:
            stage = '/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010'
            out = '/workspace/celeba_gradient_screen64_v2_results_20261010/' + rid
            remote_job = stage + '/jobs/' + rid + '.json'
        artifacts = {}
        for key, pin in ev.items():
            server = remote_job if key == 'job' else stage + '/full_scope.json' if key == 'scope' else out + '/' + Path(pin['path']).name
            artifacts[key] = dict(pin, server_path=server)
            mapping[pin['path']] = {'kind': 'server', 'server_path': server, 'sha256': pin['sha256'], 'bytes': pin['bytes']}
        acceptance = c.read(ev['acceptance']['path'])
        for name, h in acceptance['artifact_hashes'].items():
            p = Path(ev['result']['path']).parent / name
            assert c.sha(p) == h
            pin = {'path': p.as_posix(), 'server_path': out + '/' + name, 'sha256': h, 'bytes': p.stat().st_size}
            mapping[p.as_posix()] = dict(pin, kind='server')
            artifacts.setdefault(name, pin)
        for rel, h in prov['source_hashes'].items():
            # Same original900 scientific/data lineage, excluding unrelated Adult files.
            if rel.startswith(('scripts/', 'src/', 'data/celeba/')):
                assert rel not in common or common[rel] == h
                common[rel] = h
        env = prov.get('environment', prov)
        records.append({'id': rid, 'method': method, 'distribution': job['distribution'], 'attack': job['attack'],
                        'seed': job['config']['seed'], 'terminal_round': result['rounds'], 'split': 'valid', 'n_eval': 19867,
                        'actual_alpha': result['alpha'], 'identity': identity, 'runtime_output': out,
                        'runtime_artifacts': artifacts, 'original_training_device': env['device'],
                        'original_training_torch': env['torch'], 'external_proofs': {k: chain[k] for k in ('root', 'offserver', 'strict')}})
    manifest = {'scope': 'EXACT3_VALID_IMAGE_INTERFACE_GATE_CANDIDATE', 'status': 'PREPARED_NOT_EXECUTED_NOT_DISPATCH_AUTHORIZED',
                'bridge_root_review_sha256': ROOT_REVIEW, 'server_repo': '/workspace/GuardFed-celeba-expanded',
                'server_python': '/workspace/guardfed_envs/celeba-cu128-20261009/bin/python',
                'views': ['native', 'raw', 'shared_calibration'], 'native_tolerance': 1e-12,
                'cpu_affinity': c.CPUS, 'threads': 8, 'new_scientific_acceptances': 0, 'dispatch_authorized': False,
                'records': records, 'path_map': mapping, 'runtime_repo_hashes': common,
                'authorizations_and_runtime_facts_pending': ['root source review of candidate', 'actual Linux metadata/path/source/checkpoint preflight',
                  'CPU112-119 all-thread freedom and duplicate lock', 'service/GPU/cgroup/RAM/storage health',
                  'explicit exact3 root authorization with real source-review and fresh Linux-preflight SHA',
                  'three actual replay/native-difference/fit/array receipts and later independent offserver/root adoption']}
    c.validate_manifest(manifest)
    save(HERE / 'MANIFEST.json', manifest)
    save(HERE / 'STORAGE_CHECK.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'volume': volume,
                                     'operation': 'F read-only hashes; E compact source/proof/identity only', 'new_bulk_copies': 0,
                                     'source_data_weights_remain_on_F_and_original_server': True})
    print(json.dumps({'records': len(records), 'manifest_sha256': c.sha(HERE / 'MANIFEST.json'), 'new_inference': 0, 'new_fit': 0}))


if __name__ == '__main__':
    main()
