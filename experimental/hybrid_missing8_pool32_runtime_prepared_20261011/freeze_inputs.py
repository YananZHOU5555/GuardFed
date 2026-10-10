"""One metadata freeze: compact sources/proofs only; no arrays, models, Torch or SSH."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, shutil, subprocess, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIVATE = ROOT / 'tmp/celeba_hybrid_three_view_bridge_20261011'
FL = ROOT / 'tmp/celeba_flgmm_three_view_closed_batch_preparation_20261011'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    assert not (HERE / 'MANIFEST.json').exists(), 'Freeze once; preserve all failed artifacts'
    volume = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command',
        'Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'], text=True))
    assert volume['FileSystemLabel'] == 'Yanan 2TB' and volume['HealthStatus'] == 'Healthy' and volume['SizeRemaining'] > 1024**3
    inputs = json.loads((PRIVATE / 'INPUTS.json').read_bytes())
    reuse = json.loads(Path(inputs['original_reuse']['path']).read_bytes())
    mapping, sources = {}, {}
    (HERE / 'originals').mkdir(exist_ok=True)
    (HERE / 'proofs').mkdir(exist_ok=True)

    def copy_pin(pin, rel):
        src, dst = Path(pin['path']), HERE / rel
        assert src.suffix in {'.py', '.json'} and src.stat().st_size < 2_000_000, 'Compact sources/proofs only'
        assert sha(src) == pin['sha256'] and ('bytes' not in pin or src.stat().st_size == pin['bytes'])
        if not dst.exists():
            shutil.copyfile(src, dst)
        assert sha(dst) == pin['sha256']
        actual = dict(path=src.as_posix(), sha256=sha(src), bytes=src.stat().st_size)
        mapping[src.as_posix()] = dict(kind='package', relative=rel, sha256=actual['sha256'], bytes=actual['bytes'])
        sources[rel] = actual

    def pin(p):
        return dict(path=p.as_posix(), sha256=sha(p), bytes=p.stat().st_size)

    copy_pin(pin(FL / 'candidate.py'), 'originals/fl_candidate.py')
    copy_pin(pin(PRIVATE / 'bridge.py'), 'originals/hybrid_bridge.py')
    copy_pin(pin(PRIVATE / 'INPUTS.json'), 'originals/HYBRID_INPUTS.json')
    copy_pin(inputs['original_bridge'], 'originals/bridge.py')
    copy_pin(inputs['original_reuse'], 'originals/SOURCE_REUSE.json')
    for entry, name in zip(reuse['function_sources'], ('evaluator.py', 'replay.py')):
        copy_pin(entry['file'], 'originals/' + name)
    copy_pin(reuse['original_core'], 'originals/core.py')
    copy_pin(pin(FL / 'originals/saved_science.py'), 'originals/saved_science.py')
    audit = ROOT / 'tmp/celeba_flgmm47_saved_outputs_audit_source_20261011/audit_saved.py'
    copy_pin(pin(audit), 'originals/saved_output_audit.py')
    diff = ROOT / 'tmp/diagnose_added_cnn_exact3_offserver_root_20261010.py'
    copy_pin(pin(diff), 'originals/receipt_diff.py')
    copy_pin(inputs['bound_root'], 'proofs/bound_root.json')
    batch = inputs['batches']['after1']
    for key in ('root', 'strict', 'offserver', 'receipt', 'members', 'scope', 'body'):
        copy_pin(batch[key], 'proofs/' + key + Path(batch[key]['path']).suffix)
    for key, value in batch['prior'].items():
        copy_pin(value, 'proofs/' + key + '.json')
    prior = ROOT / 'tmp/celeba_added_cnn_exact3_root_execution_20261010/ROOT_SCIENTIFIC_ADOPTION.json'
    copy_pin(pin(prior), 'proofs/existing_replay_root.json')
    copy_pin(pin(ROOT / 'tmp/celeba_hybrid_three_view_canary1_prepared_20261011/CHECK.json'), 'proofs/existing_replay_reuse_check.json')
    spec = importlib.util.spec_from_file_location('_hybrid_native_actual_metadata', PRIVATE / 'bridge.py')
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    spec = importlib.util.spec_from_file_location('_missing8_candidate', HERE / 'candidate.py')
    c = importlib.util.module_from_spec(spec); sys.modules[spec.name] = c; spec.loader.exec_module(c)
    records, common = [], {}
    stage = '/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/stage'
    for rid in c.IDS:
        identity = bridge.identity_record(rid)
        selected = inputs['records'][rid]
        assert selected['batch'] == 'after1'
        out = stage + '/runs/' + rid
        artifacts = {}
        for key, value in selected['metadata'].items():
            server = stage + '/jobs/' + rid + '.json' if key == 'job' else out + '/' + Path(value['path']).name
            mapping[value['path']] = dict(kind='server', server_path=server, sha256=value['sha256'], bytes=value['bytes'])
            artifacts[key] = dict(value, server_path=server)
        model = selected['checkpoint']
        assert model['server_path'] == out + '/model.pt'
        artifacts['model'] = dict(model)
        mapping[model['path']] = dict(kind='server', server_path=model['server_path'], sha256=model['sha256'], bytes=model['bytes'])
        for rel, h in identity['source_hashes'].items():
            if rel.startswith(('scripts/', 'src/', 'data/celeba/')):
                assert rel not in common or common[rel] == h
                common[rel] = h
        records.append(dict(id=rid, method=c.METHOD, distribution='IID', attack='Benign', seed=identity['seed'],
            terminal_round=70, split='valid', n_eval=19867, actual_alpha=5000.0, identity=identity,
            runtime_output=out, runtime_artifacts=artifacts, original_training_torch=identity['training_torch'],
            original_training_device=identity['config']['device']))
    manifest = dict(scope='HYBRID_ADOPTED_BENIGN_MISSING8_CANARY_FIRST_SOURCE_ONLY',
        status='PREPARED_NOT_EXECUTED_NOT_DISPATCH_AUTHORIZED', exact_ids=c.IDS, canary_id=c.IDS[0],
        previously_accepted_skip_ids=[c.IDS[0].replace('91003', '91002')],
        excluded_screen91001_identity_gap=True, private_inputs_pin=pin(PRIVATE / 'INPUTS.json'),
        native_root_pin=batch['root'], existing_replay_root_pin=pin(prior), records=records, path_map=mapping,
        server_repo='/workspace/GuardFed-celeba-expanded', server_python='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python',
        server_namespace=c.BASE, runtime_repo_hashes=common, cpu_affinity=None, affinity_cardinality_options=[8, 32], threads=8, device='cpu', dtype='float32',
        views=['native', 'raw', 'shared_calibration'], native_tolerance=1e-12, new_three_view_accepted=0,
        dispatch_authorized=False, sources=sources,
        pending=['actual source review', 'explicit root 8-CPU mask or same-socket32 pool binding excluding FL32..63/old11..18/102..119', 'fresh Linux all-thread/resource/hash/producer preflight',
            'root authorization', 'real91003 native canary then seven sequential receipts', 'Linux original whole check', 'F exact saved-array transport',
            'Windows saved-output check with zero fit', 'root scientific adoption'])
    c.validate_manifest(manifest)
    save(HERE / 'MANIFEST.json', manifest)
    save(HERE / 'SOURCE_PINS.json', dict(files=sources, scientific_functions=reuse['function_sources'],
        scientific_function_changes=0, tolerance=1e-12, original_hybrid_checked_sha256=inputs['hybrid_function_hashes']['body.checked'],
        existing_replay_reused_not_rerun=True, new_model_array_or_archive_copies=0, F_volume=volume,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(status='EXACT8_METADATA_FROZEN_NO_SCIENCE', records=len(records), manifest_sha256=sha(HERE / 'MANIFEST.json'))))


if __name__ == '__main__':
    main()
