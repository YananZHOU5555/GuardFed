"""Strict GPU CANARY acceptance, exact repeats, and disclosed CPU/GPU differences."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def check_cpu_receipts(root, cpu_root):
    """Bind the CPU comparison to the exact independently backed-up members."""
    freeze = read(root / 'FREEZE.json')
    prerequisites = freeze['prerequisites']
    assert prerequisites['cpu_offserver_verified'] is True
    expected = prerequisites['cpu_receipt_hashes']
    assert set(expected) == {'LOCAL_ACCEPTANCE.json', 'BACKUP_SHA256.json'}
    for name, value in expected.items(): assert sha(cpu_root / name) == value, name
    local, backup = read(cpu_root / 'LOCAL_ACCEPTANCE.json'), read(cpu_root / 'BACKUP_SHA256.json')
    assert local['status'] == 'PASS' and local['actual_results_checked'] == 2
    assert sha(cpu_root / 'BACKUP_MEMBERS.json') == backup['inventory_sha256']
    inventory = read(cpu_root / 'BACKUP_MEMBERS.json')
    assert len(inventory) == backup['member_count']
    required = {'FREEZE.json', 'SCOPE.json'}
    for condition in ('IID_Benign', 'non-IID_S-DFA'):
        prefix = f'runs/FLGMM_realimage_CANARY_{condition}_seed91001_Tg1_round3/'
        required.update(prefix + name for name in ('model.pt', 'result.json', 'diagnostics.json',
            'state.json', 'job.json', 'provenance.json', 'acceptance.json',
            'round_001_state.json', 'round_002_state.json', 'round_003_state.json'))
    for name in required:
        assert name in inventory, name
        record, path = inventory[name], cpu_root / name
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
    scope, cpu_freeze = read(cpu_root / 'SCOPE.json'), read(cpu_root / 'FREEZE.json')
    assert cpu_freeze['status'] == 'FROZEN' and cpu_freeze['scope'] == 'two_3round_real_image_CANARY_only'
    for name, value in cpu_freeze['local_hashes'].items(): assert sha(cpu_root / name) == value, name
    for relative in scope['jobs']:
        job = read(cpu_root / relative)
        out = cpu_root / 'runs' / job['id']
        acceptance, provenance = read(out / 'acceptance.json'), read(out / 'provenance.json')
        assert acceptance['status'] == 'PASS' and not acceptance['formal_table_eligible']
        assert not (out / 'failure.json').exists()
        assert provenance['job_sha256'] == sha(cpu_root / relative)
        assert provenance['freeze_sha256'] == sha(cpu_root / 'FREEZE.json')
        assert provenance['local_hashes'] == cpu_freeze['local_hashes']
        assert provenance['source_hashes'] == scope['source_hashes']
        for name, value in acceptance['artifact_hashes'].items(): assert sha(out / name) == value, name
        assert read(out / 'result.json')['provenance'] == provenance
        assert read(out / 'result.json')['revision_job'] == job
        matches = [r for r in local['results'] if r['id'] == job['id']]
        assert len(matches) == 1 and matches[0]['model_sha256'] == sha(out / 'model.pt')
    return dict(cpu_receipt_hashes=expected, inventory_sha256=backup['inventory_sha256'], bound_members=len(required))


def save_report(path, report):
    """Never overwrite a prior mismatch or acceptance with different evidence."""
    content = (json.dumps(report, indent=2) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == content, 'Existing GPU acceptance differs; preserve it and use a separate reviewed output'
    else:
        path.write_bytes(content)


def check_one(root, out, require_pass=True):
    import torch
    root, out = Path(root), Path(out)
    assert not (out / 'failure.json').exists()
    freeze, scope = read(root / 'FREEZE.json'), read(root / 'SCOPE.json')
    assert freeze['status'] == 'FROZEN' and freeze['scope'] == 'four_3round_GPU_CANARY_only'
    for name, value in freeze['local_hashes'].items(): assert sha(root / name) == value, name
    job = read(out / 'job.json')
    job_path = root / 'jobs' / (job['id'] + '.json')
    assert job_path.relative_to(root).as_posix() in scope['jobs'] and read(job_path) == job
    expected = dict(scope['base_config'], rounds=3, seed=91001, device='cuda', learning_rate=.001,
        client_alpha={'IID': 5000., 'non-IID': 5.}[job['distribution']],
        experiment_suite=scope['version'], experiment_tag=job['id'])
    assert job['config'] == expected and job['adapter'] == dict(warmup_rounds=1, control_width=3.)
    assert job['repeat'] in (1, 2) and job['evidence_stage'] == 'GPU_CANARY_REAL_IMAGE_ONLY'
    acceptance = read(out / 'acceptance.json')
    assert acceptance['status'] == ('PASS' if require_pass else 'RECORDED_PENDING_CHECK')
    artifacts = {'model.pt', 'state.json', 'diagnostics.json', 'result.json', 'provenance.json', 'job.json', 'source_verification.json', 'import_rng_evidence.json'}
    artifacts.update(f'round_{i:03d}_state.json' for i in range(1, 4))
    artifacts.update(f'{prefix}_rng.{suffix}' for prefix in ['round_001', 'round_002', 'round_003', 'final'] for suffix in ['json', 'pt'])
    artifacts.update(f'{prefix}_rng_scope.json' for prefix in ['round_001', 'round_002', 'round_003', 'final'])
    assert set(acceptance['artifact_hashes']) == artifacts and not acceptance['formal_table_eligible']
    for name, value in acceptance['artifact_hashes'].items(): assert sha(out / name) == value, name
    result, provenance = read(out / 'result.json'), read(out / 'provenance.json')
    assert result['revision_job'] == job and result['config'] == job['config'] and result['provenance'] == provenance
    assert result['evidence_stage'] == 'GPU_CANARY_REAL_IMAGE_ONLY' and result['status'] == 'canary_complete'
    assert result['seed'] == 91001 and result['rounds'] == 3
    for name in ('dataset', 'method', 'distribution', 'attack'): assert result[name] == job[name], name
    assert provenance['job_sha256'] == sha(job_path) and provenance['freeze_sha256'] == sha(root / 'FREEZE.json')
    assert provenance['source_hashes'] == scope['source_hashes'] and provenance['local_hashes'] == freeze['local_hashes']
    source = read(out / 'source_verification.json')
    assert source['consistent'] and source['before'] == source['after'] == scope['source_hashes']
    assert provenance['device'] == 'cuda:0' and provenance['threads'] == 1 and provenance['visible_gpu'] == str(job['gpu'])
    assert not provenance['formal_table_eligible'] and provenance['real_data']
    contract = result['data_contract']['image_data_contract']
    assert (contract['evaluation_split'], contract['actual_train_rows'], contract['actual_evaluation_rows'], result['evaluation_stats']['prediction_count']) == ('valid', 162770, 19867, 19867)
    assert contract['train_eval_disjoint'] and contract['root_client_disjoint']
    for name in ('trajectory_metrics', 'round_summaries'): assert [r['round'] for r in result[name]] == [1, 2, 3]
    assert result['metrics'] == result['trajectory_metrics'][-1]['metrics']
    assert all(math.isfinite(v) and 0 <= v <= 1 for r in result['trajectory_metrics'] for v in r['metrics'].values())
    diagnostics = read(out / 'diagnostics.json')
    assert [r['aggregate']['stage'] for r in diagnostics] == ['per_round_gmm', 'fit_control_limit', 'monitor']
    for number, row in enumerate(diagnostics, 1):
        assert row['aggregate'] == result['round_summaries'][number - 1]['aggregate']
        state = read(out / f'round_{number:03d}_state.json')
        assert state['round_index'] == number and state['warmup_rounds'] == 1 and state['control_width'] == 3
        assert state['client_ids'] == list(range(20)) and len(state['history']) == 20
        assert all(len(h) == number and all(math.isfinite(x) for x in h) for h in state['history'])
        assert state['ucl'] == row['aggregate']['ucl']
    assert read(out / 'state.json') == read(out / 'round_003_state.json')
    model = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    assert model and all(torch.isfinite(value).all() for value in model.values())
    imported = read(out / 'import_rng_evidence.json')
    assert imported['scope'] == 'module_import_only_not_training_rng'
    assert len(imported['records']) == len(imported['boundary_states'])
    for prefix in ('round_001', 'round_002', 'round_003', 'final'):
        rng = torch.load(out / f'{prefix}_rng.pt', map_location='cpu', weights_only=True)
        assert set(rng) == {'cpu', 'cuda'} and len(rng['cuda']) == 1
        assert rng['cpu'].dtype == torch.uint8 and rng['cuda'][0].dtype == torch.uint8
        assert set(read(out / f'{prefix}_rng.json')) == {'python', 'numpy_legacy', 'numpy_generators'}
        evidence = read(out / f'{prefix}_rng_scope.json')
        assert evidence['training_states'] == read(out / f'{prefix}_rng.json')['numpy_generators']
        assert len(evidence['training_origins']) == len(evidence['training_states'])
        assert len(evidence['import_states']) == len(imported['boundary_states'])
        advanced = [i for i, value in enumerate(evidence['import_states']) if value != imported['boundary_states'][i]]
        assert advanced == evidence['import_advanced_indices']
        enrolled = []
        for state, origin in zip(evidence['training_states'], evidence['training_origins']):
            assert origin['reason'] in ('default_rng_called_after_import', 'imported_generator_advanced_after_boundary')
            if origin['import_index'] is not None:
                index = origin['import_index'];enrolled.append(index)
                assert state == evidence['import_states'][index]
        assert set(advanced).issubset(enrolled)
    return result


def compare_models(first, second):
    import torch
    a, b = [torch.load(p / 'model.pt', map_location='cpu', weights_only=True) for p in (first, second)]
    assert a.keys() == b.keys()
    return dict(exact=all(torch.equal(a[k], b[k]) for k in a),
        max_abs=max(float((a[k].double() - b[k].double()).abs().max()) for k in a), tensor_count=len(a))


def compare_repeat(first, second):
    import torch
    model = compare_models(first, second)
    ra, rb = read(first / 'result.json'), read(second / 'result.json')
    checks = dict(model_tensors=model['exact'], final_metrics=ra['metrics'] == rb['metrics'],
        trajectory_metrics=ra['trajectory_metrics'] == rb['trajectory_metrics'],
        last10_metrics=ra['last10_metrics'] == rb['last10_metrics'],
        round_summaries=ra['round_summaries'] == rb['round_summaries'],
        attack_audit=ra['attack_audit'] == rb['attack_audit'],
        evaluation_stats=ra['evaluation_stats'] == rb['evaluation_stats'],
        diagnostics=read(first / 'diagnostics.json') == read(second / 'diagnostics.json'))
    for prefix in ('round_001', 'round_002', 'round_003', 'final'):
        checks[prefix + '_rng_json'] = read(first / f'{prefix}_rng.json') == read(second / f'{prefix}_rng.json')
        a, b = [torch.load(p / f'{prefix}_rng.pt', map_location='cpu', weights_only=True) for p in (first, second)]
        checks[prefix + '_rng_tensors'] = torch.equal(a['cpu'], b['cpu']) and torch.equal(a['cuda'][0], b['cuda'][0])
        scopes = [read(p / f'{prefix}_rng_scope.json') for p in (first, second)]
        checks[prefix + '_training_rng_origins'] = scopes[0]['training_origins'] == scopes[1]['training_origins']
        checks[prefix + '_imported_rng_use'] = scopes[0]['import_advanced_indices'] == scopes[1]['import_advanced_indices']
    for number in range(1, 4):
        name = f'round_{number:03d}_state.json'
        checks[name] = read(first / name) == read(second / name)
    imports = [read(p / 'import_rng_evidence.json')['boundary_states'] for p in (first, second)]
    disclosure = dict(scope='Import-time evidence retained separately; any generator observed used enters training comparison',
        counts=list(map(len, imports)), different_import_state_indices=[i for i, (a,b) in enumerate(zip(*imports)) if a != b])
    return dict(exact=all(checks.values()), checks=checks, model=model, import_rng_disclosure=disclosure)


def verify(root, cpu_root):
    cpu_binding = check_cpu_receipts(root, cpu_root)
    scope = read(root / 'SCOPE.json')
    reports, pairs = [], {}
    for relative in scope['jobs']:
        job = read(root / relative)
        out = root / 'runs' / job['id']
        result = check_one(root, out)
        pairs.setdefault((job['distribution'], job['attack']), {})[job['repeat']] = out
        reports.append(dict(id=job['id'], metrics=result['metrics'], model_sha256=sha(out / 'model.pt')))
    assert len(reports) == 4 and set(pairs) == {('IID', 'Benign'), ('non-IID', 'S-DFA')}
    repeats, cpu_comparison = {}, {}
    for condition, paths in pairs.items():
        assert set(paths) == {1, 2}
        label = '_'.join(condition)
        repeats[label] = compare_repeat(paths[1], paths[2])
        cpu = cpu_root / 'runs' / f'FLGMM_realimage_CANARY_{condition[0]}_{condition[1]}_seed91001_Tg1_round3'
        cpu_result, gpu_result = read(cpu / 'result.json'), read(paths[1] / 'result.json')
        assert cpu_result['revision_job']['adapter'] == gpu_result['revision_job']['adapter']
        assert cpu_result['provenance']['source_hashes'] == gpu_result['provenance']['source_hashes']
        ignored = {'device', 'experiment_suite', 'experiment_tag'}
        assert {k: v for k, v in cpu_result['config'].items() if k not in ignored} == {k: v for k, v in gpu_result['config'].items() if k not in ignored}
        cpu_comparison[label] = dict(model=compare_models(cpu, paths[1]),
            final_metric_gpu_minus_cpu={k: gpu_result['metrics'][k] - v for k, v in cpu_result['metrics'].items()},
            trajectory_metrics_exact=cpu_result['trajectory_metrics'] == gpu_result['trajectory_metrics'],
            selection_and_diagnostics_exact=read(cpu / 'diagnostics.json') == read(paths[1] / 'diagnostics.json'),
            controller_history_exact=read(cpu / 'state.json') == read(paths[1] / 'state.json'),
            limitation='CPU eight threads vs GPU one thread; differences retained, no equivalence assumption')
    report = dict(status='PASS' if all(r['exact'] for r in repeats.values()) else 'REPEAT_MISMATCH',
        evidence_stage='GPU_CANARY_REAL_IMAGE_ONLY', formal_table_eligible=False,
        results=reports, same_seed_gpu_repeats=repeats, cpu_gpu=cpu_comparison, cpu_backup_binding=cpu_binding)
    save_report(root / 'GPU_ACCEPTANCE.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cpu-root', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.root, args.cpu_root)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
