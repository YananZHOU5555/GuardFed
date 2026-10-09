"""Real FedAA/LASA inputs and intentional binding corruptions; no image inference."""
import copy
import json
from pathlib import Path
import tempfile
import replay_v4 as bridge

HERE = Path(__file__).resolve().parent
v2 = bridge.v2
stock = v2.read(HERE.parent / 'inputs/model_inventory.json')
v2.require(v2.digest(HERE.parent / 'inputs/model_inventory.json') == v2.INVENTORY_SHA, 'Original inventory bytes changed')
probe = v2.read(HERE / 'verification_inputs/real900_rawjob_schema_probe.json')
accepted, rejected = [], []

def refuse(name, function):
    try:
        function()
    except (ValueError, KeyError) as exc:
        rejected.append({'case': name, 'error_type': type(exc).__name__, 'error': str(exc)})
    else:
        raise AssertionError('Corruption accepted: ' + name)

for sample in probe['samples']:
    method = sample['method']
    record = next(r for r in stock['records'] if r['id'] == sample['id'])
    job_path = HERE / ('verification_inputs/' + method + '_original_rawjob.json')
    result_path = HERE / ('verification_inputs/' + method + '_original_result.json')
    bridge.v3.full_hashes({job_path: record['raw_job']['sha256'], result_path: record['result']['sha256']})
    raw, result = v2.read(job_path), v2.read(result_path)
    saved = copy.deepcopy((raw, result, record))
    paths = bridge.expected_paths(record)
    job_view, result_view, descriptor = bridge.normalized_views(record, paths, raw, result)
    assert (raw, result, record) == saved
    assert job_view['output'] == record['original_remote_output']
    assert result_view['metrics'] == result['metrics'] == record['prior_validation_metrics']
    assert result_view['alpha'] == record['actual_alpha']
    assert result_view['data_contract']['image_data_contract'] == record['data_contract']
    assert [r['round'] for r in result_view['trajectory_metrics']] == list(range(1, 71))
    assert [r['round'] for r in result_view['round_summaries']] == list(range(1, 71))
    accepted.append({'id': record['id'], 'schema_bridge': descriptor, 'metric_values_unchanged': True, 'original_dictionaries_unchanged': True})
    def changed_job(key, value):
        bad = copy.deepcopy(raw)
        bad[key] = value
        return lambda: bridge.normalized_views(record, paths, bad, result)
    def changed_result(path, value):
        bad = copy.deepcopy(result)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return lambda: bridge.normalized_views(record, paths, raw, bad)
    bad_paths = dict(paths, result=paths['result'].parent.parent / 'other/result.json')
    refuse(method + '_wrong_runtime_result_path', lambda: bridge.normalized_views(record, bad_paths, raw, result))
    bad_record = copy.deepcopy(record)
    bad_record['original_remote_output'] += '_wrong'
    refuse(method + '_wrong_inventory_historical_output', lambda: bridge.normalized_views(bad_record, paths, raw, result))
    refuse(method + '_invented_raw_output', changed_job('output', '/workspace/wrong'))
    refuse(method + '_wrong_raw_id', changed_job('id', raw['id'] + '_wrong'))
    bad_config = dict(raw['config'], seed=91099)
    refuse(method + '_wrong_original_seed_config', changed_job('config', bad_config))
    if method == 'FedAA':
        for name, path, value in [('wrong_job_sha', ['identity', 'job_sha256'], '0' * 64), ('wrong_policy_seed', ['identity', 'policy_seed'], 91099), ('wrong_policy_recipe', ['policy_config', 'actor_lr'], 9), ('wrong_root_id', ['identity', 'data_contract', 'root_image_ids_sha256'], '0' * 64), ('wrong_checkpoint_sha', ['checkpoint_sha256'], '0' * 64), ('wrong_planned_rounds', ['planned_rounds'], 69), ('wrong_source', ['source_hashes', 'scripts/reproduce_paper_tables.py'], '0' * 64)]:
            refuse('FedAA_' + name, changed_result(path, value))
    else:
        for name, path, value in [('wrong_revision_output', ['revision_job', 'output'], '/workspace/wrong'), ('wrong_checkpoint_sha', ['revision_job', 'checkpoint_sha256'], '0' * 64), ('wrong_revision_seed', ['revision_job', 'config', 'seed'], 91099)]:
            refuse('LASA_' + name, changed_result(path, value))
    with tempfile.TemporaryDirectory(prefix='valid_replay_input_tamper_') as temp:
        wrong = Path(temp) / 'rawjob.json'
        wrong.write_bytes(job_path.read_bytes() + b'\n')
        refuse(method + '_original_raw_bytes_tamper', lambda: bridge.v3.full_hashes({wrong: record['raw_job']['sha256']}))

assert len(accepted) == 2 and len(rejected) == 22
report = {'status': 'REAL_ADAPTER_SCHEMA_AND_BINDING_SELFCHECK_PASS', 'source_sha256': v2.digest(HERE / 'replay_v4.py'), 'sealed_v3_source_sha256': bridge.V3_SHA, 'inventory_sha256': v2.INVENTORY_SHA, 'accepted_original_samples': accepted, 'rejected_cases': rejected, 'accepted_n': len(accepted), 'rejected_n': len(rejected), 'new_image_inference': 0, 'semantic_labels_loaded': False, 'all900_native_valid_replayed': False}
target = HERE / 'selfcheck_v4_final.json'
assert not target.exists()
v2.save(target, report)
print(json.dumps({'status': report['status'], 'accepted_n': len(accepted), 'rejected_n': len(rejected), 'receipt_sha256': v2.digest(target)}, indent=2))
