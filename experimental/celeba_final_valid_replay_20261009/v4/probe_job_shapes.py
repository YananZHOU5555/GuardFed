"""Read-only real900 raw-job schema probe; no image or label input is opened."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE = Path('/workspace/guardfed_checks/celeba_final_valid_replay_20261009')
STORE = Path('/workspace/guardfed_checks/celeba_validation900_restore_20261009')
OUT = BASE / 'v4/verification_inputs'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def tree(v):
    if isinstance(v, dict):
        return {k: tree(x) for k, x in v.items()}
    if isinstance(v, list):
        return {'type': 'list', 'n': len(v), 'first_item': tree(v[0]) if v else None}
    return type(v).__name__

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[16:24])
os.nice(10)
subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
inventory = BASE / 'inputs/model_inventory.json'
mapping = STORE / 'storage_map.json'
assert sha(inventory) == '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
assert sha(mapping) == 'e160416101ccb82b42224c9ff1bb337de45ef72fa251197063d1c15e2910f949'
stock = json.loads(inventory.read_text())
print('inventory_top_keys=' + repr(list(stock)), flush=True)
records = stock['records']
bound = {(r['id'], r['kind']): r for r in json.loads(mapping.read_text())['records']}
groups, samples = {}, []
OUT.mkdir(parents=True, exist_ok=False)
for r in records:
    row = bound[r['id'], 'raw_job']
    p = Path(row['target'])
    assert sha(p) == row['sha256'] == r['raw_job']['sha256']
    job = json.loads(p.read_text())
    sig = ','.join(sorted(job))
    group = groups.setdefault((r['method'], sig), {'method': r['method'], 'keys': sorted(job), 'n': 0, 'example_id': r['id']})
    group['n'] += 1
    if r['method'] in ('FedAA', 'LASA') and r['distribution'] == 'non-IID' and r['attack'] == 'Benign' and r['seed'] == 91003:
        artifacts = {}
        for kind in ('raw_job', 'result', 'checkpoint'):
            a = bound[r['id'], kind]
            assert sha(Path(a['target'])) == a['sha256'] == r[kind]['sha256']
            artifacts[kind] = a
        result = json.loads(Path(artifacts['result']['target']).read_text())
        sample = {'id': r['id'], 'method': r['method'], 'inventory_original_remote_output': r['original_remote_output'], 'record_sha256': hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(), 'artifacts': artifacts, 'raw_job_key_tree': tree(job), 'raw_job': job, 'result_top_keys': sorted(result), 'result_data_contract': result['data_contract'], 'result_identity': {k: result.get(k) for k in ('method', 'seed', 'distribution', 'attack', 'alpha', 'rounds', 'revision_job', 'adapter_job', 'adapter_source_hashes', 'config')}, 'round_summary_n': len(result['round_summaries']), 'trajectory_metrics_n': len(result['trajectory_metrics'])}
        samples.append(sample)
        (OUT / (r['method'] + '_original_rawjob.json')).write_bytes(p.read_bytes())
report = {'status': 'REAL900_RAWJOB_SCHEMAS_PROBED_NO_INFERENCE', 'inventory_sha256': sha(inventory), 'storage_map_sha256': sha(mapping), 'raw_jobs_verified': len(records), 'groups': list(groups.values()), 'samples': samples, 'test_images_or_labels_read': False}
target = OUT / 'real900_rawjob_schema_probe.json'
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'report_sha256': sha(target), 'groups': report['groups'], 'samples': [{'id': s['id'], 'raw_keys': list(s['raw_job']), 'raw_top_identity': {k: v for k, v in s['raw_job'].items() if k not in ('config', 'source_hashes', 'adapter_hashes', 'adapter_source_hashes')}, 'result_identity': s['result_identity']} for s in samples]}, indent=2))
