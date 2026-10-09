"""Independently bind the received real900 semantic receipt to stock/map/source."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RECEIPT_SHA = 'cc93d94d69478a4ee190abff76da4fb185791ece7e4e1d2ee75360482ead80b4'
SOURCE_SHA = '43b16d20d2497b7762cd0f6039f7f4bfc8fddd4e4c32979a16291b26e394ae5e'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

path = HERE / 'semantic900_inspection.json'
assert sha(path) == RECEIPT_SHA and sha(HERE / 'replay_v4.py') == SOURCE_SHA
r = read(path)
inventory = BASE / 'inputs/model_inventory.json'
assert sha(inventory) == '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
stock = {x['id']: x for x in read(inventory)['records']}
map_path = BASE / 'v3/inputs/storage_map.json'
assert sha(map_path) == 'e160416101ccb82b42224c9ff1bb337de45ef72fa251197063d1c15e2910f949'
mapping = {(x['id'], x['kind']): x for x in read(map_path)['records']}
assert r['status'] == 'ALL900_ORIGINAL_SEMANTICS_ACCEPTED_NO_INFERENCE' and r['accepted_n'] == 900 and not r['invalid']
assert r['v4_source_sha256'] == SOURCE_SHA and r['sealed_v3_source_sha256'] == 'abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e'
assert r['source_before'] == r['source_after'] and len(r['source_before']) == 43
assert r['inventory_sha256'] == sha(inventory) and r['storage_map_sha256'] == sha(map_path)
assert r['restore_acceptance_sha256'] == '5114c2cd96e5b8ffaf46e40a341619dd8b3547f89d417263c19c6e7f1f33bf77'
rows = {x['id']: x for x in r['accepted']}
assert len(rows) == len(r['accepted']) == 900 and set(rows) == set(stock) == set(r['artifacts'])
schemas, methods = collections.Counter(), collections.Counter()
for model_id, record in stock.items():
    row = rows[model_id]
    canonical = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert row['record_sha256'] == canonical
    assert row['original_native_metrics'] == record['prior_validation_metrics']
    bridge = row['schema_bridge']
    for kind, key in [('raw_job', 'raw_job_sha256'), ('result', 'result_sha256'), ('checkpoint', 'checkpoint_sha256')]:
        assert bridge[key] == record[kind]['sha256']
        item = mapping[model_id, kind]
        identity = r['artifacts'][model_id][item['target']]
        assert identity['sha256'] == item['sha256'] == record[kind]['sha256']
        assert identity['bytes'] == item['bytes'] == record[kind]['bytes']
    assert bridge['historical_output'] == record['original_remote_output']
    assert bridge['runtime_read_output'] == str(Path(mapping[model_id, 'result']['target']).parent).replace('\\', '/')
    assert bridge['original_artifact_bytes_modified'] is False
    methods[record['method']] += 1
    schemas[bridge['schema']] += 1
    for relative, expected in record['source_hashes'].items():
        assert r['source_before']['/workspace/GuardFed-celeba-expanded/' + relative]['sha256'] == expected
    assert set(record['adapter_source_hashes'].values()) <= {v['sha256'] for v in r['source_before'].values()}
assert set(methods.values()) == {100}
assert schemas == {'original_revision_job': 700, 'FedAA_identity_v1_to_private_revision_view': 100, 'LASA_rawjob_to_private_output_view': 100}
assert not r['all900_native_valid_replayed'] and r['new_image_inference'] == 0 and not r['semantic_labels_loaded'] and not r['final_protocol_frozen']
report = {'status': 'REAL900_SEMANTIC_RECEIPT_OFFSERVER_IDENTITY_VERIFIED', 'semantic_receipt_sha256': RECEIPT_SHA, 'source_sha256': SOURCE_SHA, 'accepted_original_semantic_records': 900, 'invalid_n': 0, 'artifact_file_sha_verified_in_received_receipt': 2700, 'sources_before_after_sha_equal': 43, 'schemas': dict(schemas), 'methods': dict(methods), 'native_values': 'Previously accepted inventory values only, no new inference', 'artifact_after_inspection_check': 'Remote stat tokens unchanged; each worker still requires full artifact SHA before and after inference', 'new_image_inference': 0, 'all900_native_valid_replayed': False, 'final_protocol_frozen': False}
p = HERE / 'semantic900_offserver_verification.json'
assert not p.exists()
p.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'receipt_sha256': sha(p), 'schemas': dict(schemas)}, indent=2))
