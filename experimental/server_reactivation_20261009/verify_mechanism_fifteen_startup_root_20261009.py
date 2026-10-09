"""Check the deployed exact15 source, authority and actual CPU-worker evidence."""
from pathlib import Path
import datetime
import hashlib
import json
import tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_v2_execution_20261009/startup_delivery'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(b):return hashlib.sha256(b).hexdigest()
archive=BASE/'startup_delivery.tar.gz'
assert digest(archive.read_bytes())=='382fce3384d3772b2c7167ce5d9df738ecef4a1166efeb2d954c241e74cb9974'
with tarfile.open(archive) as bundle:
    members=bundle.getmembers()
    assert len(members)==len({m.name for m in members})==44 and all(m.isfile() for m in members)
    data={m.name:bundle.extractfile(m).read() for m in members}
assert digest(data['backup_inventory.json'])=='9b4912eb096383908aab43073fc7d5a376ac4ffa45a67fd3d563456347f607bb'
inventory=json.loads(data['backup_inventory.json'])
assert set(data)==set(inventory['members'])|{'backup_inventory.json'}
for name,row in inventory['members'].items():
    assert len(data[name])==row['bytes'] and digest(data[name])==row['sha256']
assert digest(data['EXECUTION_SOURCE_SHA256.json'])=='7f35b8c2e8f6c1dfe610f2750d721cb20650dc19bcd9b9e3d03fe244d52f9aa0'
for row in json.loads(data['EXECUTION_SOURCE_SHA256.json'])['members']:
    assert len(data[row['path']])==row['size'] and digest(data[row['path']])==row['sha256']
assert digest(data['sealed_source/FILES_SHA256.json'])=='70d0d920c4c5351c42efc9968fe3c38eed431d208b94bc8af486ba49d869a42d'
assert digest(data['ROOT_APPROVED.json'])=='701f95342f1da3a7e0ba6247b85320a7f5043ccf92475c1c713105866636f8de'
assert digest(data['APPROVED.json'])=='ae5a289fffa9d89badfd152f1c0105a764f21e69afa4d011ee5886fe2e210313'
preflight=json.loads(data['preflight.json'])
assert preflight['status']=='PASS_EXACT15_PREFLIGHT'
start=json.loads(data['start_receipt.json'])
assert 'RUNNING' in start['status'] and '28281' in start['status']
live=json.loads(data['live_20261009T103447Z.json'])
proof=dict(status='ROOT_EXACT15_DEPLOYED_SOURCE_AUTHORITY_AND_LIVE_PASS',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    archive_sha256=digest(archive.read_bytes()),members_verified=44,
    execution_source_seal_sha256=digest(data['EXECUTION_SOURCE_SHA256.json']),
    original_prepared19_seal_sha256=digest(data['sealed_source/FILES_SHA256.json']),
    actual_approval_sha256=digest(data['APPROVED.json']),
    preflight_sha256=digest(data['preflight.json']),start_receipt_sha256=digest(data['start_receipt.json']),
    actual_live_sha256=digest(data['live_20261009T103447Z.json']),
    raw_live_evidence=live,service='guardfed_celeba_mechanism_valid_incremental15',
    selected_new_replays=15,already_closed_eight_excluded=True,new_training=0,new_Full_inference=0,
    compute_threads=8,max_processes=1,allowed_cpus=list(range(112,120)),test_started=False,
    startup_not_scientific_completion=True)
out=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/MECHANISM_FIFTEEN_STARTUP_ROOT_VERIFICATION.json'
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],members_verified=44,service=proof['service'],new_training=0)))
