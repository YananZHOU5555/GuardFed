"""Verify deployed CUDA gate source and a real first-round GPU observation."""
from pathlib import Path
import datetime
import hashlib
import json
import tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_hybrid_cuda_execution_20261009/execution_attachments/startup_backup'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
seal=read(BASE/'FILES_SHA256.json')
assert len(seal['members'])==10
for name,row in seal['members'].items():
    assert sha(BASE/name)==row['sha256'] and (BASE/name).stat().st_size==row['bytes']
archive=BASE/'hybrid_cuda_source_startup_20261009.tar.gz'
assert sha(archive)=='1876769e6073b00ef1131d57a855611d60ab6863f144fefb3bbd20e7ac07d791'
with tarfile.open(archive) as bundle:
    members=bundle.getmembers()
    assert len(members)==len({m.name for m in members})==81 and all(m.isfile() for m in members)
    data={m.name:bundle.extractfile(m).read() for m in members}
receipt=read(BASE/'backup_receipt.json')
assert hashlib.sha256(data['backup_inventory.json']).hexdigest()==receipt['inventory_sha256']
inventory=json.loads(data['backup_inventory.json'])
assert set(data)==set(inventory['members'])|{'backup_inventory.json'}
for name,row in inventory['members'].items():
    assert hashlib.sha256(data[name]).hexdigest()==row['sha256'] and len(data[name])==row['bytes']
assert hashlib.sha256(data['FILES_SHA256.json']).hexdigest()=='8c08bbabe0321adb4cf0f417a0784af58a7ddab3381a2b6c03e3123f79c3f112'
source_seal=json.loads(data['FILES_SHA256.json'])
assert len(source_seal['files'])==63
for name,digest in source_seal['files'].items():assert hashlib.sha256(data[name]).hexdigest()==digest
for name in ('body.py','driver.py','writer_policy.py'):
    assert data[name]==(ROOT/'tmp/celeba_hybrid_gpu_prepared_20261009'/name).read_bytes()
for name in ('worker.py','adapters.py','accept_result.py','prepare_jobs.py','protocol.json'):
    assert data['scientific_snapshot/'+name]==(ROOT/'tmp/celeba_hybrid_gpu_prepared_20261009/scientific_snapshot'/name).read_bytes()
assert hashlib.sha256(data['APPROVED_gate.json']).hexdigest()=='da49e151d2cc2766da90c3559fac972b504c40ef5d64152822706cb43a112c36'
resource=read(BASE/'resource_preflight.json')
assert resource['cpu_allocation']==[104] and resource['cuda_visible_device']=='0'
assert resource['existing_nominal_compute_threads']==106 and 107<=resource['actual_quota_cores']
assert resource['no_duplicate_worker'] and resource['no_restricted_CPU_overlap'] and resource['gpu_free_memory_mib']>=4096
provenance=json.loads(data['gate_runs/IID_Benign_hybrid_seed91001_cuda_gate3/provenance.json'])
assert provenance['device']=='cuda:0' and provenance['torch']=='2.11.0+cu128'
assert provenance['cpu_threads']==1 and provenance['cpu_affinity']==[104]
assert provenance['gpu_uuid']=='GPU-da357477-30a7-fddc-344b-a20513b9a2d0'
live=read(BASE/'live_progress_01.json')
assert 'RUNNING' in live['service'] and live['items'][0]['round']==1 and not live['queue_failure']
assert read(BASE/'offserver_verification.json')['different_host_observed']
proof=dict(status='ROOT_CUDA4_SOURCE_STARTUP_AND_FIRST_ROUND_PASS',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_sha256=sha(archive),members_verified=81,
    startup_delivery_seal_sha256=sha(BASE/'FILES_SHA256.json'),
    execution_source_seal_sha256=hashlib.sha256(data['FILES_SHA256.json']).hexdigest(),
    actual_approval_sha256=sha(BASE/'APPROVED_gate.json'),offserver_proof_sha256=sha(BASE/'offserver_verification.json'),
    observed_service=live['service'],observed_first_round=1,resource_nominal_compute_threads_including_gate=107,
    quota_cores=resource['actual_quota_cores'],runtime_provenance=provenance,
    startup_not_four_gate_acceptance=True,screen32_started=False,test_started=False)
out=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/HYBRID_CUDA_STARTUP_ROOT_VERIFICATION.json'
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],members_verified=81,first_round=1)))
