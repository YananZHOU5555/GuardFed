"""Verify incremental deployment reconstruction and actual first-round GPU evidence."""
from pathlib import Path
import datetime
import hashlib
import json
import tarfile
ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
BASE=NEW/'execution_dispatch_v1/startup_incremental_backup'
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(BASE/'FILES_SHA256.json')=='306a06badf2bdf3b8d02c1c4b1bfddcfe7aaffe5a2414436d3331fe01eb2da8e'
for name,row in read(BASE/'FILES_SHA256.json')['members'].items():
    assert sha(BASE/name)==row['sha256'] and (BASE/name).stat().st_size==row['bytes']
archive=BASE/'hybrid_screen32_source_startup_incremental_20261009.tar.gz'
assert sha(archive)=='3fc949ec7cd3d2413c1258b1719ec1aa746575fb07f113629d954e6b55f45ed1'
with tarfile.open(archive) as bundle:
    members=bundle.getmembers();assert len(members)==len({m.name for m in members})==73
    data={m.name:bundle.extractfile(m).read() for m in members}
receipt=read(BASE/'backup_receipt.json');inventory=json.loads(data['backup_inventory.json'])
assert hashlib.sha256(data['backup_inventory.json']).hexdigest()==receipt['inventory_sha256']
assert set(data)==set(inventory['members'])|{'backup_inventory.json'}
for name,row in inventory['members'].items():assert len(data[name])==row['bytes'] and hashlib.sha256(data[name]).hexdigest()==row['sha256']
source=read(BASE/'SOURCE_RESTORE_MAP.json')
assert len(source['reused_members'])==20 and len(source['fresh_source_members'])==49
prior=ROOT/'tmp/celeba_hybrid_cuda_execution_20261009/execution_attachments/startup_backup/hybrid_cuda_source_startup_20261009.tar.gz'
assert sha(prior)==source['prior_archive_sha256']=='1876769e6073b00ef1131d57a855611d60ab6863f144fefb3bbd20e7ac07d791'
with tarfile.open(prior) as bundle:
    for name,row in source['reused_members'].items():
        payload=bundle.extractfile(row['source_member']).read()
        assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
        assert name not in data;data[name]=payload
expected=read(NEW/'FILES_SHA256.json')['files']
assert expected==source['expected_source_members'] and len(expected)==69
for name,h in expected.items():assert hashlib.sha256(data[name]).hexdigest()==h and sha(NEW/name)==h
handoff=read(BASE/'STARTUP_HANDOFF.json');live=read(BASE/'live_launch_receipt.json');probe=read(BASE/'live_probe_01.json')
assert sha(BASE/'STARTUP_HANDOFF.json')=='eb6db6bcdeb1217523fdea3e4d38cd9e6079179787e0d320808216f06954d86e'
assert handoff['source_seal_sha256']==sha(NEW/'FILES_SHA256.json')=='2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f'
approval=read(BASE/'APPROVED_screen.json');resource=read(BASE/'resource_preflight.json')
assert sha(BASE/'APPROVED_screen.json')=='c984c7bf910ce87156c4901f559eac30f7aebbad28858a942dd5343226c476ee'
assert handoff['root_approval_sha256']=='9fd3bb9a9305bd2cdc95eca07a7a5eef6a5a02c3ab38005f1b63d18ab9bcab1a'
dispatch=read(BASE/'screen_dispatch.json')
assert approval['status']=='APPROVED_32_HYBRID_VALID_SCREEN_ONLY' and len(approval['selected_ids'])==32
assert approval['resource_preflight']['sha256']==sha(BASE/'resource_preflight.json')
assert 0<=dispatch['resources_before']['at_unix']-resource['at_unix']<=90
assert resource['no_duplicate_worker'] and resource['no_restricted_CPU_overlap'] and resource['gpu_free_memory_mib']>=4096
assert resource['gpu_recovery_action']=='None' and resource['existing_nominal_compute_threads']+1<=resource['actual_quota_cores']
assert len(live['workers'])==1 and live['workers'][0]['cpus']==[104] and live['workers'][0]['nice']==10
assert probe['round']==1 and live['first_progress_round']>=1 and live['first_client_rows']==146493
for r in (probe['provenance'],live['provenance']):
    assert r['device']=='cuda:0' and r['torch']=='2.11.0+cu128' and r['cuda_build']=='12.8'
    assert r['cpu_threads']==1 and r['cpu_affinity']==[104] and r['gpu_uuid']=='GPU-da357477-30a7-fddc-344b-a20513b9a2d0'
assert probe['screen_failure'] is None and probe['first_failure'] is None
assert live['source69_before_after_verified'] and live['protected28_source_data_before_after_verified'] and live['protected_formal_real_growth']
assert live['scientific_accepted_records']==0 and not live['formal100_started'] and not live['test_started'] and not live['automatic_retry']
proof=dict(status='ROOT_HYBRID32_INCREMENTAL_SOURCE_AND_REAL_GPU_STARTUP_PASS',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),service='guardfed_celeba_hybrid_screen32',
    observed_service=live['service'],observed_unix=live['at_unix'],first_round=1,snapshot_round=live['first_progress_round'],
    source_seal_sha256=handoff['source_seal_sha256'],scope_sha256=handoff['scope_sha256'],
    root_approval_sha256=handoff['root_approval_sha256'],execution_approval_sha256=sha(BASE/'APPROVED_screen.json'),
    archive_sha256=sha(archive),new_members_verified=73,reused_source_members_verified=20,source_members_restorable=69,
    startup_delivery_seal_sha256=sha(BASE/'FILES_SHA256.json'),cpu_threads=1,allowed_cpus=[104],nice=10,
    physical_gpu=0,gpu_uuid=handoff['gpu_uuid'],nominal_threads_with_new_worker=11,quota_cores=resource['actual_quota_cores'],
    source_and_data_before_after_verified=True,scientific_accepted_records=0,all32_completed=False,
    test_started=False,formal100_started=False,automatic_retry=False,goal_complete=False)
out=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/HYBRID_SCREEN32_STARTUP_ROOT_VERIFICATION.json'
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
