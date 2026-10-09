"""Check sealed CUDA gate artifacts and paired scientific values without inference."""
from pathlib import Path
import datetime
import hashlib
import io
import json
import tarfile
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_hybrid_cuda_execution_20261009/completed_four_incremental_backup'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(BASE/'FILES_SHA256.json')=='bff8361b5aa46adff768a5d521bd0bb02fcc86d955f7fb4e930317a48019f290'
for name,row in read(BASE/'FILES_SHA256.json')['members'].items():
    assert sha(BASE/name)==row['sha256'] and (BASE/name).stat().st_size==row['bytes']
archive=BASE/'hybrid_cuda_four_artifacts_20261009.tar.gz'
assert sha(archive)=='6d8ab8f15e488479cc9c3342281261d34fe54a790f5595db3a6bb6b40ed76132'
assert sha(BASE/'cuda_gate_delivery.json')=='eb09a8dccb9d9dc1ebeaf7b5da8ec9428287db5dcfaa65764d4d6db2a40adc0d'
assert sha(BASE/'offserver_verification.json')=='1be047dfa597307bda90cb135ba56dbf5d2949af3e8a223ca73011da5e2e5ef0'
with tarfile.open(archive) as bundle:
    members=bundle.getmembers()
    assert len(members)==len({m.name for m in members})==49 and all(m.isfile() for m in members)
    data={m.name:bundle.extractfile(m).read() for m in members}
receipt=read(BASE/'backup_receipt.json')
assert hashlib.sha256(data['backup_inventory.json']).hexdigest()==receipt['inventory_sha256']
inventory=json.loads(data['backup_inventory.json'])
assert set(data)==set(inventory['members'])|{'backup_inventory.json'}
for name,row in inventory['members'].items():
    assert len(data[name])==row['bytes'] and hashlib.sha256(data[name]).hexdigest()==row['sha256']
delivery=read(BASE/'cuda_gate_delivery.json')
assert delivery['status']=='FOUR_CUDA_CANARIES_STRICT_ACCEPTED_OFFSERVER'
assert delivery['new_CANARY']==4 and delivery['scientific_table_records']==0 and not delivery['test_evaluated']
assert delivery['source_seal_sha256']=='8c08bbabe0321adb4cf0f417a0784af58a7ddab3381a2b6c03e3123f79c3f112'
assert delivery['source_backup_reference']['archive_sha256']=='1876769e6073b00ef1131d57a855611d60ab6863f144fefb3bbd20e7ac07d791'
for pair in delivery['pairs']:
    a,b=('gate_runs/'+pair[k]+'/' for k in ('hybrid','legacy'))
    left,right=(torch.load(io.BytesIO(data[p+'model.pt']),map_location='cpu',weights_only=True) for p in (a,b))
    assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
    assert json.loads(data[a+'rng_final.json'])==json.loads(data[b+'rng_final.json'])
    left_diag,right_diag=(json.loads(data[p+'diagnostics.json']) for p in (a,b))
    assert [{k:v for k,v in r.items() if k!='cumulative_seconds'} for r in left_diag]==[{k:v for k,v in r.items() if k!='cumulative_seconds'} for r in right_diag]
    left_result,right_result=(json.loads(data[p+'result.json']) for p in (a,b))
    for key in ('metrics','trajectory_metrics','attack_audit','evaluation_stats','data_contract','last10_metrics'):
        assert left_result[key]==right_result[key],key
    assert len(left_result['round_summaries'])==len(right_result['round_summaries'])==3
    for l,r in zip(left_result['round_summaries'],right_result['round_summaries']):
        assert {k:v for k,v in l.items() if k!='aggregate'}=={k:v for k,v in r.items() if k!='aggregate'}
        for key in ('selected_clients','trust_scores'): assert l['aggregate'][key]==r['aggregate'][key]
for record in delivery['records']:
    prefix='gate_runs/'+record['id']+'/'
    accepted=json.loads(data[prefix+'acceptance.json'])
    assert accepted['status']=='PASS'
    for name,digest in accepted['artifact_hashes'].items(): assert hashlib.sha256(data[prefix+name]).hexdigest()==digest
    assert record['rounds']==3 and record['metrics']==dict(accuracy=0.5166859616449389,aeod=0.0,aspd=0.0)
    native=json.loads(data[prefix+'native_replay.json'])
    assert native['metrics']==record['metrics'] and native['prediction_count']==19867
    assert record['gpu_uuid']=='GPU-da357477-30a7-fddc-344b-a20513b9a2d0'
proof=dict(status='ROOT_ARCHIVE_AND_PAIRED_CUDA_TENSORS_PASS',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_seal_sha256=delivery['source_seal_sha256'],archive_sha256=sha(archive),members_verified=49,
    offserver_proof_sha256=sha(BASE/'offserver_verification.json'),actual_cuda_canaries_accepted=4,
    pairs=delivery['pairs'],paired_model_tensors_exact=True,paired_round_diagnostics_rng_exact=True,
    cumulative_wall_time_excluded_as_original_frozen_comparator=True,constant_negative_prediction_preserved=True,
    scientific_table_records=0,CPU_CUDA_or_70_round_equivalence_claim=False,test_started=False,goal_complete=False)
out=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/HYBRID_CUDA_FOUR_ROOT_VERIFICATION.json'
with out.open('x',encoding='utf8') as f: json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
