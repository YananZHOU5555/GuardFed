"""Independently check repaired CPU canary archives and exact paired tensors."""
from pathlib import Path
import datetime
import hashlib
import io
import json
import tarfile
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'tmp/celeba_hybrid_realimage_gate_20261009/repaired_two_completed_backup_20261009'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(BASE/'FILES_SHA256.json') == 'a2dcdb3d97fa83ae431ac96bed6b2de36d6af8e2ee7bcb5c2f3f4779f0e67abd'
for name,row in read(BASE/'FILES_SHA256.json')['members'].items():
    assert sha(BASE/name)==row['sha256'] and (BASE/name).stat().st_size==row['bytes']
archive=BASE/'two_repaired_CANARY_incremental.tar.gz'
assert sha(archive)=='7fe104d9a4daee550ca6d962cfab93edb8e7419823b341388e7d37b0dd81ca8d'
with tarfile.open(archive) as bundle:
    members=bundle.getmembers()
    assert len(members)==len({m.name for m in members})==56 and all(m.isfile() for m in members)
    data={m.name:bundle.extractfile(m).read() for m in members}
receipt=read(BASE/'backup_receipt.json')
assert hashlib.sha256(data['backup_inventory.json']).hexdigest()==receipt['inventory_sha256']
inventory=json.loads(data['backup_inventory.json'])
assert set(data)==set(inventory['members'])|{'backup_inventory.json'}
for name,row in inventory['members'].items():
    assert len(data[name])==row['bytes'] and hashlib.sha256(data[name]).hexdigest()==row['sha256']
delivery=read(BASE/'strict_delivery.json')
assert delivery['new_CANARY']==delivery['reused_CANARY']==2 and delivery['scientific_table_records']==0
assert not delivery['test_evaluated'] and not delivery['repeated_IID_inference_or_training']
pair=delivery['new_pairs'][0]
a,b=('new/'+pair[k]+'/' for k in ('hybrid','legacy'))
left,right=(torch.load(io.BytesIO(data[p+'model.pt']),map_location='cpu',weights_only=True) for p in (a,b))
assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
for name in ('diagnostics.json','rng_final.json'):
    assert json.loads(data[a+name])==json.loads(data[b+name]),name
for record in delivery['records']:
    assert record['metrics']==dict(accuracy=0.5166859616449389,aeod=0.0,aspd=0.0)
proof=dict(status='ROOT_ARCHIVE_AND_PAIRED_CPU_TENSORS_PASS',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_seal_sha256=sha(BASE/'FILES_SHA256.json'),archive_sha256=sha(archive),members_verified=56,
    offserver_proof_sha256=sha(BASE/'offserver_verification.json'),new_canaries=2,reused_canaries=2,
    aggregate_four_canaries_accepted=True,original_failed_attempt_preserved=True,
    paired_model_tensors_exact=True,paired_round_diagnostics_rng_exact=True,
    constant_negative_prediction_preserved=True,scientific_table_records=0,
    GPU_or_70_round_equivalence_claim=False,test_started=False)
out=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/HYBRID_REPAIRED_TWO_ROOT_VERIFICATION.json'
with out.open('x',encoding='utf8') as f: json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
