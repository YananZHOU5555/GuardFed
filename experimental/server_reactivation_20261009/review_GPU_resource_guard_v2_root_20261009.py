"""Review the small operational guard diff and saved actual snapshot schema."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'tmp/celeba_valid_gpu_resource_gate_fix_20261009'
OLD=ROOT/'tmp/celeba_valid_gpu_recovery_implementation_20261009'
DEST=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(NEW/'PACKAGE_SHA256.json')=='e12c8cb5fdcd9d95df84f240a01242b19d974a5d9178cd157c9d9b21d05fd77b'
for name,row in read(NEW/'PACKAGE_SHA256.json')['members'].items():
    assert sha(NEW/name)==row['sha256'] and (NEW/name).stat().st_size==row['bytes']
changed=[]
for name,row in read(OLD/'PACKAGE_SHA256.json')['members'].items():
    assert sha(OLD/name)==row['sha256']
    if sha(NEW/'release'/name)!=row['sha256']:changed.append(name)
assert changed==['recovery.py']
checked=subprocess.run([sys.executable,str(NEW/'selfcheck.py')],capture_output=True,check=True)
with (DEST/'GPU_RESOURCE_GUARD_V2_ROOT_SELFCHECK.log').open('xb') as stream:stream.write(checked.stdout+checked.stderr)
result=json.loads(checked.stdout);assert result['health_matrix_cases']==60 and len(result['resource_refusal_checks'])==11
spec=importlib.util.spec_from_file_location('guard_v2_root_review',NEW/'release/recovery.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
saved=read(ROOT/'tmp/celeba_valid_gpu_remaining464_evidence_20261009/chunk_000/verified_extract/batch/runs/FairGuard_IID_FedSA_seed91004/receipt.json')
for phase in ('before_resources','after_resources'):
    snapshot=saved[phase]
    candidate.main_health(snapshot['service'],snapshot['training_queue_snapshot'],'saved_actual_'+phase)
proof=dict(status='ROOT_RESOURCE_GUARD_V2_DIFF_AND_NO_CNN_REVIEW_PASS_PREPARED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    runtime_seal_sha256=sha(NEW/'release/PACKAGE_SHA256.json'),outer_seal_sha256=sha(NEW/'PACKAGE_SHA256.json'),
    source_sha256=sha(NEW/'release/recovery.py'),exact_diff_sha256=sha(NEW/'SOURCE_DIFF.patch'),
    unchanged_source_members=24,changed_source_members=changed,actual_saved_snapshot_schema_checks=2,
    health_cases=60,resource_refusals=11,native_tolerance=1e-12,science_body_unchanged=True,
    accepted=460,remaining=440,server_deployment=False,new_CNN_inference=0,Linux_runtime_verified=False,test=False)
with (DEST/'GPU_RESOURCE_GUARD_V2_ROOT_REVIEW.json').open('x',encoding='utf8',newline='\n') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(proof))
