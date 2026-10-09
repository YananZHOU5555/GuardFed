"""Preserve and independently confirm a native mismatch; no CNN inference or acceptance."""
from pathlib import Path
import datetime
import hashlib
import json
import socket
import tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_final_valid_replay_20261009/v4/remaining872_attempt1/chunk_036'
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(BASE/'FILES_SHA256')=='80e55f940500f6b5a376e65e70e52bf015aa60a414674a184a8c5b0534a655f4'
for line in (BASE/'FILES_SHA256').read_text().splitlines():
    digest,name=line.split('  ',1);assert sha(BASE/name)==digest,name
archive=BASE/'failure_chunk_evidence.tar.gz'
assert sha(archive)=='fe5c3786ead4d4878e6e6f16c89793e3d10163f75559ef7ad7e798798ccfb555'
inventory=read(BASE/'failure_remote_archive_inventory.json')
with tarfile.open(archive) as bundle:
    members=bundle.getmembers();assert len(members)==len({m.name for m in members})==65
    assert {m.name for m in members}==set(inventory['members'])
    for m in members:
        row=inventory['members'][m.name];payload=bundle.extractfile(m).read()
        assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
assert sha(BASE/'failure_offserver_verification.json')=='8e37410b4ec83f57d5550da962997f449cfd8a07934627327344a4d6eddc5bdc'
off=read(BASE/'failure_offserver_verification.json')
assert off['archive_members_verified']==65 and off['saved_arrays_consistent_models_n']==11
assert off['registered_accepted_n']==0 and off['strict_partial_n_not_registered']==10
identity='FairGuard_IID_F-Flip_seed91009';folder=BASE/'failed_worker_original'/identity
r=read(folder/'receipt.json');assert r['id']==identity and not r['native_comparison']['accepted']
assert r['weights_before']==r['weights_after'] and not r['test_inference_performed']
assert sha(folder/'validation_predictions.npz')==r['prediction_arrays_sha256']
cache=ROOT/'tmp/celeba_final_valid_replay_20261009/verification_inputs/original_valid_cache.npz'
assert sha(cache)=='39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
with np.load(cache,allow_pickle=False) as f:y,s=f['valid_y'],f['valid_sensitive']
with np.load(folder/'validation_predictions.npz',allow_pickle=False) as f:
    margins=f['valid_margins'];prediction=f['prediction_native']
assert len(prediction)==19867 and np.array_equal(prediction,margins>0)
tpr=[];rates=[];counts={}
for group in (0,1):
    mask=s==group
    c=dict(tp=int(np.sum(mask&(y==1)&(prediction==1))),fp=int(np.sum(mask&(y==0)&(prediction==1))),
        tn=int(np.sum(mask&(y==0)&(prediction==0))),fn=int(np.sum(mask&(y==1)&(prediction==0))))
    assert all(c[k]==off['cpu_native_confusion_counts'][str(group)][k] for k in c)
    counts[str(group)]=c;tpr.append(c['tp']/(c['tp']+c['fn']));rates.append((c['tp']+c['fp'])/int(mask.sum()))
observed=dict(accuracy=int(np.sum(y==prediction))/len(y),aeod=abs(tpr[0]-tpr[1]),aspd=abs(rates[0]-rates[1]))
comparison=r['native_comparison'];assert comparison['tolerance']==1e-12 and observed==comparison['observed']
original=read(BASE/'input_original_result.json');assert original['metrics']==comparison['expected']
differences={k:observed[k]-original['metrics'][k] for k in observed}
assert differences==comparison['differences'] and max(abs(v) for v in differences.values())>1e-12
proof=dict(status='ROOT_NATIVE_METRIC_MISMATCH_FAILURE_PRESERVED_OFFSERVER',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verification_host=socket.gethostname(),
    archive_sha256=sha(archive),members_verified=65,failed_model_id=identity,
    terminal_service='guardfed_celeba_valid_remaining872_20261009 EXITED Oct09 11:02 AM',
    terminal_service_observed_by_root='ssh -p60350 root@89.22.197.55 supervisorctl status; stdout inspected, exited services cause nonzero status',
    native_tolerance_unchanged=1e-12,expected_metrics=original['metrics'],observed_cpu_metrics=observed,
    metric_differences=differences,confusion_counts=counts,failed_chunk_partial_strict_not_counted=10,
    old_result_sha256=sha(BASE/'input_original_result.json'),weights_before_after_equal=True,
    closest_cpu_valid_margin=float(margins[np.argmin(abs(margins))]),
    root_cause='Unresolved; one near-zero CPU margin is only a candidate; original per-image GPU outputs absent',
    previous424_valid_results_not_invalidated=True,new_acceptances_from_failed_chunk=0,
    new_CNN_inference=0,retry_started=False,test_started=False,goal_complete=False)
out=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/BASELINE_VALID_CHUNK036_FAILURE_ROOT_VERIFICATION.json'
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
