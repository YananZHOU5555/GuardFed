"""Independently hash and recompute the first new eleven-model valid chunk."""
from pathlib import Path
import datetime
import hashlib
import io
import json
import socket
import tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_final_valid_replay_20261009'
p=BASE/'v4/remaining872_attempt1/chunk_000'
def read(q):return json.loads(q.read_text(encoding='utf8'))
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
inventory=read(p/'remote_archive_inventory.json'); archive=p/inventory['archive']
assert sha(archive)==inventory['sha256']=='ec87c8deb4600ec12f484f1bea72c69a5f4f239a7e44e03174efef79963910db'
with tarfile.open(archive) as t:
    members=t.getmembers();assert len(members)==len({m.name for m in members})==64
    data={}
    for m in members:
        row=inventory['members'][m.name];assert m.isfile() and m.size==row['bytes']
        data[m.name]=t.extractfile(m).read();assert hashlib.sha256(data[m.name]).hexdigest()==row['sha256']
manifest=json.loads(data['execution/manifest.json'])
assert hashlib.sha256(data['execution/manifest.json']).hexdigest()=='ad6eebf517f534fb8489acb241c51a9ec5328bb285406e55275f7dd9c0c3ed43'
assert inventory['accepted_ids']==manifest['chunks'][0]['ids'] and len(inventory['accepted_ids'])==11
cache=BASE/'verification_inputs/original_valid_cache.npz'
assert sha(cache)=='39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
with np.load(cache,allow_pickle=False) as z:y,sensitive=z['valid_y'],z['valid_sensitive']
assert len(y)==len(sensitive)==19867
for model_id in inventory['accepted_ids']:
    prefix='batch/runs/'+model_id
    receipt=json.loads(data[prefix+'/receipt.json'])
    assert receipt['id']==model_id and receipt['native_comparison']['max_abs_difference']==0
    assert receipt['weights_before']==receipt['weights_after'] and not receipt['test_inference_performed']
    with np.load(io.BytesIO(data[prefix+'/validation_predictions.npz']),allow_pickle=False) as arrays:
        for view in ('native','raw','shared_calibration'):
            prediction=arrays['prediction_'+view];tpr=[];rate=[]
            assert prediction.shape==y.shape
            fit=receipt['fits'][view]
            if fit['rule']=='argmax_margin_strictly_positive':rule=arrays['valid_margins']>0
            else:
                assert fit['rule']=='group_margin_greater_equal'
                rule=np.where(sensitive==0,arrays['valid_margins']>=fit['thresholds']['0'],arrays['valid_margins']>=fit['thresholds']['1'])
            assert np.array_equal(prediction,rule)
            for group in (0,1):
                mask=sensitive==group
                counts=dict(tp=int(np.sum(mask&(y==1)&(prediction==1))),fp=int(np.sum(mask&(y==0)&(prediction==1))),
                            tn=int(np.sum(mask&(y==0)&(prediction==0))),fn=int(np.sum(mask&(y==1)&(prediction==0))))
                assert all(receipt['views'][view]['group_confusion_counts'][str(group)][k]==v for k,v in counts.items())
                tpr.append(counts['tp']/(counts['tp']+counts['fn']));rate.append((counts['tp']+counts['fp'])/int(mask.sum()))
            computed=dict(accuracy=int(np.sum(prediction==y))/len(y),aeod=abs(tpr[0]-tpr[1]),aspd=abs(rate[0]-rate[1]))
            assert all(receipt['views'][view][k]==v for k,v in computed.items())
proof=read(p/'offserver_verification.json');assert proof['accepted_n']==11 and proof['status']=='PASS'
collection=read(BASE/'v4/remaining872_execution_20261009/cumulative_39_accepted.json')
assert collection['accepted_n']==len(set(collection['accepted_ids']))==39
assert set(inventory['accepted_ids'])<=set(collection['accepted_ids']) and len(collection['missing_ids'])==861
assert collection['collector_source_sha256']=='19066f63c341b9ee23b7c6f491802cfdde0c1c0c833c2fe64724a16de9bb2234'
report=dict(status='ROOT_ARCHIVE_AND_INDEPENDENT_ARRAY_CHECKS_PASS',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    verification_host=socket.gethostname(),archive_sha256=sha(archive),archive_members=64,accepted_new_models=11,
    direct_metric_checks=99,confusion_count_checks=264,prediction_rule_checks=33,native_max_abs_difference=0,
    original_model_files_repacked=0,cumulative_actual_models=39,remaining=861,
    offserver_proof_sha256=sha(p/'offserver_verification.json'),cumulative_collection_sha256=sha(BASE/'v4/remaining872_execution_20261009/cumulative_39_accepted.json'),
    test_started=False,goal_complete=False)
out=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/VALID_CHUNK000_ROOT_VERIFICATION.json'
with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
