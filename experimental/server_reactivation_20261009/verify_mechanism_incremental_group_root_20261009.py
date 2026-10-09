"""Verify one new evaluation delta against pinned native checkpoints and valid labels."""
from pathlib import Path
import argparse
import datetime
import hashlib
import io
import json
import tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch',required=True)
parser.add_argument('--archive-sha',required=True)
parser.add_argument('--offserver-sha',required=True)
args=parser.parse_args()
assert args.batch.startswith('incremental_') and '/' not in args.batch and '\\' not in args.batch
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_v2_execution_20261009/backups'/args.batch
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=BASE/'incremental_valid_three_views.tar.gz'
assert sha(archive)==args.archive_sha and sha(BASE/'OFFSERVER_VERIFICATION.json')==args.offserver_sha
receipt=read(BASE/'backup_receipt.json')
with tarfile.open(archive) as bundle:
    members=bundle.getmembers()
    assert len(members)==len({m.name for m in members})==receipt['members'] and all(m.isfile() for m in members)
    data={m.name:bundle.extractfile(m).read() for m in members}
assert hashlib.sha256(data['backup_inventory.json']).hexdigest()==receipt['inventory_sha256']
inventory=json.loads(data['backup_inventory.json'])
assert set(data)==set(inventory['members'])|{'backup_inventory.json'}
for name,row in inventory['members'].items():
    assert len(data[name])==row['bytes'] and hashlib.sha256(data[name]).hexdigest()==row['sha256']
assert not any(n.endswith('.pt') for n in data)
original_path=ROOT/'tmp/celeba_mechanism_valid_incremental_v2_20261009/inventory_actual23_Full100refs.json'
assert sha(original_path)=='288d2afb260f7eb77bcccba82e7edf6dbfe0519cd5bcc42fb546496236eeadbc'
original=read(original_path);originals={r['id']:r for r in original['records']}
ids=receipt['accepted_new_ids']
assert ids and len(ids)==len(set(ids)) and set(ids)<=set(original['selected_replay_ids'])
assert not set(ids).intersection(original['closed_replay_ids'])
cache=ROOT/'tmp/celeba_final_valid_replay_20261009/verification_inputs/original_valid_cache.npz'
assert sha(cache)=='39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
with np.load(cache,allow_pickle=False) as arrays:y,sensitive=arrays['valid_y'],arrays['valid_sensitive']
assert len(y)==len(sensitive)==19867
for identity in ids:
    prefix='runs/'+identity
    r=json.loads(data[prefix+'/receipt.json'])
    assert r['id']==identity and r['native_comparison']['max_abs_difference']==0
    for field,kind in (('checkpoint_sha256','checkpoint'),('original_result_sha256','result'),('original_job_sha256','raw_job')):
        assert r[field]==originals[identity][kind]['sha256']
    assert r['weights_before']==r['weights_after'] and not r['test_inference_performed']
    with np.load(io.BytesIO(data[prefix+'/validation_predictions.npz']),allow_pickle=False) as predictions:
        for view in ('native','raw','shared_calibration'):
            prediction=predictions['prediction_'+view];assert prediction.shape==y.shape
            fit=r['fits'][view]
            if fit['rule']=='argmax_margin_strictly_positive':rule=predictions['valid_margins']>0
            else:
                assert fit['rule']=='group_margin_greater_equal'
                rule=np.where(sensitive==0,predictions['valid_margins']>=fit['thresholds']['0'],predictions['valid_margins']>=fit['thresholds']['1'])
            assert np.array_equal(prediction,rule)
            tpr,rate=[],[]
            for group in (0,1):
                mask=sensitive==group
                counts=dict(tp=int(np.sum(mask&(y==1)&(prediction==1))),fp=int(np.sum(mask&(y==0)&(prediction==1))),
                    tn=int(np.sum(mask&(y==0)&(prediction==0))),fn=int(np.sum(mask&(y==1)&(prediction==0))))
                assert all(r['views'][view]['group_confusion_counts'][str(group)][k]==v for k,v in counts.items())
                tpr.append(counts['tp']/(counts['tp']+counts['fn']))
                rate.append((counts['tp']+counts['fp'])/int(mask.sum()))
            computed=dict(accuracy=int(np.sum(prediction==y))/len(y),aeod=abs(tpr[0]-tpr[1]),aspd=abs(rate[0]-rate[1]))
            assert all(r['views'][view][k]==v for k,v in computed.items())
proof=dict(status='ROOT_INCREMENTAL_ARCHIVE_AND_INDEPENDENT_ARRAY_CHECKS_PASS',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),batch=args.batch,
    archive_sha256=sha(archive),members_verified=len(data),offserver_verification_sha256=sha(BASE/'OFFSERVER_VERIFICATION.json'),
    accepted_new_ids=ids,metric_checks=9*len(ids),confusion_count_checks=24*len(ids),prediction_rule_checks=3*len(ids),
    native_max_abs_difference=0,original_models_repacked=0,Full_reinference=0,test_started=False,goal_complete=False)
out=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'/('MECHANISM_VALID_'+args.batch.upper()+'_ROOT_VERIFICATION.json')
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
