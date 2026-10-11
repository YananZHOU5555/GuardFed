"""Root closure of exact Full100 with explicit reuse of two accepted canaries."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
R=Path(__file__).resolve().parents[1]
H=R/'tmp/Full100_cuda_saved_closure_prepare_v2_20261011'
sys.path[:0]=[str(H),str(R/'tmp')]
from guardfed_local_storage import check_bulk_storage
from audit_saved import verify_transport,source_check
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
p=argparse.ArgumentParser();p.add_argument('--audit-sha256',required=True);p.add_argument('--transport-sha256',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True);a=p.parse_args()
assert __debug__
D=H/'root_adoption';assert not D.exists()
volume=check_bulk_storage()
seal='e6f6bd5ddedb3b1aa093d6de5b69e377b08cb10701a92dc66eb5b2db7246ed5b'
source_check(seal)
review=H/'ROOT_SOURCE_REVIEW.json'
assert sha(review)=='ded413319bd7e9198db7b1710eef992e3f7faaf812c51febea4602d9287aee33' and read(review)['source_adoptable']
ap=H/'actual/SAVED_OUTPUTS_AUDIT.json';tp=H/'ACTUAL_TRANSPORT_VERIFICATION.json'
assert sha(ap)==a.audit_sha256 and sha(tp)==a.transport_sha256 and sha(a.inputs)==a.inputs_sha256
audit=read(ap);t=read(tp);inp=read(a.inputs);scope=read(H/'SCOPE.json');ids=scope['ids']
assert audit['status']=='FULL100_CUDA_SAVED_CLOSURE_PASS_ROOT_ADOPTION_PENDING'
assert audit['ids']==ids==inp['exact_ids'] and len(ids)==len(set(ids))==100
assert audit['inputs_sha256']==a.inputs_sha256 and audit['source_seal_sha256']==seal
assert audit['transport']==inp['transport']==dict(path=str(tp),sha256=a.transport_sha256)
assert (audit['prior_adopted_reused'],audit['new_zero_fit_checked'])==(2,98)
assert (audit['new_metric_values_checked'],audit['new_integer_base_counts_checked'],audit['new_prediction_rules_checked'])==(882,2352,294)
assert (audit['total_metric_values'],audit['total_integer_base_counts'],audit['total_prediction_rules'])==(900,2400,300)
assert audit['fit_calls']==audit['new_CNN']==0 and not audit['test'] and not audit['root_adopted'] and not audit['uniform500_complete']
prior=read(H/'PRIOR_ADOPTED.json');assert audit['prior']==prior
for v in prior.values():assert sha(v['path'])==v['sha256']
pr=read(prior['root']['path']);pa=read(prior['audit']['path']);pt=read(prior['transport']['path'])
assert pr['accepted']==7 and pr['audit_sha256']==prior['audit']['sha256'] and pr['transport_sha256']==prior['transport']['sha256']
allowedold={'source/metadata.npz'}|{('base' if i.startswith('GuardFed-AD2+') else 'mechanism')+'/runs/'+i+'/validation_predictions.npz' for i in pr['ids']}
px,pm=verify_transport(pt,pr['archive']['sha256'],allowedold)
allowed={g['tag']+'/runs/'+i+'/validation_predictions.npz' for g in scope['groups'] for i in g['full_ids'] if i in scope['delta_ids']}
dx,dm=verify_transport(t,inp['archive_sha256'],allowed)
manifest=read(dx/'ROOT_MEMBERS.json')
assert manifest['scope']==scope and manifest['accepted_sha256']==inp['accepted_sha256']
assert manifest['prior_root_sha256']==prior['root']['sha256'] and manifest['prior_transport_sha256']==prior['transport']['sha256']
assert sum(n.endswith('validation_predictions.npz') for n in dm)==98 and 'source/metadata.npz' not in dm
def resolve(remote):
    pin=manifest['remote_members'][remote];storage=pin['storage'];assert storage['kind'] in ('prior','delta')
    ext,mem=(px,pm) if storage['kind']=='prior' else (dx,dm)
    assert mem[storage['member']]=={k:pin[k] for k in ['sha256','bytes']}
    p=ext/storage['member'];assert sha(p)==pin['sha256'];return p
assert [v['id'] for v in audit['records']]==ids
records={};cells=set()
for row in audit['records']:
    i=row['id'];group=next(g for g in scope['groups'] if i in g['full_ids']);base=group['base']
    cohort=read(resolve(base+'/cohort.json'));record=next(r for r in cohort['records'] if r['id']==i)
    cells.add((record['distribution'],record['attack'],record['seed']))
    aa=read(resolve(base+'/tasks/'+i+'/ACCEPTED.json'));receipt=read(resolve(base+'/runs/'+i+'/receipt.json'));binding=read(resolve(base+'/runs/'+i+'/formal_binding.json'))
    assert sha(resolve(base+'/tasks/'+i+'/ACCEPTED.json'))==inp['accepted_sha256'][i]==row['accepted_sha256']
    assert aa['receipt']['sha256']==row['receipt_sha256'] and aa['arrays']['sha256']==row['arrays_sha256']
    assert aa['acceptance']['views']==receipt['views']==row['views']
    assert aa['acceptance']['binding_sha256']==sha(resolve(base+'/runs/'+i+'/formal_binding.json'))
    assert aa['acceptance']['native_comparison']==receipt['native_comparison']==row['native_comparison']
    assert row['native_comparison']['accepted'] and row['native_comparison']['max_abs_difference']<=1e-12
    assert row['original_Linux_whole_saved_accepted'] and row['fit_calls']==0
    assert binding['source_before']==binding['source_after'] and binding['artifact_before']==binding['artifact_after']
    assert receipt['checkpoint_sha256']==row['checkpoint_sha256']==record['checkpoint']['sha256']
    assert record['terminal_round']==70 and record['original_split']=='valid' and receipt['runtime']['device']=='cuda:0'
    if i in scope['reused_adopted_ids']:assert row==next(v for v in pa['records'] if v['id']==i)
    records[i]=dict(producer=group['tag'],distribution=record['distribution'],attack=record['attack'],seed=record['seed'],checkpoint_sha256=row['checkpoint_sha256'],task_acceptance_sha256=row['accepted_sha256'],receipt_sha256=row['receipt_sha256'],arrays_sha256=row['arrays_sha256'],views=row['views'],native_comparison=row['native_comparison'],saved_fit_hashes=row['saved_fit_hashes'])
assert cells=={(r['distribution'],r['attack'],r['seed']) for r in scope['cells']}
D.mkdir()
proof=dict(status='ROOT_FULL100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_accepted=True,accepted=100,new_unique=98,reused_already_adopted=2,ids=ids,records=records,audit=dict(path=str(ap),sha256=sha(ap)),transport=dict(path=str(tp),sha256=sha(tp)),archive=dict(path=t['archive_path'],sha256=t['archive_sha256'],members=t['archive_members'],bytes=t['archive_bytes']),prior=prior,source_seal_sha256=seal,fresh_F_volume=volume,metric_values_checked=900,integer_base_counts_checked=2400,prediction_rules_checked=300,native_max_abs_difference=max(v['native_comparison']['max_abs_difference'] for v in records.values()),archive_all_members_rehashed=True,original_Linux_saved_gate='whole original root fit/prediction/metrics/native1e-12 accepted',Windows_audit='zero-fit saved predictions/counts/metrics; not whole refit',whole100_complete=True,whole500_complete=False,whole_comparison_complete=False,new_training=False,parameters_changed=False,tolerance_changed=False,test=False,old_CPU_outputs_preserved=True,old_failures_preserved=True)
target=D/'ROOT_SCIENTIFIC_ADOPTION.json'
with target.open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(accepted=100,new_unique=98,sha256=sha(target),path=str(target))))
