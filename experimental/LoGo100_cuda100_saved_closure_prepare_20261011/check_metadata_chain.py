"""Synthetic JSON fixtures only: no real approval or scientific execution."""
from pathlib import Path
import copy,hashlib,json,sys
sys.dont_write_bytecode=True
import metadata_gate as g
H=Path(__file__).resolve().parent
x=g.read(H/'ROOT_INPUTS.template.json');B='/workspace/guardfed_checks/uniform_cuda_LoGo100_SYNTHETIC_ONLY'
def p(path,h='a'):return dict(path=path,sha256=h*64)
x.update(remaining_service='guardfed_SYNTHETIC_REMAINING',canary_complete=p(B+'/canary/COMPLETE.json'),remaining_complete=p(B+'/remaining/COMPLETE.json'),root_canary_gate=p(B+'/ROOT_CANARY_ACCEPTANCE.json'),server_base=B,service='guardfed_SYNTHETIC_ONLY',root_queue_auth=p(B+'/ROOT_QUEUE_AUTH.json'),deployment_seal=p(B+'/DEPLOYMENT_SEAL.json'),task_accepted={i:p(B+'/tasks/'+i+'/ACCEPTED.json') for i in g.IDS})
row=next(r for r in g.read(H/'EXACT100_REFS.json')['records'] if r['post_id'].endswith('_seed91010'));i=row['post_id'];out=B+'/runs/'+i;task=B+'/tasks/'+i
auth=dict(status='ROOT_AUTHORIZED_LOGO100_CALLER_V1',execution_authorized=True,canary_ids=g.CANARY_IDS,exact_cohort={'sha256':g.COHORT_SHA},FedAvg_CUDA_root=x['FedAvg_CUDA_root'],FedAvg_CUDA_index=x['FedAvg_CUDA_index'])
original={r:p('/synthetic/'+r,pin['sha256'][0]) for r,pin in row['original_artifacts'].items()}
for r,pin in original.items():pin['sha256']=row['original_artifacts'][r]['sha256']
settings={'post_rounds':30};job=dict(id=i,candidate='LoGoFair-DP_07',fit_seed=1719,evaluation_split='valid',seed=91010,settings=settings,mapping_sha256=original['mapping.npz']['sha256'],mapping_metadata_sha256=original['mapping_metadata.json']['sha256'])
result=dict(job=job,status='complete',settings=settings,fit_seed=1719,history=[{'round':n} for n in range(1,31)],checkpoint_sha256=row['checkpoint_sha256'],original_result_sha256=row['original_FedAvg_result_sha256'],accepted_margin_cache_sha256='e'*64,root_image_ids_sha256=row['root_image_ids_sha256'],valid_image_ids_sha256=row['valid_image_ids_sha256'],mapping_sha256=job['mapping_sha256'],mapping_metadata_sha256=job['mapping_metadata_sha256'])
approval=dict(status='ROOT_AUTHORIZED_LOGO100_FROZEN_POST_CUDA_CACHE_REEVALUATION',execution_authorized=True,post_id=i,baseline_id=row['baseline_id'],output=out,root_queue_auth=x['root_queue_auth'],FedAvg_CUDA_root=x['FedAvg_CUDA_root'],FedAvg_CUDA_index=x['FedAvg_CUDA_index'],netcal='1.3.6',post_device='cpu',compute_threads=1,max_processes=1,adapter_sha256='4121b32dbaafaa04dcb2d212882e3e6af04ce06cdd4f0c32f949e1e0453bb3f1',adapter={'sha256':'4121b32dbaafaa04dcb2d212882e3e6af04ce06cdd4f0c32f949e1e0453bb3f1'},runtime_dependencies={'core':{'sha256':g.CORE_SHA},'bridge':{'sha256':'0527f26e9c22da94f35d647d96ab06b227568c9febd9850c6ac5f251a408c16d'}},original_artifacts=original)
for k in ['new_CNN','refit','reselect_recipe','test','automatic_retry','diagnostic_reuse','CPU_reference_equality_required']:approval[k]=False
aa=dict(post_id=i,baseline_id=row['baseline_id'],root_adopted=False,acceptance=p(out+'/formal_saved_acceptance.json'),binding=p(out+'/formal_binding.json'),approval=p(task+'/approval.json'))
formal=dict(status='FROZEN_LOGO_POST_CUDA_CACHE_WHOLE_CHECK_PASSED_NOT_ROOT_ADOPTED',post_id=i,baseline_id=row['baseline_id'],approval_sha256='a'*64,FedAvg_CUDA_root_sha256=g.ROOT_SHA,frozen_post_state_sha256=original['post_state.pkl']['sha256'],original_saved_consistency_tolerance=1e-12,new_fit=0,new_CNN=0,root_adopted=False,test=False,historical_CPU_equality_required=False,result_sha256='a'*64,acceptance_sha256='a'*64,CUDA_cache_sha256='e'*64,historical_CPU_difference={'accuracy':0.01})
accepted=dict(status='PER_RECORD_LOGO_FROZEN_POST_CUDA_CACHE_SAVED_ACCEPTED',new_fit=0,new_CNN=0,root_adopted=False,test=False,result=result,historical_CPU_difference=formal['historical_CPU_difference'])
artifact=dict(status='PASS',job_sha256=original['job']['sha256'],artifact_hashes={r:original[r]['sha256'] for r in ['post_state.pkl','mapping.npz','mapping_metadata.json']});artifact['artifact_hashes'].update({'result.json':'a'*64,'scores_predictions.npz':'f'*64})
docs={x['root_queue_auth']['path']:auth,x['task_accepted'][i]['path']:aa,aa['approval']['path']:approval,aa['binding']['path']:formal,aa['acceptance']['path']:accepted,out+'/result.json':result,out+'/acceptance.json':artifact,original['job']['path']:job}
g.input_gate(x);assert g.verify_record(x,row,lambda pin:docs[pin['path']])['scores']['sha256']=='f'*64
reject=0
wrong_seed=copy.deepcopy(docs);wrong_seed[original['job']['path']]['seed']=91001
try:g.verify_record(x,row,lambda pin:wrong_seed[pin['path']])
except ValueError:reject+=1
else:raise AssertionError('Mismatched per-row seed accepted')
mutations=[(aa['acceptance']['path'],lambda d:d.update(status='RUNNING')),(aa['approval']['path'],lambda d:d.update(refit=True)),(aa['binding']['path'],lambda d:d.update(original_saved_consistency_tolerance=1e-6)),(out+'/acceptance.json',lambda d:d['artifact_hashes'].update({'post_state.pkl':'z'*64})),(out+'/result.json',lambda d:d.update(checkpoint_sha256='z'*64)),(aa['binding']['path'],lambda d:d.update(FedAvg_CUDA_root_sha256='z'*64)),(aa['acceptance']['path'],lambda d:d.update(new_CNN=1)),(aa['approval']['path'],lambda d:d['runtime_dependencies']['bridge'].update(sha256='z'*64))]
for path,fn in mutations:
    ds=copy.deepcopy(docs);ds[path]=copy.deepcopy(ds[path]);fn(ds[path])
    try:g.verify_record(x,row,lambda pin:ds[pin['path']])
    except (ValueError,KeyError):reject+=1
    else:raise AssertionError('Tampering accepted: '+path)
report=dict(status='SYNTHETIC_METADATA_CHAIN_PASS_NOT_ACTUAL_EVIDENCE',positive=1,rejected=reject,scientific_imports=False,real_files_or_approvals_used=False)
with (H/'METADATA_CHAIN_CHECK.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
