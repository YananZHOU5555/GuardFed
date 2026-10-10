"""Offline metadata/source checks only: no future scientific record or mean is produced."""
import ast,copy,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
import build as b
inputs=b.verify_inputs();root=b.read(b.OLD/'ROOT_VERIFICATION.json')
assert root['status']=='ROOT_C50_FIVE_SCENE_THREE_VIEW_TABLES_ADOPTED' and root['unique_records']==100
old=b.module('old_C50_builder_metadata',b.OLD/'build.py')
raw=(b.OLD/'snapshot/records.json').read_text('utf8');records=b.read(b.OLD/'snapshot/records.json')['records']
assert len(records)==100 and old.record_spans(raw)==old.record_spans(json.dumps(dict(records=records),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
prior=b.read(b.C6/'inventory_actual156_Full100refs.json')
# Deliberately synthetic metadata only. No fixture written as an actual model or result.
future=copy.deepcopy(prior);future['selected_replay_ids']=b.EXPECTED;future['excluded_prior_replay_ids']=[r['id'] for r in prior['records']]
for i in b.EXPECTED:
 row=copy.deepcopy(next(r for r in prior['records'] if r['id']==b.TEN[0]));row.update(id=i,seed=int(i[-5:]));future['records'].append(row)
assert len(b.scope(prior,future))==60
refused=[]
def reject(label,fn):
 try:fn()
 except (ValueError,KeyError,FileNotFoundError):refused.append(label);return
 raise AssertionError('Accepted drift '+label)
for k,v in [('selected_replay_ids',b.EXPECTED[:-1]),('excluded_prior_replay_ids',[]),('native_tolerance',1e-6)]:
 x=copy.deepcopy(future);x[k]=v;reject(k,lambda x=x:b.scope(prior,x))
for k,v in [('variant','minus_U'),('distribution','IID'),('attack','S-DFA'),('seed',91001),('terminal_round',69),('original_split','test'),('original_n_eval',10),('actual_alpha',5000)]:
 x=copy.deepcopy(future);x['records'][-1][k]=v;reject(k,lambda x=x:b.scope(prior,x))
x=copy.deepcopy(future);x['records'][0]['terminal_round']=69;reject('old156_changed',lambda:b.scope(prior,x))
x=copy.deepcopy(future);x['full_references'][0]['checkpoint_sha256']='0'*64;reject('Full100_reference',lambda:b.scope(prior,x))
x=copy.deepcopy(future);x['records'].append(x['records'][-1]);reject('duplicate',lambda:b.scope(prior,x))
proof=dict(status='ROOT_C_AFTER56_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',prior_three_view_models=156,accepted_new=4,cumulative_three_view_models=160,accepted_new_ids=b.EXPECTED,original156_unchanged=True,source_scope_complete=True,negative_results_preserved=True,science_seal_sha256='fixture_science',execution_seal_sha256='fixture_execution',prior156_root_adoption_sha256=inputs['prior_C6_adoption_sha256'],all_native_differences_zero=True,server_strict_bound_in_saved_receipts=True,new_training=0,new_Full_inference=0,test_inference=False)
b.adoption_metadata(proof,'fixture_science','fixture_execution',inputs['prior_C6_adoption_sha256'])
for k,v in [('status','PREPARED'),('accepted_new',6),('accepted_new_ids',b.EXPECTED[:-1]),('original156_unchanged',False),('science_seal_sha256','wrong'),('execution_seal_sha256','wrong'),('prior156_root_adoption_sha256','wrong'),('all_native_differences_zero',False),('server_strict_bound_in_saved_receipts',False),('negative_results_preserved',False),('test_inference',True),('new_Full_inference',1)]:
 x=copy.deepcopy(proof);x[k]=v;reject('adoption_'+k,lambda x=x:b.adoption_metadata(x,'fixture_science','fixture_execution',inputs['prior_C6_adoption_sha256']))
reject('missing_future_actual_bindings',lambda:b.verify_future_binding({}))
oldpanel=(b.OLD/'panels.py').read_text('utf8');newpanel=(b.H/'panels.py').read_text('utf8')
expected=oldpanel.replace("('IID','Sp-DFA')]","('IID','Sp-DFA'),('non-IID','Benign')]").replace('Only exact C IID Benign10, F Flip10, FedSA10 S-DFA10 and Sp-DFA10 scenes are publishable','Only exact five IID scenes plus non-IID Benign10 are publishable')
assert newpanel==expected
ov=(b.OLD/'verify_numeric.py').read_text('utf8');nv=(b.H/'verify_numeric.py').read_text('utf8')
body=lambda s:s.split('    errors=[]',1)[1].split('    assert len(errors)',1)[0]
assert body(ov).replace('len(rows)==15','len(rows)==18')==body(nv)
counts=lambda s:s.split('    metric_checks=0;count_checks=0',1)[1].split('    assert metric_checks',1)[0]
assert counts(ov)==counts(nv)
fn=lambda s,n:ast.get_source_segment(s,next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==n))
assert fn(ov,'verify_aggregate')==fn(nv,'verify_aggregate')
for name in ['build.py','panels.py','verify_numeric.py','check_prepared.py']:ast.parse((b.H/name).read_text('utf-8-sig'))
run=subprocess.run([sys.executable,'-B',str(b.H/'build.py'),'--output',str(b.H/'NOT_CREATED')],capture_output=True)
assert run.returncode==2 and not (b.H/'NOT_CREATED').exists();refused.append('CLI_external_actual_binding_required')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules and not (b.H/'snapshot').exists()
print(json.dumps(dict(status='PREPARED_ONLY_SOURCE_METADATA_PASS',input_pins=len(inputs['files']),actual_prior_three_views=156,actual_C50_records=100,synthetic_future_scope_fixture_only=True,actual_future_C4_accepted=False,future_exact_ids=b.EXPECTED,refusals=refused,refusal_count=len(refused),statistics_body_unchanged=True,confusion_body_unchanged=True,old_IID_aggregate_function_exact=True,old100_record_bytes_exact=True,planned_records=120,planned_scalar_checks=972,planned_cells=486,planned_metric_checks=1080,planned_group_count_checks=2880,old_IID_aggregate_scalars=162,new_statistics=0,CNN=0),indent=2))
