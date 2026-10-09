"""Finite source rebind and new-boundary refusal checks; no scientific runtime."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'tmp/celeba_mechanism_valid_incremental_after71_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def funcs(text):return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
def main():
 b=load('after82_bridge_local_check',HERE/'bridge.py');batch=load('after82_batch_local_check',HERE/'execution_candidate/batch.py')
 inv=read(HERE/'inventory_actual92_Full100refs.json');baseline=read(ROOT/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json')
 assert len(b.validate_inventory(inv,baseline))==92
 oldinv=read(OLD/'inventory_actual82_Full100refs.json');assert len(oldinv['records'])==82
 assert {r['id']:r for r in oldinv['records']}=={r['id']:r for r in inv['records'] if r['id'] in b.EXCLUDED_PRIOR_IDS}
 assert inv['full_references']==oldinv['full_references']
 assert set(b.ACCEPTED_IDS)-set(b.EXCLUDED_PRIOR_IDS)==set(b.REPLAY_IDS) and len(b.REPLAY_IDS)==10
 before=(OLD/'bridge.py').read_text(encoding='utf-8');after=(HERE/'bridge.py').read_text(encoding='utf-8');of,nf=funcs(before),funcs(after)
 exact=[n for n in of if n!='validate_inventory'];assert len(exact)==11 and all(of[n]==nf[n] for n in exact)
 assert of['validate_inventory'].replace('== 82','== 92').replace('exactly82','exactly92').replace('accepted82','accepted92').replace('prior71','prior82').replace('== 718','== 708')==nf['validate_inventory']
 rebind={OLD.name:HERE.name,'INCREMENTAL_AFTER71':'INCREMENTAL_AFTER82','BOUNDED_AFTER71':'BOUNDED_AFTER82','APPROVED_AFTER71':'APPROVED_AFTER82','AFTER71_VALID':'AFTER82_VALID','PREPARED_AFTER71':'PREPARED_AFTER82','valid_after71':'valid_after82','inventory_actual82_Full100refs.json':'inventory_actual92_Full100refs.json','closed71':'closed82','Prior71':'Prior82','prior71':'prior82','exactly82':'exactly92','actual_native82':'actual_native92'}
 for name in ['bridge.py','inventory_actual82_Full100refs.json','FILES_SHA256.json','execution_candidate/RUNTIME_BINDINGS.json']:
  target=name.replace('actual82','actual92');rebind[sha(OLD/name)]=sha(HERE/target)
 oldbridge=load('after71_bridge_local_check',OLD/'bridge.py');rebind[oldbridge.INSPECTION_SHA]=b.INSPECTION_SHA
 reports={}
 for name in ['batch.py','install_once.py','resource_extra.py','verify_saved_increment.py','verify_backup.py','backup_completed.py']:
  oldtext=(OLD/'execution_candidate'/name).read_text(encoding='utf-8');newtext=(HERE/'execution_candidate'/name).read_text(encoding='utf-8');expected=oldtext
  for old,new in rebind.items():expected=expected.replace(old,new)
  if name=='batch.py':
   n=next(n for n in ast.parse(expected).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SELECTED' for t in n.targets))
   expected=expected.replace(ast.get_source_segment(expected,n),'SELECTED = '+repr(b.REPLAY_IDS),1).replace("len(scope['excluded_prior_ids']) == 71","len(scope['excluded_prior_ids']) == 82")
  if name=='install_once.py':expected=expected.replace('len(chosen)==11','len(chosen)==10').replace('reviewed11','reviewed10')
  if name=='backup_completed.py':expected=expected.replace('len(prior|set(ids))<11','len(prior|set(ids))<10').replace('ALL11_STRICT','ALL10_STRICT')
  expected=expected.replace('prepared11','prepared10').replace('reviewed11','reviewed10').replace('Only11 IDs','Only10 IDs').replace('exact11-scope','exact10-scope')
  assert funcs(expected)==funcs(newtext),name
  compile(ast.parse(newtext),name,'exec');reports[name]={'functions':len(funcs(newtext)),'source_exact_after_explicit_identity_count_rebind':True}
 assert sha(OLD/'execution_candidate/resource_extra.py')==sha(HERE/'execution_candidate/resource_extra.py')
 refusals=[]
 def reject(name,run):
  try:run()
  except (ValueError,KeyError,FileNotFoundError) as e:refusals.append({'case':name,'error':str(e)})
  else:raise AssertionError('Must reject: '+name)
 def changed(name,mutate):
  v=copy.deepcopy(inv);mutate(v);reject(name,lambda:b.validate_inventory(v,baseline))
 changed('closed82_ID_cannot_reenter_selected10',lambda v:v['selected_replay_ids'].__setitem__(0,b.EXCLUDED_PRIOR_IDS[0]))
 changed('no_future_terminal_auto_inclusion',lambda v:v['records'][0].__setitem__('id',v['pending_new_ids_no_checkpoint'][0]))
 changed('tolerance_change_refused',lambda v:v.__setitem__('native_tolerance',1e-6))
 index=next(i for i,r in enumerate(inv['records']) if r['id']==b.REPLAY_IDS[0])
 changed('new10_root_ID_drift_refused',lambda v:v['records'][index]['data_contract'].__setitem__('root_image_ids_sha256','0'*64))
 changed('new10_true_alpha_drift_refused',lambda v:v['records'][index].__setitem__('actual_alpha',5000.0))
 changed('new10_checkpoint_drift_refused',lambda v:v['records'][index]['checkpoint'].__setitem__('sha256','0'*64))
 changed('Full_inference_flag_refused',lambda v:v['full_references'][0].__setitem__('replay_required_here',True))
 scope=read(HERE/'SCOPE.json');bindings=read(HERE/'execution_candidate/RUNTIME_BINDINGS.json');batch.bind_scope(scope,bindings)
 bad=copy.deepcopy(bindings);bad['outputs'][b.REPLAY_IDS[0]]='/foreign/'+b.REPLAY_IDS[0];reject('foreign_output_namespace_refused',lambda:batch.bind_scope(scope,bad))
 reject('no_external_approval_refused',lambda:batch.approved(None,None))
 template=HERE/'execution_candidate/APPROVED_TEMPLATE.json';reject('prepared_template_not_dispatch_authority',lambda:batch.approved(template,sha(template)))
 assert 'torch' not in sys.modules and 'numpy' not in sys.modules
 out={'status':'LOCAL_AFTER82_EXACT10_REBIND_AND_REFUSAL_CHECK_PASS','bridge_scientific_functions_exact':11,'validate_inventory_only_snapshot_count_strings_changed':True,'execution_functions':reports,'selected_count':10,'excluded_count':82,'native_inventory_count':92,'Full100_references_and_original82_records_exact':True,'resource_extra_byte_exact':True,'rejections':refusals,'refusal_count':len(refusals),'old_full_suites_rerun':False,'scientific_runtime_imported':False,'CNN_or_server_execution':False,'actual_Linux_preflight':'PENDING_ROOT'}
 with (HERE/'LOCAL_REBIND_CHECK.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':out['status'],'refusals':len(refusals),'execution_functions':sum(v['functions']for v in reports.values())}))
if __name__=='__main__':main()
