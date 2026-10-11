"""Source-only fixed-ID native collector derivation; no collector execution."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/mechanism_F_nonIID_FFlip10_prepare_20261011'
B=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T014706Z'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
root=json.loads((B/'ROOT_DELTA_VERIFICATION.json').read_bytes());inspection=json.loads((B/'inspection/inspection.json').read_bytes());ledger=json.loads((B/'verified_ledger.json').read_bytes())
assert sha(B/'ROOT_DELTA_VERIFICATION.json')=='4ae0a66ca2cc48289b9a7a18272b83652c73c9da16b2d65989ecd5a1166d002e'
assert root['root_adopted'] is True and root['total_new_strict_and_offserver']==370
assert sha(B/'inspection/inspection.json')==root['inspection_sha256'] and sha(B/'verified_ledger.json')==root['ledger_sha256']
assert len(ledger['entries'])==51 and not inspection['invalid']
oldids=sorted(set(inspection['accepted_new_ids'])-set(root['observed_not_backed_up_ids']))
assert len(oldids)==root['total_new_strict_and_offserver'] and len(inspection['accepted_new_ids'])==root['observed_native_count']
ids=[f'minus_F_non-IID_FedSA_seed{s}' for s in range(91001,91011)]
assert len(ids)==len(set(ids))==10 and not set(ids)&set(oldids)
with (H/'PARENT_LEDGER.json').open('xb') as f:f.write((B/'verified_ledger.json').read_bytes())
changes={};checks={};diff=[]
for name in ('fixed_backup.py','run_native_once.py'):
 original=(P/name).read_bytes();b=original;ed=[]
 def change(old,new):
  global b
  old=old.encode();new=new.encode();positions=[];start=0
  while True:
   pos=b.find(old,start)
   if pos<0:break
   positions.append(pos);start=pos+len(old)
  assert positions,(name,old)
  for pos in reversed(positions):
   b=b[:pos]+new+b[pos+len(old):];ed.append({'offset':pos,'old':old.decode(),'new':new.decode()})
 change('minus_F_non-IID_F Flip_seed','minus_F_non-IID_FedSA_seed')
 if name=='run_native_once.py':
  change('native_delta_after360_nonIID_FFlip10_20261011','native_delta_after370_nonIID_FedSA10_20261011')
  change('mechanism_F_nonIID_FFlip10_prepare_20261011','mechanism_F_nonIID_FedSA10_prepare_20261011')
  change('67c32aced70438bf7bf614c5862792f1887ddeb4fbede5e4da06cc157fd992ef',root['ledger_sha256'])
  change('396e40654b6dfc75029d4a7a42dbfafe4caaf111a872ae2ece5045f6d1203418',sha(B/'ROOT_DELTA_VERIFICATION.json'))
  change("prior_inspected['new_count']==360 and len(prior_inspected['accepted_new_ids'])==360", "prior_inspected['new_count']==PRIOR_OBSERVED_NATIVE and len(PRIOR_ACCEPTED_IDS)==370 and set(prior_inspected['accepted_new_ids'])-set(PRIOR_PENDING_IDS)==set(PRIOR_ACCEPTED_IDS)")
  change("set(prior_inspected['accepted_new_ids'])", "set(PRIOR_ACCEPTED_IDS)")
  # Restore the full-inspection projection guard: this occurrence must use raw observed IDs.
  change("set(PRIOR_ACCEPTED_IDS)-set(PRIOR_PENDING_IDS)==set(PRIOR_ACCEPTED_IDS)", "set(prior_inspected['accepted_new_ids'])-set(PRIOR_PENDING_IDS)==set(PRIOR_ACCEPTED_IDS)")
  change("prior_native=prior_inspected['new_count']", "prior_native=len(PRIOR_ACCEPTED_IDS),prior_observed_native=prior_inspected['new_count']")
  change("inspected['new_count']>=370", "inspected['new_count']>=380")
  change("assert sha(H/'PARENT_LEDGER.json')=='"+root['ledger_sha256']+"'", "assert sha(H/'PARENT_LEDGER.json')=='"+root['ledger_sha256']+"' and len(json.loads((H/'PARENT_LEDGER.json').read_bytes())['entries'])==51")
  change("PRIOR_INSPECTION_SHA=SCOPE['parent_inspection_sha256'],TOOL=TOOL", "PRIOR_INSPECTION_SHA=SCOPE['parent_inspection_sha256'],PRIOR_ACCEPTED_IDS=SCOPE['parent_accepted_ids'],PRIOR_PENDING_IDS=SCOPE['parent_observed_pending_ids'],PRIOR_OBSERVED_NATIVE=SCOPE['parent_observed_native'],TOOL=TOOL")
  change("set(json.loads(Path(ROOT/SCOPE['parent_inspection_path']).read_bytes())['accepted_new_ids'])", "set(SCOPE['parent_accepted_ids'])")
  change('FIXED10_SELECTION_METADATA_AFTER360','FIXED10_SELECTION_METADATA_AFTER370')
  change("__file__=__file__,SCOPE=scope", "__file__=__file__,H=H,SCOPE=scope")
 restored=b
 for e in reversed(ed):
  pos=e['offset'];new=e['new'].encode();assert restored[pos:pos+len(new)]==new
  restored=restored[:pos]+e['old'].encode()+restored[pos+len(new):]
 assert restored==original
 compile(b,str(H/name),'exec')
 with (H/name).open('xb') as f:f.write(b)
 changes[name]=ed;checks[name]={'parent_path':(P/name).relative_to(R).as_posix(),'parent_sha256':sha(P/name),'source_sha256':sha(H/name),'compile':True,'full_raw_inverse_exact':True,'inverse_AST_exact':ast.dump(ast.parse(restored))==ast.dump(ast.parse(original))}
 diff.extend(difflib.unified_diff(original.decode().splitlines(True),b.decode().splitlines(True),fromfile=(P/name).relative_to(R).as_posix(),tofile=(H/name).relative_to(R).as_posix()))
scope={'status':'SOURCE_ONLY_FIXED_F_NONIID_FEDSA10_PENDING_CLOSURE','candidate_ids':ids,'parent_native':root['total_new_strict_and_offserver'],'target_native_if_exact10_root_adopted':root['total_new_strict_and_offserver']+len(ids),'Full_reused':inspection['reused_count'],'parent_root_path':(B/'ROOT_DELTA_VERIFICATION.json').relative_to(R).as_posix(),'parent_root_sha256':sha(B/'ROOT_DELTA_VERIFICATION.json'),'parent_inspection_path':(B/'inspection/inspection.json').relative_to(R).as_posix(),'parent_inspection_sha256':sha(B/'inspection/inspection.json'),'parent_remote_inspection_path':'/workspace/guardfed_checks/server_reactivation_20261009/native_delta_after360_nonIID_FFlip10_20261011/root_delta_20261011T014706Z/inspection/inspection.json','parent_ledger_sha256':sha(H/'PARENT_LEDGER.json'),'parent_ledger_entries':len(ledger['entries']),'parent_observed_native':root['observed_native_count'],'parent_observed_pending_ids':root['observed_not_backed_up_ids'],'parent_accepted_ids':oldids,'fixed_backup_source_sha256':sha(H/'fixed_backup.py'),'future_snapshot_path':None,'future_snapshot_sha256':None,'future_native_root_sha256':None,'future_offserver_sha256':None,'future_archive_members':None,'observed_extra_ids_allowed':True,'extras_exported_as_accepted':False,'execution_authorized':False,'scientific_execution_performed':False}
save('FROZEN_SCOPE.json',scope)
save('SOURCE_INVERSE.json',changes)
with (H/'SOURCE_DIFF.patch').open('x',encoding='utf8',newline='') as f:f.writelines(diff)
# Only compile the original-derived driver; its code is not executed.
tree=ast.parse((H/'run_native_once.py').read_bytes())
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='derive']
guard=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GUARD' for t in n.targets))
ns={'P':R/'tmp/celeba_native_after288_20261011/run_once.py','R':R,'H':H,'sha':sha,'Path':Path,'json':json,'GUARD':guard}
exec(compile(ast.Module(body=selected,type_ignores=[]),'<source-derivation-only>','exec'),ns)
effective,_=ns['derive']();compile(effective,'<unexecuted-native-collector>','exec')
oldtree=ast.parse((P/'fixed_backup.py').read_bytes());newtree=ast.parse((H/'fixed_backup.py').read_bytes())
for fn in ('derive_backup','main'):
 def raw(path,tree):
  n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn)
  return b''.join(path.read_bytes().splitlines(keepends=True)[n.lineno-1:n.end_lineno])
 assert raw(P/'fixed_backup.py',oldtree)==raw(H/'fixed_backup.py',newtree)
save('ROOT_INPUTS.template.json',{'status':'TEMPLATE_NOT_EXECUTION_AUTHORITY','fixed_ids':ids,'parent_native_root_path':scope['parent_root_path'],'parent_native_root_sha256':scope['parent_root_sha256'],'parent_ledger_sha256':scope['parent_ledger_sha256'],'parent_ledger_entries':51,'actual_closed_snapshot_path':None,'actual_closed_snapshot_sha256':None,'root_source_review_sha256':None,'actual_native_root_path':None,'actual_native_root_sha256':None,'actual_archive_path':None,'actual_archive_sha256':None,'actual_archive_members':None,'actual_receipt_sha256':None,'actual_offserver_sha256':None,'actual_ledger_sha256':None,'execution_authorized':False,'root_adopted':False})
save('SOURCE_CHECK.json',{'status':'SOURCE_ONLY_FIXED_FEDSA10_NATIVE_COMPILE_RAW_INVERSE_PASS_NOT_EXECUTED','files':checks,'effective_original_driver_compile':True,'original_v4_tool_sha256':'3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef','fixed_backup_derive_and_main_functions_raw_exact':True,'only_backup_FIXED_IDS_literal_changed':True,'original_strict_archive_member_verify_body_unchanged':True,'original_51ledger_prefix_assertion_retained':True,'parent_accepted_ids_derived_from_actual_root_pending_projection':True,'parent_raw_inspection_preserved':True,'parent_native_accepted':len(oldids),'parent_raw_inspection_observed':inspection['new_count'],'parent_raw_total_records':len(inspection['records']),'parent_accepted_total_records_expected_by_source_scope':len(oldids)+inspection['reused_count'],'namespace_H_repair_retained':True,'source_derivation_only_no_collector_execution':True,'SSH':False,'F_access':False,'science_execution':False,'statistics':False,'STATE_modified':False,'Git_modified':False})
print(json.dumps({'driver':sha(H/'run_native_once.py'),'fixed_backup':sha(H/'fixed_backup.py'),'source_check':sha(H/'SOURCE_CHECK.json'),'parent_accepted':len(oldids),'parent_observed':inspection['new_count']},indent=2))
