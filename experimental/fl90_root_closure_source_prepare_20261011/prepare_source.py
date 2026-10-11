"""Metadata-only derivation of the accepted FL81 complementary root adopter."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/fl_FedSA10_three_view_prepared_20261011/transport_v2/adopt_root.py'
C=R/'tmp/fl_SDFA9_three_view_source_prepare_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (H/'FILES_SHA256.json').exists(),'Source package already sealed'
def save(name,v):
 with (H/name).open('w',encoding='utf8',newline='\n') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
original=P.read_bytes();b=original;ed=[]
def change(old,new):
 global b
 old=old.encode();new=new.encode();offsets=[];start=0
 while True:
  pos=b.find(old,start)
  if pos<0:break
  offsets.append(pos);start=pos+len(old)
 assert offsets,old
 for pos in reversed(offsets):
  b=b[:pos]+new+b[pos+len(old):];ed.append({'offset':pos,'old':old.decode(),'new':new.decode()})
change('Prepared root-only exact10 complementary join.','Prepared root-only exact9 complementary join after prior81.')
change("H=Path(__file__).resolve().parent.parent;R=H.parents[1];A=H/'actual/saved002'", "H=Path(__file__).resolve().parent;R=H.parents[1];C=R/'tmp/fl_SDFA9_three_view_source_prepare_20261011';A=C/'actual/saved001'")
change("ROOT_INPUTS_v2.json","ROOT_INPUTS.json")
for rel in ('FILES_SHA256.json','MANIFEST.json','ROOT_SOURCE_REVIEW.json','saved_v2/STATIC_SOURCE_SHA256.json','saved_v2/FILES_SHA256.json','saved_v2/SOURCE_PINS.json'):
 change("H/'"+rel+"'","C/'"+rel+"'")
change("check_seal(H,'da73ce851d63cd21a33c85e63311ba8ed3300c8b5d8be9d466076d5d9d00d505')", "check_seal(C,inputs['files']['candidate_seal']['sha256'])")
change("check_seal(H/'saved_v2'", "check_seal(C/'saved_v2'")
change('tmp/fl_FFlip10_capacity_pool32_20261011/ROOT_SCIENTIFIC_ADOPTION.json','tmp/fl_FedSA10_three_view_prepared_20261011/ROOT_SCIENTIFIC_ADOPTION.json')
change('tmp/fl_native_after67_FedSA10_prepared_20261011/ROOT_ADOPTION_REVIEW.json','tmp/fl_native_after77_SDFA9_prepared_20261011/ROOT_ADOPTION_REVIEW.json')
change('fe400060961fb923cbac79443f735fab6824b06689d45fb3bbf8d56307c95421','adc7726ca05561c6903ebb153e34fb3dc69a1df0118d87e6bf7dfa3e31655799')
change('5fdf135fd13665df6d528818db81765ad3567d5d754b18c538b1772f3994c80d','904e3ee036dc4df66459f804db1903211b873ffb5de714ccb97f3900e39c824a')
change("'prior71_root'", "'prior81_root'");change("'native77_root'", "'native86_root'")
change("old['FLGMM_total_three_view_records']==71", "old['FLGMM_total_three_view_records']==81")
change("len({x['id'] for x in prior})==71", "len({x['id'] for x in prior})==81")
change('Prior71','Prior81');change('Native77','Native86');change('native-minus71','native-minus81')
change("native['accepted_total']==77", "native['accepted_total']==86")
change("'FLGMM_Tg20_L2.0_lr0.001_non-IID_FedSA_seed'+str(s)+'_fullcoverage' for s in range(91001,91011)", "'FLGMM_Tg20_L2.0_lr0.001_non-IID_S-DFA_seed'+str(s)+'_fullcoverage' for s in range(91002,91011)")
change('LINUX_ORIGINAL_FLGMM10_WHOLE_SAVED_CHECK_PASS_NOT_ROOT_ADOPTED','LINUX_ORIGINAL_FLGMM9_WHOLE_SAVED_CHECK_PASS_NOT_ROOT_ADOPTED')
change("linux['cached_root_refits']==10", "linux['cached_root_refits']==9")
change('FLGMM10_F_TRANSPORT_ALL_MEMBERS_SHA_PASS_NOT_SCIENTIFIC_ACCEPTANCE','FLGMM9_F_TRANSPORT_ALL_MEMBERS_SHA_PASS_NOT_SCIENTIFIC_ACCEPTANCE')
change("t['member_count']==24", "t['member_count']==22")
change("audit['metric_values_checked']==90 and audit['integer_base_counts_checked']==240 and audit['prediction_rules_checked']==30", "audit['metric_values_checked']==81 and audit['integer_base_counts_checked']==216 and audit['prediction_rules_checked']==27")
change('Minimum exact24 scope differs','Minimum exact22 scope differs')
change('len(z.namelist())==24', 'len(z.namelist())==22')
change('FLGMM_EXACT10_THREE_VIEW_PASS_NOT_ROOT_ADOPTED','FLGMM_EXACT9_THREE_VIEW_PASS_NOT_ROOT_ADOPTED')
change('Exact10 proof order differs','Exact9 proof order differs')
change('ROOT_FLGMM_AFTER71_EXACT10_COMPLEMENTARY_EVIDENCE_ADOPTED_PRIOR_WINDOWS47_FAIL_PRESERVED','ROOT_FLGMM_AFTER81_EXACT9_COMPLEMENTARY_EVIDENCE_ADOPTED_PRIOR_WINDOWS47_FAIL_PRESERVED')
change("'prior71_objects_order_unchanged':True", "'prior81_objects_order_unchanged':True")
change("'new_three_view_records_accepted':10", "'new_three_view_records_accepted':9")
change("'FLGMM_total_three_view_records':81", "'FLGMM_total_three_view_records':90")
change("'native_training_records':77", "'native_training_records':86")
change("'Linux_original_root_refit_records_new':10", "'Linux_original_root_refit_records_new':9")
change("'Windows_new10_refit_executed':False", "'Windows_new9_refit_executed':False")
change("'metric_values_checked_new':90", "'metric_values_checked_new':81")
change("'integer_base_counts_checked_new':240", "'integer_base_counts_checked_new':216")
change("'prediction_rules_checked_new':30", "'prediction_rules_checked_new':27")
change("'archive_members':24", "'archive_members':22")
change('New10 terminal-valid endpoints only, plus unchanged prior71.', 'New9 terminal-valid endpoints only, plus unchanged prior81. Old screen91001 remains reused, not replayed; nine complete scenes only, not full100.')
change("'new':10,'total':81", "'new':9,'total':90")
# The reused screen object is already within the parent81 and must stay unchanged.
old="need(len(prior)==len({x['id'] for x in prior})==81,'Prior81 identities differ')"
new=old+"\n reuse=read(C/'OLD_SEED91001_REUSE.json');need(reuse['source_root_sha256']==sha(oldpath) and reuse['record']==next(r for r in prior if r['id']=='FLGMM_Tg20_L2.0_lr0.001_non-IID_S-DFA_seed91001_screen') and reuse['replay_again'] is False,'Original screen91001 reuse identity differs')"
change(old,new)
restored=b
for e in reversed(ed):
 pos=e['offset'];n=e['new'].encode();assert restored[pos:pos+len(n)]==n
 restored=restored[:pos]+e['old'].encode()+restored[pos+len(n):]
assert restored==original
compile(b,str(H/'adopt_root.py'),'exec')
if (H/'adopt_root.py').exists():assert (H/'adopt_root.py').read_bytes()==b,'Only parent provenance corrected; adopter bytes must stay identical'
else:
 with (H/'adopt_root.py').open('xb') as f:f.write(b)
save('SOURCE_INVERSE.json',ed)
with (H/'SOURCE_DIFF.patch').open('w',encoding='utf8',newline='') as f:f.writelines(difflib.unified_diff(original.decode().splitlines(True),b.decode().splitlines(True),fromfile=P.relative_to(R).as_posix(),tofile=(H/'adopt_root.py').relative_to(R).as_posix()))
def func(data,name):
 n=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name==name)
 return b''.join(data.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
def record_loop(data):
 n=next(n for n in ast.walk(ast.parse(data)) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts if isinstance(x,ast.Name)]==['row','g','l','w'])
 return b''.join(data.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
loop=record_loop(b);assert loop==record_loop(original)
for n in ('sha','need','check_seal'):assert func(b,n)==func(original,n)
ids=json.loads((C/'BOUND_EXACT9.json').read_bytes())['exact_ids']
oldroot=R/'tmp/fl_FedSA10_three_view_prepared_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
nativeroot=R/'tmp/fl_native_after77_SDFA9_prepared_20261011/ROOT_ADOPTION_REVIEW.json'
prior=json.loads(oldroot.read_bytes());native=json.loads(nativeroot.read_bytes())
assert ids==[x for x in native['accepted_job_ids'] if x not in {r['id'] for r in prior['records']+prior['prior_interface_explicitly_reused']}]
assert len(prior['records'])==80 and len(prior['prior_interface_explicitly_reused'])==1
roles={'candidate_seal':C/'FILES_SHA256.json','candidate_manifest':C/'MANIFEST.json','candidate_root_review':C/'ROOT_SOURCE_REVIEW.json','saved_static_seal':C/'saved_v2/STATIC_SOURCE_SHA256.json','saved_root_review':C/'ROOT_SOURCE_REVIEW.json','audit_source_seal':C/'saved_v2/FILES_SHA256.json','audit_source_pins':C/'saved_v2/SOURCE_PINS.json','prior81_root':oldroot,'native86_root':nativeroot,'prior_Windows47_failure':R/'tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001/OFFSERVER_ARRAY_REFIT_CHECK.failure.json'}
for role,name in [('Linux_original_whole','LINUX_SAVED_CHECK.json'),('F_transport','TRANSPORT_VERIFICATION.json'),('Windows_zero_fit_audit','SAVED_OUTPUTS_AUDIT_NO_REFIT.json'),('Linux_command','LINUX_CHECK_COMMAND.json'),('Linux_exit','LINUX_CHECK_EXIT.json'),('transport_command','TRANSPORT_COMMAND.json'),('transport_exit','TRANSPORT_SSH_EXIT.json'),('Windows_command','WINDOWS_AUDIT_COMMAND.json'),('Windows_exit','WINDOWS_AUDIT_EXIT.json')]:roles[role]=C/'actual/saved001'/name
known={'candidate_seal','candidate_manifest','candidate_root_review','saved_static_seal','saved_root_review','prior81_root','native86_root','prior_Windows47_failure'}
files={role:{'path':path.relative_to(R).as_posix(),'sha256':sha(path) if role in known else None,'bytes':path.stat().st_size if role in known else None} for role,path in roles.items()}
save('ROOT_INPUTS.template.json',{'status':'TEMPLATE_NOT_ADOPTION_OR_AUTHORIZATION','adoption_source_sha256':sha(H/'adopt_root.py'),'files':files,'exact_ids':ids,'exact_root_receipt_differences':None,'root_reviewed_exact_differences_without_tolerance_substitution':False,'final_test':False})
save('SOURCE_CHECK.json',{'status':'SOURCE_ONLY_FL_AFTER81_EXACT9_ROOT_ADOPTER_COMPILE_RAW_INVERSE_PASS_NOT_EXECUTED','parent_path':P.relative_to(R).as_posix(),'parent_sha256':sha(P),'adopter_sha256':sha(H/'adopt_root.py'),'compile':True,'full_raw_inverse_exact':True,'full_inverse_AST_exact':ast.dump(ast.parse(restored))==ast.dump(ast.parse(original)),'original_per_record_loop_raw_and_AST_exact':True,'per_record_loop_sha256':hashlib.sha256(loop).hexdigest(),'sha_need_check_seal_functions_raw_exact':True,'actual_native_minus_prior81_exact9_metadata_join':True,'old81_object_construction_kept':True,'new_counts_metadata_only':{'new_records':9,'views_after_actual_root':90,'zip_members_expected_by_existing_contract':22,'new_metrics_expected':81,'new_counts_expected':216,'new_rules_expected':27},'actual_candidate_package_sha256':sha(C/'FILES_SHA256.json'),'table_build_source_sha256':sha(C/'table/build.py'),'table_verify_source_sha256':sha(C/'table/verify.py'),'SSH':False,'F_access':False,'scientific_execution':False,'root_adoption':False})
print(json.dumps({'adopter':sha(H/'adopt_root.py'),'parent':sha(P),'source_check':sha(H/'SOURCE_CHECK.json'),'roles':len(roles),'new_ids':ids},indent=2))
