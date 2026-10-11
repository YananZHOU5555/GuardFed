"""Prepare the fixed compact increment67 publication source. Never invokes it."""
from pathlib import Path
import ast, datetime, difflib, hashlib, json

R=Path(__file__).resolve().parents[2]; H=Path(__file__).resolve().parent
P=R/'tmp/publication_increment66_prepared_20261011/publish_guardfed_increment66_20261011.py'
T='docs/server_deployment_20260923/training_20260923'
REV='docs/server_deployment_20260923/revision_20260923'
PREVIOUS=T+'/publication_closed_increment66_verified_20261011.json'
PARENT='181969f241d938a93e1e72ac0245a85ae39b3e73'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def pin(rel):
 p=R/rel;assert p.is_file() and not p.is_symlink() and p.stat().st_size<2_000_000,rel
 return {'sha256':sha(p),'bytes':p.stat().st_size}
def save(name,value):
 with (H/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
previous=read(R/PREVIOUS)
assert sha(R/PREVIOUS)=='890d2c79e759de68ba7290060473d7aa8e9694b4296f86849997c9860bcba7de'
assert previous['commit']==PARENT and previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
assert sha(P)=='29de49707bd7ceb617b17b62481483e6d52d3b0a7a7bd5deab02f3f49f863aca'

# Fixed names only. Each old unchanged file becomes a parent-blob reference.
groups={
 'tmp/fl_SDFA9_three_view_source_prepare_20261011':['FILES_SHA256.json','MANIFEST.json','ROOT_SOURCE_REVIEW.json','OLD_SEED91001_REUSE.json'],
 'tmp/fl_SDFA9_three_view_source_prepare_20261011/saved_v2':['audit_saved.py','check_linux.py','contract.py','linux_saved_remote.py','saved_science.py','transport.py','transport_remote.py','verify_arrays.py','FIXED_IDS.json','SOURCE_PINS.json','STATIC_SOURCE_SHA256.json','FILES_SHA256.json'],
 'tmp/fl_SDFA9_three_view_source_prepare_20261011/actual':['bind_audit.py'],
 'tmp/fl_SDFA9_three_view_source_prepare_20261011/actual/saved001':['LINUX_CHECK_COMMAND.json','LINUX_CHECK_EXIT.json','LINUX_CHECK_STDERR.txt','LINUX_CHECK_STDOUT.txt','LINUX_SAVED_CHECK.json','ROOT_AUDIT_ORIGINAL_COMMAND.json','ROOT_AUDIT_ORIGINAL.stderr','ROOT_AUDIT_ORIGINAL.stdout','ROOT_BIND_ORIGINAL_COMMAND.json','ROOT_BIND_ORIGINAL.stderr','ROOT_BIND_ORIGINAL.stdout','SAVED_OUTPUTS_AUDIT_NO_REFIT.json','SAVED_OUTPUTS_AUDIT_NO_REFIT.started.json','TRANSPORT_COMMAND.json','TRANSPORT_REMOTE_BOUND.py','TRANSPORT_SSH_EXIT.json','TRANSPORT_STDERR.txt','TRANSPORT_VERIFICATION.json','WINDOWS_AUDIT_COMMAND.json','WINDOWS_AUDIT_EXIT.json'],
 'tmp/fl90_root_closure_source_prepare_20261011':['adopt_root.py','FILES_SHA256.json','HANDOFF.json','prepare_source.py','README.md','ROOT_INPUTS.template.json','ROOT_INPUTS.json','SOURCE_CHECK.json','SOURCE_DIFF.patch','SOURCE_INVERSE.json','TABLE_ENTRY_REVIEW.json','TABLE_ROOT_BINDING.template.json','ROOT_SCIENTIFIC_ADOPTION.json'],
 'tmp/fl90_independent_actual_review_20261011':['REVIEW.json','FILES_SHA256.json'],
 'tmp/fl_SDFA9_three_view_source_prepare_20261011/table':['build.py','verify.py','input_contract.py','SOURCE_CHECK.json','SOURCE_DIFF.patch','INVERSE_EDITS.json','ROOT_BINDING.json'],
 'outputs/guardfed_tables/celeba_flgmm_nine_scenes90_20261011':['records90.json','tables.json','TABLES.md','cells.csv','SOURCE_BINDINGS.json','VERIFICATION.json','ROOT_VERIFICATION.json'],
 'tmp/gradient14_root_closure_actual_20261011':['COMMAND.json','REMOTE.py','stderr.txt','stdout.json','ROOT_SCIENTIFIC_ADOPTION.json'],
 'tmp/gradient14_root_closure_actual_20261011/binding':['COMMAND.json','REMOTE.py','ROLE_CONTRACT.json','ROOT_BINDING.json','stderr.txt','stdout.json'],
 'tmp/gradient14_root_closure_actual_20261011/binding_repair':['COMMAND.json','REMOTE.py','ROLE_CONTRACT.json','ROOT_BINDING.json','stderr.txt','stdout.json'],
 'tmp/gradient14_root_closure_actual_20261011/export':['COMMAND.json','REMOTE.py','stderr.txt','stdout.json'],
 'tmp/gradient14_root_closure_actual_20261011/offserver':['DOWNLOAD_VERIFICATION.json','SCP_COMMAND.json','SCP.stderr','SCP.stdout','VERIFY_ORIGINAL_COMMAND.json','VERIFY_ORIGINAL.stderr','VERIFY_ORIGINAL.stdout'],
 'tmp/gradient14_offserver_closure_prepared_20261011':['export14.py','verify14.py','FILES_SHA256.json','HANDOFF.json','README.md','ROOT_INPUTS.template.json','ROOT_SOURCE_REVIEW.json','SOURCE_CHECK.json','SOURCE_DIFF.patch','SOURCE_PINS.json'],
 'tmp/gradient14_offserver_closure_prepared_20261011/actual_20261011T023737Z':['OFFSERVER_VERIFICATION.json','ROOT_NAMESPACE_CORRECTION.json','VERIFY_ORIGINAL_COMMAND.json','VERIFY_ORIGINAL.stderr','VERIFY_ORIGINAL.stdout'],
 'tmp/gradient96_parent_bootstrap_prepare_20261011':['initial_preflight.py','root_bootstrap.py','job_preflight.original.py','DERIVATION_CHECK.json','FILES_SHA256.json','HANDOFF.json','INVERSE_EDITS.json','README.md','ROOT_INPUTS.template.json','SOURCE_CHECK.json','SOURCE_DIFF.patch','SOURCE_PINS.json'],
 'tmp/gradient96_parent_bootstrap_prepare_20261011/actual001':['COMMAND.json','OBSERVATION_20261011T024849Z.json','REMOTE.py','ROOT_SOURCE_REVIEW_AND_RUNTIME_AUTHORIZATION.json','stderr.txt','stdout.json'],
 'tmp/gradient96_parent_bootstrap_prepare_20261011/actual_parent_correction001':['COMMAND.json','OBSERVATION_20261011T025034Z.json','REMOTE.py','ROOT_RUNTIME_START_VERIFIED.json','stderr.txt','stdout.json'],
 'tmp/rebuttal_FL90_editorial_prepare_20261011':['DIFF.patch','SEMANTIC_DIFF.json','VERIFICATION.json','FILES_SHA256.json'],
 'tmp/native380_adopter_prepare_20261011':['ACTUAL_INPUTS.template.json','adopt_root.py','FILES_SHA256.json','HANDOFF.json','prepare_source.py','README.md','ROOT_SOURCE_REVIEW.json','SOURCE_DIFF.patch','SOURCE_INVERSE.json'],
 'tmp/mechanism_F_nonIID_FedSA10_prepare_20261011':['FILES_SHA256.json','fixed_backup.py','FROZEN_SCOPE.json','HANDOFF.json','PARENT_LEDGER.json','prepare_source.py','README.md','ROOT_INPUTS.template.json','run_native_once.py','SOURCE_CHECK.json','SOURCE_DIFF.patch','SOURCE_INVERSE.json'],
 'tmp/mechanism_F_nonIID_FedSA10_prepare_20261011/root_delta_20261011T024637Z':['ARCHIVE_LOCATION.json','DELTA_VERIFICATION.json','OFFSERVER_VERIFICATION.json','REMOTE_COMMANDS.json','root_delta_20261011T024637Z.tar.gz.receipt.json','STORAGE_PREFLIGHT.json','verified_ledger.json'],
}
fixed=[PREVIOUS,P.relative_to(R).as_posix(),REV+'/rebuttal_working_F70_FL90_Hybrid20_candidate_20261011.md']
fixed += [base+'/'+name for base,names in groups.items() for name in names]
# Runtime wiring dependencies are already explicit in this reviewed local pin map.
fixed += list(read(R/'tmp/gradient96_parent_bootstrap_prepare_20261011/SOURCE_PINS.json'))
old_roles=['native370_root','current_three_view_root','F70_table_root','FL_native86_root','FL_three_view81_root','FL_eight_scene80_table_root','Hybrid_three_view20_root','Hybrid_table20_root','gradient64_root','gradient200_frozen_source_root','current_working_editorial_root']
fixed += [previous['proof_roles'][x]['path'] for x in old_roles]
files={};refs={}
for rel in dict.fromkeys(fixed):
 p=pin(rel);dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
 old=previous['files'].get(dest) or previous.get('parent_references',{}).get(rel)
 (refs if old and (old['sha256'],old['bytes'])==(p['sha256'],p['bytes']) else files)[rel]=p
known_roles={
 'previous66':PREVIOUS,
 'FL_three_view90_root':'tmp/fl90_root_closure_source_prepare_20261011/ROOT_SCIENTIFIC_ADOPTION.json',
 'FL_nine_scene90_table_root':'outputs/guardfed_tables/celeba_flgmm_nine_scenes90_20261011/ROOT_VERIFICATION.json',
 'gradient14_root':'tmp/gradient14_root_closure_actual_20261011/ROOT_SCIENTIFIC_ADOPTION.json',
 'gradient14_offserver':'tmp/gradient14_offserver_closure_prepared_20261011/actual_20261011T023737Z/OFFSERVER_VERIFICATION.json',
 'NGA96_runtime_start':'tmp/gradient96_parent_bootstrap_prepare_20261011/actual_parent_correction001/ROOT_RUNTIME_START_VERIFIED.json',
}
roles={k:{'path':v,'sha256':pin(v)['sha256']} for k,v in known_roles.items()}
late=['native380_root','native380_adoption_command','current_working_editorial_root','editorial_adopter','entry_writer','entry_update_proof','latest_observation_raw']
roles.update({k:{'path':None,'sha256':None} for k in late})
entries=read(R/'tmp/publication_increment66_prepared_20261011/INPUTS_TEMPLATE.json')['entry_paths']
cutoff={'native_new':380,'three_view_models':370,'F_paired_models':70,'F_complete_scenes':7,'FL_native_new':86,'FL_screen_reuse_separate':4,'FL_three_view_total':90,'FL_complete_table_records':90,'FL_complete_table_scenes':9,'gradient_accepted':64,'gradient14_gate_jobs':14,'gradient14_scientific_records':0,'NGA96_new_scientific_accepted':0,'Hybrid_native_new':19,'Hybrid_three_view_total':20,'Hybrid_complete_table_scenes':2}
template={'status':'INCREMENT67_SOURCE_ONLY_TARGET_SCOPE_NOT_ACCEPTANCE','root_accepted':False,'parent_commit':PARENT,'STATE_path':T+'/TRAINING_STATE.json','STATE_sha256':None,'files':files,'parent_references':refs,'entry_paths':entries,'current_accepted':cutoff,'draft_path':REV+'/rebuttal_working_F70_FL90_Hybrid20_release_20261011.md','draft_sha256':None,'proof_roles':roles,'late_role_names':late,'late_files':None,'excluded':['F80 table','non-IID FedSA10 three-view results','NGA96 completed scientific performance','Huber96 started/completed claim','all archives models arrays and raw datasets'],'publisher_creates_scientific_acceptances':False}
save('INPUTS_TEMPLATE.json',template)
save('SELECTION_TEMPLATE.json',{'status':'PENDING_ROOT_INCREMENT67_CLOSED_SELECTION','root_accepted':False,'parent_commit':PARENT,'STATE_sha256':None,'stable_input_sha256':None,'semantic_review_sha256':None,'files':None,'parent_references':refs,'proof_roles':roles,'root_late_bound_files':None})
save('SEMANTIC_TEMPLATE.json',{'status':'PENDING_ROOT_CURRENT_ENTRY_SEMANTIC_REVIEW','root_accepted':False,'STATE_sha256':None,'current_accepted':cutoff,'working_draft_content_cutoff':None,'historical_Git66_cutoff':previous['acceptance_cutoff'],'latest_observation_only':None,'entries':{p:{'sha256':None,'history_exact':None} for p in entries},'source_pins':roles,'gradient14_deployment_only':True,'NGA96_scientific_completed':0,'F80_excluded':True,'final_test':False,'whole_rebuttal_complete':False})
source=P.read_text(encoding='utf8'); original=source;changes=[]
def change(old,new):
 global source
 assert source.count(old)==1,(old[:80],source.count(old));pos=source.index(old);source=source[:pos]+new+source[pos+len(old):];changes.append({'offset':pos,'old':old,'new':new})
change('"""Publish closed native/views370 F70 Hybrid20 and FL86-native compact evidence; gates14 excluded."""','"""Publish root-adopted FL90, deployment14, runtime-only NGA96 and late-bound native380 metadata."""')
change("PARENT='8515952b668c3a22427a8d0db5cc0a2d4c7cf6e3'","PARENT='"+PARENT+"'")
change('ROOT_CLOSED_INCREMENT66_COMPACT_SELECTION','ROOT_CLOSED_INCREMENT67_COMPACT_SELECTION')
change('ROOT_BOUND_INCREMENT66_INPUTS','ROOT_BOUND_INCREMENT67_INPUTS')
change('d11b26e1db08ac16d3f8263b3465e4d0f0f9d35aa8ef10580ed7aaefa7a8f7f1',sha(H/'INPUTS_TEMPLATE.json'))
change("'current_accepted','draft_path','draft_sha256','late_role_names'","'current_accepted','draft_path','late_role_names'")
start=source.index("previous_rel='docs/");end=source.index("entry_paths={",start)
new="""previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment66_verified_20261011.json'
previous=pinned_json(repo_path(previous_rel),'890d2c79e759de68ba7290060473d7aa8e9694b4296f86849997c9860bcba7de')
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
assert (previous['acceptance_cutoff']['native'],previous['acceptance_cutoff']['three_view'],previous['acceptance_cutoff']['F_table_pairs'])==(370,370,70)
# Existing F70/Hybrid20/gradient64 science is referenced through the actual66 receipt, not rerun.
native=proofs['native380_root'];ids=['minus_F_non-IID_FedSA_seed'+str(s) for s in range(91001,91011)]
assert native['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and native['root_adopted'] is True
assert native['total_new_strict_and_offserver']==380 and native['new_ids']==ids and native['old470_records_exact'] is True and native['test'] is False
assert native['prior_root_sha256']==previous['proof_roles']['native370_root']['sha256']
fl=proofs['FL_three_view90_root'];ft=proofs['FL_nine_scene90_table_root']
assert fl['root_adoption'] is True and (fl['FLGMM_total_three_view_records'],fl['new_three_view_records_accepted'])==(90,9)
assert fl['prior_root_sha256']==previous['proof_roles']['FL_three_view81_root']['sha256'] and fl['prior81_objects_order_unchanged'] is True
assert fl['native_root_sha256']==previous['proof_roles']['FL_native86_root']['sha256']
assert fl['Linux_whole_original_saved_check_pass'] is True and fl['Windows_saved_outputs_audit_pass'] is True
assert fl['Windows_saved_outputs_audit_fit_calls']==0 and fl['Windows_original47_exact_refit_pass'] is False and fl['Windows_whole_saved_check_pass'] is False
assert fl['test'] is False and fl['cross_platform_bitwise_recalibration_claimed'] is False
assert ft['root_adoption'] is True and (ft['records_preserved'],ft['complete_scene_records'],ft['retained_partial_records'],ft['scenes'])==(90,90,0,9)
assert ft['source_root90_sha256']==roles['FL_three_view90_root']['sha256'] and ft['old432_statistics_preserved_not_recomputed'] is True
assert (ft['total_statistical_scalars'],ft['Markdown_cells'],ft['CSV_cells'])==(486,243,243) and ft['final_test'] is False
gate=proofs['gradient14_root'];off=proofs['gradient14_offserver'];run=proofs['NGA96_runtime_start']
assert gate['status']=='ROOT_GRADIENT14_SAME_HORIZON_STRICT_OFFSERVER_ADOPTED' and gate['root_adoption'] is True
assert (gate['accepted_gate_jobs'],gate['same_horizon_rounds_per_job'],gate['members_verified'],gate['scientific_table_records'])==(14,3,457,0)
assert gate['all14_offserver_verified'] is True and gate['offserver_original_proof_sha256']==roles['gradient14_offserver']['sha256'] and gate['test'] is False
assert off['status']=='EXACT14_OFFSERVER_ORIGINAL_SAVED_STRICT_MODEL_RNG_AND_MEMBER_PASS' and off['all14_offserver_verified'] is True
assert off['archive_verification']['members_verified']==457 and off['archive_verification']['archive_sha256']==gate['archive_sha256']
assert run['runtime_dispatch_verified'] is True and run['new_jobs']==96 and run['completed70_scientific_records']==0
assert run['original14_root']['sha256']==roles['gradient14_root']['sha256'] and run['Huber_started'] is False and run['coverage192_complete'] is False and run['test'] is False
assert run['first_progress']['record']['round']>=1 and run['current_failure_files']==0
assert entry['gradient14_deployment_only'] is True and entry['NGA96_scientific_completed']==0 and entry['F80_excluded'] is True
editorial=proofs['current_working_editorial_root']
assert editorial['status']=='ROOT_EDITORIAL_ADOPTED_COMPLETE24_F70_FL90_HYBRID20_AUTHOR_REVIEW_DRAFT' and editorial['root_accepted'] is True
assert editorial['candidate_path'].replace('\\\\','/')==template['draft_path'] and editorial['candidate_sha256']==stable_input['draft_sha256']
assert editorial['quote_blocks']==24 and editorial['quote_lines']==38 and editorial['quote_order_exact'] is True
assert editorial['full_agent_and_root_inverse_exact'] is True and editorial['new_statistics_computed']==0
assert editorial['final_test'] is False and editorial['primary_endpoint_selected'] is False and editorial['manuscript_applied'] is False and editorial['whole_rebuttal_complete'] is False
assert H(repo_path(template['draft_path']).read_bytes())==editorial['candidate_sha256']
assert editorial['fixed_scientific_content_cutoff']['mechanism_native']==370 and editorial['fixed_scientific_content_cutoff']['mechanism_three_view']==370
assert editorial['fixed_scientific_content_cutoff']['F_pairs']==70 and editorial['fixed_scientific_content_cutoff']['FL_three_view']==90
state_path=T/'TRAINING_STATE.json';state=pinned_json(state_path,entry['STATE_sha256']);main=state['celeba_mechanism_v1']
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(380,370)
assert main['three_view_counts_by_variant']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':70}
assert main['latest_native_root_proof_sha256']==roles['native380_root']['sha256'] and main['three_view_root_proof_sha256']==previous['proof_roles']['current_three_view_root']['sha256']
assert state['latest_working_rebuttal_draft']['sha256']==state['latest_rebuttal_draft']['sha256']==editorial['candidate_sha256']
assert H(repo_path(roles['entry_writer']['path']).read_bytes())==state['current_entry_writer']['sha256']
assert entry['current_accepted']==template['current_accepted']
assert entry['working_draft_content_cutoff']==state['latest_working_rebuttal_draft']['content_cutoff']==editorial['fixed_scientific_content_cutoff']
assert entry['historical_Git66_cutoff']==previous['acceptance_cutoff']
assert entry['latest_observation_only']==state['latest_five_queue_readonly_observation']
assert entry['latest_observation_only']['new_acceptance']==0 and entry['latest_observation_only']['counts_are_observation_only'] is True
"""
change(source[start:end],new)
start=source.index('own_names={');end=source.index('\nown_paths=',start)
own=['publish_guardfed_increment67_20261011.py','prepare_source.py','INPUTS_TEMPLATE.json','STABLE_INPUTS.json','SELECTION_TEMPLATE.json','SEMANTIC_TEMPLATE.json','SOURCE_CHECK.json','SOURCE_DIFF.patch','INVERSE_DIFF.json','README.md','HANDOFF.json','FILES_SHA256.json']
change(source[start:end],'own_names='+repr(set(own)))
change("bound_paths={roles[k]['path'] for k in template['late_role_names']}|entry_paths|{template['STATE_path']}","bound_paths={roles[k]['path'] for k in template['late_role_names']}|entry_paths|{template['STATE_path'],template['draft_path']}")
change("assert len(bound_paths)==len(template['late_role_names'])+6","assert len(bound_paths)==len(template['late_role_names'])+7")
change("assert not (T/'publication_closed_increment66_verified_20261011.json').exists()","assert not (T/'publication_closed_increment67_verified_20261011.json').exists()")
change("git('commit','-m','Publish adopted native370 F70 Hybrid20 and FL86 native evidence')","git('commit','-m','Publish adopted FL90 and deployment14 with native380 and NGA runtime metadata')")
start=source.index('proof=dict(status=');end=source.index("(T/'publication_closed_increment66",start)
proof="""proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=dict(native=380,three_view=370,FLGMM_native_new=86,FLGMM_screen_reuse_separate=4,FLGMM_three_views=90,FLGMM_table_complete_scene_records=90,FLGMM_table_partial_records=0,FLGMM_table_scenes=9,gradient_search=64,gradient200_frozen_new_jobs=192,gradient200_screen_reuse_separate=8,gradient200_new_scientific_accepted=0,gradient14_gate_jobs=14,gradient14_scientific_table_records=0,NGA96_runtime_started=True,NGA96_completed_scientific_records=0,Hybrid_native_new=19,Hybrid_three_views=20,Hybrid_table_scenes=2,UCA_models=300,F_table_pairs=70,F_table_scenes=7,reviewer_comments=24),proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,latest_observation_sha256=entry['latest_observation_only']['sha256'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published=dict(mechanism=0,FLGMM=9,Hybrid=0),new_native_records_published=10,new_native_records_by_queue=dict(mechanism=10,FLGMM=0,Hybrid=0),working_draft_content_cutoff=entry['working_draft_content_cutoff'],historical_Git66_cutoff=entry['historical_Git66_cutoff'],parent_references=parent_refs,gradient14_evidence_scope='Original14 same-horizon CPU3 deployment gates and457-member offserver proof; zero scientific table records',NGA96_evidence_scope='Original runner wiring and observed first progress only; retained parent-missing startup failure and bounded metadata correction; zero completed70 scientific records',F80_and_FedSA10_three_view_excluded=True,original_Windows47_refit_failure_preserved=True,Windows_whole_refit_pass=False,manuscript_applied=False,primary_endpoint_selected=False,final_test=False,whole_rebuttal_complete=False)
"""
change(source[start:end],proof)
change("(T/'publication_closed_increment66_verified_20261011.json').write_text","(T/'publication_closed_increment67_verified_20261011.json').write_text")
restored=source
for e in reversed(changes):
 pos=e['offset'];assert restored[pos:pos+len(e['new'])]==e['new'];restored=restored[:pos]+e['old']+restored[pos+len(e['new']):]
assert restored==original
compile(source,str(H/'publish_guardfed_increment67_20261011.py'),'exec')
(H/'publish_guardfed_increment67_20261011.py').write_text(source,encoding='utf8',newline='\n')
save('INVERSE_DIFF.json',changes)
(H/'SOURCE_DIFF.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile=P.relative_to(R).as_posix(),tofile=(H/'publish_guardfed_increment67_20261011.py').relative_to(R).as_posix())),encoding='utf8')
# The actual Git/F-volume/blob/push block is byte-identical except receipt and commit-message literals.
def core(s):return s[s.index("G=['git'"):s.index('proof=dict(status=')]
expected=core(original).replace('Publish adopted native370 F70 Hybrid20 and FL86 native evidence','Publish adopted FL90 and deployment14 with native380 and NGA runtime metadata')
assert core(source)==expected
save('SOURCE_CHECK.json',{'status':'SOURCE_ONLY_COMPILE_AND_ORIGINAL_PUBLISHER_INVERSE_PASS_NOT_EXECUTED','parent_publisher_sha256':sha(P),'publisher_sha256':sha(H/'publish_guardfed_increment67_20261011.py'),'input_template_sha256':sha(H/'INPUTS_TEMPLATE.json'),'full_source_inverse_exact':True,'core_Git_F_volume_parent_remote_blob_clean_push_original_except_commit_message':True,'core_AST_equal_after_literal_inverse':ast.dump(ast.parse(core(source).replace('Publish adopted FL90 and deployment14 with native380 and NGA runtime metadata','Publish adopted native370 F70 Hybrid20 and FL86 native evidence')))==ast.dump(ast.parse(core(original))),'fixed_new_files':len(files),'parent_references':len(refs),'fixed_new_bytes':sum(x['bytes'] for x in files.values()),'all_fixed_files_below_2MB':True,'late_roles_left_null':late,'draft_STATE_entries_left_null':True,'publisher_executed':False,'Git':False,'SSH':False,'F_access':False,'root_accepted':False})
print(json.dumps({'fixed_new_files':len(files),'parent_references':len(refs),'fixed_new_bytes':sum(x['bytes'] for x in files.values()),'publisher_sha256':sha(H/'publish_guardfed_increment67_20261011.py')}))
