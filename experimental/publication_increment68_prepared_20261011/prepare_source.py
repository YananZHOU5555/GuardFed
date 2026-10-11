"""Prepare increment68 from the actually published67 source; never execute publication."""
from pathlib import Path
import ast,difflib,hashlib,json
R=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent
P=R/'tmp/publication_increment67_prepared_20261011/publish_guardfed_increment67_20261011.py'
T='docs/server_deployment_20260923/training_20260923';REV='docs/server_deployment_20260923/revision_20260923'
B='tmp/mechanism_F_nonIID_FedSA10_three_view_prepare_20261011'
D='tmp/celeba_mechanism_remaining_F_nonIID_FedSA10_root_adoption_20261011'
F=T+'/celeba_mechanism_v1/three_view_F_eight_scenes80_20261011'
PREVIOUS=T+'/publication_closed_increment67_verified_20261011.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
def save(n,x):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def pin(rel):
 p=R/rel;assert p.is_file() and not p.is_symlink() and p.stat().st_size<2_000_000,rel
 return {'sha256':sha(p),'bytes':p.stat().st_size}
previous=read(R/PREVIOUS);PARENT=previous['commit']
assert sha(R/PREVIOUS)=='c58a5bf7804bcbceb30d2a3b515e23af4a40ad44b275ef284960b426037da1ae'
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and PARENT=='3ffad7675c82c4ec7a8e29db541c437e93969590'
assert sha(P)=='f5b30edade3a28ea7096ffcc7265aa517476649fca5a45b866e4d1fe70a1c265'
assert sha(R/D/'ROOT_ADOPTION.json')=='64b37040dc477aa92b08656ae9f029b419607b699646b2b4e74ce135f4851c2b'
assert sha(R/F/'ROOT_VERIFICATION.json')=='0b1a8b65d063253e1f09f60fda09e46af8758e9dcf5bcbadd103a22603f006a0'
names='accepted_scope.py adopt_root.py binding.py bind_native.py build.py check_source.py COMMANDS.template.json cpu_release_readonly.py DOWNLOAD_STARTED.json download_verify_once.py execute_once.py EXECUTION_INPUTS.json export_once.py FILES_SHA256.json finalize_actual_handoff.py HANDOFF.json join_saved.py MECHANISM_INDEX.json NATIVE_IDENTITY_JOIN.json panels.py PREPARED.json prepare_source.py RAW_STORAGE_INDEX.json README.md remote_cpu111_export.py remote_preflight.py ROOT_ACTUAL_PINS.template.json ROOT_ACTUAL_PINS.json root_adopt_actual.py ROOT_BINDING.template.json ROOT_BINDING.json ROOT_SOURCE_REVIEW.json SOURCE_ADAPTATIONS.json SOURCE_CHECK.json SOURCE_DIFF.patch SOURCE_FILES_SHA256.json SOURCE_INPUTS.json SOURCE_PINS.json verify_native_inputs.py verify_numeric.py verify_saved.py ACTUAL_DELIVERY_FILES_SHA256.json ACTUAL_HANDOFF.json CPU_RELEASE.json'.split()
# Fixed command/exit pairs and original outputs; no directory or whole-repository glob.
for stem in ['ACTUAL_BIND','ACTUAL_EXECUTE','ACTUAL_JOIN','ACTUAL_FINALIZE','ACTUAL_BUILD','ACTUAL_VERIFY','CPU_RELEASE','ROOT_ADOPT','ROOT_TABLE_ADOPT']:
 names += [stem+'_COMMAND.json',stem+'_EXIT.json',stem+'.stderr',stem+('.stdout.json' if stem=='ACTUAL_VERIFY' else '.stdout')]
names.remove('CPU_RELEASE.stdout') # the original CPU receipt is CPU_RELEASE.json
for stem in ['EXPORT','OFFSERVER','PREFLIGHT','SCP']:
 names += [stem+'_COMMAND.json',stem+'_EXIT.json',stem+'.stderr.txt',stem+('.stdout.json' if stem in ['EXPORT','PREFLIGHT'] else '.stdout.txt')]
names += ['REMOTE_ARTIFACTS_COMMAND.json','REMOTE_ARTIFACTS.stderr.txt','REMOTE_ARTIFACTS.stdout.json']
table_names=['checks.json','coverage.json','FILES_SHA256.json','paired_per_seed.json','README.md','ROOT_VERIFICATION.json','SOURCE_BINDINGS.json','tables.json','TABLES.md']
fixed=[PREVIOUS,P.relative_to(R).as_posix(),'tmp/finish_FedSA10_saved380_F80_root_once_20261011.py']+[B+'/'+n for n in names]+[F+'/'+n for n in table_names]+[D+'/ROOT_ADOPTION.json',D+'/MECHANISM380_INDEX.json']
fixed += ['tmp/FedSA10_saved380_independent_review_20261011/'+n for n in ['FILES_SHA256.json','REVIEW.json','REVIEW_FAILURE_ATTEMPT001.json','review_source.py','review_source_attempt001.py']]
old_view='tmp/celeba_mechanism_remaining_F_nonIID_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json'
old_table=T+'/celeba_mechanism_v1/three_view_F_seven_scenes70_20261011/ROOT_VERIFICATION.json'
fixed += [old_view,old_table]+[previous['proof_roles'][k]['path'] for k in ['native380_root','FL_three_view90_root','FL_nine_scene90_table_root','gradient14_root','NGA96_runtime_start','current_working_editorial_root']]
files={};refs={}
for rel in dict.fromkeys(fixed):
 p=pin(rel);dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel;old=previous['files'].get(dest) or previous.get('parent_references',{}).get(rel)
 (refs if old and (old['sha256'],old['bytes'])==(p['sha256'],p['bytes']) else files)[rel]=p
known={'previous67':PREVIOUS,'current_three_view_root':D+'/ROOT_ADOPTION.json','saved380_index':D+'/MECHANISM380_INDEX.json','F80_table_root':F+'/ROOT_VERIFICATION.json','saved380_source_seal':B+'/FILES_SHA256.json','saved380_delivery_seal':B+'/ACTUAL_DELIVERY_FILES_SHA256.json','saved380_handoff':B+'/ACTUAL_HANDOFF.json'}
roles={k:{'path':v,'sha256':pin(v)['sha256']} for k,v in known.items()};late=['current_working_editorial_root','editorial_adopter','entry_writer','entry_update_proof','latest_observation_raw'];roles.update({k:{'path':None,'sha256':None} for k in late})
entries=read(R/'tmp/publication_increment67_prepared_20261011/INPUTS_TEMPLATE.json')['entry_paths']
cutoff={'native_new':380,'three_view_models':380,'F_paired_models':80,'F_complete_scenes':8,'FL_native_new':86,'FL_screen_reuse_separate':4,'FL_three_view_total':90,'FL_complete_table_records':90,'FL_complete_table_scenes':9,'gradient_accepted':64,'gradient14_gate_jobs':14,'gradient14_scientific_records':0,'NGA96_new_scientific_accepted':0,'Hybrid_native_new':19,'Hybrid_three_view_total':20,'Hybrid_complete_table_scenes':2}
omitted={F+'/records.json':dict(read(R/F/'ROOT_VERIFICATION.json')['files']['records.json'],reason='Excluded above2MB; immutable source hash retained in actual table root and file manifest. No content discarded or truncated.')}
assert next(iter(omitted.values()))['bytes']==2258686 and not set(omitted)&set(files)
template={'status':'INCREMENT68_SOURCE_ONLY_ACTUAL_SAVED380_F80_FUTURE_METADATA_UNBOUND','root_accepted':False,'parent_commit':PARENT,'STATE_path':T+'/TRAINING_STATE.json','STATE_sha256':None,'files':files,'parent_references':refs,'entry_paths':entries,'current_accepted':cutoff,'draft_path':REV+'/rebuttal_working_F80_FL90_Hybrid20_release_20261011.md','draft_sha256':None,'proof_roles':roles,'late_role_names':late,'late_files':None,'omitted_large_artifacts':omitted,'publisher_creates_scientific_acceptances':False}
save('INPUTS_TEMPLATE.json',template)
save('SELECTION_TEMPLATE.json',{'status':'PENDING_ROOT_INCREMENT68_CLOSED_SELECTION','root_accepted':False,'parent_commit':PARENT,'STATE_sha256':None,'stable_input_sha256':None,'semantic_review_sha256':None,'files':None,'parent_references':refs,'proof_roles':roles,'root_late_bound_files':None})
save('SEMANTIC_TEMPLATE.json',{'status':'PENDING_ROOT_CURRENT_ENTRY_SEMANTIC_REVIEW','root_accepted':False,'STATE_sha256':None,'current_accepted':cutoff,'working_draft_content_cutoff':None,'historical_Git67_cutoff':previous['acceptance_cutoff'],'latest_observation_only':None,'entries':{p:{'sha256':None,'history_exact':None} for p in entries},'source_pins':roles,'gradient14_deployment_only':True,'NGA96_scientific_completed':0,'F80_root_accepted':None,'final_test':False,'whole_rebuttal_complete':False})
source=P.read_text(encoding='utf8');original=source;changes=[]
def change(old,new):
 global source
 assert source.count(old)==1,(old[:80],source.count(old));pos=source.index(old);source=source[:pos]+new+source[pos+len(old):];changes.append({'offset':pos,'old':old,'new':new})
change('"""Publish root-adopted FL90, deployment14, runtime-only NGA96 and late-bound native380 metadata."""','"""Publish root-adopted saved380/F80 compact metadata with future root-bound draft and entries."""')
change("PARENT='181969f241d938a93e1e72ac0245a85ae39b3e73'","PARENT='"+PARENT+"'")
change('ROOT_CLOSED_INCREMENT67_COMPACT_SELECTION','ROOT_CLOSED_INCREMENT68_COMPACT_SELECTION');change('ROOT_BOUND_INCREMENT67_INPUTS','ROOT_BOUND_INCREMENT68_INPUTS')
change('e30f548db530358a55300436c1631e87be0deb9de04437ce10bde9ffd003ee06',sha(H/'INPUTS_TEMPLATE.json'))
change("'draft_path','late_role_names'","'draft_path','late_role_names','omitted_large_artifacts'")
start=source.index("previous_rel='docs/");end=source.index("editorial=proofs[",start)
new="""previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment67_verified_20261011.json'
previous=pinned_json(repo_path(previous_rel),'c58a5bf7804bcbceb30d2a3b515e23af4a40ad44b275ef284960b426037da1ae')
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
assert (previous['acceptance_cutoff']['native'],previous['acceptance_cutoff']['three_view'],previous['acceptance_cutoff']['F_table_pairs'])==(380,370,70)
# Previously published native380/FL90/Hybrid20/gate14/runtime evidence stays referenced, not re-executed.
views=proofs['current_three_view_root'];index=proofs['saved380_index'];table=proofs['F80_table_root']
ids=['minus_F_non-IID_FedSA_seed'+str(s) for s in range(91001,91011)]
assert views['status']=='ROOT_F_NONIID_FEDSA10_EXACT10_SAVED_ARRAYS_REPLAY380_ADOPTED'
assert (views['prior_accepted'],views['new_accepted'],views['cumulative_accepted'],views['remaining620_new_accepted'])==(370,10,380,200)
assert views['original370_unchanged'] is True and views['accepted_new_ids']==ids and views['test'] is False
assert views['native_root_sha256']==previous['proof_roles']['native380_root']['sha256'] and views['native_max_abs_difference']==0
assert views['Full_inference']==views['new_CNN']==views['new_fit']==views['new_training']==0
assert views['records_index_sha256']==roles['saved380_index']['sha256'] and index['new_ids']==ids
assert len(index['all_ids'])==len(set(index['all_ids']))==380
prior_view_path='tmp/celeba_mechanism_remaining_F_nonIID_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json'
assert index['prior_adoption_sha256']==previous['parent_references'][prior_view_path]['sha256']
assert table['status']=='ROOT_F80_EIGHT_SCENES_COUNTS_PAIRED_STATISTICS_DISPLAY_AND_PRIOR_ROWS_PASS'
assert (table['paired_models'],table['complete_scenes'],table['records'])==(80,8,160)
assert table['old140_record_objects_order_exact'] is True and table['old1134_statistical_scalars_exact'] is True and table['old567_display_cells_preserved'] is True
assert table['actual_views380_root_sha256']==roles['current_three_view_root']['sha256'] and table['actual_views380_index_sha256']==roles['saved380_index']['sha256']
assert table['source_seal_sha256']==roles['saved380_source_seal']['sha256'] and views['delivery_seal_sha256']==roles['saved380_delivery_seal']['sha256']
assert table['seed_panels']==[10,9,6] and table['sampleSD_ddof']==1 and table['root_executed_saved_verifier_once'] is True
assert (table['original_fsum_verification']['mean_sd_scalars'],table['original_fsum_verification']['display_cells'])==(1296,648)
assert table['final_test'] is False and table['primary_endpoint_selected'] is False and table['whole_mechanism_complete'] is False
assert table['new_threshold_fits']==table['new_CNN']==table['new_training']==0
assert entry['gradient14_deployment_only'] is True and entry['NGA96_scientific_completed']==0 and entry['F80_root_accepted'] is True
"""
change(source[start:end],new)
change("assert editorial['status']=='ROOT_EDITORIAL_ADOPTED_COMPLETE24_F70_FL90_HYBRID20_AUTHOR_REVIEW_DRAFT' and editorial['root_accepted'] is True","assert editorial['status'].startswith('ROOT_') and 'ADOPTED' in editorial['status'] and editorial['root_accepted'] is True")
change("assert editorial['fixed_scientific_content_cutoff']['mechanism_native']==370 and editorial['fixed_scientific_content_cutoff']['mechanism_three_view']==370","assert editorial['fixed_scientific_content_cutoff']['mechanism_native']==380 and editorial['fixed_scientific_content_cutoff']['mechanism_three_view']==380")
change("assert editorial['fixed_scientific_content_cutoff']['F_pairs']==70 and editorial['fixed_scientific_content_cutoff']['FL_three_view']==90","assert editorial['fixed_scientific_content_cutoff']['F_pairs']==80 and editorial['fixed_scientific_content_cutoff']['FL_three_view']==90")
change("assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(380,370)","assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(380,380)")
change("assert main['three_view_counts_by_variant']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':70}","assert main['three_view_counts_by_variant']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':80}")
change("assert main['latest_native_root_proof_sha256']==roles['native380_root']['sha256'] and main['three_view_root_proof_sha256']==previous['proof_roles']['current_three_view_root']['sha256']","assert main['latest_native_root_proof_sha256']==previous['proof_roles']['native380_root']['sha256'] and main['three_view_root_proof_sha256']==roles['current_three_view_root']['sha256']")
change("assert entry['historical_Git66_cutoff']==previous['acceptance_cutoff']","assert entry['historical_Git67_cutoff']==previous['acceptance_cutoff']")
change("'publish_guardfed_increment67_20261011.py'","'publish_guardfed_increment68_20261011.py'")
anchor="selected=selection['files'];parent_refs=selection['parent_references'];late_files=selection['root_late_bound_files']"
change(anchor,anchor+"\nassert not set(template['omitted_large_artifacts'])&set(selected)\nfor rel,pin in template['omitted_large_artifacts'].items():\n    assert rel.endswith('/three_view_F_eight_scenes80_20261011/records.json')\n    assert (pin['sha256'],pin['bytes'])==(table['files']['records.json']['sha256'],table['files']['records.json']['bytes']) and pin['bytes']>2_000_000")
change("assert not (T/'publication_closed_increment67_verified_20261011.json').exists()","assert not (T/'publication_closed_increment68_verified_20261011.json').exists()")
change("git('commit','-m','Publish adopted FL90 and deployment14 with native380 and NGA runtime metadata')","git('commit','-m','Publish adopted saved380 F80 compact evidence and complete24 author-review update')")
start=source.index('proof=dict(status=');end=source.index("(T/'publication_closed_increment67",start)
newproof="""cutoff=dict(previous['acceptance_cutoff'],three_view=380,F_table_pairs=80,F_table_scenes=8)
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=cutoff,proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,latest_observation_sha256=entry['latest_observation_only']['sha256'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published=dict(mechanism=10,FLGMM=0,Hybrid=0),new_native_records_published=0,new_native_records_by_queue=dict(mechanism=0,FLGMM=0,Hybrid=0),working_draft_content_cutoff=entry['working_draft_content_cutoff'],historical_Git67_cutoff=entry['historical_Git67_cutoff'],parent_references=parent_refs,omitted_large_artifacts=template['omitted_large_artifacts'],scientific_scope='Native380 retained; exact10 non-IID FedSA saved-array records close three-view380 and F80 eight-scene tables. Two non-IID F scenes and four other controls remain. Original14 gates and NGA runtime evidence remain historical parent67 scope, not new scientific results.',original_Windows47_refit_failure_preserved=True,Windows_whole_refit_pass=False,manuscript_applied=False,primary_endpoint_selected=False,final_test=False,whole_rebuttal_complete=False)
"""
change(source[start:end],newproof)
change("(T/'publication_closed_increment67_verified_20261011.json').write_text","(T/'publication_closed_increment68_verified_20261011.json').write_text")
restored=source
for e in reversed(changes):
 pos=e['offset'];assert restored[pos:pos+len(e['new'])]==e['new'];restored=restored[:pos]+e['old']+restored[pos+len(e['new']):]
assert restored==original;compile(source,str(H/'publish_guardfed_increment68_20261011.py'),'exec')
(H/'publish_guardfed_increment68_20261011.py').write_text(source,encoding='utf8',newline='\n');save('INVERSE_DIFF.json',changes)
(H/'SOURCE_DIFF.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile=P.relative_to(R).as_posix(),tofile=(H/'publish_guardfed_increment68_20261011.py').relative_to(R).as_posix())),encoding='utf8')
def core(s,end):return s[s.index("G=['git'"):s.index(end)]
oldcore=core(original,'proof=dict(status=');newcore=core(source,"cutoff=dict(previous['acceptance_cutoff']")
restoredcore=newcore.replace('Publish adopted saved380 F80 compact evidence and complete24 author-review update','Publish adopted FL90 and deployment14 with native380 and NGA runtime metadata')
assert restoredcore==oldcore
save('SOURCE_CHECK.json',{'status':'SOURCE_ONLY_COMPILE_AND_ORIGINAL_PUBLISHER_INVERSE_PASS_NOT_EXECUTED','parent_publisher_sha256':sha(P),'parent_commit_from_actual67_receipt':PARENT,'publisher_sha256':sha(H/'publish_guardfed_increment68_20261011.py'),'input_template_sha256':sha(H/'INPUTS_TEMPLATE.json'),'full_source_inverse_exact':True,'core_Git_F_volume_parent_remote_blob_clean_push_original_except_commit_message':True,'core_AST_equal_after_literal_inverse':ast.dump(ast.parse(restoredcore))==ast.dump(ast.parse(oldcore)),'actual_saved380_and_F80_pins_received_from_root':True,'fixed_new_files':len(files),'parent_references':len(refs),'fixed_new_bytes':sum(x['bytes'] for x in files.values()),'all_fixed_files_below2MB':True,'omitted_large_artifacts':omitted,'late_roles_left_null':late,'draft_STATE_entries_left_null':True,'publisher_executed':False,'Git':False,'SSH':False,'F_access':False,'root_accepted':False})
print(json.dumps({'fixed_new_files':len(files),'parent_references':len(refs),'fixed_new_bytes':sum(x['bytes'] for x in files.values()),'publisher_sha256':sha(H/'publish_guardfed_increment68_20261011.py')}))
