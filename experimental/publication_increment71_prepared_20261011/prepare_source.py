"""Prepare a compact increment71 proposal. No Git/F/network/scientific calls."""
from pathlib import Path
import ast, difflib, hashlib, json

R=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
OLD=R/'tmp/publication_increment70_prepared_20261011'
T='docs/server_deployment_20260923/training_20260923/'
PARENT='4fccc8b8d1d56492ac9cffd0cf27ca390e78bd20'
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
    with (D/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def pin(rel):
    p=R/rel
    assert not p.is_symlink() and p.resolve().is_relative_to(R.resolve()) and p.is_file()
    assert p.stat().st_size<2_000_000,rel
    return {'sha256':H(p.read_bytes()),'bytes':p.stat().st_size}
previous_rel=T+'publication_closed_increment70_verified_20261011.json'
previous=read(R/previous_rel)
assert previous['commit']==PARENT and previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
metadata_rel='tmp/root_CUDA10_two_queues_Git70_20261011/ROOT_METADATA_ADOPTION.json'
metadata=read(R/metadata_rel)
assert (metadata['root_adopted_CUDA'],metadata['remote500_saved_accepted'],metadata['remote800_saved_accepted'])==(10,58,23)
paths=[]
def add(folder,names):
    for n in names.split():paths.append(folder+'/'+n)

# Fixed literal filenames only: no globs, directory expansion or bulk evidence copying.
add('tmp/root_CUDA10_two_queues_Git70_20261011','ROOT_METADATA_ADOPTION.json')
paths += ['tmp/record_CUDA10_two_queues_Git70_root_20261011.py',previous_rel]
add('tmp/baseline_old800_runtime_root_20261011_attempt003',
    'guard.py worker.py run_queue.py reuse_fedavg.py ROOT_PREPARATION.json INSTALL_COMMAND.json INSTALL.stdout INSTALL.stderr ROOT_FIXED3_AUDIT_INPUTS.json ROOT_FIXED3_TRANSPORT_INPUTS.json REMAINING797_START_COMMAND.json REMAINING797_START.stdout REMAINING797_START.stderr OBSERVATION_20261011T075457Z.json')
paths += ['tmp/prepare_old800_root_caller_attempt003_20261011.py','tmp/install_old800_cuda_canary_attempt003_root_once_20261011.py','tmp/prepare_old800_installer_attempt003_20261011.py']
add('tmp/baseline_old800_cuda_formal_adapter_prepare_20261011',
    'adapter.py gpu_body.py old800_saved_science.py.txt SOURCE_PINS.json SOURCE_ADAPTATIONS.json ROOT_APPROVAL.template.json FILES_SHA256.json SOURCE_CHECK.json HANDOFF.json README.md')
add('tmp/baseline_old800_cuda_formal_independent_review_20261011','SOURCE_REVIEW.json ROOT_CALLER_ATTEMPT003_REVIEW.json')
add('tmp/old800_proc_guard_semantics_review_20261011','SOURCE_REVIEW.json')
add('tmp/old800_reuse_FedAvg_guard_prepare_20261011','reuse_fedavg.py AUTH_REUSE.fragment.json REVIEW.json SYNTHETIC_CHECK.json FILES_SHA256.json README.md')
add('tmp/old800_cuda3_saved_audit_prepare_20261011',
    'audit_saved.py transport.py saved_fit_binding.py original_evaluator.py.txt original_fl_audit.py.txt original_full_saved_science.py.txt original_replay.py.txt FILES_SHA256.json SOURCE_PINS.json SOURCE_CHECK.json FIT_HELPER_INVERSE.json HANDOFF.json README.md SOURCE_PREPARATION_FAILURE.json')
C='tmp/old800_cuda3_saved_audit_prepare_20261011/mixed_producer_v2'
add(C,'audit_saved.py transport.py audit_saved.py.inverse.json transport.py.inverse.json FILES_SHA256.json SOURCE_PINS.json SOURCE_CHECK.json HANDOFF.json README.md ROOT_INPUTS.template.json OLD800_FIXED3_ATTEMPT1_BACKUP_COMMAND.json OLD800_FIXED3_ATTEMPT1_BACKUP.stdout OLD800_FIXED3_ATTEMPT1_BACKUP.stderr OLD800_FIXED3_ATTEMPT1_SCP_COMMAND.json OLD800_FIXED3_ATTEMPT1_SCP.stdout OLD800_FIXED3_ATTEMPT1_SCP.stderr OLD800_FIXED3_ATTEMPT1_TRANSPORT_VERIFICATION.json actual/STARTED.json actual/SAVED_OUTPUTS_AUDIT.json root_adoption/ROOT_SCIENTIFIC_ADOPTION.json')
add('tmp/old800_cuda3_saved_audit_independent_review_20261011','SOURCE_REVIEW.json')
add('tmp/uniform_cuda500_resume2_source_prepare_20261011',
    'dispatch_resume2.py resume_queue.py installer_proposal.py PINNED_PRIOR.json PREPARED_SOURCES.json SOURCE_DERIVATION.json DISPATCH_DIFF.patch QUEUE_DIFF.patch FILES_SHA256.json SOURCE_CHECK.json HANDOFF.json README.md ROOT_INPUTS.template.json')
add('tmp/uniform_cuda500_resume2_independent_review_20261011','SOURCE_REVIEW.json')
add('tmp/uniform_cuda500_root_dispatch_20261011',
    'RESUME1_INTERRUPTION_DIAGNOSIS_20261011T073527Z.json RESUME2_INSTALL_COMMAND.json RESUME2_INSTALL.stdout RESUME2_INSTALL.stderr RESUME2_OBSERVATION_20261011T075524Z.json')
paths.append(metadata['evidence']['health']['path'])
add('tmp/FL100_uniform_cuda_formal_adapter_prepare_20261011',
    'adapter.py gpu_body.py SOURCE_PINS.json SOURCE_ADAPTATIONS.json EXACT100.json ROOT_INPUTS.template.json FILES_SHA256.json SOURCE_CHECK.json HANDOFF.json README.md GPU_METADATA_DIFF.patch')
add('tmp/FL100_uniform_cuda_formal_independent_review_20261011','SOURCE_REVIEW.json')
add('tmp/FL100_cuda_runtime_mapping_prepare_20261011',
    'RUNTIME_DEPENDENCIES.json RECORD_INPUT_REFS.json EVIDENCE.json ROOT_ARGS.template.json FILES_SHA256.json REVIEW.json README.md')
add('tmp/FL100_cuda_runtime_caller_prepare_20261011',
    'worker.py guard.py run_queue.py install_once.py worker.py.inverse.json guard.py.inverse.json run_queue.py.inverse.json ROOT_PREPARATION.json ROOT_INSTALL_INPUTS.template.json SOURCE_PINS.json FILES_SHA256.json SOURCE_CHECK.json HANDOFF.json README.md PRESERVED_INITIAL_STATIC_INDEX.json')
add('tmp/FL100_cuda_runtime_caller_independent_review_20261011','SOURCE_REVIEW.json RUNTIME_MAPPING_REVIEW.json')
add('tmp/LoGo100_uniform_cuda_cache_adapter_prepare_20261011',
    'adapter.py original_functions.py.txt EXACT100.json SOURCE_REUSE.json SOURCE_PINS.json ROOT_INPUTS.template.json MISSING_BINDINGS.json NATIVE_GATE_SEMANTICS.md FILES_SHA256.json SOURCE_CHECK.json HANDOFF.json README.md')
add('tmp/LoGo100_uniform_cuda_cache_independent_review_20261011','REVIEW.json REPORT.md FILES_SHA256.json')

roles={'metadata':metadata_rel,'CUDA3_root':C+'/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
 'CUDA3_audit':C+'/actual/SAVED_OUTPUTS_AUDIT.json','CUDA3_transport':C+'/OLD800_FIXED3_ATTEMPT1_TRANSPORT_VERIFICATION.json',
 'CUDA7_root':metadata['evidence']['root7']['path'],'resume2_start':metadata['evidence']['resume2_start']['path'],
 'resume2_observation':metadata['evidence']['cuda500']['path'],'old800_start':metadata['evidence']['remaining797_start']['path'],
 'old800_observation':metadata['evidence']['cuda800']['path'],'health_snapshot':metadata['evidence']['health']['path'],
 'resume2_review':metadata['evidence']['resume2_review']['path'],
 'FL_caller_review':'tmp/FL100_cuda_runtime_caller_independent_review_20261011/SOURCE_REVIEW.json',
 'FL_mapping_review':'tmp/FL100_cuda_runtime_caller_independent_review_20261011/RUNTIME_MAPPING_REVIEW.json'}
paths+=list(roles.values())
prior_template=read(OLD/'INPUTS_TEMPLATE.json')
draft=prior_template['draft_path'];paths.append(draft)
parent_refs={};files={}
for rel in sorted(set(paths)):
    value=pin(rel);dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
    old=previous['files'].get(dest) or previous.get('parent_references',{}).get(rel)
    if old and all(old[k]==value[k] for k in ['sha256','bytes']):parent_refs[rel]=value
    else:files[rel]=value
assert 'tmp/baseline_old800_runtime_root_20261011_attempt003/reuse_fedavg.py' in files
late=sorted([metadata['state_after']['path'],metadata['reader']['path'],metadata['protocol']['path']]+[p['path'] for p in metadata['five_entries']])
assert len(late)==8
for rel in late:assert rel not in files and rel not in parent_refs
cutoff=dict(prior_template['current_accepted'],uniform_CUDA_root_accepted=10,uniform_CUDA_remote_observed=58,uniform_CUDA_old800_remote_observed=23,uniform_CUDA_phase=1300)
template={'status':'UNBOUND_INCREMENT71_INPUTS_PROPOSAL','root_accepted':False,'parent_commit':PARENT,
 'STATE_path':metadata['state_after']['path'],'STATE_sha256':None,'entry_paths':[p['path'] for p in metadata['five_entries']],
 'reader_path':metadata['reader']['path'],'protocol_path':metadata['protocol']['path'],'draft_path':draft,
 'draft_sha256':pin(draft)['sha256'],'current_accepted':cutoff,
 'proof_roles':{k:{'path':rel,'sha256':pin(rel)['sha256']} for k,rel in roles.items()},'late_role_names':[],
 'files':files,'late_files':{p:None for p in late},'parent_references':parent_refs,
 'omitted_large_artifacts':{},'native11_PDF_path':prior_template['native11_PDF_path'],
 'preparation_observation_only':{'metadata_sha256':pin(metadata_rel)['sha256'],'STATE_sha256':metadata['state_after']['sha256'],'observed_main_terminal':427,'CUDA_root_adopted':10,'remote500':58,'remote800':23,'not_publish_authority':True},
 'future_FL_runtime_evidence':{'include':False,'actual_install':None,'actual_start':None,'actual_scientific_adoption':None},
 'publisher_creates_scientific_acceptances':False}
save('INPUTS_TEMPLATE.json',template)
save('EXPLICIT_SELECTION_PROPOSAL.json',{'status':'PROPOSED_FIXED_COMPACT_FILES_NOT_PUBLISH_AUTHORIZATION','fixed_files':files,'parent_references':parent_refs,'late_entry_paths':late,'fixed_files_count':len(files),'fixed_files_bytes':sum(v['bytes'] for v in files.values()),'future_FL_actual_evidence_included':False})
save('OMISSIONS.json',{'status':'INTENTIONAL_COMPACT_PUBLICATION_BOUNDARIES','omitted_categories':['model checkpoints','NPZ/NPY arrays','images and datasets','archives','environment directories','full8MB original800 cohort/inventory','large per-round logs','old native1100/F80 raw records','unclosed FL runtime or scientific outcomes','future LoGo execution/cache adoption'],
 'source_seals_are_references_not_standalone_bundle_claim':True,'prior_Git70_files_not_recopied':list(parent_refs),
 'FedAvg_cache_closure':'Optional source directory not supplied at this cutoff; omitted rather than guessed',
 'future_additions':'Root may revise this explicit proposal after actual closed receipts; no wildcard or automatic discovery'})

old=(OLD/'publish_guardfed_increment70_20261011.py').read_bytes().decode()
s=old.replace('increment70','increment71').replace('INCREMENT70','INCREMENT71')
s=s.replace("PARENT='0a276220b447645020e8dca8ee092ce411367bb7'", "PARENT='"+PARENT+"'")
s=s.replace('Publish fixed FL100/CUDA7/Hybrid29/NGA9 compact evidence; preserve negative and historical results.','Publish fixed CUDA10 and two-queue compact evidence with source-only FL/LoGo proposals.')
s=s.replace('20137180b9b3d4de14c67f74c9bffa0b66db589768a41df357cd580777bfc9b4',H((D/'INPUTS_TEMPLATE.json').read_bytes()))
start=s.index("for key in ['parent_references'")
end=s.index('selected=selection[',start)
guards='''for key in ['parent_references','entry_paths','STATE_path','reader_path','protocol_path','current_accepted','draft_path','draft_sha256','late_role_names','omitted_large_artifacts','native11_PDF_path']:
    assert stable_input[key]==template[key],key
roles=stable_input['proof_roles'];assert selection['proof_roles']==roles==template['proof_roles']
for role,pin in roles.items():assert entry['source_pins'][role]==pin
proofs={role:pinned_json(repo_path(pin['path']),pin['sha256']) for role,pin in roles.items()}
previous_rel=__PREVIOUS_REL__
previous=pinned_json(repo_path(previous_rel),__PREVIOUS_SHA__)
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
assert previous['acceptance_cutoff']['uniform_CUDA_root_accepted']==7
metadata=proofs['metadata'];root3=proofs['CUDA3_root'];root7=proofs['CUDA7_root']
assert metadata['status']=='ROOT_CUDA10_TWO_QUEUES_GIT70_CURRENT_ENTRIES_RECORDED' and metadata['history_tails_exact'] is True
assert (metadata['root_adopted_CUDA'],metadata['remote500_saved_accepted'],metadata['remote800_saved_accepted'])==(10,58,23)
assert root7['accepted']==7 and root3['accepted']==3
assert root3['status']=='ROOT_FIXED3_OLD800_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED'
ids=['FedAvg_IID_Benign_seed91001','FedAA-DDPG_policy0.001_keep16_local0.001_IID_Benign_seed91001','LASA_s0.3_l2_lr0.001_IID_Benign_seed91001_screen']
assert root3['ids']==ids and root3['record_producers']==dict(zip(ids,['attempt002','attempt003','attempt003']))
assert root3['archive_all_members_rehashed'] is True and root3['archive']['members']==85
assert all(r['native_max_abs_difference']==0 for r in root3['records'].values())
assert root3['records'][ids[0]]['views']['native']['constant_predictions'] is True
assert root3['whole800_complete'] is root3['whole_comparison_complete'] is root3['test'] is False
audit=proofs['CUDA3_audit'];transport=proofs['CUDA3_transport']
assert root3['audit']['sha256']==roles['CUDA3_audit']['sha256'] and root3['transport']['sha256']==roles['CUDA3_transport']['sha256']
assert (audit['metric_values_checked'],audit['integer_base_counts_checked'],audit['prediction_rules_checked'])==(27,72,9)
assert audit['fit_calls']==audit['new_CNN']==0 and audit['root_adopted'] is False
assert transport['all_members_verified'] is True and transport['archive_members']==85 and len(transport['members'])==84
assert audit['archive_sha256']==transport['archive_sha256']==root3['archive']['sha256']
assert proofs['resume2_start']['exit_code']==proofs['old800_start']['exit_code']==0
resume=proofs['resume2_observation']['progress.json'];old800=proofs['old800_observation']
assert len(resume['remote_saved_accepted_ids'])==58 and (resume['accepted_prior_cuda'],resume['accepted_new_cuda'],resume['failed'])==(57,1,0)
assert len(old800['remaining_progress']['accepted_ids'])==20 and old800['remaining_progress']['failed']==0
assert old800['canary_progress']['planned']==3 and old800['canary_progress']['failed']==0
assert proofs['resume2_review']['status'].startswith('PASS_') and not proofs['resume2_review']['blocking_findings']
for role in ['FL_caller_review','FL_mapping_review']:
    assert proofs[role]['status'].startswith('PASS_') and not proofs[role]['blocking_findings']
state_path=T/'TRAINING_STATE.json';state=pinned_json(state_path,entry['STATE_sha256']);main=state['celeba_mechanism_v1'];cuda=state['uniform_CUDA_evaluation_v1']
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(400,380)
assert (cuda['uniform_CUDA_scientific_accepted'],cuda['current_remote_saved_accepted'],cuda['old800_remote_saved_accepted'])==(10,58,23)
assert (cuda['root_adopted_first500'],cuda['root_adopted_old800'],cuda['actual_all_frozen'])==(7,3,1300)
assert cuda['whole500_complete'] is cuda['whole_comparison_complete'] is cuda['CPU_outputs_reused_as_CUDA'] is False
assert cuda['old800_root']['sha256']==roles['CUDA3_root']['sha256'] and cuda['current_observation']['sha256']==roles['resume2_observation']['sha256']
assert state['latest_current_resource_observation']['main_terminal']==427 and state['latest_current_resource_observation']['not_new_scientific_acceptance'] is True
assert entry['latest_observation_only']==state['latest_current_resource_observation']
assert state['latest_working_rebuttal_draft']['entry']==template['draft_path'] and state['latest_working_rebuttal_draft']['sha256']==template['draft_sha256']
assert H(repo_path(template['draft_path']).read_bytes())==template['draft_sha256']
assert entry['working_draft_content_cutoff']==state['latest_working_rebuttal_draft']['content_cutoff']
assert entry['current_accepted']==template['current_accepted'] and entry['fixed_observed_cutoff']=={'main_terminal':427,'CUDA500_remote':58,'CUDA800_remote':23}
assert entry['FL_LoGo_source_only'] is True and entry['no_final_test_or_complete_comparison_claim'] is True
assert entry['historical_Git70_cutoff']==previous['acceptance_cutoff']
entry_paths=set(template['entry_paths']);assert set(entry['entries'])==entry_paths and len(entry_paths)==5
for rel,pin in entry['entries'].items():assert H(repo_path(rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for role in ['reader','protocol']:
    pin=entry[role];assert pin['path']==template[role+'_path'] and H(repo_path(pin['path']).read_bytes())==pin['sha256']
for pin in entry['source_pins'].values():assert H(repo_path(pin['path']).read_bytes())==pin['sha256']
'''.replace('__PREVIOUS_REL__',repr(previous_rel)).replace('__PREVIOUS_SHA__',repr(pin(previous_rel)['sha256']))
s=s[:start]+guards+s[end:]
own_names=['prepare_source.py','publish_guardfed_increment71_20261011.py','INPUTS_TEMPLATE.json','EXPLICIT_SELECTION_PROPOSAL.json','OMISSIONS.json','README.md','SOURCE_CHECK.json','SOURCE_DIFF.patch','INVERSE_DIFF.json','HANDOFF.json','FILES_SHA256.json','SEMANTIC_TEMPLATE.json','SELECTION_TEMPLATE.json','STABLE_INPUTS.json']
a=s.index('own_names=');b=s.index('\nown_paths=',a);s=s[:a]+'own_names='+repr(set(own_names))+s[b:]
s=s.replace("bound_paths={roles[k]['path'] for k in template['late_role_names']}|entry_paths|{template['STATE_path'],template['draft_path']}","bound_paths=entry_paths|{template['STATE_path'],template['reader_path'],template['protocol_path']}")
s=s.replace("assert len(bound_paths)==len(template['late_role_names'])+7","assert len(bound_paths)==8")
s=s.replace('Publish FL100 CUDA7 Hybrid29 NGA9 fixed-cutoff evidence and author review','Publish CUDA10 fixed-cutoff evidence and reviewed runtime source proposals')
a=s.index('cutoff=dict(previous[\'acceptance_cutoff\']');s=s[:a]+'''cutoff=dict(previous['acceptance_cutoff'],uniform_CUDA_root_accepted=10,uniform_CUDA_old800_root_accepted=3,uniform_CUDA_available_phase=1300)
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=cutoff,proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,parent_references=parent_refs,observed_only_cutoff=entry['fixed_observed_cutoff'],historical_Git70_cutoff=previous['acceptance_cutoff'],working_draft_content_cutoff=entry['working_draft_content_cutoff'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published={'uniform_CUDA':3},source_only_FL_LoGo=True,whole_comparison_complete=False,whole_rebuttal_complete=False,final_test=False,primary_endpoint_selected=False,original_negative_results_preserved=True,scientific_scope='Fixed CUDA10=root7+old800root3; remote58/23 and main427 are observations only. FL/LoGo material here is reviewed source, not runtime or acceptance. Old CPU tables and complete24 historical prose cutoff retained. Old800 constant FedAvg and prior failures retained.')
(T/'publication_closed_increment71_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
'''
compile(s,'publisher71','exec')
(D/'publish_guardfed_increment71_20261011.py').write_bytes(s.encode())
delta=list(difflib.ndiff(old.splitlines(True),s.splitlines(True)))
assert ''.join(difflib.restore(delta,1)).encode()==old.encode() and ''.join(difflib.restore(delta,2)).encode()==s.encode()
save('INVERSE_DIFF.json',{'parent':{'path':'tmp/publication_increment70_prepared_20261011/publish_guardfed_increment70_20261011.py','sha256':H(old.encode())},'derived_sha256':H(s.encode()),'ndiff':delta})
(D/'SOURCE_DIFF.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='publisher70',tofile='publisher71')),encoding='utf8')
semantic={'status':'UNBOUND_SEMANTIC_REVIEW_TEMPLATE','root_accepted':False,'STATE_sha256':None,'entries':{p:{'sha256':None,'history_exact':False} for p in template['entry_paths']},'reader':{'path':template['reader_path'],'sha256':None},'protocol':{'path':template['protocol_path'],'sha256':None},'source_pins':template['proof_roles'],'current_accepted':cutoff,'fixed_observed_cutoff':{'main_terminal':427,'CUDA500_remote':58,'CUDA800_remote':23},'working_draft_content_cutoff':None,'latest_observation_only':None,'historical_Git70_cutoff':previous['acceptance_cutoff'],'FL_LoGo_source_only':True,'no_final_test_or_complete_comparison_claim':True}
save('SEMANTIC_TEMPLATE.json',semantic)
save('SELECTION_TEMPLATE.json',{'status':'UNBOUND_INCREMENT71_SELECTION','root_accepted':False,'parent_commit':PARENT,'semantic_review_sha256':None,'stable_input_sha256':None,'STATE_sha256':None,'proof_roles':template['proof_roles'],'parent_references':parent_refs,'files':None,'root_late_bound_files':None,'own_names':own_names})
# Unchanged execution safety bodies are compared as raw text; never executed.
def section(text,a,b):return text[text.index(a):text.index(b,text.index(a))]
checks={}
for name,a,b in [('Git_F_checkout_write_blob','G=[',"git('commit'"),('committed_blob_clean_remote',"commit=git('rev-parse'","cutoff=dict"),('closed_file_validation','paths=[]',"assert not (T/'publication_closed_increment")]:
    checks[name]=section(old,a,b)==section(s,a,b)
assert all(checks.values()),checks
save('SOURCE_CHECK.json',{'status':'PASS_SOURCE_PROPOSAL_ONLY_NOT_PUBLISHED','publisher_compile':True,'old_new_raw_inverse_exact':True,'unchanged_execution_blocks':checks,'fixed_files':len(files),'fixed_bytes':sum(v['bytes'] for v in files.values()),'late_entries':8,'parent_references':len(parent_refs),'future_FL_runtime_included':False,'Git':False,'F':False,'SSH':False,'STATE_modified':False})
print(json.dumps({'fixed_files':len(files),'fixed_bytes':sum(v['bytes'] for v in files.values()),'parent_references':len(parent_refs),'late_entries':8,'publisher_sha256':H(s.encode())}))
