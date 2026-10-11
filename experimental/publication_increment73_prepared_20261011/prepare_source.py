"""Prepare a closed, compact increment73 source proposal. No Git, F, SSH or science."""
from pathlib import Path
import ast, difflib, hashlib, json

R=Path(__file__).resolve().parents[2]
Q=Path(__file__).resolve().parent
OLD=R/'tmp/publication_increment72_prepared_20261011'
H=lambda b:hashlib.sha256(b).hexdigest()
PARENT='51a27b50c6c1880d3fb3c6b5c45b750d0cc68d12'
PREVIOUS='docs/server_deployment_20260923/training_20260923/publication_closed_increment72_verified_20261011.json'
PREVIOUS_SHA='a246527f9cb72f656650f29b37a5c49475d8baf318dbc2addfa3377474368126'
OWN_NAMES={'prepare_source.py','publish_guardfed_increment73_20261011.py','INPUTS_TEMPLATE.json','SELECTION_TEMPLATE.json','SEMANTIC_TEMPLATE.json','EXPLICIT_SELECTION_PROPOSAL.json','SOURCE_DIFF.patch','INVERSE_DIFF.json','SOURCE_REUSE.json','SOURCE_CHECK.json','OMISSIONS.json','PARENT_PRE_GIT_FAILURE.json','README.md','HANDOFF.json','FILES_SHA256.json','STABLE_INPUTS.json'}

def write(name,value):
    p=Q/name
    assert not p.exists(),p
    b=value.encode('utf8') if isinstance(value,str) else (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    p.write_bytes(b)

def pin(rel):
    p=R/rel
    assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(R.resolve()),rel
    assert p.stat().st_size<2_000_000,rel
    b=p.read_bytes()
    return {'sha256':H(b),'bytes':len(b)}

def obj(rel):return json.loads((R/rel).read_bytes())
assert H((R/PREVIOUS).read_bytes())==PREVIOUS_SHA
previous=obj(PREVIOUS)
assert previous['commit']==PARENT and previous['acceptance_cutoff']['uniform_CUDA_root_accepted']==112
paths={PREVIOUS};omissions={};source_seals={}
def add(folder,names):
    for n in names.split():paths.add(folder+'/'+n)
def sealed(folder):
    rel=folder+'/FILES_SHA256.json';seal=obj(rel)
    members=seal.get('files',seal)
    assert isinstance(members,dict)
    source_seals[rel]=pin(rel)
    paths.add(rel)
    for name,p in members.items():
        assert isinstance(name,str) and not name.startswith('/') and '..' not in Path(name).parts
        rel=folder+'/'+name
        size=p.get('bytes',p.get('size')) if isinstance(p,dict) else None
        sha=p.get('sha256') if isinstance(p,dict) else p
        if (R/rel).stat().st_size>=2_000_000:
            omissions[rel]={'sha256':sha,'bytes':(R/rel).stat().st_size,'reason':'Original source input is above compact limit; original seal and root/archive lineage retained. No payload mirror claim.'}
            continue
        actual=pin(rel)
        assert actual['sha256']==sha and (size is None or actual['bytes']==size),rel
        paths.add(rel)

FL='tmp/FL100_cuda100_saved_closure_prepare_20261011'
sealed(FL)
add(FL,'ROOT_SOURCE_REVIEW.json FL100_DELTA97_TRANSPORT_VERIFICATION.json FL100_DELTA97_BACKUP_COMMAND.json FL100_DELTA97_BACKUP.stdout FL100_DELTA97_BACKUP.stderr FL100_DELTA97_SCP_COMMAND.json FL100_DELTA97_SCP.stdout FL100_DELTA97_SCP.stderr actual/SAVED_OUTPUTS_AUDIT.json actual/STARTED.json root_adoption/ROOT_SCIENTIFIC_ADOPTION.json')
L='tmp/LoGo100_cuda_actual_runtime_prerequisites_20261011'
add(L,'ACTUAL_HANDOFF.json ACTUAL_PARTIAL_ENVIRONMENT.json ACTUAL_REPAIRED_HANDOFF.json ACTUAL_TRANSPORT_VERIFICATION.json ENV_CLOSURE_COMMAND.json ENV_CLOSURE.stderr ENV_CLOSURE.stdout ENVIRONMENT_CLOSURE.json ENV_REPAIR_COMMAND.json ENV_REPAIR.stderr ENV_REPAIR.stdout ENVIRONMENT_COMMAND.json ENVIRONMENT.stderr ENVIRONMENT.stdout execute_authorized_once.py execute_env_repair_once.py FRESH_F_GUARD.json FRESH_REMOTE_COMMAND.json FRESH_REMOTE.stderr FRESH_REMOTE.stdout ORIGINAL_CPU_PACKAGE_DEPENDENCIES.json OWNED_INSTALL_STOP_COMMAND.json OWNED_INSTALL_STOP.stderr OWNED_INSTALL_STOP.stdout read_final_prerequisites.py read_repaired_environment.py REPAIR_FRESH_COMMAND.json REPAIR_FRESH.stderr REPAIR_FRESH.stdout ROOT_ENV_AUTH.json ROOT_ENV_REPAIR_AUTH.json ROOT_TRANSFER_AUTH.json stop_owned_version_drift.py TRANSFER_COMMAND.json TRANSFER.stderr TRANSFER.stdout')
for folder in [L+'/env_repair_source',L+'/caller_installer_source',L+'/caller_installer_v2_source']:
    sealed(folder)
add(L+'/caller_installer_v2_source','DEPLOY_COMMAND.json DEPLOY.stdout DEPLOY.stderr')
add('tmp/LoGo100_caller_installer_independent_review_20261011','REVIEW.json FILES_SHA256.json REVIEW_V2.json FILES_V2_SHA256.json')
LR='tmp/LoGo100_cuda_actual_root_20261011'
sealed(LR)
add(LR,'INSTALL_COMMAND.json ROOT_DEPLOYMENT_RECORD.json ROOT_INSTALL_INPUTS.json ROOT_START_COMMAND.json ROOT_START_INPUTS.json START_FRESH_PROTECTED.json start001/COMMAND.json start001/REMOTE_EXECUTED.py observe001/COMMAND.json observe001/REMOTE_EXECUTED.py')
LS='tmp/LoGo_cuda_fixed2_saved_audit_prepare_v2_20261011'
sealed(LS)
add(LS,'ROOT_SOURCE_REVIEW.json root_adoption/ROOT_SCIENTIFIC_ADOPTION.json actual_metadata/ACTUAL_FILES_SHA256.json')
actual=obj(LS+'/actual_metadata/ACTUAL_FILES_SHA256.json')
for name,p in actual['files'].items():
    rel=LS+'/'+name
    assert pin(rel)==p,rel
    paths.add(rel)
START='tmp/LoGo100_remaining98_root_start_prepare_20261011'
sealed(START)
add(START,'ROOT_SOURCE_REVIEW.json ROOT_BIND_COMMAND.json ROOT_BOUND_START_INPUTS.json ROOT_REMAINING98_START_COMMAND.json ROOT_START_INPUTS.json START_FRESH_PROTECTED.json start001/COMMAND.json start001/REMOTE_EXECUTED.py observe001/COMMAND.json observe001/REMOTE_EXECUTED.py')
add('tmp/publication_increment72_prepared_20261011','FAILED_PRE_GIT_COMPACT_SELECTION.json ROOT_SELECTION_REPAIR.json SUFFIX_FIX.json ROOT_PUBLISH_COMMAND.json ROOT_PUBLISH.stderr ROOT_PUBLISH.stdout ROOT_PUBLISH_attempt002_COMMAND.json ROOT_PUBLISH_attempt002.stderr ROOT_PUBLISH_attempt002.stdout')
add('tmp','adopt_FL100_all100_root_once_20261011.py adopt_LoGo_fixed2_root_once_20261011.py deploy_LoGo100_caller_root_once_20261011.py start_LoGo100_fixed2_root_once_20261011.py start_LoGo100_remaining98_root_once_20261011.py review_FL100_delta97_closure_root_20261011.py record_CUDA211_FL100_LoGo2_Git72_root_20261011.py')
META='tmp/root_CUDA211_FL100_LoGo2_Git72_20261011/ROOT_METADATA_ADOPTION.json'
metadata=obj(META)
paths.add(META)
for p in metadata['evidence'].values():paths.add(p['path'])
roles={name:{'path':rel,**pin(rel)} for name,rel in {
    'metadata':META,'FL100_root':FL+'/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
    'FL100_audit':FL+'/actual/SAVED_OUTPUTS_AUDIT.json','FL100_transport':FL+'/FL100_DELTA97_TRANSPORT_VERIFICATION.json',
    'LoGo2_root':LS+'/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json','LoGo2_audit':LS+'/actual/SAVED_AUDIT.json',
    'LoGo2_transport':LS+'/LOGO_FIXED2_TRANSPORT_VERIFICATION.json',
    'LoGo_env':L+'/ACTUAL_REPAIRED_HANDOFF.json','LoGo_restore':L+'/ACTUAL_TRANSPORT_VERIFICATION.json',
    'LoGo_deployment':LR+'/ROOT_DEPLOYMENT_RECORD.json','LoGo_remaining98_start':START+'/ROOT_REMAINING98_START_COMMAND.json',
    'LoGo_installer_v1_review':'tmp/LoGo100_caller_installer_independent_review_20261011/REVIEW.json',
    'LoGo_installer_v2_review':'tmp/LoGo100_caller_installer_independent_review_20261011/REVIEW_V2.json',
    'health':metadata['evidence']['health']['path'],'cuda500':metadata['evidence']['cuda500']['path'],
    'cuda800':metadata['evidence']['cuda800']['path']}.items()}
for p in roles.values():paths.add(p['path'])
oldtemplate=obj('tmp/publication_increment72_prepared_20261011/INPUTS_TEMPLATE.json')
current=dict(oldtemplate['current_accepted'])
current.update(uniform_CUDA_root_accepted=211,uniform_CUDA_remote_observed=95,uniform_CUDA_phase=1500,uniform_CUDA_old800_remote_observed=220,uniform_CUDA_FL_root_accepted=100,uniform_CUDA_FL_remote_observed=100,uniform_CUDA_LoGo_root_accepted=2)
observed={'main_terminal':440,'CUDA500_remote':95,'CUDA800_remote':220,'FL100_remote':100}
files={};refs={}
for rel in sorted(paths):
    p=pin(rel);dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
    old=previous['files'].get(dest) or previous.get('parent_references',{}).get(rel)
    if old and (old['sha256'],old['bytes'])==(p['sha256'],p['bytes']) and rel not in {x['path'] for x in roles.values()}:
        refs[rel]=p
    else:files[rel]=p
for rel,p in oldtemplate['parent_references'].items():
    assert pin(rel)==p
    refs[rel]=p
template={k:oldtemplate[k] for k in ['entry_paths','STATE_path','reader_path','protocol_path','draft_path','draft_sha256','native11_PDF_path']}
template.update(status='SOURCE_ONLY_INCREMENT73_INPUTS_TEMPLATE',root_accepted=False,parent_commit=PARENT,files=files,parent_references=refs,proof_roles=roles,current_accepted=current,fixed_observed_cutoff=observed,late_role_names=[],late_mutable_names=oldtemplate['entry_paths']+[oldtemplate[k] for k in ['STATE_path','reader_path','protocol_path']],omitted_large_artifacts=omissions,own_names=sorted(OWN_NAMES),STATE_sha256=None,semantic_review_sha256=None,scope='CLOSED_CUDA211_FL100_LOGO2_REMAINING98_START_NOT_ACCEPTANCE',scientific_results_created_by_preparation=0)
write('INPUTS_TEMPLATE.json',template)
write('EXPLICIT_SELECTION_PROPOSAL.json',{'status':'SOURCE_ONLY_CLOSED_FILES_NOT_GIT_STAGED','files':files,'parent_references':refs,'source_seals':source_seals,'future_mutable_pins':None,'no_directory_expansion':True})
write('SELECTION_TEMPLATE.json',{'status':'ROOT_CLOSED_INCREMENT73_COMPACT_SELECTION','root_accepted':False,'parent_commit':PARENT,'STATE_sha256':None,'semantic_review_sha256':None,'stable_input_sha256':None,'proof_roles':roles,'files':None,'parent_references':refs,'root_late_bound_files':None,'own_names':sorted(OWN_NAMES),'binding_rule':'Root fills STABLE_INPUTS.files=fixed files + exact8 current bytes; COMPACT_SELECTION.files=stable files + own_names (STABLE_INPUTS included); root_late_bound_files=exact own_names. No arbitrary extras.'})
semantic={'status':'PASS_LIMITED_CURRENT_ENTRY_SEMANTIC_REVIEW','root_accepted':False,'STATE_sha256':None,'source_pins':roles,'current_accepted':current,'fixed_observed_cutoff':observed,'latest_observation_only':None,'entries':{p:{'sha256':None,'history_exact':False} for p in template['entry_paths']},'reader':{'path':template['reader_path'],'sha256':None},'protocol':{'path':template['protocol_path'],'sha256':None},'working_draft_content_cutoff':None,'no_final_test_or_complete_comparison_claim':False,'historical_Git72_cutoff':previous['acceptance_cutoff'],'LoGo_scientific_accepted':2,'LoGo_remaining98_started_not_accepted':True}
write('SEMANTIC_TEMPLATE.json',semantic)
write('OMISSIONS.json',{'large_source_inputs':omissions,'bulk':'No F/remote arrays, model, pickle, images, tar or NPZ is selected or read by this preparation. Original archive/member recovery pins remain in roots/transports.','duplicate_compact_source_bundle_omitted':L+'/caller_installer_v2_source/ACTUAL_DEPLOY_BUNDLE.json','extensionless_runtime_logs_omitted_with_command_sha_provenance':[LR+'/start001/stdout',LR+'/start001/stderr',LR+'/observe001/stdout',LR+'/observe001/stderr',START+'/start001/stdout',START+'/start001/stderr',START+'/observe001/stdout',START+'/observe001/stderr'],'source_mirror_is_partial':True,'not_in_scope':'Future LoGo98 acceptance/new Full100 adoption; fixed CUDA211 cutoff, not complete comparison/test.'})
write('PARENT_PRE_GIT_FAILURE.json',{'origin':'Root actual increment72 publisher command and original stderr','failure_phase':'Closed selection assertion, before F/Git mutation','problem':'SELECTION_TEMPLATE own_names omitted SUFFIX_FIX.json while publisher required it.','preserved_failure':{'path':'tmp/publication_increment72_prepared_20261011/ROOT_PUBLISH.stderr',**pin('tmp/publication_increment72_prepared_20261011/ROOT_PUBLISH.stderr')},'repair':{'path':'tmp/publication_increment72_prepared_20261011/ROOT_SELECTION_REPAIR.json',**pin('tmp/publication_increment72_prepared_20261011/ROOT_SELECTION_REPAIR.json')},'increment73_prevention':'One OWN_NAMES set feeds source and both templates; literal set equality checked statically.','scientific_failure':False})

old=(OLD/'publish_guardfed_increment72_20261011.py').read_bytes()
assert b'\r' not in old
text=old.decode('utf8')
text=text.replace('increment72','increment73').replace('INCREMENT72','INCREMENT73')
text=text.replace('ba302df51cbeeee87d0c0f6bd08352a3fd826d03',PARENT)
text=text.replace('"""Publish CUDA112/FedAvg100/NGA19 compact closure; LoGo prerequisites remain source-only."""','"""Publish closed CUDA211/FL100/LoGo2 compact proofs; remaining98 start is not acceptance."""')
text=text.replace('938174c24922f6b99229eee56b6365764c04936a295526085b314bac6135a8b9',H((Q/'INPUTS_TEMPLATE.json').read_bytes()))
start=text.index("for key in ['parent_references'")
end=text.index("selected=selection['files']",start)
guard='''for key in ['parent_references','entry_paths','STATE_path','reader_path','protocol_path','current_accepted','draft_path','draft_sha256','late_role_names','omitted_large_artifacts','native11_PDF_path','fixed_observed_cutoff']:
    assert stable_input[key]==template[key],key
roles=stable_input['proof_roles'];assert selection['proof_roles']==roles==template['proof_roles']
for role,pin in roles.items():assert entry['source_pins'][role]==pin
proofs={role:pinned_json(repo_path(pin['path']),pin['sha256']) for role,pin in roles.items()}
previous_rel=PREVIOUS_LITERAL
previous=pinned_json(repo_path(previous_rel),PREVIOUS_SHA_LITERAL)
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
assert previous['acceptance_cutoff']['uniform_CUDA_root_accepted']==112
metadata=proofs['metadata'];fl=proofs['FL100_root'];logo=proofs['LoGo2_root']
assert metadata['status']=='ROOT_CUDA211_FL100_LOGO2_GIT72_CURRENT_ENTRIES_RECORDED' and metadata['history_tails_exact'] is True
assert (metadata['root_adopted_CUDA'],metadata['old800_unique_accepted'],metadata['FL100_accepted'],metadata['LoGo2_accepted'])==(211,102,100,2)
assert metadata['LoGo98_started'] is True and metadata['new_scientific_adoptions_in_metadata_writer']==0
assert (metadata['remote500_saved_accepted'],metadata['remote800_saved_accepted'])==(95,220)
assert 7+102+100+2==metadata['root_adopted_CUDA']
assert fl['status']=='ROOT_ALL100_FL100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED'
assert (fl['accepted'],fl['new_unique'],fl['reused_already_adopted'])==(100,97,3)
assert fl['prior3_root']['sha256']==previous['proof_roles']['FL3_root']['sha256']
fa=proofs['FL100_audit'];ft=proofs['FL100_transport']
assert fl['audit']['sha256']==roles['FL100_audit']['sha256'] and fl['transport']['sha256']==roles['FL100_transport']['sha256']
assert fl['archive_all_members_rehashed'] is True and fl['archive']['members']==959
assert fl['ids']==fa['exact_ids'] and len(fl['records'])==100 and len(set(fl['ids']))==100
assert (fa['metric_values_checked'],fa['integer_base_counts_checked'],fa['prediction_rules_checked'])==(900,2400,300)
assert fa['fit_calls']==fa['new_CNN']==fa['new_training']==0 and fa['root_adopted'] is False
assert ft['all_members_verified'] is True and ft['archive_members']==959
assert fa['archive_sha256']==ft['archive_sha256']==fl['archive']['sha256']
assert all(r['native_max_abs_difference']==0 for r in fl['records'].values())
assert logo['status']=='ROOT_LOGO_FIXED2_CUDA_CACHE_FROZEN_POST_AND_OFFSERVER_SAVED_AUDIT_ADOPTED' and logo['root_accepted'] is True
assert logo['accepted']==len(logo['records'])==len(logo['ids'])==2
assert logo['ids']==['LoGoFair-DP_07_FedAvg_IID_Benign_seed91001','LoGoFair-DP_07_FedAvg_IID_F Flip_seed91001']
assert logo['FedAvg_CUDA_root']['sha256']==previous['proof_roles']['FedAvg100_root']['sha256']
assert logo['FedAvg_CUDA_index']['sha256']=='6a21265f08fb26da130d95c67bf4130039b25916f44a543cd695e15665a25b64'
la=proofs['LoGo2_audit'];lt=proofs['LoGo2_transport']
assert logo['offserver_audit']['sha256']==roles['LoGo2_audit']['sha256'] and logo['transport']['sha256']==roles['LoGo2_transport']['sha256']
assert logo['archive_all_members_rehashed'] is logo['original_artifacts_immutable_rehashed'] is True
assert logo['archive']['members']==lt['archive_members']==34 and lt['all_members_verified'] is True
assert la['archive_sha256']==lt['archive_sha256']==logo['archive']['sha256']
assert (la['metric_values_checked'],la['integer_confusion_counts_recomputed'],la['prediction_items_checked'],la['max_metric_error'])==(6,16,39734,0.0)
assert la['new_fit']==la['new_CNN']==la['pickle_loads']==0 and la['root_adopted'] is False
assert logo['records'][logo['ids'][0]]['constant_prediction']==0
for root in [fl,logo]:
    assert root['whole_comparison_complete'] is root['test'] is root['parameters_changed'] is root['tolerance_changed'] is False
    assert root['old_CPU_outputs_preserved'] is True
assert logo['whole100_complete'] is False
assert proofs['LoGo_remaining98_start']['exit_code']==0
assert proofs['LoGo_installer_v1_review']['source_adoptable'] is False
assert proofs['LoGo_installer_v2_review']['source_adoptable'] is True
for role,evidence_name in [('FL100_root','FL_root'),('LoGo2_root','LoGo_root'),('LoGo_remaining98_start','LoGo_start98'),('LoGo_deployment','LoGo_deployment'),('LoGo_env','LoGo_env'),('LoGo_restore','LoGo_restore'),('health','health'),('cuda500','cuda500'),('cuda800','cuda800')]:
    assert metadata['evidence'][evidence_name]['path']==roles[role]['path'] and metadata['evidence'][evidence_name]['sha256']==roles[role]['sha256']
state_path=T/'TRAINING_STATE.json';state=pinned_json(state_path,entry['STATE_sha256']);main=state['celeba_mechanism_v1'];cuda=state['uniform_CUDA_evaluation_v1']
assert metadata['state_after']['sha256']==entry['STATE_sha256']
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(400,380)
assert (cuda['uniform_CUDA_scientific_accepted'],cuda['root_adopted_old800_unique'],cuda['FedAvg100_scientific_accepted'],cuda['root_adopted_FL100'],cuda['root_adopted_LoGo100'])==(211,102,100,100,2)
assert (cuda['current_remote_saved_accepted'],cuda['old800_remote_saved_accepted'],cuda['FL100_server_saved_accepted'],cuda['actual_all_frozen'])==(95,220,100,1500)
assert cuda['whole500_complete'] is cuda['whole_comparison_complete'] is cuda['CPU_outputs_reused_as_CUDA'] is cuda['LoGo_whole100_complete'] is False
assert cuda['FL100_runtime_started'] is cuda['LoGo100_runtime_started'] is cuda['LoGo_remaining98_started'] is True
assert cuda['FL100_root']['sha256']==roles['FL100_root']['sha256'] and cuda['LoGo100_root']['sha256']==roles['LoGo2_root']['sha256']
assert state['latest_current_resource_observation']['main_terminal']==440 and state['latest_current_resource_observation']['NGA_new_accepted']==19
assert state['latest_current_resource_observation']['not_new_scientific_acceptance'] is True
assert entry['latest_observation_only']==state['latest_current_resource_observation']
assert state['latest_working_rebuttal_draft']['entry']==template['draft_path'] and state['latest_working_rebuttal_draft']['sha256']==template['draft_sha256']
assert H(repo_path(template['draft_path']).read_bytes())==template['draft_sha256']
assert entry['working_draft_content_cutoff']==state['latest_working_rebuttal_draft']['content_cutoff']
assert entry['current_accepted']==template['current_accepted'] and entry['fixed_observed_cutoff']==template['fixed_observed_cutoff']
assert entry['LoGo_scientific_accepted']==2 and entry['LoGo_remaining98_started_not_accepted'] is True
assert entry['no_final_test_or_complete_comparison_claim'] is True and entry['historical_Git72_cutoff']==previous['acceptance_cutoff']
entry_paths=set(template['entry_paths']);assert set(entry['entries'])==entry_paths and len(entry_paths)==5
assert {p['path']:p['sha256'] for p in metadata['five_entries']}=={p:pin['sha256'] for p,pin in entry['entries'].items()}
for rel,pin in entry['entries'].items():assert H(repo_path(rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for role in ['reader','protocol']:
    pin=entry[role];assert pin['path']==template[role+'_path'] and H(repo_path(pin['path']).read_bytes())==pin['sha256'] and metadata[role]==pin
for pin in entry['source_pins'].values():assert H(repo_path(pin['path']).read_bytes())==pin['sha256']
'''.replace('PREVIOUS_LITERAL',repr(PREVIOUS)).replace('PREVIOUS_SHA_LITERAL',repr(PREVIOUS_SHA))
text=text[:start]+guard+text[end:]
start=text.index('own_names=');end=text.index('\nown_paths=',start)
text=text[:start]+'own_names='+repr(OWN_NAMES)+"\nassert own_names==set(template['own_names'])"+text[end:]
needle="EXACT_SHELL_SOURCE_PATHS={"
start=text.index(needle);end=text.index('\npaths=list',start)
text=text[:start]+"EXACT_SHELL_SOURCE_PATHS="+repr({L+'/env_repair_source/create_cpu_environment_once.sh','tmp/LoGo100_cuda_actual_prerequisites_20261011/create_cpu_environment_proposal.sh','tmp/LoGo100_cuda_actual_prerequisites_v2_20261011/create_cpu_environment_proposal.sh'})+text[end:]
text=text.replace("Publish CUDA112 FedAvg100 NGA19 and reviewed LoGo prerequisites","Publish CUDA211 FL100 LoGo2 and preserved prerequisite failures")
start=text.index('cutoff=dict(')
text=text[:start]+'''cutoff=dict(previous['acceptance_cutoff'],uniform_CUDA_root_accepted=211,uniform_CUDA_old800_root_accepted=102,uniform_CUDA_available_phase=1500,uniform_CUDA_FedAvg100=100,uniform_CUDA_FL100_root_accepted=100,uniform_CUDA_LoGo_root_accepted=2,NGA96_completed_scientific_records=19,gradient200_new_scientific_accepted=19)
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=cutoff,proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,parent_references=parent_refs,observed_only_cutoff=entry['fixed_observed_cutoff'],historical_Git72_cutoff=previous['acceptance_cutoff'],working_draft_content_cutoff=entry['working_draft_content_cutoff'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published={'FL100_new_unique':97,'LoGo_fixed2':2},LoGo_remaining98_started=True,LoGo_remaining98_scientific_accepted=0,whole_comparison_complete=False,whole_rebuttal_complete=False,final_test=False,primary_endpoint_selected=False,original_negative_results_preserved=True,scientific_scope='CUDA211=7 prior500+102 unique old800+100 FL+2 LoGo. FL100=prior3+new97, without duplication. LoGo uses adopted CUDA FedAvg caches and frozen CPU postprocessing; remaining98 start is not scientific acceptance. Observed95/220/main440 are separate. Original failures, CPU/mixed tables and fixed full24 prose are preserved. Compact evidence only, not an archive/source mirror or complete comparison/test.')
(T/'publication_closed_increment73_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
'''
new=text.encode('utf8')
write('publish_guardfed_increment73_20261011.py',text)
compile(new,'publisher73','exec')
blocks=[(b"G=['git'",b'\n# No source mutation'),(b'# No source mutation',b"\ngit('commit'"),(b"raw=git('cat-file'",b'\ncutoff=dict(')]
checks=[]
for start,end in blocks:
    x=old[old.index(start):old.index(end,old.index(start))]
    y=new[new.index(start):new.index(end,new.index(start))]
    assert x==y
    assert ast.dump(ast.parse(x),include_attributes=False)==ast.dump(ast.parse(y),include_attributes=False)
    checks.append({'start':start.decode(),'bytes':len(x),'sha256':H(x),'raw_bytes_exact':True,'AST_exact':True})
ndiff=list(difflib.ndiff(old.decode('utf8').splitlines(keepends=True),text.splitlines(keepends=True)))
assert ''.join(difflib.restore(ndiff,1)).encode('utf8')==old and ''.join(difflib.restore(ndiff,2)).encode('utf8')==new
write('INVERSE_DIFF.json',{'parent_path':'tmp/publication_increment72_prepared_20261011/publish_guardfed_increment72_20261011.py','parent_sha256':H(old),'new_sha256':H(new),'ndiff':ndiff,'old_raw_byte_restore_exact':True,'new_raw_byte_restore_exact':True})
write('SOURCE_DIFF.patch',''.join(difflib.unified_diff(old.decode('utf8').splitlines(keepends=True),text.splitlines(keepends=True),fromfile='increment72',tofile='increment73')))
write('SOURCE_REUSE.json',{'parent_source_path':'tmp/publication_increment72_prepared_20261011/publish_guardfed_increment72_20261011.py','parent_source_sha256':H(old),'parent_commit':PARENT,'three_original_blocks':checks,'changes':'Only namespace/parent/template pin, closed metadata roles and explicit files, fixed CUDA211 assertions, one exact environment-repair .sh path, own_names consistency and receipt wording. No change to original Git/F/copy/stage/blob/push/remote loops.','scientific_execution':False})
tree=ast.parse(text)
assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='own_names' for t in n.targets))
assert ast.literal_eval(assign.value)==OWN_NAMES==set(obj('tmp/publication_increment73_prepared_20261011/SELECTION_TEMPLATE.json')['own_names'])==set(template['own_names'])
assert b'\r' not in new
write('SOURCE_CHECK.json',{'status':'SOURCE_ONLY_STATIC_COMPILE_RAW_INVERSE_AND_CLOSED_SELECTION_PASS','publisher_sha256':H(new),'parent_sha256':H(old),'compiled':['prepare_source.py','publish_guardfed_increment73_20261011.py'],'core_blocks':checks,'both_full_raw_inverse_exact':True,'LF_only':True,'own_names_source_input_template_selection_template_exact':True,'fixed_files':len(files),'fixed_bytes':sum(p['bytes'] for p in files.values()),'parent_refs':len(refs),'proof_roles':len(roles),'late_mutable_files':8,'future_root_authorization_in_templates':False,'future_pins_are_null':True,'Git_F_SSH_science_execution':False,'no_source_directory_glob_at_publication':True})
write('README.md',f'''# Increment73 source-only handoff

Closed cutoff: CUDA211 = 7 prior500 + 102 unique old800 + FL100 + LoGo2. FL adds only97 beyond prior3. NGA19/native400/views380/FL native100/Hybrid29 native and20 views/gradient64 remain unchanged. LoGo remaining98 has an actual start, with zero additional accepted records in this cutoff. Observations95/500,220/800 and mechanism440 are dated09:19, not acceptance. The existing full24 draft remains at its original content cutoff.

No publisher or Git/F operation was executed. All three original72 Git/F/copy/blob/push blocks are raw-byte and AST identical. Source and metadata selection are explicit; no directories expand at runtime. Original source seals and actual failure/repair evidence are retained, with source-only historical handoffs distinguished from actual outcomes. The2.37MB FL input, F archives/arrays/models/images/pickle and redundant embedded deployment bundle remain outside Git; transport/root pins locate their original copies. Extensionless observer/start logs are referenced by exact original command SHA, not renamed or modified.

Root binding: fill STABLE_INPUTS.json (ROOT_BOUND_INCREMENT73_INPUTS/root_accepted true) from INPUTS_TEMPLATE, binding only the exact8 mutable files; retain all fixed files/roles/parent references/counters verbatim. Create actual semantic review from SEMANTIC_TEMPLATE after reading the eight entries. Fill COMPACT_SELECTION.json from SELECTION_TEMPLATE with exact stable files plus exact own_names, including STABLE_INPUTS. Hash all actual bytes. Templates with false/null cannot publish. Their own_names sets are checked against publisher literal, preventing the increment72 pre-Git omission.

Then root alone executes once:
`python -B tmp/publication_increment73_prepared_20261011/publish_guardfed_increment73_20261011.py --semantic-review <actual root review.json> --semantic-review-sha256 <actual SHA> --compact-selection tmp/publication_increment73_prepared_20261011/COMPACT_SELECTION.json --compact-selection-sha256 <actual SHA>`

Parent {PARENT}; branch codex/revision-evidence-baselines-20260928; origin GuardFed unchanged. Fresh F label/Healthy/capacity and clean checkout/branch/origin/remote-parent checks remain in the original block. Parent same-byte references are checked against the72 receipt and actual parent blobs only at root execution. No science, training, inference, calibration, test or new scientific acceptance is created by this publisher.
''')
compile(Path(__file__).read_bytes(),str(Path(__file__)),'exec')
write('HANDOFF.json',{'status':'SOURCE_ONLY_INCREMENT73_CLOSED_CUDA211_READY_FOR_ROOT_BINDING','publisher':{'path':(Q/'publish_guardfed_increment73_20261011.py').relative_to(R).as_posix(),'sha256':H(new)},'parent_commit':PARENT,'previous_receipt':{'path':PREVIOUS,'sha256':PREVIOUS_SHA},'fixed_files':len(files),'fixed_bytes':sum(p['bytes'] for p in files.values()),'parent_references':len(refs),'proof_roles':len(roles),'accepted_CUDA':211,'FL100':100,'LoGo2':2,'remaining98_started_not_accepted':True,'late_binding':['exact8 mutable source bytes','actual semantic review path/SHA','STABLE_INPUTS and exact COMPACT_SELECTION hashes'],'source_checks':'SOURCE_CHECK.json','source_diff':'SOURCE_DIFF.patch','raw_inverse':'INVERSE_DIFF.json','scientific_results_created':0,'Git_F_SSH_execution':False,'root_authorized_publisher_execution':False})
sealed_names=sorted(OWN_NAMES-{'FILES_SHA256.json','STABLE_INPUTS.json'})
write('FILES_SHA256.json',{'files':{name:{'sha256':H((Q/name).read_bytes()),'bytes':(Q/name).stat().st_size} for name in sealed_names},'source_only':True})
print(json.dumps({'source_sha256':H(new),'seal_sha256':H((Q/'FILES_SHA256.json').read_bytes()),'handoff_sha256':H((Q/'HANDOFF.json').read_bytes()),'files':len(files),'bytes':sum(p['bytes'] for p in files.values()),'refs':len(refs),'roles':len(roles)}))
