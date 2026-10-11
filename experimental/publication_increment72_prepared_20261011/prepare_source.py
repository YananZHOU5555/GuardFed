"""Prepare increment72 source/explicit compact pins only; no Git/F/network/science."""
from pathlib import Path
import ast,difflib,hashlib,json

R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
OLD=R/'tmp/publication_increment71_prepared_20261011'
T='docs/server_deployment_20260923/training_20260923/'
PARENT='ba302df51cbeeee87d0c0f6bd08352a3fd826d03'
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(name,value):
    with (D/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def pin(rel):
    p=R/rel
    assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(R.resolve()),rel
    assert p.stat().st_size<2_000_000,rel
    return {'sha256':H(p.read_bytes()),'bytes':p.stat().st_size}
previous_rel=T+'publication_closed_increment71_verified_20261011.json'
previous=read(R/previous_rel)
assert previous['commit']==PARENT and previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
prior_template=read(OLD/'INPUTS_TEMPLATE.json')
metadata_rel='tmp/root_CUDA112_FedAvg100_20261011/ROOT_METADATA_ADOPTION.json'
metadata=read(R/metadata_rel)
assert metadata['root_adopted_CUDA']==112 and metadata['old800_unique_accepted']==102
paths=[]
def add(folder,names):paths.extend(folder+'/'+n for n in names.split())
def sealed(folder):
    # Resolve this one explicitly named compact seal; never discover a directory tree.
    seal=read(R/folder/'FILES_SHA256.json')
    for name,value in seal['files'].items():
        assert '/' not in name and '\\' not in name
        assert pin(folder+'/'+name)==value,(folder,name)
    add(folder,' '.join(seal['files'])+' FILES_SHA256.json')

roles={
 'metadata':metadata_rel,
 'FedAvg100_root':metadata['evidence']['FedAvg_root']['path'],
 'FedAvg100_audit':'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/actual/SAVED_OUTPUTS_AUDIT.json',
 'FedAvg100_transport':'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/OLD800_FIXED3_ATTEMPT1_TRANSPORT_VERIFICATION.json',
 'CUDA3_root':metadata['evidence']['old3_root']['path'],
 'CUDA7_root':previous['proof_roles']['CUDA7_root']['path'],
 'FL3_root':'tmp/FL100_cuda3_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
 'FL3_audit':'tmp/FL100_cuda3_saved_closure_prepare_20261011/actual/SAVED_OUTPUTS_AUDIT.json',
 'FL3_transport':'tmp/FL100_cuda3_saved_closure_prepare_20261011/OLD800_FIXED3_ATTEMPT1_TRANSPORT_VERIFICATION.json',
 'FL_remaining97_start':'tmp/FL100_cuda_actual_root_20261011/REMAINING97_START_COMMAND.json',
 'NGA19_root':'tmp/NGA96_IID_FFlip10_root_adopter_20261011/ROOT_ADOPTION.json',
 'NGA19_offserver':'tmp/NGA96_IID_FFlip10_delta_prepare_20261011/OFFSERVER_ACCEPTANCE.json',
 'LoGo_v1_review':'tmp/LoGo100_cuda_actual_prerequisites_independent_review_20261011/REVIEW.json',
 'LoGo_v2_review':'tmp/LoGo100_cuda_actual_prerequisites_v2_independent_review_20261011/REVIEW.json',
 'health_snapshot':metadata['evidence']['health']['path'],
 'resume2_observation':metadata['evidence']['cuda500']['path'],
 'old800_observation':metadata['evidence']['cuda800']['path'],
 'FL_observation':metadata['evidence']['FL']['path'],
}
paths.extend(roles.values());paths.append(previous_rel)
add('tmp','record_CUDA112_FedAvg100_root_20261011.py adopt_FedAvg100_cuda_root_once_20261011.py bind_FedAvg100_cuda_actual_root_once_20261011.py adopt_FL100_fixed3_root_once_20261011.py bind_FL100_cuda3_transport_root_once_20261011.py review_FL100_cuda3_saved_closure_root_20261011.py prepare_FL100_actual_root_inputs_20261011.py start_FL100_remaining97_root_once_20261011.py')
for folder in ['tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011','tmp/FL100_cuda3_saved_closure_prepare_20261011','tmp/LoGo100_cuda_actual_prerequisites_20261011','tmp/LoGo100_cuda_actual_prerequisites_v2_20261011','tmp/LoGo100_cuda_actual_prerequisites_independent_review_20261011','tmp/LoGo100_cuda_actual_prerequisites_v2_independent_review_20261011','tmp/FedAvg_cuda100_saved_cache_closure_independent_review_20261011']:
    sealed(folder)
for folder in ['tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011','tmp/FL100_cuda3_saved_closure_prepare_20261011']:
    add(folder,'ACTUAL_AUDIT_COMMAND.json ACTUAL_AUDIT_HANDOFF.json ACTUAL_AUDIT_INPUTS.json ACTUAL_AUDIT_F_VOLUME.json ACTUAL_AUDIT.stdout ACTUAL_AUDIT.stderr OLD800_FIXED3_ATTEMPT1_BACKUP_COMMAND.json OLD800_FIXED3_ATTEMPT1_BACKUP.stdout OLD800_FIXED3_ATTEMPT1_BACKUP.stderr OLD800_FIXED3_ATTEMPT1_SCP_COMMAND.json OLD800_FIXED3_ATTEMPT1_SCP.stdout OLD800_FIXED3_ATTEMPT1_SCP.stderr')
add('tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011','ACTUAL_TRANSPORT_START.json ACTUAL_TRANSPORT_COMMAND.json ACTUAL_TRANSPORT.stdout ACTUAL_TRANSPORT.stderr ACTUAL_AUDIT_START.json actual/STARTED.json')
add('tmp/FL100_cuda_actual_root_20261011','ROOT_FIXED3_TRANSPORT_INPUTS.json ROOT_INSTALL_INPUTS.json REMAINING97_START.stdout REMAINING97_START.stderr install_receipts/COMMAND.json install_receipts/EXIT.json install_receipts/REMOTE.py')
add('tmp/NGA96_IID_FFlip10_delta_prepare_20261011','FILES_SHA256.json SOURCE_FILES_SHA256.json SOURCE_PARAMETER_CHECK.json ROOT_PARAMETERS_PROPOSAL.json AUTHORIZED_SNAPSHOT.json ROOT_REVIEW.json ROOT_READY_HANDOFF.json DELIVERY_FILES_SHA256.json OFFSERVER_ACCEPTANCE.json SAVED_TENSOR_STATE_CHECK.json SERVER_COLLECTOR_RECEIPT.json COLLECTOR_RELEASE.json RAW_STORAGE_INDEX.json DEPENDENCIES.json SCOPE.json README.md run_once.py collect_delta.py pinned_collect_one.py verify_delta.py scope.py prepare_parameters.py DOWNLOAD.json COLLECT_COMMAND.json COLLECT.stdout COLLECT.stderr VERIFY_COMMAND.json VERIFY.stdout VERIFY.stderr RELEASE_COMMAND.json RELEASE.stdout RELEASE.stderr SCP_COMMAND.json SCP.stdout SCP.stderr F_VOLUME_BEFORE_COLLECT.json F_VOLUME_BEFORE_DOWNLOAD.json F_VOLUME_BEFORE_RESTORE.json')
for phase in ['COLLECT','DOWNLOAD','VERIFY','CLOSE']:
    add('tmp/NGA96_IID_FFlip10_delta_prepare_20261011',' '.join('ACTUAL_'+phase+s for s in ['_COMMAND.json','_EXIT.json','.stdout','.stderr']))
add('tmp/NGA96_IID_FFlip10_root_adopter_20261011','adopt_root.py ROOT_ADOPTION.json ROOT_INPUTS.json SOURCE_DIFF.patch ROOT_ADOPT_COMMAND.json ROOT_ADOPT.stdout ROOT_ADOPT.stderr')

# Root supplies only actual compact Linux-index provenance after closure, including the held first attempt.
# This preparation input contains exact paths and hashes, never a directory/wildcard or future PASS.
index_input=read(D/'FEDAVG_INDEX_CLOSED_INPUT.json')
assert index_input['status']=='ACTUAL_INDEX002_COMPACT_INPUTS_NOT_PUBLICATION_AUTHORITY'
for rel,value in index_input['files'].items():assert pin(rel)==value
paths.extend(index_input['files'])
roles.update(index_input['proof_roles'])
for role,rel in roles.items():pin(rel)

draft=prior_template['draft_path'];paths.append(draft)
late=sorted([metadata['state_after']['path'],metadata['reader']['path'],prior_template['protocol_path']]+[p['path'] for p in metadata['five_entries']])
assert len(late)==len(set(late))==8
files={};parent_refs={}
for rel in sorted(set(paths)):
    assert rel not in late,rel
    value=pin(rel);dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
    old=previous['files'].get(dest) or previous.get('parent_references',{}).get(rel)
    if old and all(old[k]==value[k] for k in ['sha256','bytes']):parent_refs[rel]=value
    else:files[rel]=value
cutoff=dict(prior_template['current_accepted'],NGA96_new_scientific_accepted=19,uniform_CUDA_root_accepted=112,uniform_CUDA_remote_observed=77,uniform_CUDA_old800_remote_observed=126,uniform_CUDA_phase=1400,uniform_CUDA_FedAvg100=100,uniform_CUDA_old800_unique=102,uniform_CUDA_FL_root_accepted=3,uniform_CUDA_FL_remote_observed=40)
template={'status':'UNBOUND_INCREMENT72_INPUTS_PROPOSAL','root_accepted':False,'parent_commit':PARENT,'STATE_path':metadata['state_after']['path'],'STATE_sha256':None,'entry_paths':[p['path'] for p in metadata['five_entries']],'reader_path':metadata['reader']['path'],'protocol_path':prior_template['protocol_path'],'draft_path':draft,'draft_sha256':pin(draft)['sha256'],'current_accepted':cutoff,'proof_roles':{k:{'path':rel,'sha256':pin(rel)['sha256']} for k,rel in roles.items()},'late_role_names':[],'files':files,'late_files':{rel:None for rel in late},'parent_references':parent_refs,'omitted_large_artifacts':{},'native11_PDF_path':prior_template['native11_PDF_path'],'index_closed_input_sha256':H((D/'FEDAVG_INDEX_CLOSED_INPUT.json').read_bytes()),'preparation_observation_only':{'metadata_sha256':pin(metadata_rel)['sha256'],'STATE_sha256':metadata['state_after']['sha256'],'main_terminal':432,'CUDA_root_adopted':112,'remote500':77,'remote800':126,'remoteFL':40,'not_publish_authority':True},'LoGo_transfer_install_science_included':False,'publisher_creates_scientific_acceptances':False}
save('INPUTS_TEMPLATE.json',template)
save('EXPLICIT_SELECTION_PROPOSAL.json',{'status':'PROPOSED_FIXED_COMPACT_FILES_NOT_PUBLISH_AUTHORIZATION','fixed_files':files,'parent_references':parent_refs,'late_entry_paths':late,'fixed_files_count':len(files),'fixed_files_bytes':sum(v['bytes'] for v in files.values())})
save('OMISSIONS.json',{'status':'COMPACT_ONLY_NOT_COMPLETE_ARCHIVE_MIRROR','categories':['NPZ/NPY/pickle/checkpoint/model/data/environment/archive','FedAvg100 F archive21076042B and saved arrays','old large original800/900/fullcohort inventories','future LoGo transfer/environment/postprocess outcomes','later running queue results'],'FedAvg100_archive':read(R/roles['FedAvg100_root'])['archive'],'source_seals_referenced_not_standalone_mirror':True,'parent_same_bytes_not_recopied':list(parent_refs)})

old=(OLD/'publish_guardfed_increment71_20261011.py').read_bytes().decode('utf8')
s=old.replace('increment71','increment72').replace('INCREMENT71','INCREMENT72')
s=s.replace("PARENT='4fccc8b8d1d56492ac9cffd0cf27ca390e78bd20'","PARENT='"+PARENT+"'")
s=s.replace('Publish fixed CUDA10 and two-queue compact evidence with source-only FL/LoGo proposals.','Publish CUDA112/FedAvg100/NGA19 compact closure; LoGo prerequisites remain source-only.')
s=s.replace('b3e15991c999a6c511ee59154fbb14bcb676ede99bbd6cb99b0441a880a090d3',H((D/'INPUTS_TEMPLATE.json').read_bytes()))
start=s.index("for key in ['parent_references'");end=s.index('selected=selection[',start)
guards='''for key in ['parent_references','entry_paths','STATE_path','reader_path','protocol_path','current_accepted','draft_path','draft_sha256','late_role_names','omitted_large_artifacts','native11_PDF_path']:
    assert stable_input[key]==template[key],key
roles=stable_input['proof_roles'];assert selection['proof_roles']==roles==template['proof_roles']
for role,pin in roles.items():assert entry['source_pins'][role]==pin
proofs={role:pinned_json(repo_path(pin['path']),pin['sha256']) for role,pin in roles.items()}
previous_rel='''+repr(previous_rel)+'''
previous=pinned_json(repo_path(previous_rel),'''+repr(pin(previous_rel)['sha256'])+''')
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
assert previous['acceptance_cutoff']['uniform_CUDA_root_accepted']==10
metadata=proofs['metadata'];avg=proofs['FedAvg100_root'];fl=proofs['FL3_root'];nga=proofs['NGA19_root']
assert metadata['status']=='ROOT_CUDA112_FEDAVG100_CURRENT_ENTRIES_RECORDED' and metadata['history_tails_exact'] is True
assert (metadata['root_adopted_CUDA'],metadata['old800_unique_accepted'],metadata['remote500_saved_accepted'],metadata['remote800_saved_accepted'],metadata['remoteFL_saved_accepted'])==(112,102,77,126,40)
assert avg['status']=='ROOT_FEDAVG100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED' and avg['accepted']==100
assert fl['status']=='ROOT_FIXED3_FL100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED' and fl['accepted']==3
assert proofs['CUDA7_root']['accepted']==7 and proofs['CUDA3_root']['accepted']==3
assert set(proofs['CUDA3_root']['ids'])&set(avg['ids'])=={'FedAvg_IID_Benign_seed91001'}
assert len(set(proofs['CUDA3_root']['ids'])|set(avg['ids']))==102
assert 7+102+3==metadata['root_adopted_CUDA']
for root,auditrole,transportrole,n,metrics,counts,rules,members in [(avg,'FedAvg100_audit','FedAvg100_transport',100,900,2400,300,1058),(fl,'FL3_audit','FL3_transport',3,27,72,9,77)]:
    audit=proofs[auditrole];transport=proofs[transportrole]
    assert root['audit']['sha256']==roles[auditrole]['sha256'] and root['transport']['sha256']==roles[transportrole]['sha256']
    assert root['archive_all_members_rehashed'] is True and root['archive']['members']==members
    assert root['ids']==audit['exact_ids'] and len(root['records'])==n
    assert (audit['metric_values_checked'],audit['integer_base_counts_checked'],audit['prediction_rules_checked'])==(metrics,counts,rules)
    assert audit['fit_calls']==audit['new_CNN']==audit['new_training']==0 and audit['root_adopted'] is False
    assert transport['all_members_verified'] is True and transport['archive_members']==members
    assert audit['archive_sha256']==transport['archive_sha256']==root['archive']['sha256']
    assert all(r['native_max_abs_difference']==0 for r in root['records'].values())
    assert root['whole_comparison_complete'] is root['test'] is root['parameters_changed'] is root['tolerance_changed'] is False
assert avg['records']['FedAvg_IID_Benign_seed91001']['views']['native']['constant_predictions'] is True
assert nga['root_adopted'] is True and (nga['accepted_before'],len(nga['accepted_new_ids']),nga['accepted_total'])==(9,10,19)
assert nga['old9_ordered_record_prefix_exact'] is nga['all_negative_results_retained'] is nga['constant_predictor_not_fairness_success'] is True
assert nga['offserver']['sha256']==roles['NGA19_offserver']['sha256'] and nga['reused_screen4_count_separate']==4
assert proofs['NGA19_offserver']['members_verified']==196 and proofs['NGA19_offserver']['previous_accepted_ordered_prefix_exact'] is True
assert proofs['FL_remaining97_start']['exit_code']==0
assert proofs['LoGo_v2_review']['source_adoptable'] is True and not proofs['LoGo_v2_review']['blocking_source_findings']
assert proofs['LoGo_v1_review']['source_adoptable_for_direct_execution'] is False
index=proofs['FedAvg_linux_index']
assert index['status']=='ACTUAL_LINUX_FEDAVG100_ROOT_CACHE_INDEX_VERIFIED' and (index['records'],index['exit_code'])==(100,0)
assert index['linux_pins']['ROOT_SCIENTIFIC_ADOPTION.json']['sha256']==roles['FedAvg100_root']['sha256']
assert index['source_bytes_unchanged'] is index['metadata_canonical_alias_only'] is True
assert index['new_CNN']==index['new_fit']==index['arrays_downloaded']==0 and index['prior_failed_index_usable'] is False
state_path=T/'TRAINING_STATE.json';state=pinned_json(state_path,entry['STATE_sha256']);main=state['celeba_mechanism_v1'];cuda=state['uniform_CUDA_evaluation_v1']
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(400,380)
assert (cuda['uniform_CUDA_scientific_accepted'],cuda['root_adopted_old800_unique'],cuda['FedAvg100_scientific_accepted'],cuda['root_adopted_FL100'])==(112,102,100,3)
assert (cuda['current_remote_saved_accepted'],cuda['old800_remote_saved_accepted'],cuda['FL100_server_saved_accepted'],cuda['actual_all_frozen'])==(77,126,40,1400)
assert cuda['whole500_complete'] is cuda['whole_comparison_complete'] is cuda['CPU_outputs_reused_as_CUDA'] is False
assert cuda['FL100_runtime_started'] is True and cuda['LoGo100_runtime_started'] is False
assert cuda['FedAvg100_root']['sha256']==roles['FedAvg100_root']['sha256'] and cuda['FL100_root']['sha256']==roles['FL3_root']['sha256']
assert state['latest_current_resource_observation']['main_terminal']==432 and state['latest_current_resource_observation']['NGA_new_accepted']==19
assert state['latest_current_resource_observation']['not_new_scientific_acceptance'] is True
assert entry['latest_observation_only']==state['latest_current_resource_observation']
assert state['latest_working_rebuttal_draft']['entry']==template['draft_path'] and state['latest_working_rebuttal_draft']['sha256']==template['draft_sha256']
assert H(repo_path(template['draft_path']).read_bytes())==template['draft_sha256']
assert entry['working_draft_content_cutoff']==state['latest_working_rebuttal_draft']['content_cutoff']
assert entry['current_accepted']==template['current_accepted'] and entry['fixed_observed_cutoff']=={'main_terminal':432,'CUDA500_remote':77,'CUDA800_remote':126,'FL100_remote':40}
assert entry['LoGo_source_only'] is True and entry['no_final_test_or_complete_comparison_claim'] is True
assert entry['historical_Git71_cutoff']==previous['acceptance_cutoff']
entry_paths=set(template['entry_paths']);assert set(entry['entries'])==entry_paths and len(entry_paths)==5
for rel,pin in entry['entries'].items():assert H(repo_path(rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for role in ['reader','protocol']:
    pin=entry[role];assert pin['path']==template[role+'_path'] and H(repo_path(pin['path']).read_bytes())==pin['sha256']
for pin in entry['source_pins'].values():assert H(repo_path(pin['path']).read_bytes())==pin['sha256']
'''
s=s[:start]+guards+s[end:]
shell_paths={'tmp/LoGo100_cuda_actual_prerequisites_20261011/create_cpu_environment_proposal.sh','tmp/LoGo100_cuda_actual_prerequisites_v2_20261011/create_cpu_environment_proposal.sh'}
s=s.replace('paths=list(dict.fromkeys(paths))','EXACT_SHELL_SOURCE_PATHS='+repr(shell_paths)+'\npaths=list(dict.fromkeys(paths))')
s=s.replace('or p.relative_to(R).as_posix()==template[\'native11_PDF_path\']','or p.relative_to(R).as_posix() in EXACT_SHELL_SOURCE_PATHS or p.relative_to(R).as_posix()==template[\'native11_PDF_path\']')
own_names=['prepare_source.py','publish_guardfed_increment72_20261011.py','FEDAVG_INDEX_CLOSED_INPUT.json','INPUTS_TEMPLATE.json','EXPLICIT_SELECTION_PROPOSAL.json','OMISSIONS.json','README.md','SOURCE_CHECK.json','SOURCE_REUSE.json','SUFFIX_FIX.json','SOURCE_DIFF.patch','INVERSE_DIFF.json','HANDOFF.json','FILES_SHA256.json','SEMANTIC_TEMPLATE.json','SELECTION_TEMPLATE.json','STABLE_INPUTS.json']
start=s.index('own_names=');end=s.index('\nown_paths=',start);s=s[:start]+'own_names='+repr(set(own_names))+s[end:]
s=s.replace('Publish CUDA10 fixed-cutoff evidence and reviewed runtime source proposals','Publish CUDA112 FedAvg100 NGA19 and reviewed LoGo prerequisites')
start=s.index("cutoff=dict(previous['acceptance_cutoff']")
s=s[:start]+'''cutoff=dict(previous['acceptance_cutoff'],uniform_CUDA_root_accepted=112,uniform_CUDA_old800_root_accepted=102,uniform_CUDA_available_phase=1400,uniform_CUDA_FedAvg100=100,uniform_CUDA_FL100_root_accepted=3,NGA96_completed_scientific_records=19,gradient200_new_scientific_accepted=19)
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=cutoff,proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,parent_references=parent_refs,observed_only_cutoff=entry['fixed_observed_cutoff'],historical_Git71_cutoff=previous['acceptance_cutoff'],working_draft_content_cutoff=entry['working_draft_content_cutoff'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published={'uniform_CUDA_unique':102,'NGA_native':10},LoGo_prerequisites_source_only=True,whole_comparison_complete=False,whole_rebuttal_complete=False,final_test=False,primary_endpoint_selected=False,original_negative_results_preserved=True,scientific_scope='CUDA112=7 prior500+102 unique old800+3 FL; FedAvg100 includes the prior fixed3 FedAvg single and is not counted twice. NGA19 native retains constant-negative results. Remote77/126/40/main432 are observations only. LoGo transfer/environment/runtime outcomes are excluded. Old CPU/mixed tables and historical full24 prose retained; no complete1300/1400/final claim.')
(T/'publication_closed_increment72_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
'''
compile(s,'publisher72','exec');(D/'publish_guardfed_increment72_20261011.py').write_bytes(s.encode('utf8'))
delta=list(difflib.ndiff(old.splitlines(True),s.splitlines(True)))
assert ''.join(difflib.restore(delta,1)).encode()==old.encode() and ''.join(difflib.restore(delta,2)).encode()==s.encode()
save('INVERSE_DIFF.json',{'parent':{'path':'tmp/publication_increment71_prepared_20261011/publish_guardfed_increment71_20261011.py','sha256':H(old.encode())},'derived_sha256':H(s.encode()),'ndiff':delta})
(D/'SOURCE_DIFF.patch').write_bytes(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='publisher71',tofile='publisher72')).encode())
semantic={'status':'UNBOUND_SEMANTIC_REVIEW_TEMPLATE','root_accepted':False,'STATE_sha256':None,'entries':{rel:{'sha256':None,'history_exact':False} for rel in template['entry_paths']},'reader':{'path':template['reader_path'],'sha256':None},'protocol':{'path':template['protocol_path'],'sha256':None},'source_pins':template['proof_roles'],'current_accepted':cutoff,'fixed_observed_cutoff':{'main_terminal':432,'CUDA500_remote':77,'CUDA800_remote':126,'FL100_remote':40},'working_draft_content_cutoff':None,'latest_observation_only':None,'historical_Git71_cutoff':previous['acceptance_cutoff'],'LoGo_source_only':True,'no_final_test_or_complete_comparison_claim':True}
save('SEMANTIC_TEMPLATE.json',semantic)
save('SELECTION_TEMPLATE.json',{'status':'UNBOUND_INCREMENT72_SELECTION','root_accepted':False,'parent_commit':PARENT,'semantic_review_sha256':None,'stable_input_sha256':None,'STATE_sha256':None,'proof_roles':template['proof_roles'],'parent_references':parent_refs,'files':None,'root_late_bound_files':None,'own_names':own_names})
checks={}
def section(b,a,z):return b[b.index(a):b.index(z,b.index(a))]
for name,a,z in [('Git_F_checkout_write_blob',b'G=[',b"git('commit'"),('committed_blob_clean_remote',b"commit=git('rev-parse'",b'cutoff=dict'),('commit_source_mutation_guard',b'# No source mutation',b"git('commit'")]:
    x=section(old.encode(),a,z);y=section(s.encode(),a,z)
    checks[name]={'raw_bytes_exact':x==y,'AST_exact':ast.dump(ast.parse(x))==ast.dump(ast.parse(y)),'sha256':H(x)}
assert all(v['raw_bytes_exact'] and v['AST_exact'] for v in checks.values())
reverted=s.replace('EXACT_SHELL_SOURCE_PATHS='+repr(shell_paths)+'\n','').replace('or p.relative_to(R).as_posix() in EXACT_SHELL_SOURCE_PATHS ','')
assert section(old.encode(),b'paths=[]',b"assert not (T/'publication_closed_increment")==section(reverted.encode(),b'paths=[]',b"assert not (T/'publication_closed_increment")
assert len(shell_paths)==2 and 'tmp/other.sh' not in shell_paths
save('SOURCE_REUSE.json',{'parent_publisher_sha256':H(old.encode()),'unchanged_blocks':checks,'raw_ndiff_both_directions_exact':True,'closed_file_validation_raw_exact_after_two_explicit_shell_paths_inverse':True,'metadata_only_adaptations':['parent71/receipt72/source namespace','fixed explicit compact sources and actual root proof schema','CUDA112 FedAvg100 unique102 FL3 NGA19 factual scope','root-late semantic and eight entry pins','exact two LoGo shell paths only; no global .sh suffix'],'no_science_Git_F_network_executed':True})
save('SOURCE_CHECK.json',{'status':'PASS_SOURCE_PROPOSAL_ONLY_NOT_PUBLISHED','publisher_compile':True,'old_new_raw_inverse_exact':True,'unchanged_execution_blocks':checks,'exact_shell_paths':sorted(shell_paths),'other_sh_rejected':True,'fixed_files':len(files),'fixed_bytes':sum(v['bytes'] for v in files.values()),'late_entries':8,'parent_references':len(parent_refs),'Git':False,'F':False,'SSH':False,'STATE_modified':False,'auxiliary_read_failure':{'origin':'tool_output375b1e','exit':1,'cause':'Reader range extended beyond existing prepare_source.py EOF; no mutation or scientific execution'}})
print(json.dumps({'fixed_files':len(files),'fixed_bytes':sum(v['bytes'] for v in files.values()),'parent_references':len(parent_refs),'late_entries':8,'publisher_sha256':H(s.encode())}))
