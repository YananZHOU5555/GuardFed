"""Publish fixed FL100/CUDA7/Hybrid29/NGA9 compact evidence; preserve negative and historical results."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[2]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='0a276220b447645020e8dca8ee092ce411367bb7'
BRANCH='codex/revision-evidence-baselines-20260928'
H=lambda b:hashlib.sha256(b).hexdigest()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--semantic-review',type=Path,required=True)
parser.add_argument('--semantic-review-sha256',required=True)
parser.add_argument('--compact-selection',type=Path,required=True)
parser.add_argument('--compact-selection-sha256',required=True)
a=parser.parse_args()

def pinned_json(path,sha):
    assert re.fullmatch('[0-9a-f]{64}',sha)
    assert path.is_file() and not path.is_symlink() and path.stat().st_size<2_000_000 and H(path.read_bytes())==sha,path
    return json.loads(path.read_bytes())

def repo_path(rel):
    assert isinstance(rel,str) and '\\' not in rel and ':' not in rel and not rel.startswith('/')
    p=R/rel
    assert p.relative_to(R).as_posix()==rel and '..' not in p.relative_to(R).parts and p.resolve().is_relative_to(R.resolve()),rel
    return p

semantic_review=(R/a.semantic_review).resolve()
assert semantic_review.relative_to(R).parts[0]=='tmp' and semantic_review.suffix=='.json'
HERE=Path(__file__).resolve().parent
scope_path=(R/a.compact_selection).resolve()
assert scope_path==HERE/'COMPACT_SELECTION.json'
entry=pinned_json(semantic_review,a.semantic_review_sha256)
selection=pinned_json(scope_path,a.compact_selection_sha256)
assert entry['status']=='PASS_LIMITED_CURRENT_ENTRY_SEMANTIC_REVIEW' and entry['root_accepted'] is True
assert selection['status']=='ROOT_CLOSED_INCREMENT70_COMPACT_SELECTION' and selection['root_accepted'] is True
assert selection['parent_commit']==PARENT and selection['semantic_review_sha256']==a.semantic_review_sha256
template=pinned_json(HERE/'INPUTS_TEMPLATE.json','20137180b9b3d4de14c67f74c9bffa0b66db589768a41df357cd580777bfc9b4')
stable_input=pinned_json(HERE/'STABLE_INPUTS.json',selection['stable_input_sha256'])
assert stable_input['status']=='ROOT_BOUND_INCREMENT70_INPUTS' and stable_input['root_accepted'] is True
assert stable_input['parent_commit']==template['parent_commit']==PARENT
assert selection['STATE_sha256']==entry['STATE_sha256']==stable_input['STATE_sha256']
for key in ['parent_references','entry_paths','STATE_path','current_accepted','draft_path','late_role_names','omitted_large_artifacts','native11_PDF_path']:
    assert stable_input[key]==template[key],key
roles=stable_input['proof_roles'];assert selection['proof_roles']==roles
assert set(roles)==set(template['proof_roles'])
for role,pin in roles.items():
    assert set(pin)=={'path','sha256'} and isinstance(pin['path'],str)
    assert entry['source_pins'][role]==pin
    if role not in template['late_role_names']:assert pin==template['proof_roles'][role]
    else:
        assert pin['path'].startswith(('tmp/','docs/server_deployment_20260923/')) and Path(pin['path']).suffix in {'.py','.json'}
proofs={role:pinned_json(repo_path(pin['path']),pin['sha256']) for role,pin in roles.items() if Path(pin['path']).suffix=='.json'}
previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment69_verified_20261011.json'
previous=pinned_json(repo_path(previous_rel),'4667bab2574b294c66cb48c537e77c5586d86f54934304134ec4d862fa7b47e6')
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
assert previous['acceptance_cutoff']['native']==400 and previous['acceptance_cutoff']['NGA96_completed_scientific_records']==1
fl=proofs['FL100_root'];ft=proofs['FL100_table'];nga=proofs['NGA9_root'];hy=proofs['Hybrid29_root'];cu=proofs['CUDA7_root']
assert fl['root_adoption'] is True and (fl['new_three_view_records_accepted'],fl['FLGMM_total_three_view_records'])==(10,100)
assert fl['prior90_objects_order_unchanged'] is True and fl['Linux_whole_original_saved_check_pass'] is True
assert fl['Windows_saved_outputs_audit_fit_calls']==0 and fl['Windows_original47_exact_refit_pass'] is False and fl['cross_platform_bitwise_recalibration_claimed'] is False
assert ft['root_adopted'] is True and (ft['records'],ft['scenes'],ft['seed_panels'])==(100,10,[10,9,6])
assert ft['root100']['sha256']==roles['FL100_root']['sha256'] and ft['old90_objects_and_486_scalars_243_cells_exact'] is True
assert ft['uniform_CUDA_cohort_result'] is False and ft['Windows_whole_refit_pass'] is False and ft['final_test'] is False
assert nga['root_adopted'] is True and (nga['accepted_before'],nga['accepted_total'])==(1,9)
assert nga['constant_predictor_not_fairness_success'] is True and nga['all_negative_results_retained'] is True and nga['coverage100_completed_claim'] is False
assert (hy['prior_accepted'],hy['new_accepted'],hy['cumulative_accepted'],hy['saved_tensor_count'])==(19,10,29,80)
assert hy['accepted_new_ids']==['CosineFairness_lam20.0_tau0.1_lr0.001_IID_FedSA_seed'+str(s)+'_fullcoverage' for s in range(91001,91011)]
assert hy['archive_all_members_rehashed'] is True and hy['whole100_complete'] is False and hy['table_adopted'] is False
ids=['GuardFed-AD2+_IID_Benign_seed91001','minus_U_IID_Benign_seed91001','minus_C_IID_Benign_seed91001','minus_A_IID_Benign_seed91001','minus_F_IID_Benign_seed91001','GuardFed-AD2+_non-IID_Sp-DFA_seed91004','minus_F_non-IID_Sp-DFA_seed91004']
assert cu['status']=='ROOT_FIXED7_UNIFORM_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED' and cu['ids']==ids and cu['accepted']==7
assert all(v==0 for v in cu['native_max_abs_differences'].values()) and cu['archive']['members']==110 and cu['archive']['payload_members']==109
assert cu['audit_sha256']==roles['CUDA7_audit']['sha256'] and cu['transport_sha256']==roles['CUDA7_transport']['sha256']
assert cu['whole500_complete'] is False and cu['whole_comparison_complete'] is False and cu['diagnostic_retroactively_adopted'] is False and cu['old_CPU_outputs_preserved'] is True
audit=proofs['CUDA7_audit'];transport=proofs['CUDA7_transport']
assert audit['exact_ids']==ids and (audit['fit_calls'],audit['new_CNN'],audit['metric_values_checked'],audit['integer_base_counts_checked'],audit['prediction_rules_checked'])==(0,0,63,168,21)
assert transport['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and transport['all_members_verified'] is True and (transport['archive_members'],transport['payload_members'],len(transport['members']))==(110,109,109)
assert cu['archive']['sha256']==transport['archive_sha256']==audit['archive_sha256']
assert proofs['CUDA_author']['approved']['uniform_CUDA_evaluation'] and proofs['CUDA_author']['approved']['corresponding_Full_included'] and not proofs['CUDA_author']['existing_diagnostic_retroactively_adopted']
assert proofs['resume_install']['exit_code']==0
resume=proofs['resume_observation'];assert len(resume['progress.json']['remote_saved_accepted_ids'])==11 and resume['progress.json']['failed']==0 and resume['progress.json']['pending']==488
assert not resume.get('QUEUE_FAILURE.json') and resume['progress.json']['active']=='GuardFed-AD2+_IID_Benign_seed91006'
editorial=proofs['current_working_editorial_root']
assert editorial['root_adopted'] is True and editorial['status']=='ROOT_COMPLETE24_FL100_LEGACY_CPU_NATIVE11_EDITORIAL_UPDATE_ADOPTED'
assert editorial['adopted_draft']['path']==template['draft_path'] and editorial['adopted_draft']['sha256']==stable_input['draft_sha256']
assert (editorial['comment_blocks'],editorial['reviewer_quote_lines_exact'])==(24,38)
assert editorial['all_original_headings_and_performance_tables_exact'] is True and editorial['inverse_reconstructs_parent'] is True
assert editorial['fixed_scientific_cutoff']=={'mechanism_native':380,'mechanism_saved':380,'F_pairs':80,'FL_native':100,'FL_saved_CPU':100,'Hybrid_pairs':20,'NGA_new_accepted_at_cutoff':0,'native_methods':11,'native_evaluation_records':1100}
assert editorial['uniform_CUDA_results_claimed'] is False and editorial['submission_ready'] is False and editorial['final_test'] is False
assert H(repo_path(template['draft_path']).read_bytes())==editorial['adopted_draft']['sha256']
metadata=proofs['entry_update_proof'];alias=proofs['entry_alias_fix']
assert metadata['status']=='ROOT_CUDA7_RESUME1_HYBRID29_CURRENT_ENTRIES_RECORDED' and metadata['history_tails_exact'] is True
assert alias['status']=='STALE_LATEST_ALIASES_UPDATED_TO_ROOT_ACCEPTED_FL100_DRAFT' and alias['state_before_sha256']==metadata['state_after']['sha256']
assert alias['new_science']==0 and alias['draft_content_changed'] is False and alias['five_entry_historical_tails_untouched'] is True
state_path=T/'TRAINING_STATE.json';state=pinned_json(state_path,entry['STATE_sha256']);main=state['celeba_mechanism_v1']
assert entry['STATE_sha256']==alias['state_after_sha256']=='8f803c7b0c4789a369dfce1997a9262909e2b5072e4f5e019cc3be652c80a2a4'
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'],main['queue_completed_observed'])==(400,380,416)
assert state['NGA96_coverage_acceptance']['new_accepted']==9 and state['NGA96_coverage_acceptance']['root_proof']['sha256']==roles['NGA9_root']['sha256']
assert state['hybrid100_fullcoverage_20261010']['new_accepted']==29 and state['hybrid100_fullcoverage_20261010']['latest_delta_root_sha256']==roles['Hybrid29_root']['sha256']
cuda=state['uniform_CUDA_evaluation_v1'];assert (cuda['uniform_CUDA_scientific_accepted'],cuda['actual_phase_frozen'],cuda['current_remote_saved_accepted'])==(7,500,11)
assert cuda['first7_root']['sha256']==roles['CUDA7_root']['sha256'] and cuda['current_observation']['sha256']==roles['resume_observation']['sha256']
assert cuda['CPU_outputs_reused_as_CUDA'] is False and cuda['whole500_complete'] is False and cuda['whole_comparison_complete'] is False
assert state['latest_working_rebuttal_draft']['entry']==state['latest_rebuttal_draft']['entry']==template['draft_path']
assert state['latest_working_rebuttal_draft']['sha256']==state['latest_rebuttal_draft']['sha256']==editorial['adopted_draft']['sha256']
assert state['current_entry_writer']['sha256']==roles['entry_writer']['sha256']
assert entry['current_accepted']==template['current_accepted'] and entry['working_draft_content_cutoff']==state['latest_working_rebuttal_draft']['content_cutoff']
assert entry['historical_Git69_cutoff']==previous['acceptance_cutoff']
assert entry['latest_observation_only']==state['latest_current_resource_observation'] and entry['latest_observation_only']['not_new_scientific_acceptance'] is True
assert entry['NGA96_scientific_completed']==9 and entry['uniform_CUDA_root_accepted']==7 and entry['uniform_CUDA_remote_observed']==11
assert entry['gradient14_deployment_only'] is True and entry['F80_root_accepted'] is True and entry['replay_failure_preserved'] is True
assert state['mechanism_replay_failure_20261011']['comparison']['accepted'] is False
assert state['uniform_CUDA_infrastructure_interruption_20261011']['original_partial_formally_adopted'] is False
entry_paths={x['path'] for x in metadata['five_entries']}
assert set(entry['entries'])==entry_paths==set(template['entry_paths']) and len(entry_paths)==5
for rel,pin in entry['entries'].items():assert H(repo_path(rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for pin in entry['source_pins'].values():assert H(repo_path(pin['path']).read_bytes())==pin['sha256']
selected=selection['files'];parent_refs=selection['parent_references'];late_files=selection['root_late_bound_files']
assert not set(template['omitted_large_artifacts'])&set(selected)
for rel,pin in template['omitted_large_artifacts'].items():
    assert rel not in selected and re.fullmatch('[0-9a-f]{64}',pin['sha256']) and pin['bytes']>2_000_000
assert parent_refs==stable_input['parent_references'] and not set(parent_refs)&set(selected)
own_names={'fact_guards.py.txt', 'README.md', 'STABLE_INPUTS.json', 'INVERSE_DIFF.json', 'HANDOFF.json', 'OMISSIONS.json', 'SELECTION_TEMPLATE.json', 'prepare_source.py', 'receipt_footer.py.txt', 'SEMANTIC_TEMPLATE.json', 'SOURCE_DIFF.patch', 'INPUTS_TEMPLATE.json', 'FILES_SHA256.json', 'SOURCE_CHECK.json', 'publish_guardfed_increment70_20261011.py'}
own_paths={(HERE/n).relative_to(R).as_posix() for n in own_names}
bound_paths={roles[k]['path'] for k in template['late_role_names']}|entry_paths|{template['STATE_path'],template['draft_path']}
assert len(bound_paths)==len(template['late_role_names'])+7
assert set(stable_input['late_files'])==bound_paths
assert set(stable_input['files'])==set(template['files'])|bound_paths
for rel,pin in template['files'].items():assert stable_input['files'][rel]==pin
for rel,pin in stable_input['late_files'].items():assert stable_input['files'][rel]==pin
assert set(selected)==set(stable_input['files'])|own_paths and set(late_files)==own_paths
for rel,pin in stable_input['files'].items():assert selected[rel]==pin
for rel,pin in late_files.items():assert selected[rel]==pin
for rel,pin in parent_refs.items():
    p=repo_path(rel);assert set(pin)=={'sha256','bytes'} and p.is_file() and not p.is_symlink()
    assert p.stat().st_size==pin['bytes']<2_000_000 and H(p.read_bytes())==pin['sha256'],rel
    dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
    old=previous['files'].get(dest) or previous.get('parent_references',{}).get(rel)
    assert old and (old['sha256'],old['bytes'])==(pin['sha256'],pin['bytes']),rel
assert scope_path.relative_to(R).as_posix() not in selected
paths=[]
for rel,pin in selected.items():
    p=repo_path(rel)
    assert set(pin)=={'sha256','bytes'} and re.fullmatch('[0-9a-f]{64}',pin['sha256'])
    assert isinstance(pin['bytes'],int) and not isinstance(pin['bytes'],bool) and pin['bytes']>=0
    assert p.is_file() and not p.is_symlink() and H(p.read_bytes())==pin['sha256'] and p.stat().st_size==pin['bytes'],rel
    paths.append(p)
# Exactly the closed list plus these two independently hash-bound input JSONs; no glob or directory expansion.
paths += [semantic_review,scope_path]
compact_limit=lambda p:2_000_000
allowed={'.py','.json','.md','.patch','.txt','.stderr','.stdout','.sha256','.csv','.tex','.diff'}
NDIFF_ALLOWED_PATHS={
    'tmp/hybrid_screen91001_saved_runtime_prepared_20261011/check_linux.py.ndiff',
    'tmp/hybrid_screen91001_saved_runtime_prepared_20261011/contract.py.ndiff',
    'tmp/hybrid_screen91001_saved_runtime_prepared_20261011/linux_saved_remote.py.ndiff',
    'tmp/hybrid_screen91001_saved_runtime_prepared_20261011/transport.py.ndiff',
    'tmp/hybrid_screen91001_saved_runtime_prepared_20261011/transport_remote.py.ndiff',
    'tmp/hybrid_screen91001_saved_runtime_prepared_20261011/windows_runtime_wrapper.py.ndiff',
}
paths=list(dict.fromkeys(paths))
for p in paths:
    assert p.is_file() and (p.suffix in allowed or p.relative_to(R).as_posix() in NDIFF_ALLOWED_PATHS or p.relative_to(R).as_posix()==template['native11_PDF_path']) and p.resolve().is_relative_to(R.resolve())
assert sum(p.stat().st_size for p in paths)<100_000_000
for p in paths:
    assert p.stat().st_size<compact_limit(p) and not p.is_symlink()
destinations=['experimental/'+p.relative_to(R).as_posix()[4:] if p.relative_to(R).as_posix().startswith('tmp/') else p.relative_to(R).as_posix() for p in paths]
assert len(destinations)==len(set(destinations)),'Publication destination collision'
assert not (T/'publication_closed_increment70_verified_20261011.json').exists(),'One publication attempt only'
G=['git','-c','safe.directory='+W.as_posix(),'-c','core.longpaths=true','-C',str(W)]
def git(*args,data=None):
    p=subprocess.run(G+list(args),input=data,capture_output=True)
    assert p.returncode==0,p.stderr.decode(errors='replace')
    return p.stdout
volume=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',"Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json"]))
assert volume['FileSystemLabel']=='Yanan 2TB' and volume['HealthStatus']=='Healthy' and volume['SizeRemaining']>1_000_000_000
assert git('rev-parse','HEAD').decode().strip()==PARENT and not git('status','--porcelain')
assert git('branch','--show-current').decode().strip()==BRANCH
assert git('remote','get-url','origin').decode().strip()=='https://github.com/YananZHOU5555/GuardFed.git'
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==PARENT
for rel,pin in parent_refs.items():
    dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
    content=git('show',PARENT+':'+dest)
    assert H(content)==pin['sha256'] and len(content)==pin['bytes'],rel
pins={}
for p in paths:
    assert p.stat().st_size<compact_limit(p) and not p.is_symlink()
    rel=p.relative_to(R).as_posix();dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
    b=p.read_bytes();target=W/dest;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
    pins[dest]={'source':rel,'sha256':H(b),'bytes':len(b)}
attr=W/'.gitattributes';lines=attr.read_text(encoding='utf8').splitlines()
for dest in pins:
    line=json.dumps(dest,ensure_ascii=False)+' -text'
    if line not in lines:lines.append(line)
attr.write_bytes(('\n'.join(lines)+'\n').encode('utf8'));pins['.gitattributes']={'sha256':H(attr.read_bytes()),'bytes':attr.stat().st_size}
names=list(pins)
for i in range(0,len(names),30):git('add','--sparse','-f','--',*names[i:i+30])
changed=[p for p in git('diff','--cached','--name-only','-z').decode('utf8').split('\0') if p]
assert changed and set(changed)<=set(pins)
# No source mutation may pass from the closed selection into a commit.
for dest,pin in pins.items():
    if dest=='.gitattributes':continue
    rel=pin['source']
    expected=selected.get(rel)
    if expected is not None:
        assert (pin['sha256'],pin['bytes'])==(expected['sha256'],expected['bytes']),rel
    else:
        assert rel in {semantic_review.relative_to(R).as_posix(),scope_path.relative_to(R).as_posix()}
        expected_sha=a.semantic_review_sha256 if rel==semantic_review.relative_to(R).as_posix() else a.compact_selection_sha256
        assert pin['sha256']==expected_sha,rel
assert H(state_path.read_bytes())==entry['STATE_sha256'],'Current STATE drifted before commit'
git('commit','-m','Publish FL100 CUDA7 Hybrid29 NGA9 fixed-cutoff evidence and author review')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
cutoff=dict(previous['acceptance_cutoff'],FLGMM_three_views=100,FLGMM_table_complete_scene_records=100,FLGMM_table_scenes=10,NGA96_completed_scientific_records=9,gradient200_new_scientific_accepted=9,Hybrid_native_new=29,uniform_CUDA_root_accepted=7,uniform_CUDA_available_phase=500)
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=cutoff,proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,latest_observation_sha256=entry['latest_observation_only']['source_ref']['sha256'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published=dict(mechanism_legacy_CPU=0,FLGMM_legacy_CPU=10,Hybrid_legacy_CPU=0,uniform_CUDA=7),new_native_records_published=18,new_native_records_by_queue=dict(mechanism=0,FLGMM=0,NGA=8,Hybrid=10),working_draft_content_cutoff=entry['working_draft_content_cutoff'],historical_Git69_cutoff=entry['historical_Git69_cutoff'],observed_only_cutoff=dict(main_terminal=416,uniform_CUDA_remote_saved=11,uniform_CUDA_root_adopted=7,Hybrid_native_adopted=29,NGA_native_adopted=9),parent_references=parent_refs,omitted_large_artifacts=template['omitted_large_artifacts'],scientific_scope='Fixed cutoff: native400 and legacy CPU mechanism views380/F80 retained; FL100 legacy CPU three views/ten scenes complete; Hybrid29 native and legacy CPU views20; NGA9 negative constant predictors preserved. Author approved uniform CUDA complete comparison; actual first phase frozen500, root7 adopted, remote11 observed at fixed snapshot. Later live progress is excluded. Original CUDA proc permission failure preserved and one explicit resume in a new namespace is documented; old CPU failure not relabeled. Full24 draft keeps fixed380/F80, FL100, Hybrid20/native11 and NGA0 prose cutoff; no complete500/2500/17methods/finaltest claim.',original_Windows47_refit_failure_preserved=True,Windows_whole_refit_pass=False,remaining213_numerical_failure_preserved=True,formal_CUDA_recovery_executed=True,old_CPU_replay_recovery_claimed=False,FL_final10_complete_legacy_CPU=True,manuscript_applied=False,primary_endpoint_selected=False,final_test=False,whole_rebuttal_complete=False)
(T/'publication_closed_increment70_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
