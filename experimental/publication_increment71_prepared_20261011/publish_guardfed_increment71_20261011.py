"""Publish fixed CUDA10 and two-queue compact evidence with source-only FL/LoGo proposals."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[2]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='4fccc8b8d1d56492ac9cffd0cf27ca390e78bd20'
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
assert selection['status']=='ROOT_CLOSED_INCREMENT71_COMPACT_SELECTION' and selection['root_accepted'] is True
assert selection['parent_commit']==PARENT and selection['semantic_review_sha256']==a.semantic_review_sha256
template=pinned_json(HERE/'INPUTS_TEMPLATE.json','b3e15991c999a6c511ee59154fbb14bcb676ede99bbd6cb99b0441a880a090d3')
stable_input=pinned_json(HERE/'STABLE_INPUTS.json',selection['stable_input_sha256'])
assert stable_input['status']=='ROOT_BOUND_INCREMENT71_INPUTS' and stable_input['root_accepted'] is True
assert stable_input['parent_commit']==template['parent_commit']==PARENT
assert selection['STATE_sha256']==entry['STATE_sha256']==stable_input['STATE_sha256']
for key in ['parent_references','entry_paths','STATE_path','reader_path','protocol_path','current_accepted','draft_path','draft_sha256','late_role_names','omitted_large_artifacts','native11_PDF_path']:
    assert stable_input[key]==template[key],key
roles=stable_input['proof_roles'];assert selection['proof_roles']==roles==template['proof_roles']
for role,pin in roles.items():assert entry['source_pins'][role]==pin
proofs={role:pinned_json(repo_path(pin['path']),pin['sha256']) for role,pin in roles.items()}
previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment70_verified_20261011.json'
previous=pinned_json(repo_path(previous_rel),'d9d28f34b443306f67f2828fd7c74b7dba44cd8422129181e8b4b5d17c6aeb4d')
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
selected=selection['files'];parent_refs=selection['parent_references'];late_files=selection['root_late_bound_files']
assert not set(template['omitted_large_artifacts'])&set(selected)
for rel,pin in template['omitted_large_artifacts'].items():
    assert rel not in selected and re.fullmatch('[0-9a-f]{64}',pin['sha256']) and pin['bytes']>2_000_000
assert parent_refs==stable_input['parent_references'] and not set(parent_refs)&set(selected)
own_names={'INVERSE_DIFF.json', 'HANDOFF.json', 'SELECTION_TEMPLATE.json', 'STABLE_INPUTS.json', 'INPUTS_TEMPLATE.json', 'README.md', 'SEMANTIC_TEMPLATE.json', 'EXPLICIT_SELECTION_PROPOSAL.json', 'OMISSIONS.json', 'SOURCE_DIFF.patch', 'publish_guardfed_increment71_20261011.py', 'prepare_source.py', 'SOURCE_CHECK.json', 'FILES_SHA256.json'}
own_paths={(HERE/n).relative_to(R).as_posix() for n in own_names}
bound_paths=entry_paths|{template['STATE_path'],template['reader_path'],template['protocol_path']}
assert len(bound_paths)==8
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
assert not (T/'publication_closed_increment71_verified_20261011.json').exists(),'One publication attempt only'
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
git('commit','-m','Publish CUDA10 fixed-cutoff evidence and reviewed runtime source proposals')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
cutoff=dict(previous['acceptance_cutoff'],uniform_CUDA_root_accepted=10,uniform_CUDA_old800_root_accepted=3,uniform_CUDA_available_phase=1300)
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=cutoff,proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,parent_references=parent_refs,observed_only_cutoff=entry['fixed_observed_cutoff'],historical_Git70_cutoff=previous['acceptance_cutoff'],working_draft_content_cutoff=entry['working_draft_content_cutoff'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published={'uniform_CUDA':3},source_only_FL_LoGo=True,whole_comparison_complete=False,whole_rebuttal_complete=False,final_test=False,primary_endpoint_selected=False,original_negative_results_preserved=True,scientific_scope='Fixed CUDA10=root7+old800root3; remote58/23 and main427 are observations only. FL/LoGo material here is reviewed source, not runtime or acceptance. Old CPU tables and complete24 historical prose cutoff retained. Old800 constant FedAvg and prior failures retained.')
(T/'publication_closed_increment71_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
