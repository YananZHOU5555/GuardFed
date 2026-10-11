"""Publish closed CUDA407/Full100/LoGo100 and partial four-method native400 proofs."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[2]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='f1148225d99e5b5f469e18bcd3d44f0a1c817875'
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
assert selection['status']=='ROOT_CLOSED_INCREMENT74_COMPACT_SELECTION' and selection['root_accepted'] is True
assert selection['parent_commit']==PARENT and selection['semantic_review_sha256']==a.semantic_review_sha256
template=pinned_json(HERE/'INPUTS_TEMPLATE.json','4ec65c97c450fb059f1206b191bd71b55b111309e79599646f4dacb625a7d8b0')
stable_input=pinned_json(HERE/'STABLE_INPUTS.json',selection['stable_input_sha256'])
assert stable_input['status']=='ROOT_BOUND_INCREMENT74_INPUTS' and stable_input['root_accepted'] is True
assert stable_input['parent_commit']==template['parent_commit']==PARENT
assert selection['STATE_sha256']==entry['STATE_sha256']==stable_input['STATE_sha256']
for key in ['parent_references','entry_paths','STATE_path','reader_path','protocol_path','current_accepted','draft_path','draft_sha256','late_role_names','omitted_large_artifacts','native11_PDF_path','fixed_observed_cutoff']:
    assert stable_input[key]==template[key],key
roles=stable_input['proof_roles'];assert selection['proof_roles']==roles==template['proof_roles']
for role,pin in roles.items():assert entry['source_pins'][role]==pin
proofs={role:pinned_json(repo_path(pin['path']),pin['sha256']) for role,pin in roles.items()}
previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment73_verified_20261011.json'
previous=pinned_json(repo_path(previous_rel),'dd7145de8cc8b9ec311e545836204e7dfd83b2e96e018195a086bcf7f3efcdd6')
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
assert previous['acceptance_cutoff']['uniform_CUDA_root_accepted']==211
metadata=proofs['metadata'];full=proofs['Full100_root'];logo=proofs['LoGo100_root'];table=proofs['four_method_root']
assert metadata['status']=='ROOT_CUDA407_FULL100_LOGO100_GIT73_CURRENT_ENTRIES_RECORDED' and metadata['history_tails_exact'] is True
assert (metadata['root_adopted_CUDA'],metadata['first500_unique_accepted'],metadata['old800_unique_accepted'],metadata['FL100_accepted'],metadata['LoGo100_accepted'],metadata['Full100_accepted'])==(407,105,102,100,100,100)
assert metadata['first500_Full100_overlap_prior7']==2 and metadata['first500_other_controls']==5
assert metadata['new_scientific_adoptions_in_metadata_writer']==0 and metadata['remote_observations_do_not_adopt'] is True
assert (metadata['remote500_saved_accepted'],metadata['remote800_saved_accepted'])==(108,277)
assert 105+102+100+100==metadata['root_adopted_CUDA'] and metadata['Git74_published'] is False
assert full['status']=='ROOT_FULL100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED'
assert (full['accepted'],full['new_unique'],full['reused_already_adopted'])==(100,98,2)
assert full['prior']['root']['sha256']==previous['parent_references']['tmp/uniform_cuda7_saved_audit_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json']['sha256']
fa=proofs['Full100_audit'];ft=proofs['Full100_transport']
assert full['audit']['sha256']==roles['Full100_audit']['sha256'] and full['transport']['sha256']==roles['Full100_transport']['sha256']
assert full['archive_all_members_rehashed'] is True and full['archive']['members']==ft['archive_members']==609
assert full['ids']==fa['ids'] and len(full['records'])==len(set(full['ids']))==100
assert (fa['total_metric_values'],fa['total_integer_base_counts'],fa['total_prediction_rules'])==(900,2400,300)
assert (fa['new_zero_fit_checked'],fa['prior_adopted_reused'])==(98,2)
assert fa['fit_calls']==fa['new_CNN']==0 and fa['root_adopted'] is False
assert ft['all_members_verified'] is True and ft['archive_sha256']==full['archive']['sha256']
assert full['native_max_abs_difference']==0
assert logo['status']=='ROOT_LOGO_ALL100_CUDA_CACHE_FROZEN_POST_AND_OFFSERVER_SAVED_AUDIT_ADOPTED'
assert (logo['accepted'],logo['newly_adopted'],logo['previously_adopted_reused'])==(100,98,2)
assert logo['prior2']['sha256']==previous['proof_roles']['LoGo2_root']['sha256'] and logo['prior2_records_exact'] is True
assert logo['FedAvg_CUDA_index']['sha256']=='6a21265f08fb26da130d95c67bf4130039b25916f44a543cd695e15665a25b64'
la=proofs['LoGo100_audit'];lt=proofs['LoGo100_transport']
assert logo['offserver_audit']['sha256']==roles['LoGo100_audit']['sha256'] and logo['transport']['sha256']==roles['LoGo100_transport']['sha256']
assert logo['ids']==la['ids'] and len(logo['records'])==len(set(logo['ids']))==100
assert logo['archive_all_members_rehashed'] is logo['original_artifacts_immutable_rehashed'] is logo['prior33_members_rehashed'] is True
assert logo['archive']['members']==lt['archive_members']==885 and lt['all_members_verified'] is True
assert la['archive_sha256']==lt['archive_sha256']==logo['archive']['sha256']
assert (la['metric_values_checked'],la['integer_confusion_counts_recomputed'],la['prediction_items_checked'],la['max_metric_error'])==(300,800,1986700,0.0)
assert la['new_fit']==la['new_CNN']==la['pickle_loads']==0 and la['root_adopted'] is False
assert logo['offserver_whole_predict_reexecuted'] is logo['historical_CPU_equality_required'] is False
assert [k for k,v in logo['records'].items() if v['constant_prediction'] is not None]==['LoGoFair-DP_07_FedAvg_IID_Benign_seed91001']
assert logo['records']['LoGoFair-DP_07_FedAvg_IID_Benign_seed91001']['constant_prediction']==0
for root in [full,logo]:
    assert root['root_accepted'] is root['whole100_complete'] is root['old_CPU_outputs_preserved'] is True
    assert root['whole_comparison_complete'] is root['test'] is root['parameters_changed'] is root['tolerance_changed'] is False
build=proofs['four_method_build'];review=proofs['four_method_review']
assert table['status']=='ROOT_PARTIAL_FOUR_METHOD_UNIFORM_CUDA_VALIDATION_TABLE_ADOPTED' and table['root_accepted'] is True
assert table['records']==build['records']==review['identity_grid_records']==400
assert table['methods']==build['methods']==['FedAvg','GuardFed-AD2+','FLGMM','LoGoFair-DP']
assert build['view']=='native' and build['ddof']==table['ddof']==1 and table['seed_panels']==[10,9,6]
assert table['seed_first_aggregation'] is table['outputs_all_SHA_rechecked'] is True
assert table['constant_predictors']=={'FedAvg':1,'GuardFed-AD2+':0,'FLGMM':0,'LoGoFair-DP':1}
assert table['numeric_comparisons']==2772 and table['new_scientific_adoptions']==0
assert table['build_receipt']['sha256']==roles['four_method_build']['sha256']==review['build_receipt_sha256']
assert table['independent_review']['sha256']==roles['four_method_review']['sha256']
assert review['status']=='PASS_INDEPENDENT_ARITHMETIC_AND_FORMAT_VERIFICATION' and not review['unit_findings'] and not review['blocking_findings']
assert table['roots']['GuardFed-AD2+']['sha256']==roles['Full100_root']['sha256'] and table['roots']['LoGoFair-DP']['sha256']==roles['LoGo100_root']['sha256']
assert table['roots']['FLGMM']['sha256']==previous['proof_roles']['FL100_root']['sha256']
assert build['source_seal_sha256']=='4ffb108466144a3c619b149bf41b414b940a77174a5bc9359b8408d602850b14'
assert build['new_fit']==build['new_inference']==0 and build['test'] is build['new_primary_endpoint'] is False
assert table['whole_comparison_complete'] is table['whole_rebuttal_complete'] is table['test'] is table['primary_endpoint_selected'] is False
for name,pin in build['files'].items():
    rel='tmp/celeba_uniform_cuda_fourmethod_partial_20261011/actual_partial400/'+name
    assert template['files'][rel]==pin and H(repo_path(rel).read_bytes())==pin['sha256']
assert table['table']['sha256']==build['files']['TABLES.md']['sha256']
for role,evidence_name in [('Full100_root','Full_root'),('LoGo100_root','LoGo_root'),('four_method_root','partial400_root'),('four_method_build','partial400_build'),('four_method_review','partial400_review'),('health','health'),('cuda500','cuda500'),('cuda800','cuda800')]:
    assert metadata['evidence'][evidence_name]['path']==roles[role]['path'] and metadata['evidence'][evidence_name]['sha256']==roles[role]['sha256']
state_path=T/'TRAINING_STATE.json';state=pinned_json(state_path,entry['STATE_sha256']);main=state['celeba_mechanism_v1'];cuda=state['uniform_CUDA_evaluation_v1']
assert metadata['state_after']['sha256']==entry['STATE_sha256']
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(400,380)
assert (cuda['uniform_CUDA_scientific_accepted'],cuda['root_adopted_first500'],cuda['root_adopted_old800_unique'],cuda['root_adopted_FL100'],cuda['root_adopted_LoGo100'])==(407,105,102,100,100)
assert (cuda['root_adopted_Full100'],cuda['FedAvg100_scientific_accepted'],cuda['current_remote_saved_accepted'],cuda['old800_remote_saved_accepted'],cuda['actual_all_frozen'])==(100,100,108,277,1500)
assert cuda['whole500_complete'] is cuda['whole_comparison_complete'] is cuda['CPU_outputs_reused_as_CUDA'] is False
assert cuda['Full100_whole_complete'] is cuda['LoGo_whole100_complete'] is cuda['LoGo_remaining98_complete'] is True
assert cuda['Full100_root']['sha256']==roles['Full100_root']['sha256'] and cuda['LoGo100_root']['sha256']==roles['LoGo100_root']['sha256']
partial=cuda['partial_fourmethod_native400'];assert partial['root']['sha256']==roles['four_method_root']['sha256'] and partial['root_adopted'] is True and partial['final_17_method_table'] is False
observed=state['latest_current_resource_observation']
assert (observed['main_terminal'],observed['NGA_terminal_observed'],observed['Hybrid_terminal_observed'],observed['NGA_new_accepted'],observed['Hybrid_native_new_accepted'])==(447,27,37,19,29)
assert observed['not_new_scientific_acceptance'] is True and entry['latest_observation_only']==observed
assert state['latest_working_rebuttal_draft']['entry']==template['draft_path'] and state['latest_working_rebuttal_draft']['sha256']==template['draft_sha256']
assert H(repo_path(template['draft_path']).read_bytes())==template['draft_sha256']
assert entry['working_draft_content_cutoff']==state['latest_working_rebuttal_draft']['content_cutoff']
assert entry['current_accepted']==template['current_accepted'] and entry['fixed_observed_cutoff']==template['fixed_observed_cutoff']
assert entry['no_final_test_or_complete_comparison_claim'] is True and entry['historical_Git73_cutoff']==previous['acceptance_cutoff']
entry_paths=set(template['entry_paths']);assert set(entry['entries'])==entry_paths and len(entry_paths)==5
assert {p['path']:p['sha256'] for p in metadata['five_entries']}=={p:pin['sha256'] for p,pin in entry['entries'].items()}
for rel,pin in entry['entries'].items():assert H(repo_path(rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for role in ['reader','protocol']:
    pin=entry[role];assert pin['path']==template[role+'_path'] and H(repo_path(pin['path']).read_bytes())==pin['sha256'] and metadata[role]==pin
for pin in entry['source_pins'].values():assert H(repo_path(pin['path']).read_bytes())==pin['sha256']

selected=selection['files'];parent_refs=selection['parent_references'];late_files=selection['root_late_bound_files']
assert not set(template['omitted_large_artifacts'])&set(selected)
for rel,pin in template['omitted_large_artifacts'].items():
    assert rel not in selected and re.fullmatch('[0-9a-f]{64}',pin['sha256']) and pin['bytes']>2_000_000
assert parent_refs==stable_input['parent_references'] and not set(parent_refs)&set(selected)
own_names={'publish_guardfed_increment74_20261011.py', 'OMISSIONS.json', 'SOURCE_DIFF.patch', 'FILES_SHA256.json', 'STABLE_INPUTS.json', 'README.md', 'SELECTION_TEMPLATE.json', 'INPUTS_TEMPLATE.json', 'current_guards.py.txt', 'FREEZE_PLAN.json', 'SOURCE_CHECK.json', 'SOURCE_REUSE.json', 'EXPLICIT_SELECTION_PROPOSAL.json', 'HANDOFF.json', 'PROVISIONAL_SELECTION.json', 'finalize_source.py', 'SEMANTIC_TEMPLATE.json', 'prepare_source.py', 'INVERSE_DIFF.json'}
assert own_names==set(template['own_names'])
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
EXACT_SHELL_SOURCE_PATHS={'tmp/LoGo100_cuda_actual_runtime_prerequisites_20261011/env_repair_source/create_cpu_environment_once.sh', 'tmp/LoGo100_cuda_actual_prerequisites_v2_20261011/create_cpu_environment_proposal.sh', 'tmp/LoGo100_cuda_actual_prerequisites_20261011/create_cpu_environment_proposal.sh'}
paths=list(dict.fromkeys(paths))
for p in paths:
    assert p.is_file() and (p.suffix in allowed or p.relative_to(R).as_posix() in NDIFF_ALLOWED_PATHS or p.relative_to(R).as_posix() in EXACT_SHELL_SOURCE_PATHS or p.relative_to(R).as_posix()==template['native11_PDF_path']) and p.resolve().is_relative_to(R.resolve())
assert sum(p.stat().st_size for p in paths)<100_000_000
for p in paths:
    assert p.stat().st_size<compact_limit(p) and not p.is_symlink()
destinations=['experimental/'+p.relative_to(R).as_posix()[4:] if p.relative_to(R).as_posix().startswith('tmp/') else p.relative_to(R).as_posix() for p in paths]
assert len(destinations)==len(set(destinations)),'Publication destination collision'
assert not (T/'publication_closed_increment74_verified_20261011.json').exists(),'One publication attempt only'
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
git('commit','-m','Publish CUDA407 Full100 LoGo100 and partial four-method validation table')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
cutoff=dict(previous['acceptance_cutoff'],uniform_CUDA_root_accepted=407,uniform_CUDA_first500_root_accepted=105,uniform_CUDA_Full100=100,uniform_CUDA_old800_root_accepted=102,uniform_CUDA_FL100_root_accepted=100,uniform_CUDA_LoGo_root_accepted=100,uniform_CUDA_partial_four_method_records=400)
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=cutoff,proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,parent_references=parent_refs,observed_only_cutoff=entry['fixed_observed_cutoff'],historical_Git73_cutoff=previous['acceptance_cutoff'],working_draft_content_cutoff=entry['working_draft_content_cutoff'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published={'Full100_new_unique':98,'LoGo100_new_unique':98},whole_comparison_complete=False,whole_rebuttal_complete=False,final_test=False,primary_endpoint_selected=False,original_negative_results_preserved=True,scientific_scope='CUDA407=105 first500 +102 unique old800 +100 FL +100 LoGo. Full100 and LoGo100 each reuse prior2 and add98. Four-method native400 validation table is partial and adds zero scientific records. Observations108/277/main447/NGA27/Hybrid37 are separate from native400/NGA19/Hybrid29 acceptance. Fixed24 draft and old CPU tables/failures preserved. Compact proof publication only, no final comparison or test claim.')
(T/'publication_closed_increment74_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
