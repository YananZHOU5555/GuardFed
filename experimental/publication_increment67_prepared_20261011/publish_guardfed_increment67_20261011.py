"""Publish root-adopted FL90, deployment14, runtime-only NGA96 and late-bound native380 metadata."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[2]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='181969f241d938a93e1e72ac0245a85ae39b3e73'
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
assert selection['status']=='ROOT_CLOSED_INCREMENT67_COMPACT_SELECTION' and selection['root_accepted'] is True
assert selection['parent_commit']==PARENT and selection['semantic_review_sha256']==a.semantic_review_sha256
template=pinned_json(HERE/'INPUTS_TEMPLATE.json','e30f548db530358a55300436c1631e87be0deb9de04437ce10bde9ffd003ee06')
stable_input=pinned_json(HERE/'STABLE_INPUTS.json',selection['stable_input_sha256'])
assert stable_input['status']=='ROOT_BOUND_INCREMENT67_INPUTS' and stable_input['root_accepted'] is True
assert stable_input['parent_commit']==template['parent_commit']==PARENT
assert selection['STATE_sha256']==entry['STATE_sha256']==stable_input['STATE_sha256']
for key in ['parent_references','entry_paths','STATE_path','current_accepted','draft_path','late_role_names']:
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
previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment66_verified_20261011.json'
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
assert editorial['candidate_path'].replace('\\','/')==template['draft_path'] and editorial['candidate_sha256']==stable_input['draft_sha256']
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
entry_paths={x['path'] for x in proofs['entry_update_proof']['entries']}
assert set(entry['entries'])==entry_paths==set(template['entry_paths']) and len(entry_paths)==5
for rel,pin in entry['entries'].items():assert H(repo_path(rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for pin in entry['source_pins'].values():assert H(repo_path(pin['path']).read_bytes())==pin['sha256']
selected=selection['files'];parent_refs=selection['parent_references'];late_files=selection['root_late_bound_files']
assert parent_refs==stable_input['parent_references'] and not set(parent_refs)&set(selected)
own_names={'README.md', 'SELECTION_TEMPLATE.json', 'INPUTS_TEMPLATE.json', 'STABLE_INPUTS.json', 'SEMANTIC_TEMPLATE.json', 'INVERSE_DIFF.json', 'SOURCE_CHECK.json', 'prepare_source.py', 'SOURCE_DIFF.patch', 'FILES_SHA256.json', 'HANDOFF.json', 'publish_guardfed_increment67_20261011.py'}
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
    assert p.is_file() and (p.suffix in allowed or p.relative_to(R).as_posix() in NDIFF_ALLOWED_PATHS) and p.resolve().is_relative_to(R.resolve())
assert sum(p.stat().st_size for p in paths)<100_000_000
for p in paths:
    assert p.stat().st_size<compact_limit(p) and not p.is_symlink()
destinations=['experimental/'+p.relative_to(R).as_posix()[4:] if p.relative_to(R).as_posix().startswith('tmp/') else p.relative_to(R).as_posix() for p in paths]
assert len(destinations)==len(set(destinations)),'Publication destination collision'
assert not (T/'publication_closed_increment67_verified_20261011.json').exists(),'One publication attempt only'
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
git('commit','-m','Publish adopted FL90 and deployment14 with native380 and NGA runtime metadata')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=dict(native=380,three_view=370,FLGMM_native_new=86,FLGMM_screen_reuse_separate=4,FLGMM_three_views=90,FLGMM_table_complete_scene_records=90,FLGMM_table_partial_records=0,FLGMM_table_scenes=9,gradient_search=64,gradient200_frozen_new_jobs=192,gradient200_screen_reuse_separate=8,gradient200_new_scientific_accepted=0,gradient14_gate_jobs=14,gradient14_scientific_table_records=0,NGA96_runtime_started=True,NGA96_completed_scientific_records=0,Hybrid_native_new=19,Hybrid_three_views=20,Hybrid_table_scenes=2,UCA_models=300,F_table_pairs=70,F_table_scenes=7,reviewer_comments=24),proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,latest_observation_sha256=entry['latest_observation_only']['sha256'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published=dict(mechanism=0,FLGMM=9,Hybrid=0),new_native_records_published=10,new_native_records_by_queue=dict(mechanism=10,FLGMM=0,Hybrid=0),working_draft_content_cutoff=entry['working_draft_content_cutoff'],historical_Git66_cutoff=entry['historical_Git66_cutoff'],parent_references=parent_refs,gradient14_evidence_scope='Original14 same-horizon CPU3 deployment gates and457-member offserver proof; zero scientific table records',NGA96_evidence_scope='Original runner wiring and observed first progress only; retained parent-missing startup failure and bounded metadata correction; zero completed70 scientific records',F80_and_FedSA10_three_view_excluded=True,original_Windows47_refit_failure_preserved=True,Windows_whole_refit_pass=False,manuscript_applied=False,primary_endpoint_selected=False,final_test=False,whole_rebuttal_complete=False)
(T/'publication_closed_increment67_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
