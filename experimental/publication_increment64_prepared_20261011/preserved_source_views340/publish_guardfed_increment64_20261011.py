"""Publish closed native350/FL81/gradient64 compact evidence; no192 result claim."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[2]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='ccb5207c24762d3ac92ed76b40bd40b3e2330954'
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
assert selection['status']=='ROOT_CLOSED_INCREMENT64_COMPACT_SELECTION' and selection['root_accepted'] is True
assert selection['parent_commit']==PARENT and selection['semantic_review_sha256']==a.semantic_review_sha256
assert selection['STATE_sha256']==entry['STATE_sha256']
stable_path=HERE/'STABLE_INPUTS.json'
stable_input=pinned_json(stable_path,'40c918c13cecec94a6e87688220349438a624855bed4b410c98e6a8c7c43f09c');stable=stable_input['files']
roles=selection['proof_roles'];assert roles==stable_input['proof_roles']
for role,pin in roles.items():assert entry['source_pins'][role]==pin
proofs={role:pinned_json(repo_path(pin['path']),pin['sha256']) for role,pin in roles.items()}
native_count=350;native_pin=roles['native350_root'];native=proofs['native350_root']
assert native['root_adopted'] is True and native['total_new_strict_and_offserver']==350
assert native['old441_records_exact'] is True and native['test'] is False
flnative=proofs['FL_native77_root'];flproof=proofs['FL_three_view81_root'];ftproof=proofs['FL_eight_scene80_table_root']
assert (flnative['accepted_total'],flnative['reused_separately'])==(77,4) and flnative['final_test'] is False
assert flproof['root_adoption'] is True and (flproof['FLGMM_total_three_view_records'],flproof['new_three_view_records_accepted'])==(81,10)
assert flproof['native_root_sha256']==roles['FL_native77_root']['sha256']
assert flproof['Linux_root_fit_verified'] is True and flproof['Windows_saved_outputs_audit_pass'] is True
assert flproof['Windows_saved_outputs_audit_fit_calls']==0 and flproof['Windows_whole_saved_check_pass'] is False and flproof['Windows_original47_exact_refit_pass'] is False
assert ftproof['root_adoption'] is True and (ftproof['complete_scene_records'],ftproof['retained_partial_records'],ftproof['scenes'])==(80,1,8)
assert ftproof['source_root81_sha256']==roles['FL_three_view81_root']['sha256'] and ftproof['final_test'] is False
assert ftproof['prior_windows_failure_preserved'] is True and ftproof['whole_windows_refit_equality_claim'] is False
grad=proofs['gradient64_root'];chosen=proofs['gradient64_selected_root'];frozen=proofs['gradient200_frozen_source_root']
assert (grad['accepted_before'],grad['accepted_new'],grad['accepted_total'])==(46,18,64) and grad['screen64_complete'] is True
assert grad['all_negative_results_retained'] is True and grad['final_test'] is False
assert chosen['accepted_count']==64 and chosen['all64_offserver_verified'] is True and chosen['final_test'] is False
assert chosen['all64_native_root_sha256']==roles['gradient64_root']['sha256'] and chosen['execution_authorized'] is False
summary=pinned_json(repo_path('tmp/gradient64_final_acceptance_prepared_20261011/COMPLETE64_SELECTION.json'),chosen['summary_sha256'])
assert summary['selected_candidates']==chosen['selected_candidates'] and summary['accuracy_champions']==chosen['accuracy_champions'] and summary['pareto_candidates']==chosen['pareto_candidates']
assert frozen['status']=='ROOT_GRADIENT200_FROZEN_SOURCE_ADOPTED' and frozen['root_selected64_sha256']==roles['gradient64_selected_root']['sha256']
assert (frozen['new_frozen_jobs'],frozen['reused_original_screen_records'])==(192,8)
assert frozen['coverage192_authorized'] is False and frozen['workers_started']==0 and frozen['test'] is False
editorial=proofs['working_editorial_root'];EDITORIAL_ROOT=roles['working_editorial_root']['sha256'];EDITORIAL_REL=roles['working_editorial_root']['path'];WORKING_REL=editorial['draft_path']
assert editorial['status']=='ROOT_EDITORIAL_F40_GRADIENT64_ADOPTED_FIXED_CUTOFF'
assert editorial['original24_comments_and38_quote_lines_exact'] is True and editorial['negative_findings_and_adaptation_limits_preserved'] is True
assert (editorial['accepted_cutoff_native'],editorial['accepted_cutoff_three_view'],editorial['F_complete_scenes'],editorial['gradient_screen'])==(341,340,4,64)
assert editorial['FL81_or_native350_or_gate14_included'] is False and editorial['test'] is False and editorial['submission_ready'] is False
assert H(repo_path(WORKING_REL).read_bytes())==editorial['draft_sha256']
previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment63_verified_20261011.json'
previous=pinned_json(repo_path(previous_rel),'88b77f1d4fe27016c57bf45931fff2a03be582b4c01be361ca865a0525fb0d8c')
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
assert (previous['acceptance_cutoff']['native'],previous['acceptance_cutoff']['three_view'],previous['acceptance_cutoff']['gradient_search'])==(341,340,46)
state_path=T/'TRAINING_STATE.json';state=pinned_json(state_path,entry['STATE_sha256']);main=state['celeba_mechanism_v1']
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(350,340)
assert main['three_view_counts_by_variant']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':40}
assert main['three_view_root_proof_sha256']==previous['mechanism_three_view_root_sha256']
assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==77
assert state['gradient64_validation_search_20261010']['offserver_accepted']==64
assert state['hybrid100_fullcoverage_20261010']['new_accepted']==12
assert main['A_three_view_ten_scene_table']['paired_models']==100
accepted=entry['current_accepted']
assert (accepted['native_new'],accepted['three_view_models'],accepted['FL_native_new'],accepted['FL_three_view_total'],accepted['gradient_accepted'],accepted['Hybrid_native_new'])==(350,340,77,81,64,12)
assert (accepted['F_paired_models'],accepted['F_complete_scenes'],accepted['FL_complete_table_records'],accepted['FL_complete_table_scenes'])==(40,4,80,8)
assert accepted['gradient200_new_scientific_accepted']==0 and accepted['by_variant']==main['three_view_counts_by_variant']
assert entry['working_draft_content_cutoff']=={'native':341,'three_view':340,'F_pairs':40,'F_scenes':4,'gradient_search':64}
assert state['latest_five_queue_readonly_observation']==entry['latest_observation_only']
assert entry['latest_observation_only']['new_acceptance']==0 and entry['latest_observation_only']['counts_are_observation_only'] is True
entry_paths={
 'docs/server_deployment_20260923/training_20260923/RUNNING.md',
 'docs/server_deployment_20260923/training_20260923/REBUTTAL_COMPLETION_20261009.md',
 'docs/返修实验总览.md',
 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/MONITOR_HANDOFF.md',
 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/EXECUTION.md'}
assert set(entry['entries'])==entry_paths
for rel,pin in entry['entries'].items():assert H(repo_path(rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for pin in entry['source_pins'].values():assert H(repo_path(pin['path']).read_bytes())==pin['sha256']
selected=selection['files'];parent_refs=selection['parent_references'];late_files=selection['root_late_bound_files']
assert isinstance(selected,dict) and selected and isinstance(parent_refs,dict) and isinstance(late_files,dict)
assert not set(parent_refs)&set(selected)
own_names={'publish_guardfed_increment64_20261011.py','prepare_selection.py','STABLE_INPUTS.json','SELECTION_CANDIDATE.json','SOURCE_CHECK.json','SOURCE_DIFF.patch','INVERSE_EDITS.json','README.md','HANDOFF.json','FILES_SHA256.json','PREPARATION_FAILURE.json'}
own_paths={(HERE/n).relative_to(R).as_posix() for n in own_names}
LATE_ROLES={'entry_writer','entry_update_proof','latest_observation_raw','latest_observation_growth','gradient14_current_metadata'}
late_role_paths={pin['path'] for role,pin in entry['source_pins'].items() if role in LATE_ROLES}
assert {'entry_writer','entry_update_proof'}<=set(entry['source_pins'])
required=set(stable)|entry_paths|own_paths|{state_path.relative_to(R).as_posix()}|late_role_paths
assert required==set(selected)|set(parent_refs)
assert set(late_files)==(entry_paths|own_paths|{state_path.relative_to(R).as_posix()}|late_role_paths)-set(stable)
for rel,pin in stable.items():assert selected.get(rel,parent_refs.get(rel))==pin
for rel,pin in late_files.items():assert selected.get(rel)==pin
# Closed fact roles, current entries and this source must be actual source copies.
assert {p['path'] for p in roles.values()}|entry_paths|own_paths|{state_path.relative_to(R).as_posix(),previous_rel,WORKING_REL}<=set(selected)
for rel,pin in parent_refs.items():
 p=repo_path(rel);assert set(pin)=={'sha256','bytes'} and p.is_file() and not p.is_symlink()
 assert p.stat().st_size==pin['bytes']<2_000_000 and H(p.read_bytes())==pin['sha256']
assert scope_path.relative_to(R).as_posix() not in selected,'Selection cannot pin itself'
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
assert not (T/'publication_closed_increment64_verified_20261011.json').exists(),'One publication attempt only'
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
git('commit','-m','Publish adopted native350 FL81 and gradient64 compact evidence')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=dict(native=350,three_view=340,FLGMM_native_new=77,FLGMM_screen_reuse_separate=4,FLGMM_three_views=81,FLGMM_table_complete_scene_records=80,FLGMM_table_partial_records=1,FLGMM_table_scenes=8,gradient_search=64,gradient200_frozen_new_jobs=192,gradient200_screen_reuse_separate=8,gradient200_new_scientific_accepted=0,Hybrid_native_new=12,UCA_models=300,F_table_pairs=40,F_table_scenes=4,reviewer_comments=24),proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,latest_observation_sha256=entry['latest_observation_only']['sha256'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published=dict(mechanism=0,FLGMM=10,Hybrid=0),new_native_records_published=9,new_gradient_search_records_published=18,working_draft_content_cutoff=entry['working_draft_content_cutoff'],gradient14_evidence_scope='Only explicitly selected compact actual item/preflight/authorization metadata; not192 results or all14 completion',original_Windows47_refit_failure_preserved=True,Windows_whole_refit_pass=False,manuscript_applied=False,final_test=False,whole_rebuttal_complete=False)
(T/'publication_closed_increment64_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
