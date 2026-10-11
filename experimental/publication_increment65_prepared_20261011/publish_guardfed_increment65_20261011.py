"""Publish closed native360/F60/FL81/gradient64 compact evidence; no192 result claim."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[2]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='3edbd954d91bbc5e4993546950202645d82ccbb0'
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
assert selection['status']=='ROOT_CLOSED_INCREMENT65_COMPACT_SELECTION' and selection['root_accepted'] is True
assert selection['parent_commit']==PARENT and selection['semantic_review_sha256']==a.semantic_review_sha256
stable_input=pinned_json(HERE/'STABLE_INPUTS.json','25eae03a22761c930e9feed8eb480595c1c50784515850c9fa91da656058c143')
assert stable_input['parent_commit']==PARENT
assert selection['STATE_sha256']==entry['STATE_sha256']==stable_input['STATE_sha256']
roles=stable_input['proof_roles'];assert selection['proof_roles']==roles
for role,pin in roles.items():assert entry['source_pins'][role]==pin
proofs={role:pinned_json(repo_path(pin['path']),pin['sha256']) for role,pin in roles.items() if Path(pin['path']).suffix=='.json'}
previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment64_verified_20261011.json'
previous=pinned_json(repo_path(previous_rel),'ceb120bf2ebc7fe8d635c2fd554f1c69787f70dc0e112398600ca265cdc9add2')
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
assert (previous['acceptance_cutoff']['native'],previous['acceptance_cutoff']['three_view'],previous['acceptance_cutoff']['F_table_pairs'])==(350,350,50)
ids=['minus_F_non-IID_Benign_seed'+str(s) for s in range(91001,91011)]
native=proofs['native360_root'];views=proofs['current_three_view_root'];table=proofs['F60_table_root']
assert native['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and native['root_adopted'] is True
assert native['total_new_strict_and_offserver']==360 and native['new_ids']==ids and native['old450_records_exact'] is True and native['test'] is False
assert views['status']=='ROOT_F_NONIID_BENIGN10_EXACT10_SAVED_ARRAYS_REPLAY360_ADOPTED'
assert (views['prior_accepted'],views['new_accepted'],views['cumulative_accepted'],views['remaining620_new_accepted'])==(350,10,360,180)
assert views['original350_unchanged'] is True and views['accepted_new_ids']==ids and views['test'] is False
assert views['native_root_sha256']==roles['native360_root']['sha256'] and views['native_max_abs_difference']==0
assert views['Full_inference']==views['new_CNN']==views['new_fit']==views['new_training']==0
assert table['status']=='ROOT_F60_SIX_SCENES_COUNTS_PAIRED_STATISTICS_DISPLAY_AND_PRIOR_ROWS_PASS'
assert (table['paired_models'],table['complete_scenes'],table['records'])==(60,6,120)
assert table['old100_record_objects_order_exact'] is True and table['old810_statistical_scalars_exact'] is True and table['old405_display_cells_preserved'] is True
assert table['actual_views360_root_sha256']==roles['current_three_view_root']['sha256'] and table['actual_views360_index_sha256']==views['records_index_sha256']
assert table['seed_panels']==[10,9,6] and table['sampleSD_ddof']==1 and table['root_executed_saved_verifier_once'] is True
assert table['final_test'] is False and table['primary_endpoint_selected'] is False
assert (table['original_fsum_verification']['mean_sd_scalars'],table['original_fsum_verification']['display_cells'])==(972,486)
flnative=proofs['FL_native77_root'];flproof=proofs['FL_three_view81_root'];ftproof=proofs['FL_eight_scene80_table_root']
assert (flnative['accepted_total'],flnative['reused_separately'])==(77,4) and flnative['final_test'] is False
assert flproof['root_adoption'] is True and flproof['FLGMM_total_three_view_records']==81
assert flproof['native_root_sha256']==roles['FL_native77_root']['sha256']
assert flproof['Linux_root_fit_verified'] is True and flproof['Windows_saved_outputs_audit_pass'] is True
assert flproof['Windows_saved_outputs_audit_fit_calls']==0 and flproof['Windows_whole_saved_check_pass'] is False and flproof['Windows_original47_exact_refit_pass'] is False
assert ftproof['root_adoption'] is True and (ftproof['complete_scene_records'],ftproof['retained_partial_records'],ftproof['scenes'])==(80,1,8)
assert ftproof['source_root81_sha256']==roles['FL_three_view81_root']['sha256'] and ftproof['final_test'] is False and ftproof['prior_windows_failure_preserved'] is True
grad=proofs['gradient64_root'];chosen=proofs['gradient64_selected_root'];frozen=proofs['gradient200_frozen_source_root']
assert grad['accepted_total']==64 and grad['screen64_complete'] is True and grad['all_negative_results_retained'] is True and grad['final_test'] is False
assert chosen['accepted_count']==64 and chosen['all64_native_root_sha256']==roles['gradient64_root']['sha256'] and chosen['final_test'] is False
assert frozen['root_selected64_sha256']==roles['gradient64_selected_root']['sha256'] and (frozen['new_frozen_jobs'],frozen['reused_original_screen_records'])==(192,8)
assert frozen['workers_started']==0 and frozen['test'] is False
editorial=proofs['current_working_editorial_root'];prior_editorial=proofs['working_FL81_F50_editorial_root']
assert editorial['status']=='ROOT_UNIFIED_COMPLETE24_F60_FL81_GRADIENT64_WORKING_REBUTTAL_ADOPTED_FOR_AUTHOR_REVIEW'
assert editorial['parent_sha256']==prior_editorial['draft_sha256']
assert prior_editorial['parent_sha256']==proofs['working_F50_editorial_root']['draft_sha256']
assert editorial['quote_blocks']==24 and editorial['quote_lines']==38 and editorial['headings_order_exact'] is True
assert editorial['eighteen_mean_SD_pairs_from_saved_JSON_exact'] is True and editorial['nine_panel_directions_retained'] is True
assert editorial['final_test'] is False and editorial['submitted_manuscript_edited'] is False and editorial['whole_rebuttal_complete'] is False
for ed in (editorial,prior_editorial):assert H(repo_path(ed['draft_path']).read_bytes())==ed['draft_sha256']
state_path=T/'TRAINING_STATE.json';state=pinned_json(state_path,entry['STATE_sha256']);main=state['celeba_mechanism_v1']
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(360,360)
assert main['three_view_counts_by_variant']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':60}
assert main['latest_native_root_proof_sha256']==roles['native360_root']['sha256'] and main['three_view_root_proof_sha256']==roles['current_three_view_root']['sha256']
assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==77 and state['gradient64_validation_search_20261010']['offserver_accepted']==64
assert state['hybrid100_fullcoverage_20261010']['new_accepted']==12 and state['mechanism_remaining620_valid_20261010']['new_offserver_accepted']==180
assert state['latest_working_rebuttal_draft']['sha256']==editorial['draft_sha256']
alias=proofs['editorial_alias_reconciliation']
assert alias['STATE_sha256']==entry['STATE_sha256'] and alias['current_full_draft_sha256']==editorial['draft_sha256']
assert alias['all_scientific_stage_objects_unchanged'] is True and alias['five_entry_bytes_unchanged'] is True
assert state['latest_rebuttal_draft']['sha256']==editorial['draft_sha256']
assert H(repo_path(roles['entry_writer']['path']).read_bytes())==state['current_entry_writer']['sha256']
accepted=entry['current_accepted']
assert accepted=={'native_new':360,'three_view_models':360,'F_paired_models':60,'F_complete_scenes':6,'FL_native_new':77,'FL_screen_reuse_separate':4,'FL_three_view_total':81,'FL_complete_table_records':80,'FL_complete_table_scenes':8,'gradient_accepted':64,'Hybrid_native_new':12,'Hybrid_three_view_total':10,'gradient200_new_scientific_accepted':0,'by_variant':{'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':60}}
assert entry['working_draft_content_cutoff']==editorial['content_cutoff']
assert entry['historical_Git64_cutoff']==previous['acceptance_cutoff']
assert entry['latest_observation_only']==state['latest_five_queue_readonly_observation']
assert entry['latest_observation_only']['new_acceptance']==0 and entry['latest_observation_only']['counts_are_observation_only'] is True
gates=proofs['gradient14_current_metadata']
assert gates['completed']==7 and gates['total']==14 and len(gates['items'])==7
assert gates['whole_comparison'] is False and gates['offserver_whole14'] is False and gates['coverage192_started'] is False and gates['scientific_table_records']==0
assert state['gradient200_fullcoverage_source_preparation_20261010']['actual_jobs']==0
assert state['gradient200_fullcoverage_source_preparation_20261010']['exact14_completed_at_cutoff']==7
assert entry['gradient14_cutoff']=={'individual_CPU3_passes':7,'whole14_accepted':False,'coverage192_scientific_records':0}
for item in gates['items']:
    p=pinned_json(repo_path(item['path']),item['sha256'])
    assert p['acceptance']['status']=='PASS'
entry_paths={x['path'] for x in proofs['entry_update_proof']['entries']}
assert set(entry['entries'])==entry_paths and len(entry_paths)==5
for rel,pin in entry['entries'].items():assert H(repo_path(rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for pin in entry['source_pins'].values():assert H(repo_path(pin['path']).read_bytes())==pin['sha256']
selected=selection['files'];parent_refs=selection['parent_references'];late_files=selection['root_late_bound_files']
assert parent_refs==stable_input['parent_references'] and not set(parent_refs)&set(selected)
own_names={'publish_guardfed_increment65_20261011.py','prepare_source.py','derive_publisher.py','STABLE_INPUTS.json','SELECTION_CANDIDATE.json','SEMANTIC_CANDIDATE.json','SOURCE_CHECK.json','SOURCE_DIFF.patch','INVERSE_DIFF.json','README.md','HANDOFF.json','FILES_SHA256.json','PREPARATION_FAILURE.json','prepare_source_initial_STATE7ec.py'}
own_paths={(HERE/n).relative_to(R).as_posix() for n in own_names}
assert set(selected)==set(stable_input['files'])|own_paths
assert set(late_files)==own_paths
for rel,pin in stable_input['files'].items():assert selected[rel]==pin
for rel,pin in late_files.items():assert selected[rel]==pin
for rel,pin in parent_refs.items():
    p=repo_path(rel);assert set(pin)=={'sha256','bytes'} and p.is_file() and not p.is_symlink()
    assert p.stat().st_size==pin['bytes']<2_000_000 and H(p.read_bytes())==pin['sha256'],rel
    dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
    old=previous['files'][dest]
    assert (old['sha256'],old['bytes'])==(pin['sha256'],pin['bytes']),rel
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
assert not (T/'publication_closed_increment65_verified_20261011.json').exists(),'One publication attempt only'
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
git('commit','-m','Publish adopted native360 F60 and FL81 editorial evidence')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=dict(native=360,three_view=360,FLGMM_native_new=77,FLGMM_screen_reuse_separate=4,FLGMM_three_views=81,FLGMM_table_complete_scene_records=80,FLGMM_table_partial_records=1,FLGMM_table_scenes=8,gradient_search=64,gradient200_frozen_new_jobs=192,gradient200_screen_reuse_separate=8,gradient200_new_scientific_accepted=0,Hybrid_native_new=12,Hybrid_three_views=10,UCA_models=300,F_table_pairs=60,F_table_scenes=6,reviewer_comments=24),proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,latest_observation_sha256=entry['latest_observation_only']['sha256'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published=dict(mechanism=10,FLGMM=0,Hybrid=0),new_native_records_published=10,new_gradient_search_records_published=0,working_draft_content_cutoff=entry['working_draft_content_cutoff'],historical_Git64_cutoff=entry['historical_Git64_cutoff'],parent_references=parent_refs,gradient14_evidence_scope='Exactly7 original individual CPU3 item proofs and cutoff metadata; no whole14 acceptance or192 scientific results',original_Windows47_refit_failure_preserved=True,Windows_whole_refit_pass=False,manuscript_applied=False,final_test=False,whole_rebuttal_complete=False)
(T/'publication_closed_increment65_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
