"""Publish root-closed views340/F40 compact evidence; unified working draft remains333/330."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[2]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='fb1f31031aad89206f252969a6c26af6cde9efa4'
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
scope_path=(R/a.compact_selection).resolve()
assert scope_path==R/'tmp/publication_increment63_F40_source_prepared_20261011/COMPACT_SELECTION.json'
entry=pinned_json(semantic_review,a.semantic_review_sha256)
selection=pinned_json(scope_path,a.compact_selection_sha256)
assert entry['status']=='PASS_LIMITED_CURRENT_ENTRY_SEMANTIC_REVIEW' and entry['root_accepted'] is True
assert selection['status']=='ROOT_CLOSED_INCREMENT63_COMPACT_SELECTION' and selection['root_accepted'] is True
assert selection['parent_commit']==PARENT and selection['semantic_review_sha256']==a.semantic_review_sha256
assert selection['STATE_sha256']==entry['STATE_sha256']
stable_path=R/'tmp/publication_increment63_F40_source_prepared_20261011/STABLE_INPUTS.json'
stable=pinned_json(stable_path,'2f5a0cc1f7798f494a4a7f84a4ed240745c292e8ed2b2bfa75de7742b15e9864')['files']
native_count=selection['current_native_accepted'];native_pin=selection['current_native_root']
assert isinstance(native_count,int) and not isinstance(native_count,bool) and 333<=native_count<=800
assert set(native_pin)=={'path','sha256'} and entry['source_pins']['mechanism_current_native_root']==native_pin
views_pin=selection['current_three_view_root'];f40pin=selection['F40_table_root']
assert set(views_pin)==set(f40pin)=={'path','sha256'}
assert entry['source_pins']['mechanism_current_three_view_root']==views_pin
assert entry['source_pins']['F_four_scene40_table_root']==f40pin
assert native_count>=340
state_path=T/'TRAINING_STATE.json'
state=pinned_json(state_path,entry['STATE_sha256'])
main=state['celeba_mechanism_v1']
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(native_count,340)
assert main['three_view_counts_by_variant']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':40}
assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==67
fl=state['FLGMM_after61_valid_three_view_20261011']
assert (fl['new_three_view_records_accepted'],fl['FLGMM_total_three_view_records'])==(10,71)
ft=fl['seven_scene_table']
assert (ft['complete_scene_records'],ft['retained_partial_records'],ft['scenes'])==(70,1,7)
assert ft['validation_only'] is True and ft['final_test'] is False
hybrid=state['Hybrid_IID_Benign_three_view10_20261011']
assert (hybrid['new_three_view_records_accepted'],hybrid['total_three_view_records'],hybrid['prior_seed91002_reused'])==(9,10,1)
assert hybrid['screen_seed91001_phase_preserved'] is True and hybrid['validation_only'] is True and hybrid['final_test'] is False
assert state['gradient64_validation_search_20261010']['offserver_accepted']==46
assert state['hybrid100_fullcoverage_20261010']['new_accepted']==12
assert main['A_three_view_ten_scene_table']['paired_models']==100
F20_ROOT='9d222cea9e55a00442ff278eb47fa1b622938611049c44d0565644047a8894ac'
assert any(isinstance(v,dict) and v.get('root_proof_sha256')==F20_ROOT and (v.get('paired_models'),v.get('complete_scenes'))==(20,2) for v in main.values())
accepted=entry['current_accepted']
assert (accepted['native_new'],accepted['three_view_models'],accepted['F_paired_models'],accepted['F_complete_scenes'])==(native_count,340,40,4)
assert (accepted['FL_native_new'],accepted['FL_three_view_total'],accepted['FL_complete_table_records'],accepted['gradient_accepted'],accepted['Hybrid_native_new'])==(67,71,70,46,12)
assert (accepted['FL_complete_table_scenes'],accepted['Hybrid_three_view_total'],accepted['Hybrid_new_three_view_records'],accepted['Hybrid_prior_seed91002_reused'])==(7,10,9,1)
assert accepted['by_variant']==main['three_view_counts_by_variant']
FL_ROOT_REL='tmp/fl_FFlip10_capacity_pool32_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
FL_ROOT_SHA='fe400060961fb923cbac79443f735fab6824b06689d45fb3bbf8d56307c95421'
FL_TABLE_REL='outputs/guardfed_tables/celeba_flgmm_seven_scenes70_20261011/ROOT_VERIFICATION.json'
FL_TABLE_SHA='bcf0b16a6838114444080f95b6bd282b16f79f0052bee0598c7c0728e54fc8d8'
assert (fl['root_proof_path'],fl['root_proof_sha256'])==(FL_ROOT_REL,FL_ROOT_SHA)
assert (ft['root_proof_path'],ft['root_proof_sha256'])==(FL_TABLE_REL,FL_TABLE_SHA)
proof_roles={'FLGMM_three_view71_root':(FL_ROOT_REL,FL_ROOT_SHA),'FLGMM_seven_scene70_table_root':(FL_TABLE_REL,FL_TABLE_SHA),'Hybrid_three_view10_root':(hybrid['root_proof_path'],hybrid['root_proof_sha256'])}
for role,(rel,digest) in proof_roles.items():
    assert entry['source_pins'][role]=={'path':rel,'sha256':digest}
    proof=pinned_json(repo_path(rel),digest)
    assert proof['root_adoption'] is True,role
flproof=pinned_json(repo_path(FL_ROOT_REL),FL_ROOT_SHA)
assert (flproof['FLGMM_total_three_view_records'],flproof['new_three_view_records_accepted'])==(71,10)
assert flproof['Linux_root_fit_verified'] is True and flproof['Windows_saved_outputs_audit_pass'] is True
assert flproof['Windows_saved_outputs_audit_fit_calls']==0 and flproof['Windows_whole_saved_check_pass'] is False and flproof['Windows_original47_exact_refit_pass'] is False
ftproof=pinned_json(repo_path(FL_TABLE_REL),FL_TABLE_SHA)
assert (ftproof['complete_scene_records'],ftproof['retained_partial_records'],ftproof['scenes'])==(70,1,7)
assert ftproof['source_root71_sha256']==FL_ROOT_SHA and ftproof['final_test'] is False
# Parent62 acceptance baseline remains unchanged; any later native proof must be actually root-adopted.
NATIVE333_REL='docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T212808Z/ROOT_DELTA_VERIFICATION.json'
NATIVE333_SHA='654ce595db6ab06bf63c1c673c2da48c6e04aaee3c327fa44f4d3c3cac0a1830'
VIEWS330_REL='tmp/celeba_mechanism_remaining_F_FedSA10_root_adoption_20261011/ROOT_ADOPTION.json'
VIEWS330_SHA='62475bb89325d0b229977629bb879d99994c4d313cb7924db9a5efbce6c9ec25'
assert entry['source_pins']['mechanism_native333_root']=={'path':NATIVE333_REL,'sha256':NATIVE333_SHA}
assert entry['source_pins']['mechanism_three_view330_root']=={'path':VIEWS330_REL,'sha256':VIEWS330_SHA}
np=pinned_json(repo_path(NATIVE333_REL),NATIVE333_SHA)
assert np['root_adopted'] is True and np['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS'
assert np['total_new_strict_and_offserver']==333 and np['test'] is False and np['old420_records_exact'] is True
vp=pinned_json(repo_path(VIEWS330_REL),VIEWS330_SHA)
assert vp['status']=='ROOT_F_FEDSA10_EXACT10_SAVED_ARRAYS_REPLAY330_ADOPTED'
assert (vp['new_accepted'],vp['prior_accepted'],vp['cumulative_accepted'])==(10,320,330)
assert vp['native_root_sha256']==NATIVE333_SHA and vp['original320_unchanged'] is True and vp['test'] is False
current_native=pinned_json(repo_path(native_pin['path']),native_pin['sha256'])
assert current_native['root_adopted'] is True and current_native['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS'
assert current_native['total_new_strict_and_offserver']==native_count and current_native['test'] is False
if native_count==333:assert native_pin=={'path':NATIVE333_REL,'sha256':NATIVE333_SHA}
# Current replay340 is external actual evidence; no prepared binding is accepted.
views=pinned_json(repo_path(views_pin['path']),views_pin['sha256'])
assert views['status']=='ROOT_F_SDFA10_EXACT10_SAVED_ARRAYS_REPLAY340_ADOPTED'
assert (views['prior_accepted'],views['new_accepted'],views['cumulative_accepted'])==(330,10,340)
assert views['original330_unchanged'] is True and views['test'] is False
sd_ids=['minus_F_IID_S-DFA_seed'+str(i) for i in range(91001,91011)]
assert views['accepted_new_ids']==sd_ids and views['native_max_abs_difference']==0
assert views['Full_inference']==views['new_CNN']==views['new_fit']==views['new_training']==0
current_index_pin={'path':views['records_index_path'],'sha256':views['records_index_sha256']}
current_index=pinned_json(repo_path(current_index_pin['path']),current_index_pin['sha256'])
prior_index_rel='tmp/celeba_mechanism_remaining_F_FedSA10_root_adoption_20261011/MECHANISM330_INDEX.json'
prior_index_sha='d580cbe700a139cd073b79e7d2defd7e99fcd0d016a53347d459713a5023525b'
prior_index=pinned_json(repo_path(prior_index_rel),prior_index_sha)
assert current_index['prior_index_sha256']==prior_index_sha and current_index['prior_adoption_sha256']==VIEWS330_SHA
assert current_index['new_ids']==sd_ids and current_index['all_ids']==prior_index['all_ids']+sd_ids and len(set(current_index['all_ids']))==340
assert [r['id'] for r in current_index['new_records']]==sd_ids
for key in ('new_bindings','new_binding_files','new_artifacts'):assert set(current_index[key])==set(sd_ids)
bound_native_pin={'path':views['native_root_path'],'sha256':views['native_root_sha256']}
bound_native=pinned_json(repo_path(bound_native_pin['path']),bound_native_pin['sha256'])
assert bound_native['root_adopted'] is True and bound_native['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS'
assert 340<=bound_native['total_new_strict_and_offserver']<=native_count and bound_native['test'] is False
assert bound_native['inspection_sha256']==views['native_inspection_sha256']==current_index['native_inspection_sha256']
assert main['three_view_root_proof_sha256']==views_pin['sha256']
assert vp['accepted_new_ids']==['minus_F_IID_FedSA_seed'+str(i) for i in range(91001,91011)]
# Actual table pins are bound again by the later root semantic review and closed selection.
f30pin=entry['source_pins']['F_three_scene30_table_root']
assert f30pin=={'path':'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_three_scenes30_20261011/ROOT_VERIFICATION.json','sha256':'ea4a3ea8caaa3c05a3bbd857eb2ae660785b6c9177a1b970f0f7ebf7cc3937d5'}
f30proof=pinned_json(repo_path(f30pin['path']),f30pin['sha256'])
assert f30proof['status']=='ROOT_F_IID_THREE_SCENES30_THREE_VIEW_TABLE_ADOPTED'
assert f30proof['root_adoption'] is True and (f30proof['paired_models'],f30proof['complete_scenes'])==(30,3)
assert f30proof['seed_panels']==[10,9,6] and f30proof['source_acceptance_sha256']==VIEWS330_SHA
assert f30proof['validation_only'] is True and f30proof['final_test'] is False and f30proof['whole_rebuttal_complete'] is False
assert f30proof['source_native_root_sha256']==NATIVE333_SHA and f30proof['displayed_records']==60
assert f30proof['old40_records_exact'] is True and f30proof['old324_scalars_exact'] is True and f30proof['old162_cells_exact'] is True
f30state=main['F_three_view_three_scene_table']
assert (f30state['root_proof_path'],f30state['root_proof_sha256'])==(f30pin['path'],f30pin['sha256'])
assert (f30state['paired_models'],f30state['complete_scenes'])==(30,3) and f30state['final_test'] is False
# F30 remains the parent's evidence and working-draft cutoff; F40 is separately adopted.
f40proof=pinned_json(repo_path(f40pin['path']),f40pin['sha256'])
assert f40proof['root_adoption'] is True and (f40proof['paired_models'],f40proof['complete_scenes'])==(40,4)
assert f40proof['displayed_records']==80 and f40proof['seed_panels']==[10,9,6]
assert f40proof['source_acceptance_sha256']==views_pin['sha256'] and f40proof['source_native_root_sha256']==bound_native_pin['sha256']
assert f40proof['validation_only'] is True and f40proof['final_test'] is False and f40proof['whole_rebuttal_complete'] is False
f40state=main['F_three_view_four_scene_table']
assert (f40state['root_proof_path'],f40state['root_proof_sha256'])==(f40pin['path'],f40pin['sha256'])
assert (f40state['paired_models'],f40state['complete_scenes'])==(40,4) and f40state['final_test'] is False
f40_table_rel=repo_path(f40pin['path']).with_name('TABLES.md').relative_to(R).as_posix()
assert H(repo_path(f40_table_rel).read_bytes())==f40proof['files']['TABLES.md']['sha256']
htpin=entry['source_pins']['Hybrid_IID_Benign10_table_root']
htproof=pinned_json(repo_path(htpin['path']),htpin['sha256'])
assert htpin=={'path':'outputs/guardfed_tables/celeba_hybrid_three_view_IID_Benign10_20261011/ROOT_VERIFICATION.json','sha256':'7f64be4721f1eaeeb42881c4633fa03de5f1ae19e32fc01842de4da0867a0b32'}
assert htproof['status']=='ROOT_HYBRID_IID_BENIGN10_THREE_VIEW_TABLE_ADOPTED' and htproof['root_adoption'] is True
assert (htproof['unique_checkpoints'],htproof['complete_scenes'])==(10,1) and htproof['seed_panels']==[10,9,6]
assert htproof['source_three_view_root_sha256']==hybrid['root_proof_sha256']
assert htproof['validation_only'] is True and htproof['final_test'] is False
assert htproof['root_independent_mean_SD_scalars']==54
assert htproof['selected_screen91001_preserved'] is True and htproof['original_Windows_failure_and_diagnostic_difference_preserved'] is True
htstate=hybrid['IID_Benign_table']
assert (htstate['root_proof_path'],htstate['root_proof_sha256'])==(htpin['path'],htpin['sha256'])
assert (htstate['unique_checkpoints'],htstate['complete_scenes'])==(10,1) and htstate['seed_panels']==[10,9,6]
assert htstate['validation_only'] is True and htstate['final_test'] is False
assert (accepted['Hybrid_table_checkpoints'],accepted['Hybrid_table_complete_scenes'])==(10,1)
new_table_paths={repo_path(f30pin['path']).with_name('TABLES.md').relative_to(R).as_posix(),repo_path(htpin['path']).with_name('TABLES.md').relative_to(R).as_posix()}
new_table_paths.add(f40_table_rel)
assert H(repo_path(f30pin['path']).with_name('TABLES.md').read_bytes())==f30proof['files']['TABLES.md']['sha256']
assert H(repo_path(htpin['path']).with_name('TABLES.md').read_bytes())==htproof['files']['TABLES.md']['sha256']
assert state['latest_five_queue_readonly_observation']==entry['latest_observation_only']
assert entry['latest_observation_only']['new_acceptance']==0 and entry['latest_observation_only']['counts_are_observation_only'] is True
EDITORIAL_REL='docs/server_deployment_20260923/revision_20260923/rebuttal_working_20261011.root_editorial_review.json'
EDITORIAL_ROOT='0cf74dba2727092895f6792d0aaea91c27957791aa3ba1bd93be2fc3e943833b'
WORKING_REL='docs/server_deployment_20260923/revision_20260923/rebuttal_working_20261011.md'
editorial=pinned_json(repo_path(EDITORIAL_REL),EDITORIAL_ROOT)
assert entry['source_pins']['unified_working_rebuttal_root']=={'path':EDITORIAL_REL,'sha256':EDITORIAL_ROOT}
assert editorial['status']=='ROOT_UNIFIED_COMPLETE24_F30_HYBRID10_WORKING_REBUTTAL_ADOPTED_FOR_AUTHOR_REVIEW'
assert editorial['root_read_all_changed_passages'] is True and editorial['original_comment_blocks']==24 and editorial['original_quote_lines_and_order_exact'] is True
assert editorial['author_review_only'] is True and editorial['submitted_manuscript_applied'] is False and editorial['final_test'] is False and editorial['whole_rebuttal_complete'] is False
assert editorial['no_new_scientific_acceptance'] is True and editorial['no_statistical_recomputation'] is True
assert editorial['draft_path']==WORKING_REL and H(repo_path(WORKING_REL).read_bytes())==editorial['draft_sha256']
assert editorial['F30_root_sha256']==f30pin['sha256'] and editorial['Hybrid10_root_sha256']==htpin['sha256']
assert (editorial['native_accepted'],editorial['three_view_accepted'])==(333,330)
assert editorial['index_variant_counts']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':30}
previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment62_verified_20261011.json'
previous=pinned_json(repo_path(previous_rel),'ec82d144d9a1a1e185d7d2d264e374539294ca247d50e44d6f971ca546aff919')
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
entry_paths={
 'docs/server_deployment_20260923/training_20260923/RUNNING.md',
 'docs/server_deployment_20260923/training_20260923/REBUTTAL_COMPLETION_20261009.md',
 'docs/返修实验总览.md',
 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/MONITOR_HANDOFF.md',
 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/EXECUTION.md'}
assert set(entry['entries'])==entry_paths
for rel,pin in entry['entries'].items():
    assert H(repo_path(rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for pin in entry['source_pins'].values():
    assert H(repo_path(pin['path']).read_bytes())==pin['sha256']
# Root closes this complete input set only after the actual current entry review.
selected=selection['files'];assert isinstance(selected,dict) and selected
required={prior_index_rel,current_index_pin['path'],bound_native_pin['path']}|entry_paths|new_table_paths|set(stable)|{state_path.relative_to(R).as_posix(),EDITORIAL_REL,previous_rel,'tmp/publication_increment63_F40_source_prepared_20261011/publish_guardfed_increment63_20261011.py','tmp/publication_increment63_F40_source_prepared_20261011/STABLE_INPUTS.json'}|{p['path'] for p in entry['source_pins'].values()}
parent_refs=selection['parent_references'];late_files=selection['root_late_bound_files']
assert isinstance(parent_refs,dict) and isinstance(late_files,dict) and not set(parent_refs)&set(selected)
assert required<=set(selected)|set(parent_refs)
assert set(selected)<=required|set(late_files)
for rel,pin in stable.items():assert selected.get(rel,parent_refs.get(rel))==pin
for rel,pin in late_files.items():assert selected.get(rel)==pin
# References are exact source paths/hash/size, never directory or suffix exemptions.
for rel,pin in parent_refs.items():
    p=repo_path(rel);assert set(pin)=={'sha256','bytes'} and p.is_file() and not p.is_symlink()
    assert p.stat().st_size==pin['bytes']<2_000_000 and H(p.read_bytes())==pin['sha256']
assert state_path.relative_to(R).as_posix() in selected and entry_paths<=set(selected)
assert previous_rel in selected and EDITORIAL_REL in selected and WORKING_REL in selected
required|={prior_index_rel,current_index_pin['path'],bound_native_pin['path']}
assert required<=set(selected)|set(parent_refs)
assert {views_pin['path'],current_index_pin['path'],f40pin['path'],f40_table_rel}<=set(selected)
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
assert not (T/'publication_closed_increment63_verified_20261011.json').exists(),'One publication attempt only'
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
git('commit','-m','Publish adopted views340 F40 evidence and unchanged working-draft cutoff')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=dict(native=native_count,three_view=340,FLGMM_native_new=67,FLGMM_three_views=71,FLGMM_table_complete_scene_records=70,FLGMM_table_scenes=7,Hybrid_three_views=10,Hybrid_table_checkpoints=10,Hybrid_table_scenes=1,Hybrid_new_three_views=9,Hybrid_prior_seed91002_reused=1,gradient_search=46,Hybrid_native_new=12,UCA_models=300,F_table_pairs=40,F_table_scenes=4,reviewer_comments=24),unified_working_editorial_root_sha256=EDITORIAL_ROOT,F20_table_root_sha256=F20_ROOT,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,latest_observation_sha256=entry['latest_observation_only']['sha256'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published=dict(mechanism=10,FLGMM=0,Hybrid=0),new_native_records_published=native_count-333,mechanism_native_root_sha256=native_pin['sha256'],mechanism_three_view_root_sha256=views_pin['sha256'],parent_three_view330_root_sha256=VIEWS330_SHA,F40_table_root_sha256=f40pin['sha256'],working_draft_content_cutoff=dict(native=333,three_view=330,F_pairs=30,F_scenes=3),F30_table_root_sha256=f30pin['sha256'],Hybrid_table_root_sha256=htpin['sha256'],FLGMM_root_sha256=FL_ROOT_SHA,FLGMM_table_root_sha256=FL_TABLE_SHA,Hybrid_root_sha256=hybrid['root_proof_sha256'],manuscript_applied=False,final_test=False,whole_rebuttal_complete=False)
(T/'publication_closed_increment63_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
