"""Publish adopted native400/native11 compact evidence with preserved held replay failure and fixed full24 draft."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[2]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='f252db2c041379dc9c49d6e56db559d0a2943985'
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
assert selection['status']=='ROOT_CLOSED_INCREMENT69_COMPACT_SELECTION' and selection['root_accepted'] is True
assert selection['parent_commit']==PARENT and selection['semantic_review_sha256']==a.semantic_review_sha256
template=pinned_json(HERE/'INPUTS_TEMPLATE.json','f735d4edfee9a3f0944bbc2f41553dbea2ba1e9bfabfd87573787c95751e43fb')
stable_input=pinned_json(HERE/'STABLE_INPUTS.json',selection['stable_input_sha256'])
assert stable_input['status']=='ROOT_BOUND_INCREMENT69_INPUTS' and stable_input['root_accepted'] is True
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
previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment68_verified_20261011.json'
previous=pinned_json(repo_path(previous_rel),'cc07ed80bf338e062a7ced2e5e8b83502b91abaa5fd9fc29263837dcfb6cc964')
assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT
assert (previous['acceptance_cutoff']['native'],previous['acceptance_cutoff']['three_view'],previous['acceptance_cutoff']['F_table_pairs'])==(380,380,80)
native=proofs['native400_root'];fl=proofs['FL96_native_root'];native11=proofs['native11_table_root'];pdf=proofs['native11_PDF_root'];nga=proofs['NGA_first1_root']
ids=['minus_F_non-IID_'+attack+'_seed'+str(seed) for attack in ['S-DFA','Sp-DFA'] for seed in range(91001,91011)]
assert native['root_adopted'] is True and native['total_new_strict_and_offserver']==400 and native['new_ids']==ids
assert native['prior_root_sha256']==roles['native380_parent_root']['sha256'] and native['old480_records_exact'] is True
assert native['canonical_local_and_remote_ledger_promoted'] is True and native['root_checked_archive_members']==190 and native['test'] is False
assert (fl['accepted_before'],fl['accepted_new'],fl['accepted_total'],fl['reused_separately'])==(86,10,96,4)
assert fl['native_coverage_complete'] is True and fl['old86_ordered_prefix_exact'] is True and fl['negative_results_preserved'] is True
assert (native11['methods'],native11['records'],native11['distributions'],native11['scenes_per_distribution'],native11['seed_counts'])==(11,1100,2,5,[10,9,6])
assert native11['original1000_objects_and_order_exact'] is True and native11['old900_metric_objects_exact'] is True
assert native11['native96_root']['sha256']==roles['FL96_native_root']['sha256']
assert native11['final_test'] is False and native11['full17_complete'] is False and native11['primary_endpoint_selected'] is False
assert pdf['root_adopted'] is True and pdf['source_root_sha256']==roles['native11_table_root']['sha256']
assert (pdf['pages'],pdf['mean_SD_cells'],pdf['IEEE_manuscript_fit_verified'])==(3,990,False)
assert pdf['pdf']['path']==template['native11_PDF_path'] and H(repo_path(pdf['pdf']['path']).read_bytes())==pdf['pdf']['sha256']
assert nga['root_adopted'] is True and nga['accepted_total']==1 and nga['accepted_ids']==['FedNGA_eta0.001_IID_Benign_seed91002_coverage']
assert nga['constant_predictor_not_fairness_success'] is True and nga['all_negative_results_retained'] is True
assert nga['coverage100_completed_claim'] is False and nga['test'] is False
editorial=proofs['current_working_editorial_root']
assert editorial['status']=='ROOT_COMPLETE24_NATIVE11_INCREMENT_AND_CURRENT_PDF_LINK_ADOPTED'
assert editorial['adopted_draft']['path']==template['draft_path'] and editorial['adopted_draft']['sha256']==stable_input['draft_sha256']
assert (editorial['comment_blocks'],editorial['reviewer_quote_lines'])==(24,38)
assert editorial['quotes_and_headers_exact'] is True and editorial['full_parent_inverse_exact'] is True and editorial['statistics_scope_correction_retained'] is True
assert editorial['native11_root']['sha256']==roles['native11_table_root']['sha256'] and editorial['PDF_root']['sha256']==roles['native11_PDF_root']['sha256']
assert editorial['fixed_other_cutoff']=={'mechanism_native':380,'mechanism_views':380,'F_pairs':80,'Hybrid':20,'gradient_screen':64,'NGA_coverage_accepted':0}
assert editorial['new_statistics']==editorial['new_CNN']==editorial['new_fit']==0 and editorial['final_test'] is False and editorial['whole_rebuttal_complete'] is False
assert H(repo_path(template['draft_path']).read_bytes())==editorial['adopted_draft']['sha256']
scope=proofs['statistical_scope_correction_root']
assert scope['complete_raw_inverse_exact'] is True and scope['all24_comment_blocks_unchanged'] is True and scope['scientific_numbers_changed']==0
assert scope['candidate']['path']==editorial['parent']['path'] and scope['candidate']['sha256']==editorial['parent']['sha256']
metadata=proofs['entry_update_proof']
assert metadata['status']=='ROOT_NATIVE400_NATIVE11_AND_UNRESOLVED_REPLAY_FAILURE_RECORDED'
assert (metadata['native'],metadata['saved'],metadata['FL_native'],metadata['FL_saved'],metadata['main_methods'])==(400,380,100,90,11)
assert metadata['old_historical_tails_exact'] is True and metadata['new_training_or_inference']==0
state_path=T/'TRAINING_STATE.json';state=pinned_json(state_path,entry['STATE_sha256']);main=state['celeba_mechanism_v1']
assert entry['STATE_sha256']==metadata['state_after']['sha256']=='ed689b5750d41c8954b36c2672b141b67d40ced750231eebd99fd0ce16a982f9'
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(400,380)
assert main['three_view_counts_by_variant']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':80}
assert main['latest_native_root_proof_sha256']==roles['native400_root']['sha256']
assert main['three_view_root_proof_sha256']==previous['proof_roles']['current_three_view_root']['sha256']
assert state['latest_working_rebuttal_draft']['sha256']==state['latest_rebuttal_draft']['sha256']==editorial['adopted_draft']['sha256']
assert H(repo_path(roles['entry_writer']['path']).read_bytes())==state['current_entry_writer']['sha256']==metadata['source']['sha256']
assert entry['current_accepted']==template['current_accepted']
assert entry['working_draft_content_cutoff']==state['latest_working_rebuttal_draft']['content_cutoff']
assert entry['working_draft_content_cutoff']['mechanism_native']==entry['working_draft_content_cutoff']['mechanism_three_view']==380
assert entry['working_draft_content_cutoff']['native_methods']==11 and entry['working_draft_content_cutoff']['native_evaluation_records']==1100
assert entry['historical_Git68_cutoff']==previous['acceptance_cutoff']
assert entry['latest_observation_only']==state['latest_five_queue_readonly_observation']
assert entry['latest_observation_only']['new_scientific_acceptance']==0 and entry['latest_observation_only']['counts_are_observation_only'] is True
assert entry['gradient14_deployment_only'] is True and entry['NGA96_scientific_completed']==1 and entry['F80_root_accepted'] is True
failure=state['mechanism_replay_failure_20261011']
assert failure['identity']=='minus_F_non-IID_Sp-DFA_seed91004' and failure['status']=='NUMERICAL_MISMATCH_DIAGNOSIS_PENDING_NOT_EXTERNAL_INTERRUPTION'
assert failure['evidence']['sha256']==roles['remaining213_numerical_failure']['sha256']
assert failure['comparison']['accepted'] is False and failure['tolerance_changed'] is False and failure['known_cause'] is False
assert failure['existing_saved380_unchanged'] is True and failure['native400_still_valid'] is True
assert failure['FL_final10_launch_refused'] is True and failure['FL_final10_started'] is False and entry['replay_failure_preserved'] is True
assert state['NGA96_coverage_acceptance']['root_proof']['sha256']==roles['NGA_first1_root']['sha256']
assert state['NGA96_coverage_acceptance']['new_accepted']==1 and state['NGA96_coverage_acceptance']['complete_method100_table'] is False
entry_paths={x['path'] for x in metadata['five_entries']}
assert set(entry['entries'])==entry_paths==set(template['entry_paths']) and len(entry_paths)==5
for rel,pin in entry['entries'].items():assert H(repo_path(rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for pin in entry['source_pins'].values():assert H(repo_path(pin['path']).read_bytes())==pin['sha256']
selected=selection['files'];parent_refs=selection['parent_references'];late_files=selection['root_late_bound_files']
assert not set(template['omitted_large_artifacts'])&set(selected)
for rel,pin in template['omitted_large_artifacts'].items():
    assert rel not in selected and re.fullmatch('[0-9a-f]{64}',pin['sha256']) and pin['bytes']>2_000_000
assert template['omitted_large_artifacts']["outputs/guardfed_tables/celeba_eleven_method_native_20261011/records_native_1100.json"]['sha256']==native11['files_sha256']['records_native_1100.json']
assert parent_refs==stable_input['parent_references'] and not set(parent_refs)&set(selected)
own_names={'fact_guards.py.txt', 'README.md', 'STABLE_INPUTS.json', 'INVERSE_DIFF.json', 'HANDOFF.json', 'OMISSIONS.json', 'SELECTION_TEMPLATE.json', 'prepare_source.py', 'receipt_footer.py.txt', 'SEMANTIC_TEMPLATE.json', 'SOURCE_DIFF.patch', 'INPUTS_TEMPLATE.json', 'FILES_SHA256.json', 'SOURCE_CHECK.json', 'publish_guardfed_increment69_20261011.py'}
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
assert not (T/'publication_closed_increment69_verified_20261011.json').exists(),'One publication attempt only'
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
git('commit','-m','Publish native400 native11 and full24 author-review evidence; preserve held replay failure')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
cutoff=dict(previous['acceptance_cutoff'],native=400,FLGMM_native_new=96,NGA96_completed_scientific_records=1,gradient200_new_scientific_accepted=1,native_methods=11,native_evaluation_records=1100)
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=cutoff,proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,latest_observation_sha256=entry['latest_observation_only']['snapshot']['sha256'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published=dict(mechanism=0,FLGMM=0,Hybrid=0),new_native_records_published=31,new_native_records_by_queue=dict(mechanism=20,FLGMM=10,NGA=1,Hybrid=0),working_draft_content_cutoff=entry['working_draft_content_cutoff'],historical_Git68_cutoff=entry['historical_Git68_cutoff'],parent_references=parent_refs,omitted_large_artifacts=template['omitted_large_artifacts'],scientific_scope='Native400 accepted; mechanism saved views remain380/F80 after unresolved numerical native mismatch at non-IID Sp-DFA seed91004. FLnative96+4 is complete100 while FLthreeviews remain90. Native11 table contains1100 records; full24 working draft keeps fixed mechanism380/F80 and NGA0 prose cutoff. NGA first1 accepted coverage is reported separately, with its constant-negative outcome preserved. Gate14 is deployment-only, FL final10 has not started; no replay recovery or final evaluation claimed.',original_Windows47_refit_failure_preserved=True,Windows_whole_refit_pass=False,remaining213_numerical_failure_preserved=True,replay_recovery_claimed=False,FL_final10_started=False,manuscript_applied=False,primary_endpoint_selected=False,final_test=False,whole_rebuttal_complete=False)
(T/'publication_closed_increment69_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
