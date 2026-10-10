"""Publish adopted native320/views320 and two F IID scenes; FL native67, all other cutoffs unchanged; compact metadata only."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[1]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='31b55b6e925af02db5fdb7d8deec4e25ac54307d'
BRANCH='codex/revision-evidence-baselines-20260928'
H=lambda b:hashlib.sha256(b).hexdigest()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--semantic-review',type=Path,required=True,help='Actual root-accepted increment59 entry semantic review JSON under this repository')
parser.add_argument('--semantic-review-sha256',required=True)
parser.add_argument('--F20-root',type=Path,required=True,help='Actual adopted canonical two-scene F20 ROOT_VERIFICATION.json; no prepared proof')
parser.add_argument('--F20-root-sha256',required=True)
a=parser.parse_args()
semantic_review=(R/a.semantic_review).resolve()
assert semantic_review.relative_to(R).parts[0]=='tmp' and semantic_review.suffix=='.json'
assert semantic_review.is_file() and not semantic_review.is_symlink() and semantic_review.stat().st_size<2_000_000
assert re.fullmatch('[0-9a-f]{64}',a.semantic_review_sha256) and H(semantic_review.read_bytes())==a.semantic_review_sha256=='71519a3620bca13467415142ca3d6d575227eee10f10da9ec5a05c7b47c65189'
assert semantic_review.relative_to(R).as_posix()=='tmp/entry_increment59_independent_review_20261011/ROOT_INDEPENDENT_REVIEW.json'
F20_path=(R/a.F20_root).resolve()
assert F20_path==(T/'celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/ROOT_VERIFICATION.json').resolve()
assert re.fullmatch('[0-9a-f]{64}',a.F20_root_sha256) and a.F20_root_sha256=='9d222cea9e55a00442ff278eb47fa1b622938611049c44d0565644047a8894ac'

def pinned_json(path,sha):
    assert path.is_file() and not path.is_symlink() and H(path.read_bytes())==sha,path
    return json.loads(path.read_bytes())

TABLE_ROOT='adbc7f81c5952e85ee1a38ff4b5515d98a8706e4e60acd2e5a259425adcf1237'
DETAILED_ROOT='195eb98c26cc2a4f0c3576cbb7a6b57a73fbc0d4ccc17a683f9817569e164c54'
CLEAR_ROOT='6c071c87b07e8fc6b36f93553875cbce33bf034b8b3d71584d37e6b70535096c'
revision=R/'docs/server_deployment_20260923/revision_20260923'
detailed=pinned_json(revision/'rebuttal_integrated_A100_reader_20261011/ROOT_REVIEW.json',DETAILED_ROOT)
clear=pinned_json(revision/'rebuttal_clear_A100_20261011/ROOT_REVIEW.json',CLEAR_ROOT)
assert detailed['status']=='ROOT_COMPLETE24_A100_READER_REBUTTAL_CANDIDATES_ADOPTED_FOR_AUTHOR_REVIEW' and detailed['root_editorial_review'] is True
assert clear['status']=='ROOT_CLEAR_A100_COMPLETE24_EDITORIAL_CANDIDATE_ADOPTED_FOR_AUTHOR_REVIEW' and clear['root_adoption'] is True
for adopted in (detailed,clear):
    assert adopted['A100_incorporated'] is True and (adopted['original_comments'],adopted['A_complete_scenes'],adopted['A_paired_models'])==(24,10,100)
    assert adopted['A100_table_root_sha256']==TABLE_ROOT and adopted['whole_rebuttal_complete'] is False
assert detailed['author_review_only'] is True and detailed['manuscript_applied'] is False and detailed['final_test'] is False
assert clear['submitted_manuscript_applied'] is False and clear['test'] is False and clear['detailed_A100_root_sha256']==DETAILED_ROOT
for name,sha in detailed['documents_sha256'].items():
    assert H((revision/'rebuttal_integrated_A100_reader_20261011'/name).read_bytes())==sha
assert H((R/clear['entry']).read_bytes())==clear['draft_sha256']
table=pinned_json(T/'celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/ROOT_VERIFICATION.json',TABLE_ROOT)
assert table['root_adoption'] is True and (table['preserved_records'],table['paired_models'],table['complete_scenes'])==(200,100,10)
assert table['test'] is False and table['primary_endpoint_selected'] is False
NATIVE_ROOT='749577f73d151c958f6dc646a1e15714db7c72a8b099a75f22298b58b8c728bb'
REPLAY_ROOT='b427a751a127242d75a6f47426f60067b15345ef47b8704ba000b28925185317'
F20_ROOT=a.F20_root_sha256
native=pinned_json(T/'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T194755Z/ROOT_DELTA_VERIFICATION.json',NATIVE_ROOT)
assert native['root_adopted'] is True and native['total_new_strict_and_offserver']==320 and native['old412_records_exact'] is True and native['test'] is False
assert native['root_checked_archive_members']==94 and native['previous_ledger_sha256']=='9baf997fcb87ec7da604af076945aab5637ef828b6cf8bccb560a4d93c2a26c2'
assert native['ledger_sha256']=='0674614906f4667f03fae0033aa5f6dfdd87b8275db5d38a7dcd90643dc3fb32'
assert native['new_ids']==[f'minus_F_IID_F Flip_seed{s}' for s in range(91003,91011)]
replay=pinned_json(R/'tmp/celeba_mechanism_remaining_F_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json',REPLAY_ROOT)
assert replay['status']=='ROOT_F_FFLIP10_EXACT10_SAVED_ARRAYS_REPLAY320_ADOPTED'
assert (replay['prior_accepted'],replay['new_accepted'],replay['cumulative_accepted'],replay['remaining620_new_accepted'])==(310,10,320,140) and replay['original310_unchanged'] is True
assert replay['native_root_sha256']==NATIVE_ROOT and replay['test'] is False
assert replay['accepted_new_ids']==[f'minus_F_IID_F Flip_seed{s}' for s in range(91001,91011)]
assert (replay['archive_members'],replay['independent_metrics'],replay['independent_counts'],replay['prediction_rules'],replay['native_max_abs_difference'])==(92,90,240,30,0)
assert replay['new_scene_table_created'] is False and replay['Full_inference']==replay['new_CNN']==replay['new_fit']==0
F20=pinned_json(F20_path,F20_ROOT)
assert F20['root_adoption'] is True and F20['status']=='ROOT_F20_TWO_COMPLETE_SCENE_THREE_VIEW_TABLE_ADOPTED'
assert (F20['preserved_records'],F20['paired_models'],F20['complete_scenes'])==(40,20,2) and F20['seed_panels']==[10,9,6]
assert F20['source_acceptance_sha256']==REPLAY_ROOT and F20['source_native_root_sha256']==NATIVE_ROOT
assert F20['actual_root_command_exit']==0 and F20['new_CNN']==F20['new_fits']==F20['new_training']==0 and not F20['test'] and not F20['whole_rebuttal_complete']
f20_names={'records.json','tables.json','TABLES.md','coverage.json','paired_per_seed.json','checks.json','SOURCE_BINDINGS.json','ROOT_BINDING.json','ROOT_SOURCE_REVIEW.json','FILES_SHA256.json','ACTUAL_HANDOFF.json','OUTPUTS_SHA256.json','ROOT_VERIFY_COMMAND.json','ROOT_VERIFY.stdout','ROOT_VERIFY.stderr','ROOT_VERIFY.stdout.json','ROOT_VERIFY.stderr.txt','ROOT_VERIFY_EXIT.json'}
assert set(F20['files_sha256'])<=f20_names and {'records.json','tables.json','TABLES.md','checks.json'}<=set(F20['files_sha256'])
for name,pin in F20['files_sha256'].items():
    assert H((F20_path.parent/name).read_bytes())==pin
checks=json.loads((F20_path.parent/'checks.json').read_bytes())
assert (checks['mean_sd_scalars'],checks['display_mean_sd_cells'],checks['receipt_metrics_from_group_counts'],checks['base_confusion_counts_structurally_checked'],checks['table_record_count'])==(324,162,360,960,40)
assert checks['old20_record_JSON_bytes_and_order_exact'] and checks['old162_scalars_exact'] and checks['old81_cells_preserved'] and checks['native_shared_metrics_and_counts_exact']
assert checks['max_abs_difference']<=1e-12 and checks['partial_pairs_excluded']==0 and checks['new_threshold_fits']==checks['new_CNN']==checks['new_training']==0 and checks['test'] is False
FL=pinned_json(R/'tmp/fl_native_after63_20261011/ROOT_ADOPTION_REVIEW.json','a11cd9ed94ab136b49dd41975d45f684d88236b6918c295461089a98adbc7db0')
assert (FL['accepted_before'],FL['accepted_new'],FL['accepted_total'],FL['reused_separately'])==(63,4,67,4) and FL['old63_ordered_prefix_exact'] is True and FL['final_test'] is False
gradient=pinned_json(R/'tmp/gradient_native_after42_20261011/ROOT_ADOPTION_REVIEW.json','e5875e4d6a5e15c95d7fb94686411a106b1db39681341c92e51e9e821867f1e6')
assert (gradient['accepted_before'],gradient['accepted_new'],gradient['accepted_total'])==(42,4,46)
assert not gradient['screen64_complete'] and not gradient['method_champion_claim'] and not gradient['final_test']
hybrid=pinned_json(R/'tmp/celeba_hybrid_native12_root_adoption_20261011/ROOT_ADOPTION.json','181e4219f967811a25446f682d94f4db511a5d7917b0650aabff380eba4c4ab1')
assert (hybrid['prior_accepted'],hybrid['new_accepted'],hybrid['cumulative_accepted'],hybrid['reused_separate'])==(9,3,12,4)
assert not hybrid['whole100_complete'] and not hybrid['final_test'] and not hybrid['table_adopted']
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
state=json.loads((T/'TRAINING_STATE.json').read_bytes())
assert state['celeba_mechanism_v1']['scientific_results_offserver_verified']==320
assert state['celeba_mechanism_v1']['three_view_new_models_offserver_verified']==320
assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==67
assert state['gradient64_validation_search_20261010']['offserver_accepted']==46
assert state['hybrid100_fullcoverage_20261010']['new_accepted']==12
assert state['FLGMM_after48_valid_three_view_20261011']['FLGMM_total_three_view_records']==61
assert state['FLGMM_after48_valid_three_view_20261011']['six_scene_table']['complete_scene_records']==60
assert state['celeba_mechanism_v1']['A_three_view_ten_scene_table']['paired_models']==100
assert state['celeba_mechanism_v1']['A_three_view_ten_scene_table']['root_proof_sha256']==TABLE_ROOT
assert state['latest_rebuttal_draft']['A100_incorporated'] and state['latest_rebuttal_draft']['clear_A100_incorporated']
assert state['latest_rebuttal_draft']['root_proof_sha256']==DETAILED_ROOT and state['latest_rebuttal_draft']['clear_reader_root_sha256']==CLEAR_ROOT
assert any(isinstance(v,dict) and v.get('root_proof_sha256')==F20_ROOT and (v.get('paired_models'),v.get('complete_scenes'))==(20,2) for v in state['celeba_mechanism_v1'].values())
entry=pinned_json(semantic_review,a.semantic_review_sha256)
assert entry['status']=='PASS_LIMITED_CURRENT_ENTRY_SEMANTIC_REVIEW'
assert H((T/'TRAINING_STATE.json').read_bytes())==entry['STATE_sha256']
accepted=entry['current_accepted']
assert (accepted['native_new'],accepted['three_view_models'],accepted['F_paired_models'],accepted['F_complete_scenes'])==(320,320,20,2)
assert (accepted['FL_native_new'],accepted['FL_three_view_total'],accepted['FL_complete_table_records'],accepted['gradient_accepted'],accepted['Hybrid_native_new'])==(67,61,60,46,12)
assert state['celeba_mechanism_v1']['three_view_counts_by_variant']==accepted['by_variant']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':20}
assert state['latest_five_queue_readonly_observation']==entry['latest_observation_only']
assert entry['latest_observation_only']['sha256']=='78a5ef2b40184bcd652678cb22620dd10811b98c5db53833427b2795373969a5'
assert entry['latest_observation_only']['growth_proof_sha256']=='1ff95525cce03031a5a1d155dcc34319b2a5f601e5d0655331c0ecd8bc686929'
assert (entry['latest_observation_only']['FLGMM_terminal'],entry['latest_observation_only']['gradient_terminal'],entry['latest_observation_only']['Hybrid_terminal'])==(69,53,15)
assert entry['latest_observation_only']['new_acceptance']==0 and entry['latest_observation_only']['counts_are_observation_only'] is True
for rel,pin in entry['entries'].items():
    assert H((R/rel).read_bytes())==pin['sha256'] and pin['history_exact'] is True
for pin in entry['source_pins'].values():
    assert H((R/pin['path']).read_bytes())==pin['sha256']
# Current entry writer is the actual executed969c source, not the historical three generators.
assert H((R/'tmp/update_increment59_entries_20261011.py').read_bytes())=='969c567f5a7277a2da8576db7acb654aaf4292cab4cd7d346dc90576f4344305'
update=pinned_json(R/'tmp/entry_increment59_actual_20261011/UPDATE_PROOF.json','ca824b30d6bcaa307ff567b26fcdfe13cffba4fadf4bdc00b6fd26d9da6a103f')
assert update['status']=='CURRENT_ENTRIES_UPDATED_FROM_ACTUAL320_F20_FL67_PROOFS' and update['STATE_sha256']==entry['STATE_sha256']
assert update['entries']=={rel:{k:v for k,v in pin.items() if k!='prefix_sha256'} for rel,pin in entry['entries'].items()}
actual_exit=pinned_json(R/'tmp/entry_increment59_execution_20261011/EXIT.json','27b45bd8cb2e6679ec05e2c6464d29884687fd7ff013763aee6a7feaa1eba028')
assert actual_exit['exit_code']==0 and actual_exit['source_sha256']=='969c567f5a7277a2da8576db7acb654aaf4292cab4cd7d346dc90576f4344305'
for stream in ('stdout','stderr'):
    assert H((R/'tmp/entry_increment59_execution_20261011'/stream).read_bytes())==actual_exit[stream+'_sha256']
deployment=pinned_json(R/'tmp/fl_three_view_FFlip10_cpu136_20261011/runtime/SOURCE_DEPLOYMENT.json','1070c5f7640477c55faa247c352612b22d76ad1c34b4043d7ae08dbedbf13f85')
assert deployment['status']=='ROOT_SMALL_SOURCE_DEPLOYED_BYTES_VERIFIED_NOT_STARTED' and deployment['new_inference']==deployment['new_fit']==deployment['new_training']==0
refusal=pinned_json(R/'tmp/fl_three_view_FFlip10_cpu136_20261011/runtime/LAUNCH_REFUSAL.json','a974849e5b617dbdf708a1ebc60ccc9b6328b7d5e99ed6fbcdde6fcac5ec3f0a')
assert refusal['status']=='CPU136_LAUNCH_REFUSED_ACTIVE_WIDE_THREAD_NO_RETRY' and refusal['no_new_CNN_fit_training'] and not refusal['retry_authorized'] and not any(refusal['gate_status_files_present'].values())
previous=pinned_json(T/'publication_closed_increment58_verified_20261011.json','efa2025fb3e499f1ee7241878ebd37dbe091410e56bf440e8d3ae519092c8815')
assert previous['commit']==PARENT and previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
paths=[T/n for n in ['RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md','publication_closed_increment58_verified_20261011.json']]
paths.append(T/'celeba_mechanism_v1/EXECUTION.md')
paths += [R/'docs/返修实验总览.md',T/'server_reactivation_20261009/MONITOR_HANDOFF.md']
paths += [R/'tmp/update_increment59_entries_20261011.py',R/'tmp/publish_guardfed_increment59_20261011.py']
paths += [semantic_review,F20_path]
scope_path=R/'tmp/publication_increment59_source_prepared_20261011/COMPACT_SELECTION.json'
selection=pinned_json(scope_path,'7868b3bcc811d1ca73064be8fd358a0544549bbb849a71994a2e3556949b122e')
for rel,pin in selection['files'].items():
    p=R/rel
    assert p.is_file() and H(p.read_bytes())==pin['sha256'] and p.stat().st_size==pin['bytes'],rel
    paths.append(p)
# Actual semantic source pins must be members of this closed selection or the exact F20 root; no arbitrary extra paths.
for pin in entry['source_pins'].values():
    assert pin['path'] in selection['files'] or pin['path']==F20_path.relative_to(R).as_posix()
    paths.append(R/pin['path'])
# Publish one canonical F20 output copy, exactly the root-pinned names; no candidate output duplicates.
paths += [F20_path.parent/name for name in F20['files_sha256']]
source_dir=R/'tmp/publication_increment59_source_prepared_20261011'
paths += [source_dir/name for name in ['COMPACT_SELECTION.json','SOURCE_DIFF.patch','SOURCE_CHECK.json','INVERSE_EDITS.json','README.md','HANDOFF.json','FILES_SHA256.json']]
compact_limit=lambda p:2_000_000
allowed={'.py','.json','.md','.patch','.txt','.stderr','.stdout','.sha256','.csv','.tex','.diff'}
paths=list(dict.fromkeys(paths))
for p in paths:
    assert p.is_file() and p.suffix in allowed and p.resolve().is_relative_to(R.resolve())
assert sum(p.stat().st_size for p in paths)<100_000_000
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
git('commit','-m','Accept native320 replay320 two paired F scenes and FL67 native')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=dict(native=320,three_view=320,FLGMM_native_new=67,FLGMM_three_views=61,FLGMM_table_complete_scene_records=60,FLGMM_table_complete_scenes=6,gradient_search=46,Hybrid_native_new=12,Hybrid_native_complete_scenes=1,A_checkpoint_records=100,A_table_pairs=100,A_table_scenes=10,F_checkpoint_records=20,F_table_pairs=20,F_table_scenes=2,F_table_records=40,reviewer_comments=24),A100_detailed_reply_root_sha256=state['latest_rebuttal_draft']['root_proof_sha256'],A100_clear_reply_root_sha256=state['latest_rebuttal_draft']['clear_reader_root_sha256'],F20_table_root_sha256=F20_ROOT,native320_root_sha256=NATIVE_ROOT,replay320_root_sha256=REPLAY_ROOT,semantic_review_sha256=a.semantic_review_sha256,latest_observation_sha256=entry['latest_observation_only']['sha256'],new_terminal_results_adopted=True,final_test=False,whole_rebuttal_complete=False)
(T/'publication_closed_increment59_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
