"""Publish root-adopted native300/views300 and A100 author-review candidates; compact metadata only."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[1]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='5749c96bb2c9a04904bfb56caeeda45c42fe7233'
BRANCH='codex/revision-evidence-baselines-20260928'
H=lambda b:hashlib.sha256(b).hexdigest()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--semantic-review',type=Path,required=True,help='Actual completed A100 semantic review report under this repository')
parser.add_argument('--semantic-review-sha256',required=True)
parser.add_argument('--entry-check-sha256',required=True,help='SHA of the final root56 entries check after root freezes current entries')
a=parser.parse_args()
semantic_review=(R/a.semantic_review).resolve()
assert semantic_review.relative_to(R).parts[0]=='tmp' and semantic_review.suffix in {'.md','.json'}
assert semantic_review.is_file() and not semantic_review.is_symlink() and semantic_review.stat().st_size<2_000_000
assert re.fullmatch('[0-9a-f]{64}',a.semantic_review_sha256) and H(semantic_review.read_bytes())==a.semantic_review_sha256

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
native=pinned_json(T/'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T173022Z/ROOT_DELTA_VERIFICATION.json','b52484c71ea0e602e13f38f6505bc446d2ca2a80d95a859cf0236b078e087d1d')
assert native['root_adopted'] is True and native['total_new_strict_and_offserver']==300 and native['test'] is False
replay=pinned_json(R/'tmp/celeba_mechanism_remaining_after295_root_adoption_20261011/ROOT_ADOPTION.json','e44122480c85e08461687bd58332aa9f30cc0ed807157966c3fd1769deb74a1e')
assert replay['status']=='ROOT_AFTER295_EXACT5_SAVED_ARRAYS_REPLAY300_ADOPTED'
assert (replay['prior_accepted'],replay['new_accepted'],replay['cumulative_accepted'])==(295,5,300) and replay['original295_unchanged'] is True
assert replay['native_root_sha256']=='b52484c71ea0e602e13f38f6505bc446d2ca2a80d95a859cf0236b078e087d1d' and replay['test'] is False
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
assert state['celeba_mechanism_v1']['scientific_results_offserver_verified']==300
assert state['celeba_mechanism_v1']['three_view_new_models_offserver_verified']==300
assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==59
assert state['gradient64_validation_search_20261010']['offserver_accepted']==42
assert state['hybrid100_fullcoverage_20261010']['new_accepted']==9
assert state['FLGMM_after48_valid_three_view_20261011']['FLGMM_total_three_view_records']==61
assert state['FLGMM_after48_valid_three_view_20261011']['six_scene_table']['complete_scene_records']==60
assert state['celeba_mechanism_v1']['A_three_view_ten_scene_table']['paired_models']==100
assert state['celeba_mechanism_v1']['A_three_view_ten_scene_table']['root_proof_sha256']==TABLE_ROOT
assert state['latest_rebuttal_draft']['A100_incorporated'] and state['latest_rebuttal_draft']['clear_A100_incorporated']
assert state['latest_rebuttal_draft']['root_proof_sha256']==DETAILED_ROOT and state['latest_rebuttal_draft']['clear_reader_root_sha256']==CLEAR_ROOT
entry_check=R/'tmp/entry_increment56_root_20261011/FINAL_ENTRIES_CHECK.json'
assert re.fullmatch('[0-9a-f]{64}',a.entry_check_sha256)
entry=pinned_json(entry_check,a.entry_check_sha256)
assert entry['status']=='ROOT56_CURRENT_ENTRIES_AND_ACTUAL_A100_EDITORIAL_ADOPTIONS_PASS'
assert (entry['native'],entry['three_views'],entry['A_pairs'],entry['A_scenes'])==(300,300,100,10)
assert entry['detailed_A100_root']==DETAILED_ROOT and entry['clear_A100_root']==CLEAR_ROOT
for rel,pin in entry['entries'].items():
    assert H((R/rel).read_bytes())==pin['sha256'] and pin['historical_suffix_unchanged'] is True
for rel,sha in entry['generators_sha256'].items():
    assert H((R/rel).read_bytes())==sha
previous=json.loads((T/'publication_closed_increment55_verified_20261011.json').read_bytes())
assert previous['commit']==PARENT and previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
paths=[T/n for n in ['RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md','publication_closed_increment55_verified_20261011.json']]
paths.append(T/'celeba_mechanism_v1/EXECUTION.md')
paths += [R/'docs/返修实验总览.md',T/'server_reactivation_20261009/MONITOR_HANDOFF.md']
current=(T/'RUNNING.md').read_text(encoding='utf8').split('# HISTORICAL:',1)[0]
details=re.findall(r'detailed_current_[0-9a-f]+\.md',current)
assert len(set(details))==1
paths.append(T/'server_reactivation_20261009/entry_history'/details[0])
paths += [R/'tmp'/n for n in ['update_reactivation_state_20261009.py','update_completion_current_20261009.py','update_overview_closure100_root_20261009.py','publish_guardfed_increment56_20261011.py']]
paths += [semantic_review,R/'tmp/gradient14_execution_dependency_20261011/REVIEW.md',R/'tmp/gradient14_execution_dependency_20261011/SUMMARY_MAPPING.md']
dirs=['tmp/celeba_native_after295_20261011','tmp/celeba_remaining_after295_20261011',
      'tmp/celeba_mechanism_remaining_after295_root_adoption_20261011',
      'tmp/celeba_mechanism_A100_candidate_20261011',
      'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011',
      'tmp/entry_increment56_root_20261011','tmp/entry56_semantic_review_20261011','tmp/live_after_A100_20261011',
      'tmp/rebuttal_A100_increment_20261011',
      'tmp/rebuttal_clear_A100_20261011','tmp/A100_independent_read_review_20261011',
      'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A100_reader_20261011',
      'docs/server_deployment_20260923/revision_20260923/rebuttal_clear_A100_20261011']
# Only explicit scientific metadata records may exceed the normal compact-file limit.
metadata_records={
 'tmp/celeba_mechanism_A100_candidate_20261011/records.json',
 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/records.json'}
compact_limit=lambda p:3_000_000 if p.relative_to(R).as_posix() in metadata_records else 2_000_000
allowed={'.py','.json','.md','.patch','.txt','.stderr','.stdout','.sha256','.csv','.tex','.diff'}
for directory in dirs:
    base=R/directory
    assert base.is_dir(),directory
    for p in base.rglob('*'):
        if not p.is_file() or p.suffix not in allowed:continue
        if p.stat().st_size>=compact_limit(p) or p.name in ['OBSERVATION.stdout','FRESH_OBSERVATION.stdout']:continue
        paths.append(p)
base=T/'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T173022Z'
paths += [p for p in base.rglob('*') if p.is_file() and p.suffix in allowed]
for name in ['root_five_queue_20261010T180131Z.raw.json','root_live_20261010T173354Z.json','latest_formal_live.json',
             'root_five_queue_20261010T172543Z.raw.json','ROOT_FIVE_QUEUE_GROWTH_20261010T1725.json',
             'root_five_queue_20261010T174308Z.raw.json','ROOT_FIVE_QUEUE_GROWTH_20261010T1743.json']:
    paths.append(T/'server_reactivation_20261009'/name)
paths.append(R/'tmp/entry_increment55_postpublication_20261011.json')
paths.append(R/'tmp/adopt_five_queue_observation_1743_root_20261011.py')
# Exact small pre-patch sources retain the current navigation fix's reversible evidence.
paths += [R/'tmp/entry56_semantic_review_20261011'/name for name in
          ['update_completion_current_20261009.py.before','update_overview_closure100_root_20261009.py.before']]
paths=list(dict.fromkeys(paths))
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
git('commit','-m','Accept A100 ten-scene evidence and native300 with author-review limits')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=dict(native=300,three_view=300,FLGMM_native_new=59,FLGMM_three_views=61,FLGMM_table_complete_scene_records=60,FLGMM_table_complete_scenes=6,gradient_search=42,Hybrid_native_new=9,Hybrid_native_complete_scenes=1,A_checkpoint_records=100,A_table_pairs=100,A_table_scenes=10,reviewer_comments=24),A100_detailed_reply_root_sha256=state['latest_rebuttal_draft']['root_proof_sha256'],A100_clear_reply_root_sha256=state['latest_rebuttal_draft']['clear_reader_root_sha256'],semantic_review_sha256=a.semantic_review_sha256,new_terminal_results_adopted=True,final_test=False,whole_rebuttal_complete=False)
(T/'publication_closed_increment56_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
