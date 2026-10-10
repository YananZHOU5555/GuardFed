"""Publish accepted native295/views295, A90 and baseline increments; no weights or arrays."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re
R=Path(__file__).resolve().parents[1]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='0f67597e34f3b3454bbc1600708c649904976d7a'
BRANCH='codex/revision-evidence-baselines-20260928'
H=lambda b:hashlib.sha256(b).hexdigest()
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
assert state['celeba_mechanism_v1']['scientific_results_offserver_verified']==295
assert state['celeba_mechanism_v1']['three_view_new_models_offserver_verified']==295
assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==59
assert state['gradient64_validation_search_20261010']['offserver_accepted']==42
assert state['hybrid100_fullcoverage_20261010']['new_accepted']==9
assert state['FLGMM_after48_valid_three_view_20261011']['FLGMM_total_three_view_records']==61
assert state['FLGMM_after48_valid_three_view_20261011']['six_scene_table']['complete_scene_records']==60
assert state['celeba_mechanism_v1']['A_three_view_nine_scene_table']['paired_models']==90
assert state['celeba_mechanism_v1']['A_three_view_nine_scene_table']['root_proof_sha256']=='445904a761cad8de89b57a9e0dd65fab298dbd097d7a0bd5f3d75458a6f65cdd'
assert state['latest_rebuttal_draft']['A90_incorporated'] and state['latest_rebuttal_draft']['clear_A90_incorporated']
assert state['latest_rebuttal_draft']['root_proof_sha256']=='17ad95f3c558c78b5c9cadd495c3d7802389ecf780840680fc09e958ffe05d11'
entry_check=R/'tmp/entry_increment55_root_20261011/FINAL_ENTRIES_CHECK.json'
assert H(entry_check.read_bytes())=='1241ec268628c9863764ee56f3b7167cb6743d954b8d028294ba92111f062c55'
assert json.loads(entry_check.read_bytes())['status']=='ROOT55_CURRENT_ENTRIES_AND_ACTUAL_A90_EDITORIAL_ADOPTIONS_PASS'
paths=[T/n for n in ['RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md','publication_closed_increment54_verified_20261011.json']]
paths.append(T/'celeba_mechanism_v1/EXECUTION.md')
paths += [R/'docs/返修实验总览.md',T/'server_reactivation_20261009/MONITOR_HANDOFF.md']
current=(T/'RUNNING.md').read_text(encoding='utf8').split('# HISTORICAL:',1)[0]
details=re.findall(r'detailed_current_[0-9a-f]+\.md',current)
assert len(set(details))==1
paths.append(T/'server_reactivation_20261009/entry_history'/details[0])
paths += [R/'tmp'/n for n in ['update_reactivation_state_20261009.py','update_completion_current_20261009.py','update_overview_closure100_root_20261009.py','publish_guardfed_increment55_20261011.py']]
dirs=['tmp/celeba_native_after288_20261011','tmp/fl_native_after57_20261011',
      'tmp/gradient_native_after39_20261011','tmp/celeba_remaining_after288_20261011',
      'tmp/celeba_mechanism_remaining_after288_root_adoption_20261011',
      'tmp/celeba_mechanism_A90_candidate_20261011',
      'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_nine_scenes90_20261011',
      'tmp/entry_increment55_root_20261011','tmp/entry55_independent_review_20261011',
      'tmp/rebuttal_A90_increment_20261011','tmp/rebuttal_clear_A90_20261011',
      'tmp/clear_A90_independent_review_20261011',
      'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A90_reader_20261011',
      'docs/server_deployment_20260923/revision_20260923/rebuttal_clear_A90_20261011']
# Only explicit scientific metadata records may exceed the normal compact-file limit.
metadata_records={
 'tmp/celeba_mechanism_A90_candidate_20261011/records.json',
 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_nine_scenes90_20261011/records.json'}
compact_limit=lambda p:3_000_000 if p.relative_to(R).as_posix() in metadata_records else 2_000_000
allowed={'.py','.json','.md','.patch','.txt','.log','.stderr','.stdout','.sha256','.csv','.tex','.diff'}
for directory in dirs:
    base=R/directory
    assert base.is_dir(),directory
    for p in base.rglob('*'):
        if not p.is_file() or p.suffix not in allowed:continue
        if p.stat().st_size>=compact_limit(p) or p.name in ['OBSERVATION.stdout','FRESH_OBSERVATION.stdout']:continue
        paths.append(p)
base=T/'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T164936Z'
paths += [p for p in base.rglob('*') if p.is_file() and p.suffix in allowed]
for name in ['root_live_20261010T170048Z.json','root_live_20261010T171120Z.json','latest_formal_live.json',
             'root_five_queue_20261010T164728Z.raw.json','ROOT_FIVE_QUEUE_GROWTH_20261010T1647.json',
             'root_five_queue_20261010T171113Z.raw.json','ROOT_FIVE_QUEUE_GROWTH_20261010T1711.json']:
    paths.append(T/'server_reactivation_20261009'/name)
paths.append(R/'tmp/entry_increment54_postpublication_20261011.json')
paths.append(R/'tmp/adopt_five_queue_observation_1647_root_20261011.py')
paths.append(R/'tmp/adopt_five_queue_observation_1711_root_20261011.py')
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
git('commit','-m','Accept A90 mechanism scene and native295 baseline increments with validation limits')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=dict(native=295,three_view=295,FLGMM_native_new=59,FLGMM_three_views=61,FLGMM_table_complete_scene_records=60,FLGMM_table_complete_scenes=6,gradient_search=42,Hybrid_native_new=9,Hybrid_native_complete_scenes=1,A_checkpoint_records=95,A_table_pairs=90,A_table_scenes=9,reviewer_comments=24),A90_detailed_reply_root_sha256=state['latest_rebuttal_draft']['root_proof_sha256'],A90_clear_reply_root_sha256=state['latest_rebuttal_draft']['clear_reader_root_sha256'],new_terminal_results_adopted=True,final_test=False,whole_rebuttal_complete=False)
(T/'publication_closed_increment55_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
