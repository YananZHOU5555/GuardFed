"""Publish root-accepted current F20 author drafts and compact operational evidence; no new scientific acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,re,argparse
R=Path(__file__).resolve().parents[1]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='27ebed940ce822f87976e0d37b90576e9a43f36a'
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
assert scope_path==R/'tmp/publication_increment60_prepared_20261011/COMPACT_SELECTION.json'
entry=pinned_json(semantic_review,a.semantic_review_sha256)
selection=pinned_json(scope_path,a.compact_selection_sha256)
assert entry['status']=='PASS_LIMITED_CURRENT_ENTRY_SEMANTIC_REVIEW' and entry['root_accepted'] is True
assert selection['status']=='ROOT_CLOSED_INCREMENT60_COMPACT_SELECTION' and selection['root_accepted'] is True
assert selection['parent_commit']==PARENT and selection['semantic_review_sha256']==a.semantic_review_sha256
assert selection['STATE_sha256']==entry['STATE_sha256']
state_path=T/'TRAINING_STATE.json'
state=pinned_json(state_path,entry['STATE_sha256'])
main=state['celeba_mechanism_v1']
assert (main['scientific_results_offserver_verified'],main['three_view_new_models_offserver_verified'])==(320,320)
assert main['three_view_counts_by_variant']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':20}
assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==67
assert state['FLGMM_after48_valid_three_view_20261011']['FLGMM_total_three_view_records']==61
assert state['FLGMM_after48_valid_three_view_20261011']['six_scene_table']['complete_scene_records']==60
assert state['gradient64_validation_search_20261010']['offserver_accepted']==46
assert state['hybrid100_fullcoverage_20261010']['new_accepted']==12
assert main['A_three_view_ten_scene_table']['paired_models']==100
F20_ROOT='9d222cea9e55a00442ff278eb47fa1b622938611049c44d0565644047a8894ac'
assert any(isinstance(v,dict) and v.get('root_proof_sha256')==F20_ROOT and (v.get('paired_models'),v.get('complete_scenes'))==(20,2) for v in main.values())
accepted=entry['current_accepted']
assert (accepted['native_new'],accepted['three_view_models'],accepted['F_paired_models'],accepted['F_complete_scenes'])==(320,320,20,2)
assert (accepted['FL_native_new'],accepted['FL_three_view_total'],accepted['FL_complete_table_records'],accepted['gradient_accepted'],accepted['Hybrid_native_new'])==(67,61,60,46,12)
assert accepted['by_variant']==main['three_view_counts_by_variant']
assert state['latest_five_queue_readonly_observation']==entry['latest_observation_only']
assert entry['latest_observation_only']['new_acceptance']==0 and entry['latest_observation_only']['counts_are_observation_only'] is True
EDITORIAL_ROOT='60bc906155a6dbb9e4c73f42d788b66579b731e8fe843b6f1c712202892409ab'
EDITORIAL_REL='docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_F20_20261011/ROOT_EDITORIAL_REVIEW.json'
latest=state['latest_rebuttal_draft']
assert latest['status']=='ROOT_F20_COMPLETE24_EDITORIAL_CANDIDATES_ADOPTED_FOR_AUTHOR_REVIEW'
assert latest['root_proof_path']==EDITORIAL_REL and latest['root_proof_sha256']==EDITORIAL_ROOT
assert latest['comments_verbatim']==24 and latest['A100_incorporated'] is True and latest['F20_incorporated'] is True
assert (latest['F_paired_models'],latest['F_complete_scenes'],latest['remaining_control_variants'],latest['remaining_control_runs'],latest['remaining_method_coverages'])==(20,2,5,480,7)
assert latest['author_review_only'] is True
assert latest['submitted_manuscript_edited'] is False and latest['manuscript_applied'] is False and latest['final_test'] is False and latest['whole_rebuttal_complete'] is False
editorial=pinned_json(repo_path(EDITORIAL_REL),EDITORIAL_ROOT)
assert editorial['status']==latest['status'] and editorial['root_editorial_review'] is True and editorial['root_read_all_changed_passages'] is True
assert editorial['original_comments']==24 and editorial['quotes_and_order_exact'] is True and editorial['F20_table_root_sha256']==F20_ROOT
assert (editorial['F_paired_models'],editorial['F_complete_scenes'])==(20,2) and editorial['fixed_seed_panels']==[10,9,6]
assert editorial['author_review_only'] is True and editorial['manuscript_applied'] is False and editorial['final_test'] is False and editorial['whole_rebuttal_complete'] is False
for rel,digest in editorial['source_and_documents'].items():
    assert H(repo_path(rel).read_bytes())==digest
assert H(repo_path(latest['clear_reader_entry']).read_bytes())==latest['clear_reader_sha256']
for key in ('entry','manuscript_candidate','clear_reader_entry'):
    assert latest[key] in editorial['source_and_documents']
previous_rel='docs/server_deployment_20260923/training_20260923/publication_closed_increment59_verified_20261011.json'
previous=pinned_json(repo_path(previous_rel),'e2964aa9d269bb84404ef7d0b91498191a336b22cf9b5eaad01112b2685baddd')
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
required=entry_paths|{state_path.relative_to(R).as_posix(),EDITORIAL_REL,previous_rel,'tmp/publish_guardfed_increment60_20261011.py'}|set(editorial['source_and_documents'])|{p['path'] for p in entry['source_pins'].values()}
assert required<=set(selected)
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
paths=list(dict.fromkeys(paths))
for p in paths:
    assert p.is_file() and p.suffix in allowed and p.resolve().is_relative_to(R.resolve())
assert sum(p.stat().st_size for p in paths)<100_000_000
for p in paths:
    assert p.stat().st_size<compact_limit(p) and not p.is_symlink()
destinations=['experimental/'+p.relative_to(R).as_posix()[4:] if p.relative_to(R).as_posix().startswith('tmp/') else p.relative_to(R).as_posix() for p in paths]
assert len(destinations)==len(set(destinations)),'Publication destination collision'
assert not (T/'publication_closed_increment60_verified_20261011.json').exists(),'One publication attempt only'
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
git('commit','-m','Publish adopted F20 response drafts and bounded operational evidence')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed).encode('utf8'));pos=0
for path in changed:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
git('push','origin','HEAD:refs/heads/'+BRANCH)
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==commit
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=dict(native=320,three_view=320,FLGMM_native_new=67,FLGMM_three_views=61,FLGMM_table_complete_scene_records=60,gradient_search=46,Hybrid_native_new=12,UCA_models=300,F_table_pairs=20,F_table_scenes=2,reviewer_comments=24),F20_editorial_root_sha256=EDITORIAL_ROOT,F20_table_root_sha256=F20_ROOT,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,latest_observation_sha256=entry['latest_observation_only']['sha256'],new_scientific_acceptances=0,manuscript_applied=False,final_test=False,whole_rebuttal_complete=False)
(T/'publication_closed_increment60_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
