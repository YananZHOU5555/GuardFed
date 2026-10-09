"""Review and stage only eight closed GPU chunks and eight new mechanism terminals."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923');CHECKS=TRAIN/'server_reactivation_20261009'
PREVIOUS='795f4b09c60c4a81de3d4aa67dad5beac5075827';MAPPING={}
def git(*args,**kwargs):return subprocess.check_output(['git','-c','core.longpaths=true',*args],cwd=REPO,**kwargs)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def copy(src,dst,expected=None):
    dst=Path(dst);target=REPO/dst;digest=sha(src)
    assert target.resolve().is_relative_to(REPO.resolve()) and src.stat().st_size<100_000_000
    assert expected is None or digest==expected
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    assert sha(target)==digest and MAPPING.setdefault(dst.as_posix(),digest)==digest
assert git('rev-parse','HEAD',text=True).strip()==PREVIOUS and not git('status','--porcelain',text=True).strip()
state=read(ROOT/TRAIN/'TRAINING_STATE.json');main=state['celeba_mechanism_v1']
assert main['scientific_results_offserver_verified']==68 and main['three_view_new_models_offserver_verified']==28
assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted']==658
evidence=ROOT/'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009'
prior=evidence/'chunk_009/cumulative_570_accepted.json';prior_data=read(prior);reviews=[]
assert sha(prior)=='d80b4bea9692144e28a78e3a0871ef7301beea762574f680254d25aaa13b681c'
for index in range(10,18):
    folder=evidence/f'chunk_{index:03d}';proof_path=folder/'ROOT_OFFSERVER_VERIFICATION.json';proof=read(proof_path)
    collector_path=folder/f'cumulative_{460+11*(index+1)}_accepted.json';collector=read(collector_path)
    assert proof['status']=='ROOT_GPU440_RESOURCE_GUARD_V2_CHUNK_OFFSERVER_SAVED_ARRAY_PASS'
    assert proof['chunk_index']==index and proof['native_max_abs_difference']==0
    assert proof['saved_metrics_verified']==99 and proof['saved_confusion_counts_verified']==264 and proof['saved_prediction_rules_verified']==33
    assert proof['new_CNN_inference']==0 and not proof['final_test']
    assert sha(folder/'chunk_evidence.tar.gz')==proof['archive_sha256']
    assert collector['previous_collector_sha256']==sha(prior) and collector['new_proof_sha256']==sha(proof_path)
    assert collector['accepted_n']==len(set(collector['accepted_ids']))==prior_data['accepted_n']+11
    assert set(collector['accepted_ids'])-set(prior_data['accepted_ids'])==set(proof['accepted_new_ids'])
    assert not set(proof['accepted_new_ids']).intersection(prior_data['accepted_ids'])
    reviews.append(dict(chunk=index,archive_sha256=proof['archive_sha256'],proof_sha256=sha(proof_path),collector_sha256=sha(collector_path)))
    for p in folder.iterdir():
        if p.is_file():copy(p,Path('experimental')/evidence.name/folder.name/p.name)
    prior,prior_data=collector_path,collector
assert prior_data['accepted_n']==658
handoff=evidence/'BOUNDED_DELTA_010_017_HANDOFF.json'
copy(handoff,Path('experimental')/evidence.name/handoff.name)
backup=ROOT/CHECKS/'mechanism_science_backups_20261009';tag='root_delta_20261009T142459Z'
proof=read(backup/tag/'ROOT_DELTA_VERIFICATION.json')
assert proof['total_new_strict_and_offserver']==68 and len(proof['new_ids'])==8
assert proof['oldFull_models_repacked']==0 and not proof['test']
for name in (tag+'.tar.gz',tag+'.tar.gz.receipt.json',tag+'_offserver_verification.json','verified_ledger.json'):
    copy(backup/name,CHECKS/'mechanism_science_backups_20261009'/name)
for folder in (backup/tag,backup/('mechanism_inspection_v4_'+tag)):
    for p in folder.rglob('*'):
        if p.is_file():copy(p,CHECKS/'mechanism_science_backups_20261009'/p.relative_to(backup))
runtime=ROOT/'tmp/celeba_valid_gpu_remaining440_resource_gate_v2_execution_20261009'
for name in ('live_20261009T142327Z.json','live_20261009T142327Z.ROOT.json'):
    copy(runtime/name,Path('experimental')/runtime.name/name)
for name in ('RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md',
             'celeba_mechanism_v1/EXECUTION.md','publication_closed_increment16_verified_20261009.json'):
    copy(ROOT/TRAIN/name,TRAIN/name)
for name in ('MONITOR_HANDOFF.md','latest_formal_live.json','root_live_20261009T143236Z.json'):
    copy(ROOT/CHECKS/name,CHECKS/name)
for name in ('update_reactivation_state_20261009.py','update_completion_current_20261009.py',
             'verify_publication_increment16_20261009.py',Path(__file__).name):
    copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
receipt_rel=TRAIN/'publication_closed_increment17_20261009.json'
attributes=REPO/'.gitattributes';content=attributes.read_text();pattern=receipt_rel.as_posix()+' -text'
if pattern not in content:attributes.write_text(content+'\n'+pattern+'\n',newline='\n')
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PREVIOUS,
    copied_sha256=MAPPING.copy(),new_GPU_chunks_root_reviewed=reviews,baseline_valid_replays_accepted=658,
    mechanism_offserver_verified=68,mechanism_three_view_offserver_verified=28,FLGMM_offserver_verified=13,
    Hybrid_offserver_verified=4,native_tolerance=1e-12,duplicated_old_models=0,test_started=False,scientific_goal_complete=False)
with (ROOT/receipt_rel).open('x',encoding='utf8',newline='\n') as stream:
    json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
copy(ROOT/receipt_rel,receipt_rel);names=[*MAPPING,'.gitattributes']
for start in range(0,len(names),25):
    git('add','-f','--',*names[start:start+25]);git('add','--renormalize','--',*names[start:start+25])
changed=git('diff','--cached','--name-only','-z').decode().split('\0')[:-1];assert set(changed)<=set(names)
payload=git('cat-file','--batch',input=''.join(':'+n+'\n' for n in MAPPING).encode());position=0
for name,digest in MAPPING.items():
    end=payload.index(b'\n',position);header=payload[position:end].split();assert header[1]==b'blob';size=int(header[2])
    assert hashlib.sha256(payload[end+1:end+1+size]).hexdigest()==digest,name;position=end+size+2
assert position==len(payload)
print(json.dumps(dict(status='ROOT_EIGHT_CHUNKS_AND_INDEX_BLOB_SHA_PASS_NO_COMMIT_OR_PUSH',
    changed_files=len(changed),byte_verified_files=len(MAPPING),baseline_accepted=658,mechanism_native_accepted=68)))
