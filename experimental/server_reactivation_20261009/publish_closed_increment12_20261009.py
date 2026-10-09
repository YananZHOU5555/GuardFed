"""Stage only newly closed scientific evidence and current failure state."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923')
CHECKS=TRAIN/'server_reactivation_20261009'
PREVIOUS='7149322f00b2e0d2e528711853612c42ac01d9c9'
MAPPING={}

def git(*args,**kwargs):
    return subprocess.check_output(['git','-c','core.longpaths=true',*args],cwd=REPO,**kwargs)

def safe(path):
    path=Path(path).resolve()
    return Path('\\\\?\\'+str(path)) if len(str(path))>235 else path

def sha(path):return hashlib.sha256(safe(path).read_bytes()).hexdigest()
def read(path):return json.loads(safe(path).read_bytes())

def copy(source,target,expected=None):
    target=Path(target); destination=REPO/target; digest=sha(source)
    assert destination.resolve().is_relative_to(REPO.resolve())
    assert expected is None or digest==expected
    assert safe(source).stat().st_size<100_000_000
    safe(destination.parent).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(safe(source),safe(destination));assert sha(destination)==digest
    assert MAPPING.setdefault(target.as_posix(),digest)==digest

def seal(source,target,name):
    if name.endswith('.json'):
        rows=read(source/name); rows=rows.get('files',rows.get('members',rows))
        for path,item in rows.items():
            copy(source/path,target/path,item['sha256'] if isinstance(item,dict) else item)
    else:
        for line in safe(source/name).read_text().splitlines():
            digest,path=line.split('  ',1);copy(source/path,target/path,digest)
    copy(source/name,target/name)

assert git('rev-parse','HEAD',text=True).strip()==PREVIOUS
assert not git('status','--porcelain',text=True).strip()
state=read(ROOT/TRAIN/'TRAINING_STATE.json')
assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted']==458
assert state['celeba_mechanism_v1']['scientific_results_offserver_verified']==54
assert not state['baseline_valid_GPU_recovery_20261009']['queue_running']
assert state['baseline_valid_GPU_recovery_20261009']['failure_before_CNN']
tag='root_delta_20261009T124918Z'
main=ROOT/CHECKS/'mechanism_science_backups_20261009'
delta=read(main/tag/'ROOT_DELTA_VERIFICATION.json')
assert delta['total_new_strict_and_offserver']==54 and len(delta['new_ids'])==14
assert sha(main/(tag+'.tar.gz'))==delta['archive_sha256']
assert sha(main/'verified_ledger.json')==delta['ledger_sha256']
for name in (tag+'.tar.gz',tag+'.tar.gz.receipt.json',tag+'_offserver_verification.json','verified_ledger.json'):
    copy(main/name,CHECKS/'mechanism_science_backups_20261009'/name)
for folder in (main/tag,main/('mechanism_inspection_v4_'+tag)):
    for path in folder.iterdir():
        if path.is_file():copy(path,CHECKS/'mechanism_science_backups_20261009'/folder.name/path.name)
table=ROOT/TRAIN/'celeba_mechanism_v1/interim_tables_20261009T124918Z'
assert read(table/'tables.json')['complete_paired_scenes']==5
for path in table.iterdir():
    assert path.is_file();copy(path,TRAIN/'celeba_mechanism_v1'/table.name/path.name)
evidence=ROOT/'tmp/celeba_valid_gpu_remaining464_evidence_20261009'
seal(evidence,Path('experimental')/evidence.name,'FILES_SHA256.txt')
for index in (0,1):
    folder=evidence/('chunk_%03d'%index)
    proof=read(folder/'ROOT_OFFSERVER_VERIFICATION.json')
    assert proof['chunk_index']==index and proof['native_max_abs_difference']==0 and proof['saved_metrics_verified']==99
    assert sha(folder/'chunk_evidence.tar.gz')==proof['archive_sha256']
    for path in folder.iterdir():
        if path.is_file():copy(path,Path('experimental')/evidence.name/folder.name/path.name)
execution=ROOT/'tmp/celeba_valid_gpu_remaining464_execution_20261009'
latest=max(execution.glob('live_*.json'))
assert read(latest)['queue_failure']['failed_chunk']==2
for path in (latest,execution/'ROOT_FAILURE_DIAGNOSIS_20261009.json'):
    copy(path,Path('experimental')/execution.name/path.name)
failure=execution/'failure_chunk002'
assert sha(failure/'failure_chunk_evidence.tar.gz')==read(failure/'ROOT_OFFSERVER_FAILURE_VERIFICATION.json')['archive_sha256']
for path in failure.iterdir():
    assert path.is_file();copy(path,Path('experimental')/execution.name/failure.name/path.name)
fl=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch/observations/bounded_review_20261009T130206Z'
seal(fl,Path('experimental/celeba_flgmm_screen_20261009_v2_dispatch/observations')/fl.name,'FILES_SHA256.json')
for name in ('RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md','celeba_mechanism_v1/EXECUTION.md','publication_closed_increment11_verified_20261009.json'):
    copy(ROOT/TRAIN/name,TRAIN/name)
for name in ('MONITOR_HANDOFF.md','latest_formal_live.json','root_live_20261009T130114Z.json'):
    copy(ROOT/CHECKS/name,CHECKS/name)
for name in ('update_reactivation_state_20261009.py','update_completion_current_20261009.py',
    'backup_mechanism_current_delta_root_20261009.py','promote_mechanism_delta_root_20261009.py',
    'review_gpu_chunk0_adoption_root_20261009.py','observe_remaining464_gpu_root_20261009.py',
    'diagnose_remaining464_failstop_root_20261009.py','preserve_remaining464_failure_root_20261009.py',
    'verify_publication_increment12_20261009.py',Path(__file__).name):
    copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
receipt_path=TRAIN/'publication_closed_increment12_20261009.json'
attributes=REPO/'.gitattributes';content=attributes.read_text()
for pattern in ('experimental/celeba_valid_gpu_remaining464_evidence_20261009/** -text',
    'experimental/celeba_flgmm_screen_20261009_v2_dispatch/observations/** -text',receipt_path.as_posix()+' -text'):
    if pattern not in content:content+='\n'+pattern+'\n'
attributes.write_text(content,newline='\n')
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PREVIOUS,copied_sha256=MAPPING.copy(),
    mechanism_offserver_verified=54,new_mechanism_ids=14,Full_models_repacked=0,baseline_valid_replays_accepted=458,
    new_GPU_offserver_records=22,CPU_provenance_n=434,GPU_provenance_n=24,
    saved_GPU_metrics_verified=198,saved_GPU_confusion_counts_verified=528,saved_GPU_rules_verified=66,
    GPU_queue_stopped=True,failure_before_CNN=True,failure_archive_members=81,partial2_not_registered=True,
    native_tolerance=1e-12,unique_handover_cause_proved=False,original436_unchanged=True,
    complete_IID_Full_minus_U_scenes=5,FLGMM_offserver_accepted=6,FLGMM_observed_terminal=11,
    goal_complete=False,uniform_device_comparison=False,final_protocol_frozen=False,test_started=False)
with (ROOT/receipt_path).open('x',encoding='utf8',newline='\n') as stream:
    json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
copy(ROOT/receipt_path,receipt_path)
names=[*MAPPING,'.gitattributes']
for start in range(0,len(names),25):
    git('add','-f','--',*names[start:start+25]);git('add','--renormalize','--',*names[start:start+25])
changed=git('diff','--cached','--name-only','-z').decode().split('\0')[:-1]
assert set(changed)<=set(names)
payload=git('cat-file','--batch',input=''.join(':'+n+'\n' for n in MAPPING).encode());position=0
for name,digest in MAPPING.items():
    end=payload.index(b'\n',position);header=payload[position:end].split();assert header[1]==b'blob';size=int(header[2])
    assert hashlib.sha256(payload[end+1:end+1+size]).hexdigest()==digest,name
    position=end+size+2
assert position==len(payload)
print(json.dumps(dict(status='SCOPED_INDEX_BLOB_SHA_PASS_NO_COMMIT_OR_PUSH',changed_files=len(changed),byte_verified_files=len(MAPPING),baseline_accepted=458,mechanism_accepted=54)))
