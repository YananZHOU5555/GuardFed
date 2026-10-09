"""Publish the actual V2 recovery and newly closed evidence, without old model repacks."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923')
CHECKS=TRAIN/'server_reactivation_20261009'
PREVIOUS='510e96a55911f01f7f17b7976aa09833087ea924'
MAPPING={}
def git(*args,**kwargs):return subprocess.check_output(['git','-c','core.longpaths=true',*args],cwd=REPO,**kwargs)
def safe(path):
    path=Path(path).resolve();return Path('\\\\?\\'+str(path)) if len(str(path))>235 else path
def sha(path):return hashlib.sha256(safe(path).read_bytes()).hexdigest()
def read(path):return json.loads(safe(path).read_bytes())
def copy(source,target,expected=None):
    target=Path(target);destination=REPO/target;digest=sha(source)
    assert destination.resolve().is_relative_to(REPO.resolve()) and safe(source).stat().st_size<100_000_000
    assert expected is None or digest==expected
    safe(destination.parent).mkdir(parents=True,exist_ok=True);shutil.copyfile(safe(source),safe(destination))
    assert sha(destination)==digest and MAPPING.setdefault(target.as_posix(),digest)==digest

assert git('rev-parse','HEAD',text=True).strip()==PREVIOUS and not git('status','--porcelain',text=True).strip()
state=read(ROOT/TRAIN/'TRAINING_STATE.json')
baseline=state['final_evaluator_runtime_20261009'];v2=state['baseline_valid_GPU_remaining440_v2_20261009']
assert baseline['actual_native_valid_image_replays_accepted']>=471 and v2['queue_running']
assert not state['baseline_valid_GPU_recovery_20261009']['queue_running']
assert state['celeba_mechanism_v1']['scientific_results_offserver_verified']>=60
assert v2['root_review_sha256']=='f8b337ddabc225ec3e313e686d19efa5999a98fca419adc42113fb4059557d13'
source_dirs=('celeba_valid_gpu_remaining440_resource_gate_v2_prepared_20261009',
             'celeba_valid_gpu_remaining440_v2_evidence_20261009')
for name in source_dirs:
    folder=ROOT/'tmp'/name;seal=folder/'FILES_SHA256.txt'
    for line in seal.read_text().splitlines():
        digest,member=line.split('  ',1);copy(folder/member,Path('experimental')/name/member,digest)
    copy(seal,Path('experimental')/name/seal.name)
execution=ROOT/'tmp/celeba_valid_gpu_remaining440_resource_gate_v2_execution_20261009'
for path in execution.iterdir():
    if path.is_file():copy(path,Path('experimental')/execution.name/path.name)
evidence=ROOT/'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009'
for chunk in sorted(evidence.glob('chunk_*')):
    if not (chunk/'ROOT_OFFSERVER_VERIFICATION.json').exists():continue
    for path in chunk.iterdir():
        if path.is_file():copy(path,Path('experimental')/evidence.name/chunk.name/path.name)
fl_base=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch'
fl_delta=fl_base/'accepted_delta_after6_20261009'
assert state['flgmm_screen32_20261009']['offserver_accepted70round_jobs']==13
for path in fl_delta.iterdir():
    if path.is_file():copy(path,Path('experimental')/fl_base.name/fl_delta.name/path.name)
for name in ('BACKUP_CHAIN_increment_after6_20261009.json','LATEST_BACKUP.json'):
    copy(fl_base/name,Path('experimental')/fl_base.name/name)

backup=ROOT/CHECKS/'mechanism_science_backups_20261009';tag='root_delta_20261009T133319Z'
for name in (tag+'.tar.gz',tag+'.tar.gz.receipt.json',tag+'_offserver_verification.json','verified_ledger.json'):
    copy(backup/name,CHECKS/'mechanism_science_backups_20261009'/name)
for folder in (backup/tag,backup/('mechanism_inspection_v4_'+tag)):
    for path in folder.rglob('*'):
        if path.is_file():copy(path,CHECKS/'mechanism_science_backups_20261009'/path.relative_to(backup))
table=ROOT/TRAIN/'celeba_mechanism_v1/interim_tables_20261009T133319Z'
for path in table.iterdir():
    if path.is_file():copy(path,TRAIN/'celeba_mechanism_v1'/table.name/path.name)
for name in ('RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md','celeba_mechanism_v1/EXECUTION.md','publication_closed_increment13_verified_20261009.json'):
    copy(ROOT/TRAIN/name,TRAIN/name)
copy(ROOT/CHECKS/'MONITOR_HANDOFF.md',CHECKS/'MONITOR_HANDOFF.md')
for path in (ROOT/CHECKS).glob('root_live_20261009T1329*.json'):copy(path,CHECKS/path.name)
copy(ROOT/CHECKS/'latest_formal_live.json',CHECKS/'latest_formal_live.json')
for name in ('update_reactivation_state_20261009.py','update_completion_current_20261009.py',
             'deploy_remaining440_valid_recovery_root_20261009.py','fix_remaining440_supervisor_review_path_root_20261009.py',
             'observe_remaining440_gpu_root_20261009.py','verify_publication_increment14_20261009.py',Path(__file__).name):
    copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
copy(ROOT/'tmp/adopt_FLGMM_delta_after6_root_20261009.py',Path('experimental/server_reactivation_20261009/adopt_FLGMM_delta_after6_root_20261009.py'))
receipt_path=TRAIN/'publication_closed_increment14_20261009.json'
attributes=REPO/'.gitattributes';content=attributes.read_text()
for pattern in [*(('experimental/'+name+'/** -text') for name in (*source_dirs,execution.name)),receipt_path.as_posix()+' -text']:
    if pattern not in content:content+='\n'+pattern+'\n'
attributes.write_text(content,newline='\n')
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PREVIOUS,copied_sha256=MAPPING.copy(),
    mechanism_offserver_verified=state['celeba_mechanism_v1']['scientific_results_offserver_verified'],
    FLGMM_offserver_verified=13,
    baseline_valid_replays_accepted=baseline['actual_native_valid_image_replays_accepted'],
    new_V2_queue_started=True,actual_spawn_resources_verified=True,precontract_filename_failure_preserved=True,
    old464_not_restarted=True,native_tolerance=1e-12,scientific_goal_complete=False,test_started=False)
with (ROOT/receipt_path).open('x',encoding='utf8',newline='\n') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
copy(ROOT/receipt_path,receipt_path)
names=[*MAPPING,'.gitattributes']
for start in range(0,len(names),25):
    git('add','-f','--',*names[start:start+25]);git('add','--renormalize','--',*names[start:start+25])
changed=git('diff','--cached','--name-only','-z').decode().split('\0')[:-1];assert set(changed)<=set(names)
payload=git('cat-file','--batch',input=''.join(':'+n+'\n' for n in MAPPING).encode());position=0
for name,digest in MAPPING.items():
    end=payload.index(b'\n',position);header=payload[position:end].split();assert header[1]==b'blob';size=int(header[2])
    assert hashlib.sha256(payload[end+1:end+1+size]).hexdigest()==digest,name;position=end+size+2
assert position==len(payload)
print(json.dumps(dict(status='SCOPED_INDEX_BLOB_SHA_PASS_NO_COMMIT_OR_PUSH',changed_files=len(changed),byte_verified_files=len(MAPPING),baseline_accepted=baseline['actual_native_valid_image_replays_accepted'])))
