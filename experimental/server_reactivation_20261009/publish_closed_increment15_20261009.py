"""Stage only newly closed replays, reviewed Hybrid4 and exact37 preparation/dispatch."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923')
CHECKS=TRAIN/'server_reactivation_20261009'
PREVIOUS='a6c18b41826289a6ee5574789732ca548d9dae92'
MAPPING={}
def git(*args,**kwargs):return subprocess.check_output(['git','-c','core.longpaths=true',*args],cwd=REPO,**kwargs)
def safe(p):
    p=Path(p).resolve();return Path('\\\\?\\'+str(p)) if len(str(p))>235 else p
def sha(p):return hashlib.sha256(safe(p).read_bytes()).hexdigest()
def read(p):return json.loads(safe(p).read_bytes())
def copy(src,dst,expected=None):
    dst=Path(dst);target=REPO/dst;digest=sha(src)
    assert target.resolve().is_relative_to(REPO.resolve()) and safe(src).stat().st_size<100_000_000
    assert expected is None or digest==expected
    safe(target.parent).mkdir(parents=True,exist_ok=True);shutil.copyfile(safe(src),safe(target))
    assert sha(target)==digest and MAPPING.setdefault(dst.as_posix(),digest)==digest
assert git('rev-parse','HEAD',text=True).strip()==PREVIOUS and not git('status','--porcelain',text=True).strip()
state=read(ROOT/TRAIN/'TRAINING_STATE.json')
assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted']>=526
assert state['hybrid_screen32_20261009']['offserver_accepted70round_jobs']==4
evidence=ROOT/'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009'
new_chunks=[]
for folder in sorted(evidence.glob('chunk_*')):
    if folder.name=='chunk_000' or not (folder/'ROOT_OFFSERVER_VERIFICATION.json').exists():continue
    new_chunks.append(folder.name)
    for p in folder.iterdir():
        if p.is_file():copy(p,Path('experimental')/evidence.name/folder.name/p.name)
hybrid=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
delta=hybrid/'accepted_delta_first_20261009'
assert sha(delta/'ROOT_RECORD_REVIEW.json')=='5715216f9828dde66b917e943737886ec79d2348b7acb316156e05f29912cfaf'
for p in delta.iterdir():
    if p.is_file():copy(p,Path('experimental')/hybrid.name/delta.name/p.name)
for p in (delta/'local_record_bridge_v1').iterdir():
    if p.is_file():copy(p,Path('experimental')/hybrid.name/delta.name/'local_record_bridge_v1'/p.name)
for name in ('BACKUP_CHAIN_first4_20261009.json','LATEST_BACKUP.json'):copy(hybrid/name,Path('experimental')/hybrid.name/name)
science=ROOT/'tmp/celeba_mechanism_valid_incremental_next37_20261009'
assert sha(science/'FILES_SHA256.json')=='95978fa42c28e9b4ff5b855b33c2dda56edc2b14fcfd56c3a29b0a9ba98135fd'
for row in read(science/'FILES_SHA256.json')['members']:
    copy(science/row['path'],Path('experimental')/science.name/row['path'],row['sha256'])
copy(science/'FILES_SHA256.json',Path('experimental')/science.name/'FILES_SHA256.json')
for p in (science/'root_source_review').iterdir():
    if p.is_file():copy(p,Path('experimental')/science.name/'root_source_review'/p.name)
execution=science/'execution_candidate'
if (execution/'EXECUTION_SOURCE_SHA256.json').exists():
    for row in read(execution/'EXECUTION_SOURCE_SHA256.json')['members']:
        copy(execution/row['path'],Path('experimental')/science.name/execution.name/row['path'],row['sha256'])
    copy(execution/'EXECUTION_SOURCE_SHA256.json',Path('experimental')/science.name/execution.name/'EXECUTION_SOURCE_SHA256.json')
    for name in ('ROOT_APPROVED.json','EXECUTION_DRAFT.json','APPROVED.json','APPROVED.sha256','preflight.json','start_receipt.json',
                 'ROOT_EXECUTION_REVIEW.json','ROOT_STARTUP_OBSERVATION.json','deployment_receipt.json'):
        if (execution/name).exists():copy(execution/name,Path('experimental')/science.name/execution.name/name)
    for name in ('FILES_SHA256.txt','root_deployment_source.tar.gz'):
        if (execution/name).exists():copy(execution/name,Path('experimental')/science.name/execution.name/name)
runtime=ROOT/'tmp/celeba_valid_gpu_remaining440_resource_gate_v2_execution_20261009'
for p in runtime.glob('live_*.json'):
    if p.name>'live_20261009T134141Z.json':
        copy(p,Path('experimental')/runtime.name/p.name)
        if p.with_suffix('.ROOT.json').exists():copy(p.with_suffix('.ROOT.json'),Path('experimental')/runtime.name/p.with_suffix('.ROOT.json').name)
for name in ('RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md','celeba_mechanism_v1/EXECUTION.md',
             'publication_closed_increment14_verified_20261009.json'):
    copy(ROOT/TRAIN/name,TRAIN/name)
for name in ('MONITOR_HANDOFF.md','latest_formal_live.json'):copy(ROOT/CHECKS/name,CHECKS/name)
for p in (ROOT/CHECKS).glob('root_live_20261009T*.json'):
    if p.name>'root_live_20261009T134141Z.json':copy(p,CHECKS/p.name)
helper_names=['review_hybrid_first4_root_20261009.py','adopt_hybrid_first4_root_20261009.py',
    'review_mechanism_next37_root_20261009.py','review_mechanism_next37_execution_root_20261009.py',
    'update_reactivation_state_20261009.py','update_completion_current_20261009.py',
    'observe_remaining440_gpu_root_20261009.py','verify_publication_increment15_20261009.py',Path(__file__).name]
for optional in ('deploy_mechanism_next37_root_20261009.py','observe_mechanism_next37_root_20261009.py'):
    if (ROOT/'tmp'/optional).exists():helper_names.append(optional)
for name in helper_names:copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
receipt_rel=TRAIN/'publication_closed_increment15_20261009.json'
attributes=REPO/'.gitattributes';content=attributes.read_text()
for pattern in ('experimental/'+science.name+'/** -text','experimental/'+hybrid.name+'/** -text',receipt_rel.as_posix()+' -text'):
    if pattern not in content:content+='\n'+pattern+'\n'
attributes.write_text(content,newline='\n')
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PREVIOUS,copied_sha256=MAPPING.copy(),
    new_V2_closed_chunks=new_chunks,baseline_valid_replays_accepted=state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted'],
    mechanism_offserver_verified=state['celeba_mechanism_v1']['scientific_results_offserver_verified'],FLGMM_offserver_verified=13,
    Hybrid_offserver_verified=4,mechanism_next37_source_reviewed=True,native_tolerance=1e-12,test_started=False,scientific_goal_complete=False)
with (ROOT/receipt_rel).open('x',encoding='utf8',newline='\n') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
copy(ROOT/receipt_rel,receipt_rel)
names=[*MAPPING,'.gitattributes']
for start in range(0,len(names),25):
    git('add','-f','--',*names[start:start+25]);git('add','--renormalize','--',*names[start:start+25])
changed=git('diff','--cached','--name-only','-z').decode().split('\0')[:-1];assert set(changed)<=set(names)
payload=git('cat-file','--batch',input=''.join(':'+n+'\n' for n in MAPPING).encode());position=0
for name,digest in MAPPING.items():
    end=payload.index(b'\n',position);header=payload[position:end].split();assert header[1]==b'blob';size=int(header[2])
    assert hashlib.sha256(payload[end+1:end+1+size]).hexdigest()==digest,name;position=end+size+2
assert position==len(payload)
print(json.dumps(dict(status='SCOPED_INDEX_BLOB_SHA_PASS_NO_COMMIT_OR_PUSH',changed_files=len(changed),byte_verified_files=len(MAPPING),
    baseline_accepted=receipt['baseline_valid_replays_accepted'],Hybrid_accepted=4,next37_source_prepared=True)))
