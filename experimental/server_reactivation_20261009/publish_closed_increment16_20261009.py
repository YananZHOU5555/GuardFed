"""Stage the newly adopted five mechanism replays and current measured entries."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923')
CHECKS=TRAIN/'server_reactivation_20261009'
PREVIOUS='d659e0bb37bbe89b8390927c54ef5f37f602e6b6'
MAPPING={}
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
assert main['three_view_new_models_offserver_verified']==28
assert main['next37_valid_replay']['offserver_new_accepted']==5
base=ROOT/'tmp/celeba_mechanism_valid_incremental_next37_20261009'
delta=base/'execution_candidate/backups/incremental_20261009T141304Z'
assert sha(delta/'ROOT_ADOPTION_REVIEW.json')=='48b219abe106d4edfb2a8bf1425928c712fcdac7919ed3f1c5e2f688ac9e1955'
assert sha(delta/'incremental_valid_three_views.tar.gz')=='4f16c82638807b3432610d31600a34b4c4fa3ffe092babdbf63d18e1ab260f3f'
for p in delta.iterdir():
    if p.is_file():copy(p,Path('experimental')/base.name/'execution_candidate/backups'/delta.name/p.name)
for name in ('RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md',
             'celeba_mechanism_v1/EXECUTION.md','publication_closed_increment15_verified_20261009.json'):
    copy(ROOT/TRAIN/name,TRAIN/name)
for name in ('MONITOR_HANDOFF.md','latest_formal_live.json','root_live_20261009T142124Z.json'):
    copy(ROOT/CHECKS/name,CHECKS/name)
for name in ('adopt_mechanism_next37_first5_root_20261009.py','update_reactivation_state_20261009.py',
             'update_completion_current_20261009.py','verify_publication_increment16_20261009.py',Path(__file__).name):
    copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
receipt_rel=TRAIN/'publication_closed_increment16_20261009.json'
attributes=REPO/'.gitattributes';content=attributes.read_text()
pattern=receipt_rel.as_posix()+' -text'
if pattern not in content:attributes.write_text(content+'\n'+pattern+'\n',newline='\n')
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PREVIOUS,
    copied_sha256=MAPPING.copy(),new_mechanism_three_view_ids=read(delta/'ROOT_ADOPTION_REVIEW.json')['accepted_new_ids'],
    baseline_valid_replays_accepted=state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted'],
    mechanism_offserver_verified=main['scientific_results_offserver_verified'],mechanism_three_view_offserver_verified=28,
    FLGMM_offserver_verified=13,Hybrid_offserver_verified=4,native_tolerance=1e-12,
    duplicated_old_models=0,test_started=False,scientific_goal_complete=False)
with (ROOT/receipt_rel).open('x',encoding='utf8',newline='\n') as stream:
    json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
copy(ROOT/receipt_rel,receipt_rel)
names=[*MAPPING,'.gitattributes']
for start in range(0,len(names),25):
    git('add','-f','--',*names[start:start+25]);git('add','--renormalize','--',*names[start:start+25])
changed=git('diff','--cached','--name-only','-z').decode().split('\0')[:-1]
assert set(changed)<=set(names)
payload=git('cat-file','--batch',input=''.join(':'+n+'\n' for n in MAPPING).encode());position=0
for name,digest in MAPPING.items():
    end=payload.index(b'\n',position);header=payload[position:end].split();assert header[1]==b'blob';size=int(header[2])
    assert hashlib.sha256(payload[end+1:end+1+size]).hexdigest()==digest,name;position=end+size+2
assert position==len(payload)
print(json.dumps(dict(status='SCOPED_INDEX_BLOB_SHA_PASS_NO_COMMIT_OR_PUSH',changed_files=len(changed),
    byte_verified_files=len(MAPPING),mechanism_three_views=28,accepted_new=5)))
