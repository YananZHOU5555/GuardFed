"""Stage saved GPU partials and the separately prepared resource guard repair."""
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
PREVIOUS='e464f773a628dc49515d2433ca835f94356cc55d'
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
    assert destination.resolve().is_relative_to(REPO.resolve()) and safe(source).stat().st_size<100_000_000
    assert expected is None or digest==expected
    safe(destination.parent).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(safe(source),safe(destination));assert sha(destination)==digest
    assert MAPPING.setdefault(target.as_posix(),digest)==digest

assert git('rev-parse','HEAD',text=True).strip()==PREVIOUS and not git('status','--porcelain',text=True).strip()
state=read(ROOT/TRAIN/'TRAINING_STATE.json')
assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted']==460
assert not state['baseline_valid_GPU_recovery_20261009']['queue_running']
partial=ROOT/'tmp/celeba_valid_gpu_remaining464_execution_20261009/partial2_explicit_import'
proof=read(partial/'ROOT_OFFSERVER_IMPORT_VERIFICATION.json')
assert proof['new_CNN_inference']==0 and proof['new_n']==2 and proof['missing9_not_accepted']
assert sha(partial/'cumulative_460_accepted.json')=='d6e396fb327ee81322027027fb3bf185f6903b469359346412d7d92c705aa512'
for path in partial.iterdir():
    if path.is_file():copy(path,Path('experimental/celeba_valid_gpu_remaining464_execution_20261009/partial2_explicit_import')/path.name)
prepared=ROOT/'tmp/celeba_valid_gpu_resource_gate_fix_20261009'
seal_path=prepared/'FILES_SHA256.txt'
for line in seal_path.read_text().splitlines():
    digest,name=line.split('  ',1);copy(prepared/name,Path('experimental')/prepared.name/name,digest)
copy(seal_path,Path('experimental')/prepared.name/seal_path.name)
for name in ('RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md','celeba_mechanism_v1/EXECUTION.md','publication_closed_increment12_verified_20261009.json'):
    copy(ROOT/TRAIN/name,TRAIN/name)
copy(ROOT/CHECKS/'MONITOR_HANDOFF.md',CHECKS/'MONITOR_HANDOFF.md')
for name in ('GPU_RESOURCE_GUARD_V2_ROOT_REVIEW.json','GPU_RESOURCE_GUARD_V2_ROOT_SELFCHECK.log'):
    copy(ROOT/CHECKS/name,CHECKS/name)
for name in ('update_reactivation_state_20261009.py','update_completion_current_20261009.py','import_preserved_GPU_partial2_root_20261009.py','review_GPU_resource_guard_v2_root_20261009.py','verify_publication_increment13_20261009.py',Path(__file__).name):
    copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
receipt_path=TRAIN/'publication_closed_increment13_20261009.json'
attributes=REPO/'.gitattributes';content=attributes.read_text()
for pattern in ('experimental/celeba_valid_gpu_resource_gate_fix_20261009/** -text',receipt_path.as_posix()+' -text'):
    if pattern not in content:content+='\n'+pattern+'\n'
attributes.write_text(content,newline='\n')
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PREVIOUS,copied_sha256=MAPPING.copy(),
    baseline_valid_replays_accepted=460,CPU_provenance_n=434,GPU_provenance_n=26,partial2_new_CNN_inference=0,
    partial_strict_accepted=2,partial_strict_requested=11,partial_strict_missing=9,missing=440,
    guard_fix_prepared_only=True,new_GPU_queue_started=False,native_tolerance=1e-12,scientific_goal_complete=False,test_started=False)
with (ROOT/receipt_path).open('x',encoding='utf8',newline='\n') as stream:
    json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
copy(ROOT/receipt_path,receipt_path)
names=[*MAPPING,'.gitattributes']
for start in range(0,len(names),25):
    git('add','-f','--',*names[start:start+25]);git('add','--renormalize','--',*names[start:start+25])
changed=git('diff','--cached','--name-only','-z').decode().split('\0')[:-1];assert set(changed)<=set(names)
payload=git('cat-file','--batch',input=''.join(':'+n+'\n' for n in MAPPING).encode());position=0
for name,digest in MAPPING.items():
    end=payload.index(b'\n',position);header=payload[position:end].split();assert header[1]==b'blob';size=int(header[2])
    assert hashlib.sha256(payload[end+1:end+1+size]).hexdigest()==digest,name
    position=end+size+2
assert position==len(payload)
print(json.dumps(dict(status='SCOPED_INDEX_BLOB_SHA_PASS_NO_COMMIT_OR_PUSH',changed_files=len(changed),byte_verified_files=len(MAPPING),baseline_accepted=460,guard_fix_prepared_only=True)))
