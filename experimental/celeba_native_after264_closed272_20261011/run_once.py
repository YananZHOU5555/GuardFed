"""Run the existing inspector/archiver and close exactly one new mechanism delta."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from guardfed_local_storage import STORAGE_ROOT, check_bulk_storage

ROOT=Path(__file__).resolve().parents[2]
CHECKS=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
LOCAL=Path(__file__).resolve().parent
REMOTE='/workspace/guardfed_checks/server_reactivation_20261009'
PY='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
TOOL=REMOTE+'/evidence_v4.py'
HASH='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
save=lambda p,v:p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
ledger=LOCAL/'PARENT_LEDGER.json'; previous=sha(ledger); before=read(ledger)
tag='root_delta_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
storage_check = check_bulk_storage()
folder=LOCAL/tag; assert not folder.exists(); folder.mkdir()
save(folder/'STORAGE_PREFLIGHT.json', storage_check)
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
inspection=REMOTE+'/native_delta_after264_20261011/'+tag+'/inspection'
remote_archive=REMOTE+'/native_delta_after264_20261011/'+tag+'/'+tag+'.tar.gz'
code="""from pathlib import Path
import hashlib,json,subprocess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(TOOL)==HASH
canonical=Path(REMOTE+'/mechanism_science_backups_20261009/verified_ledger.json')
assert sha(canonical)==PREVIOUS
import os,datetime
busy=[];workers=[]
for p in Path('/proc').glob('[0-9]*'):
 if int(p.name)==os.getpid():continue
 try:
  a=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(bytes([0])) if v]
  if any(v.endswith('deployment/celeba_mechanism_20261009/worker.py') for v in a):workers.append({'pid':int(p.name),'argv':a})
  for t in (p/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(aff)<=16 and 107 in aff:busy.append({'pid':p.name,'tid':t.name,'affinity':sorted(aff),'argv':a})
 except (FileNotFoundError,ProcessLookupError):continue
svc=subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_formal'],capture_output=True,text=True)
q=json.loads(Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json').read_bytes())
preflight=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU107_restricted_owners=busy,main_service=svc.stdout,main_workers=workers,queue=q,cgroup_cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text(),memory_current=Path('/sys/fs/cgroup/memory.current').read_text(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text(),memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),canonical_ledger_sha256=sha(canonical),ledger_copy_only=True)
assert not busy and svc.returncode==0 and 'RUNNING' in svc.stdout and 1<=len(workers)<=8 and not q['failed'],json.dumps(preflight)
base=Path(LEDGER).parent
assert not base.exists() and not base.is_symlink()
base.mkdir(parents=True)
Path(LEDGER).write_bytes(canonical.read_bytes());assert sha(LEDGER)==PREVIOUS
(base/'PREFLIGHT.json').write_text(json.dumps(preflight,indent=2))
assert not Path(INSPECTION).exists() and not Path(ARCHIVE).exists()
prefix=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','taskset','-c','107','nice','-n','10','ionice','-c','3',PY,TOOL]
commands=[prefix+['inspect','--repo','/workspace/GuardFed-celeba-expanded','--stage','/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1','--adapter-dir','/workspace/GuardFed-celeba-expanded/deployment/celeba_mechanism_20261009','--full-inventory',REMOTE+'/model_inventory.json','--output',INSPECTION],prefix+['backup','--inspection',INSPECTION+'/inspection.json','--ledger',LEDGER,'--archive',ARCHIVE]]
logs=[]
for argv in commands:
    result=subprocess.run(argv,capture_output=True,text=True)
    logs.append({'argv':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    if result.returncode:break
print(json.dumps({'preflight':preflight,'canonical_ledger_unchanged':sha(canonical)==PREVIOUS,'logs':logs,'ledger_sha256':sha(LEDGER),'tool_sha256':sha(TOOL),'archive_size_bytes':Path(ARCHIVE).stat().st_size if Path(ARCHIVE).is_file() else None}))
raise SystemExit(logs[-1]['returncode'])
"""
bindings=dict(TOOL=TOOL,HASH=HASH,LEDGER=REMOTE+'/native_delta_after264_20261011/'+tag+'/verified_ledger.json',PREVIOUS=previous,
              INSPECTION=inspection,ARCHIVE=remote_archive,PY=PY,REMOTE=REMOTE)
bound='\n'.join(k+'='+repr(v) for k,v in bindings.items())+'\n'+code
result=subprocess.run(ssh+['python -c '+shlex.quote(bound)],capture_output=True)
(folder/'REMOTE_COMMANDS.json').write_bytes(result.stdout)
(folder/'REMOTE_STDERR.log').write_bytes(result.stderr)
assert result.returncode==0, 'Preserve failed delta; do not overwrite or blind retry'
after=read(folder/'REMOTE_COMMANDS.json')
def fetch(remote,local):
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350','root@89.22.197.55:'+remote,str(local)],check=True,timeout=180)
storage_check = check_bulk_storage(after['archive_size_bytes'])
bulk_folder = STORAGE_ROOT/'celeba_native_after264_closed272_20261011'/tag
assert not bulk_folder.exists(), 'Never overwrite external evidence'
bulk_folder.mkdir(parents=True)
archive=bulk_folder/(tag+'.tar.gz'); receipt=folder/(tag+'.tar.gz.receipt.json')
save(folder/'ARCHIVE_LOCATION.json', dict(storage_check, archive_local_path=archive.as_posix(), remote_archive=remote_archive))
fetch(remote_archive,archive); fetch(remote_archive+'.receipt.json',receipt)
closed=read(receipt)
assert closed['accepted_new_ids'] and sha(archive)==closed['archive_sha256']
output=folder/'OFFSERVER_VERIFICATION.json'
verified=subprocess.run([sys.executable,str(ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'),'verify','--archive',str(archive),'--receipt',str(receipt),'--output',str(output)],capture_output=True,check=True)
(folder/'ROOT_VERIFY.log').write_bytes(verified.stdout+verified.stderr)
fetch(bindings['LEDGER'],folder/'verified_ledger.json')
updated=read(folder/'verified_ledger.json')
assert sha(folder/'verified_ledger.json')==after['ledger_sha256']
assert updated['entries'][:-1]==before['entries'] and updated['entries'][-1]['receipt_sha256']==sha(receipt)
assert updated['entries'][-1]['archive']==remote_archive and sha(ledger)==previous
promoted=folder/'inspection'; promoted.mkdir()
for name in ['inspection.json','inspection.sha256','statistics.json','per_seed.csv','paired_per_seed.csv','per_scene_summary.csv']:
    fetch(inspection+'/'+name,promoted/name)
report=read(promoted/'inspection.json')
assert report['source_script_sha256']==HASH and not report['invalid']
assert len(set(report['accepted_new_ids']))==report['new_count'] and set(closed['accepted_new_ids'])<=set(report['accepted_new_ids'])
assert sha(promoted/'inspection.json')==(promoted/'inspection.sha256').read_text().strip()==closed['inspection_sha256']
proof={'status':'ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS_ROOT_PENDING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'tag':tag,'new_ids':closed['accepted_new_ids'],'total_new_strict_and_offserver':report['new_count'],
       'archive_sha256':sha(archive),'receipt_sha256':sha(receipt),'offserver_proof_sha256':sha(output),
       'archive_local_path':archive.as_posix(),'storage_preflight_sha256':sha(folder/'ARCHIVE_LOCATION.json'),
       'previous_ledger_sha256':previous,'ledger_sha256':sha(folder/'verified_ledger.json'),
       'inspection_sha256':sha(promoted/'inspection.json'),'oldFull_models_repacked':0,'test':False,'whole_rebuttal_complete':False}
save(folder/'DELTA_VERIFICATION.json',proof)
assert sha(ledger)==previous  # Parent receipt remains immutable; root separately promotes the new ledger.
print(json.dumps(proof))
