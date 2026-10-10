"""Future root-approved one-shot canary launcher transport; no implicit approval or retry."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,subprocess,sys,traceback
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
HERE=Path(__file__).resolve().parent
REMOTE='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v2_20261010/canary_operations'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')

def main():
 parser=argparse.ArgumentParser()
 for key in ('approval','bound-offserver','bound-root-review','baseline'):
  parser.add_argument('--'+key,type=Path,required=True);parser.add_argument('--'+key+'-sha256',required=True)
 parser.add_argument('--package-sha256',required=True);parser.add_argument('--helper-seal-sha256',required=True)
 a=parser.parse_args();assert sha(HERE/'FILES_SHA256.json')==a.helper_seal_sha256
 for n,pin in read(HERE/'FILES_SHA256.json')['files'].items():assert sha(HERE/n)==pin['sha256']
 inputs={'APPROVAL.json':a.approval,'BOUND_OFFSERVER.json':a.bound_offserver,'BOUND_ROOT_REVIEW.json':a.bound_root_review,'BASELINE.json':a.baseline}
 for path,pin in [(a.approval,a.approval_sha256),(a.bound_offserver,a.bound_offserver_sha256),(a.bound_root_review,a.bound_root_review_sha256),(a.baseline,a.baseline_sha256)]:assert sha(path)==pin
 approval=read(a.approval)
 assert approval['status']=='ROOT_AUTHORIZED_SEVEN_HYBRID_CANARIES' and approval['scope']=='seven_same_horizon_3round_canaries'
 assert approval['package_sha256']==a.package_sha256 and approval['helper_seal_sha256']==a.helper_seal_sha256
 assert approval['bound_offserver_sha256']==a.bound_offserver_sha256 and approval['bound_root_review_sha256']==a.bound_root_review_sha256
 assert approval['baseline_snapshot_sha256']==a.baseline_sha256
 assert approval['service']=='guardfed_celeba_hybrid_fullcoverage_canary' and approval['cpus']==[104] and approval['cpu_threads']==1
 assert approval['formal100_started'] is False and approval['final_test'] is False
 for path in (a.bound_offserver,a.bound_root_review):assert read(path)['package_sha256']==a.package_sha256
 # One exact adopted proof can legitimately cover both offserver verification and root review.
 assert read(a.bound_offserver)['status']=='BOUND_METADATA_MEMBERS_RECEIVED_NOT_TRAINING'
 root=read(a.bound_root_review);assert root['status']=='ROOT_HYBRID100_BOUND_METADATA_ADOPTED'
 assert root['bound_offserver_sha256']==a.bound_offserver_sha256 and (root['new'],root['reused'],root['canaries'])==(96,4,7)
 assert root['execution_authorized'] is False and root['final_test'] is False
 source={n:HERE/n for n in ('remote_launch.py','main_health.py','HEALTH_LINEAGE.json','FILES_SHA256.json')};source.update(inputs)
 members={n:dict(sha256=sha(path),bytes=path.stat().st_size) for n,path in source.items()}
 payload={n:base64.b64encode(path.read_bytes()).decode('ascii') for n,path in source.items()}
 transfer=json.dumps(dict(members=members),indent=2).encode();payload['TRANSFER.json']=base64.b64encode(transfer).decode('ascii')
 attempt=HERE/('attempt_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));attempt.mkdir()
 save(attempt/'TRANSFER.json',dict(members=members))
 script="""from pathlib import Path
import base64,hashlib,json,os,subprocess,sys
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
target=Path(%r);assert target.parent==Path('/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v2_20261010') and target.parent.resolve()==target.parent
assert not target.exists() and not target.is_symlink()
payload=json.loads(%r);members=json.loads(base64.b64decode(payload['TRANSFER.json']))['members']
assert set(payload)==set(members)|{'TRANSFER.json'}
decoded={}
for name,value in payload.items():
 p=Path(name);assert len(p.parts)==1 and not p.is_absolute()
 data=base64.b64decode(value,validate=True)
 if name!='TRANSFER.json':assert hashlib.sha256(data).hexdigest()==members[name]['sha256'] and len(data)==members[name]['bytes']
 decoded[name]=data
# All-thread restricted reservations: broad scheduler masks do not claim CPU107.
for proc in Path('/proc').glob('[0-9]*'):
 if proc.name==str(os.getpid()):continue
 try:
  for task in (proc/'task').iterdir():
   try:cpus=set(os.sched_getaffinity(int(task.name)))
   except ProcessLookupError:continue
   assert not (len(cpus)<=16 and 107 in cpus),('Restricted CPU107 owner',proc.name,task.name)
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
target.mkdir()
for name,data in decoded.items():
 with (target/name).open('xb') as f:f.write(data)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',GUARDFED_CPU_THREADS='1',NUMEXPR_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(['taskset','-c','107','ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(target/'remote_launch.py')],env=env,capture_output=True,text=True)
print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
"""%(REMOTE,json.dumps(payload))
 ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
 try:
  r=subprocess.run(ssh+['python -B -'],input=script.encode(),capture_output=True,timeout=360)
  save(attempt/'REMOTE_COMMAND.json',dict(returncode=r.returncode,stdout=r.stdout.decode(errors='replace'),stderr=r.stderr.decode(errors='replace')))
  r.check_returncode();result=json.loads(r.stdout);assert result['returncode']==0,result
  # Read evidence bytes only; no second launch command, tar APIs or model transfer.
  collect="from pathlib import Path\nimport base64,json\np=Path(%r)\nnames=['START.json','START_COMMANDS.json','RESOURCE.json','SUPERVISOR.conf','APPROVAL.json']\nprint(json.dumps({n:base64.b64encode((p/n).read_bytes()).decode('ascii') for n in names}))\n"%REMOTE
  r2=subprocess.run(ssh+['python -B -'],input=collect.encode(),capture_output=True,timeout=45)
  save(attempt/'RECEIPT_TRANSFER_COMMAND.json',dict(returncode=r2.returncode,stderr=r2.stderr.decode(errors='replace')));r2.check_returncode()
  for name,data in json.loads(r2.stdout).items():
   assert name in ('START.json','START_COMMANDS.json','RESOURCE.json','SUPERVISOR.conf','APPROVAL.json')
   with (attempt/name).open('xb') as f:f.write(base64.b64decode(data,validate=True))
  assert sha(attempt/'APPROVAL.json')==a.approval_sha256
  start=read(attempt/'START.json');assert start['package_sha256']==a.package_sha256 and start['resource_sha256']==sha(attempt/'RESOURCE.json')
  save(attempt/'LOCAL_START_RECEIPTS.json',dict(status='START_RECEIPTS_RECEIVED_NOT_CANARY_ACCEPTANCE',files={n:sha(attempt/n) for n in ('START.json','START_COMMANDS.json','RESOURCE.json','SUPERVISOR.conf','APPROVAL.json')},fullcoverage_started=False))
  print(json.dumps(dict(attempt=str(attempt),service=start['service'],status=start['status'])))
 except BaseException as e:
  save(attempt/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,
   timeout_stdout=(e.stdout or b'').decode(errors='replace') if isinstance(e,subprocess.TimeoutExpired) else None,
   timeout_stderr=(e.stderr or b'').decode(errors='replace') if isinstance(e,subprocess.TimeoutExpired) else None,
   next='Inspect exact remote canary_operations and service; start may already have happened. Never rerun blindly.'))
  raise

if __name__=='__main__':main()
