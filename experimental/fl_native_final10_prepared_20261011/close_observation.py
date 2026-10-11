from pathlib import Path
import datetime,hashlib,json,os,subprocess
read=lambda p:json.loads(Path(p).read_bytes())
def service_contract(name, rc, state, exitstatus, worker_count, queue, expected_ids, completed_ids):
 assert name=='guardfed_celeba_flgmm_fullcoverage'
 assert not queue['failed']
 if state=='RUNNING':
  assert rc==0 and 1<=worker_count<=2
  return 'ORIGINAL_RUNNING_WITH_1_TO_2_STAGE_WORKERS'
 assert state=='EXITED' and rc==3 and exitstatus==0 and worker_count==0
 assert len(expected_ids)==len(set(expected_ids))==96
 assert queue['completed_new']==96 and queue['pending']==0 and not queue['active']
 assert len(completed_ids)==len(set(completed_ids))==96 and set(completed_ids)==set(expected_ids)
 return 'NORMAL_EXITED_ZERO_WORKER_EXACT96_TERMINAL_NOT_SCIENTIFIC_ACCEPTANCE'

def service_gate(service, workers, queue, manifest, stage):
 fields=service.stdout.split();assert len(fields)>=2
 name,state=fields[:2];completed=[];exitstatus=None
 expected=[item['id'] for item in manifest['jobs']]
 if state=='EXITED':
  # Query the existing Supervisor installation through supervisorctl's own
  # interpreter. Missing/mismatched RPC evidence fails closed; no restart.
  import shlex,shutil
  control=Path(shutil.which('supervisorctl'));shebang=control.read_text().splitlines()[0]
  assert shebang.startswith('#!')
  query="from supervisor.supervisorctl import ClientOptions; from supervisor.xmlrpc import SupervisorTransport; from xmlrpc.client import ServerProxy; import json; o=ClientOptions(); o.realize([]); p=ServerProxy('http://127.0.0.1',transport=SupervisorTransport(o.username,o.password,o.serverurl)).supervisor.getProcessInfo('guardfed_celeba_flgmm_fullcoverage'); print(json.dumps({k:p[k] for k in ('name','statename','pid','exitstatus','spawnerr')}))"
  info=json.loads(subprocess.check_output(shlex.split(shebang[2:])+['-c',query],text=True))
  assert info['name']==name and info['statename']=='EXITED' and info['pid']==0 and not info['spawnerr']
  exitstatus=info['exitstatus']
  for item in manifest['jobs']:
   out=stage/'runs'/item['id']
   assert all((out/n).is_file() for n in ('progress.json','result.json','acceptance.json','screen_identity.json','model.pt','job.json'))
   progress=read(out/'progress.json');accepted=read(out/'acceptance.json')
   assert progress['round']==accepted['rounds']==70
   assert (accepted['status'],accepted['evaluation_split'],accepted['train_rows'],accepted['evaluation_rows'])==('PASS','valid',162770,19867)
   assert sha(out/'job.json')==item['job_sha256']
   completed.append(item['id'])
 mode=service_contract(name,service.returncode,state,exitstatus,len(workers),queue,expected,completed)
 return dict(mode=mode,supervisor_status_returncode=service.returncode,supervisor_exitstatus=exitstatus,completed_ids_checked=completed,scientific_acceptance=False)

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
stage=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage')
busy=[];collectors=[];workers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if str(stage/'run_one.py') in argv:workers.append(dict(pid=int(p.name),argv=argv))
  if any('fl_native_final10_prepared_20261011/collect_delta.py' in x for x in argv):collectors.append(dict(pid=int(p.name),argv=argv))
  for t in (p/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(aff)<=16 and 110 in aff:busy.append(dict(pid=int(p.name),tid=int(t.name),affinity=sorted(aff)))
 except (FileNotFoundError,ProcessLookupError):continue
assert not collectors and not busy
stage=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage');q=json.loads((stage/'queue_progress.json').read_bytes())
service=subprocess.run(['supervisorctl','status','guardfed_celeba_flgmm_fullcoverage'],capture_output=True,text=True);assert not list(stage.glob('*FAILURE*.json'))
assert sha(stage/'PACKAGE_SHA256.json')=='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
manifest=read(stage/'manifest.json');assert sha(stage/'manifest.json')==read(stage/'PACKAGE_SHA256.json')['files']['manifest.json']
service_check=service_gate(service,workers,q,manifest,stage)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),collector_processes=collectors,restricted_CPU110_threads=busy,CPU110_released=True,service=service.stdout.strip(),service_terminal_gate=service_check,actual_workers=workers,queue=q,package_sha256=sha(stage/'PACKAGE_SHA256.json'),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text(),memory_current=Path('/sys/fs/cgroup/memory.current').read_text().strip(),memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),disk_free_bytes=__import__('shutil').disk_usage('/workspace').free,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True),new_acceptance_performed=False)))
