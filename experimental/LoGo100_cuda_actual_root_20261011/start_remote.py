"""Root-bound, once-only supervisor start; stdlib plus original metadata preflight."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shlex,subprocess,time
BASE=Path('/workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001')
SERVICE='guardfed_uniform_cuda_LoGo100_canary_valid'
GUIDE='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
FIRST='LoGoFair-DP_07_FedAvg_IID_Benign_seed91001'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def scan_duplicates():
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if str(BASE) in argv:raise ValueError('Existing LoGo namespace process: '+p.name)
  except (FileNotFoundError,ProcessLookupError):continue
  except PermissionError:
   try:state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
   except (FileNotFoundError,ProcessLookupError):continue
   if state in ('Z','X'):continue
   raise
need(sha('/etc/vast-agents-guide.md')==GUIDE,'Guide changed')
need(BINDING['execution_authorized'] is True,'No actual start authorization')
need(0<=time.time()-BINDING['issued_unix']<=300,'Stale root start input')
need(BASE.is_dir() and BASE.resolve()==BASE,'Deployed fixed namespace required')
need(sha(BASE/'DEPLOYMENT_SEAL.json')==BINDING['deployment_seal_sha256'],'Deployment changed')
need(sha(BASE/'ROOT_QUEUE_AUTH.json')==BINDING['queue_auth_sha256'],'Queue authorization changed')
a=json.loads((BASE/'ROOT_QUEUE_AUTH.json').read_text());seal=json.loads((BASE/'DEPLOYMENT_SEAL.json').read_text())
need(a['execution_authorized'] is True and a['status']=='ROOT_AUTHORIZED_LOGO100_CALLER_V1','Queue not authorized')
need(time.time()<a['deadline_unix'],'Expired deadline')
for name,pin in seal['files'].items():
 path=BASE/name;need(path.resolve()==path and sha(path)==pin['sha256'],'Changed source: '+name)
need(sha(a['python']['path'])==a['python']['sha256'],'Python changed')
need(len(os.sched_getaffinity(0))>16,'Starter must keep broad affinity')
for name in ['canary','remaining','tasks','runs','ROOT_START_ATTEMPT.json','ROOT_START.json','QUEUE.lock']:
 need(not (BASE/name).exists() and not (BASE/name).is_symlink(),'Existing attempt '+name)
conf=Path('/etc/supervisor/conf.d')/(SERVICE+'.conf');need(not conf.exists() and not conf.is_symlink(),'Existing supervisor configuration')
scan_duplicates()
r=subprocess.run(['supervisorctl','status'],capture_output=True,text=True,timeout=20)
need(r.returncode in (0,3),'Cannot inspect supervisor')
status={v.split()[0]:v.split()[1] for v in r.stdout.splitlines() if len(v.split())>1}
need(SERVICE not in status,'Service already registered')
need(set(BINDING['protected_services_observed'])==set(a['protected_services']),'Incomplete actual status binding')
for name,allowed in a['protected_services'].items():
 need(status.get(name)==BINDING['protected_services_observed'][name] and status[name] in allowed,'Fresh protected status changed: '+name)
commands=[]
def run(argv,timeout=90):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
 row={'argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};commands.append(row)
 print(json.dumps({'command_record':row}),flush=True)
 need(r.returncode==0,'Command failed; preserve evidence, no retry: '+repr(argv))
 return r.stdout
# This is the unchanged caller metadata guard; no scientific import/CNN/fit.
pre=run(['nice','-n','10','ionice','-c','3','taskset','-c','113',a['python']['path'],'-B',str(BASE/'guard.py'),'--id',FIRST,'--root-auth-sha256',BINDING['queue_auth_sha256']])
pre=json.loads(pre);need(pre['pass'] is True and pre['allowed_cpus']==[113],'Original fresh gate rejected')
scan_duplicates()
save(BASE/'ROOT_START_ATTEMPT.json',{'binding':BINDING,'fresh_guard':pre,'utc':datetime.now(timezone.utc).isoformat(),'no_retry':True})
cmd=['/usr/bin/nice','-n','10','/usr/bin/ionice','-c','3',a['python']['path'],'-B',str(BASE/'run_queue.py'),'canary','--root-auth-sha256',BINDING['queue_auth_sha256']]
text='[program:'+SERVICE+']\ncommand='+shlex.join(cmd)+'\ndirectory='+str(BASE)+'\nautostart=false\nautorestart=false\nstartsecs=1\nstartretries=0\nstopasgroup=true\nkillasgroup=true\nenvironment=CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1"\nstdout_logfile='+str(BASE/'canary.supervisor.stdout')+'\nstderr_logfile='+str(BASE/'canary.supervisor.stderr')+'\n'
try:
 with conf.open('x',encoding='utf8') as f:f.write(text)
 for argv in [['supervisorctl','reread'],['supervisorctl','update',SERVICE],['supervisorctl','start',SERVICE],['supervisorctl','status',SERVICE]]:run(argv)
 report={'status':'ROOT_START_COMMANDS_COMPLETED_NOT_SCIENTIFIC_ADOPTION','commands':commands,'binding':BINDING,'config_sha256':sha(conf),'science_acceptance':None}
 save(BASE/'ROOT_START.json',report);print(json.dumps(report))
finally:save(BASE/'ROOT_START_COMMANDS.json',{'commands':commands,'no_retry':True})
