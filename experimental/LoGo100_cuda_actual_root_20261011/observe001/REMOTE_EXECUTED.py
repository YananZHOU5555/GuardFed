"""Read-only compact LoGo fixed2 observation. No arrays/pickle/scientific imports."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
BASE=Path('/workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001')
GUIDE='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()==GUIDE
ids=['LoGoFair-DP_07_FedAvg_IID_Benign_seed91001','LoGoFair-DP_07_FedAvg_IID_F Flip_seed91001']
def record(p):
 if not p.exists():return None
 raw=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'record':json.loads(raw)}
def tail(p):
 if not p.exists():return None
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));return f.read().decode(errors='replace')
r=subprocess.run(['supervisorctl','status','guardfed_uniform_cuda_LoGo100_canary_valid'],capture_output=True,text=True,timeout=20)
processes=[];unreadable=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(BASE) in argv:processes.append({'pid':int(p.name),'argv':argv,'affinity':sorted(os.sched_getaffinity(int(p.name)))})
 except (FileNotFoundError,ProcessLookupError):continue
 except PermissionError:unreadable.append(int(p.name))
print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'service':{'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode},'processes':processes,'unreadable_pids':unreadable,'progress':record(BASE/'canary/progress.json'),'complete':record(BASE/'canary/COMPLETE.json'),'failure':record(BASE/'canary/QUEUE_FAILURE.json'),'tasks':{i:{'accepted':record(BASE/'tasks'/i/'ACCEPTED.json'),'worker_command':record(BASE/'tasks'/i/'worker_command.json'),'stderr_tail':tail(BASE/'tasks'/i/'worker.stderr')} for i in ids},'supervisor_stderr_tail':tail(BASE/'canary.supervisor.stderr'),'root_adopted':False,'observation_only':True}))
