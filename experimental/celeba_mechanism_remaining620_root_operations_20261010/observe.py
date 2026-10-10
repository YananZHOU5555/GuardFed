"""One read-only compact actual observation; no raw environment or unrelated cmdlines."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
code = r'''from pathlib import Path
import datetime,hashlib,json,os,subprocess
base=Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010');runtime=base/'attempt1'
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
def read(p):return json.loads(p.read_bytes()) if p.exists() else None
name='guardfed_celeba_mechanism_remaining620_valid_v2a'
s=subprocess.run(['supervisorctl','status',name],capture_output=True,text=True,timeout=20)
processes=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=[x for x in p.read_bytes().decode(errors='replace').split('\0') if x]
  if not any(x==str(base/'evaluate_remaining.py') for x in a):continue
  if (p.parent/'stat').read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):continue
  safe_keys=('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')
  env=dict(x.split('=',1) for x in (p.parent/'environ').read_text().split('\0') if '=' in x)
  tasks=[dict(tid=int(t.name),cpus=sorted(os.sched_getaffinity(int(t.name)))) for t in (p.parent/'task').iterdir()]
  processes.append(dict(pid=int(p.parent.name),argv=a,threads=tasks,nice=os.getpriority(os.PRIO_PROCESS,int(p.parent.name)),
   safe_env={k:env.get(k) for k in safe_keys},ionice=subprocess.check_output(['ionice','-p',p.parent.name],text=True).strip()))
 except (OSError,ValueError):pass
progress=read(runtime/'progress.json');ids=[];tasks=[]
for p in sorted((runtime/'tasks').glob('*')):
 b=read(p/'binding.json');c=read(p/'REMOTE_COMPLETE.json')
 if c:ids.append(p.name)
 out=runtime/'runs'/p.name
 tasks.append(dict(id=p.name,binding=b,remote_complete=c,worker_log_tail=(p/'worker.log').read_text(errors='replace')[-3500:] if (p/'worker.log').exists() else None,
  output_files=[dict(name=q.name,bytes=q.stat().st_size) for q in out.glob('*') if q.is_file()],
  hashes={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ('binding.json','REMOTE_COMPLETE.json') if (p/n).exists()}))
failures={str(p.relative_to(base)):read(p) for p in runtime.rglob('*FAILURE*.json')}
q=base/'queue_v2a.log'
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),service=dict(stdout=s.stdout,stderr=s.stderr,returncode=s.returncode),
 progress=progress,remote_closed_ids=ids,remote_closed_n=len(ids),accepted_offserver=0,processes=processes,tasks=tasks,
 failures=failures,queue_log_tail=q.read_text(errors='replace')[-3500:] if q.exists() else None,
 queue_complete=read(runtime/'QUEUE_COMPLETE.json'),source_seal_sha256=hashlib.sha256((base/'FILES_SHA256.json').read_bytes()).hexdigest(),
 root_approval_sha256=hashlib.sha256((base/'ROOT_APPROVED.json').read_bytes()).hexdigest(),new_training=0,test=False)))
'''
r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -'],
                   input=code.encode(),capture_output=True,timeout=75)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
(HERE / ('observe_'+stamp+'.stdout')).write_bytes(r.stdout)
(HERE / ('observe_'+stamp+'.stderr')).write_bytes(r.stderr)
r.check_returncode()
d=json.loads(r.stdout)
p=HERE / ('OBSERVATION_'+stamp+'.json')
p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
(HERE / 'LATEST_OBSERVATION.json').write_bytes(p.read_bytes())
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),utc=d['utc'],service=d['service'],
 remote_closed=d['remote_closed_n'],active=[dict(pid=x['pid'],action=x['argv'][x['argv'].index('evaluate_remaining.py')+1] if 'evaluate_remaining.py' in x['argv'] else 'bound_evaluator',threads=len(x['threads'])) for x in d['processes']],
 failures=d['failures'],accepted_offserver=0)))
