from pathlib import Path
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
all_tasks=sorted((runtime/'tasks').glob('*'))
closed_tasks=[p for p in all_tasks if (p/'REMOTE_COMPLETE.json').exists()]
ids=[p.name for p in closed_tasks]
selected={p.name for p in closed_tasks[-1:]}
if progress:selected.add(progress['id'])
for p in all_tasks:
 if p.name not in selected:continue
 b=read(p/'binding.json');c=read(p/'REMOTE_COMPLETE.json')
 out=runtime/'runs'/p.name
 tasks.append(dict(id=p.name,binding=b,remote_complete=c,worker_log_tail=(p/'worker.log').read_text(errors='replace')[-3500:] if (p/'worker.log').exists() else None,
  output_files=[dict(name=q.name,bytes=q.stat().st_size) for q in out.glob('*') if q.is_file()],
  hashes={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ('binding.json','REMOTE_COMPLETE.json') if (p/n).exists()}))
failures={str(p.relative_to(base)):read(p) for p in runtime.rglob('*FAILURE*.json')}
q=base/'queue_v2a.log'
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),service=dict(stdout=s.stdout,stderr=s.stderr,returncode=s.returncode),
 progress=progress,remote_closed_ids=ids,remote_closed_n=len(ids),accepted_offserver=0,processes=processes,tasks=tasks,task_details='Current dependency and last closed task only; all IDs indexed, bulk bindings retained on server',
 failures=failures,queue_log_tail=q.read_text(errors='replace')[-3500:] if q.exists() else None,
 queue_complete=read(runtime/'QUEUE_COMPLETE.json'),source_seal_sha256=hashlib.sha256((base/'FILES_SHA256.json').read_bytes()).hexdigest(),
 root_approval_sha256=hashlib.sha256((base/'ROOT_APPROVED.json').read_bytes()).hexdigest(),new_training=0,test=False)))
