"""Read-only exact Hybrid terminal/source/resource snapshot; no torch or dispatch."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
sys.dont_write_bytecode=True
H=Path('/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
assert sha(Path('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(H/'FILES_SHA256.json')=='2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f'
assert sha(H/'screen_scope.json')=='d76d5fdff375c58b3b42354fc4256e19b26b29f3fdb4387530f0ae8617920f94'
assert sha(H/'runtime_protocol.json')=='bcc66477d22096eaf647e31f065f59ed6727716dcf01db814d5edcbe4ad525f1'
seal=read(H/'FILES_SHA256.json')
for name,digest in seal['files'].items():assert sha(H/name)==digest,name
sys.path.insert(0,str(H));from driver import approve
scope,approval=approve('screen',H/'APPROVED_screen.json',sha(H/'APPROVED_screen.json'),dispatch=False)
assert len(scope['jobs'])==32 and len({e['id'] for e in scope['jobs']})==32
rows=[]
for entry in scope['jobs']:
 out=H/entry['output'];progress=read(out/'progress.json') if (out/'progress.json').exists() else None
 result=read(out/'result.json') if (out/'result.json').exists() else None
 receipt=read(out/'acceptance.json') if (out/'acceptance.json').exists() else None
 if receipt:
  assert receipt['status']=='PASS' and receipt['scope_sha256']==sha(H/'screen_scope.json') and receipt['job_sha256']==entry['job_sha256']
 rows.append(dict(id=entry['id'],terminal=receipt is not None,result_exists=result is not None,
     round=progress['round'] if progress else None,result_rounds=result.get('rounds') if result else None,
     acceptance_sha256=sha(out/'acceptance.json') if receipt else None,
     failures=[str(f) for f in out.glob('failure*.json')]))
all_restricted_threads=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  for task in (proc/'task').iterdir():
   try: affinity=os.sched_getaffinity(int(task.name))
   except ProcessLookupError:continue
   if len(affinity)<=16:
    all_restricted_threads.append(dict(pid=int(proc.name),tid=int(task.name),affinity=sorted(affinity)))
    assert 106 not in affinity,('CPU106 occupied by restricted thread',proc.name,task.name)
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
processes=[];nominal=0;restricted=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
  if not argv or 'python' not in Path(argv[0]).name:continue
  affinity=sorted(os.sched_getaffinity(int(proc.name)))
  for task in (proc/'task').iterdir():
   try: thread_affinity=os.sched_getaffinity(int(task.name))
   except ProcessLookupError:continue
   assert not(len(thread_affinity)<=16 and 106 in thread_affinity),('CPU106 thread conflict',proc.name,task.name)
  if len(affinity)<=16:restricted.extend(affinity)
  if any(s.startswith(('/workspace/guardfed_checks/','/workspace/GuardFed-')) for s in argv[1:]):
   env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x)
   threads=max([int(env[k]) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','GUARDFED_CPU_THREADS') if env.get(k,'').isdigit()]+[1]);nominal+=threads
   processes.append(dict(pid=int(proc.name),argv=argv,declared_threads=threads,affinity=affinity if len(affinity)<=16 else dict(count=len(affinity))))
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();assert quota!='max'
snapshot=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_host=os.uname().nodename,
 remote_root=str(H),rows=rows,service=subprocess.check_output(['supervisorctl','status','guardfed_celeba_hybrid_screen32'],text=True).strip(),
 processes=processes,restricted_cpus=sorted(set(restricted)),helper_cpu106_free=106 not in restricted,nominal_threads=nominal,
 cpu_quota=int(quota)/int(period),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free','--format=csv,noheader'],text=True).strip(),
 guide_sha256=sha(Path('/etc/vast-agents-guide.md')),source_seal_sha256=sha(H/'FILES_SHA256.json'),source_members_verified=len(seal['files']),
 scope_sha256=sha(H/'screen_scope.json'),runtime_protocol_sha256=sha(H/'runtime_protocol.json'),approval_sha256=sha(H/'APPROVED_screen.json'),
 screen_failure=(H/'screen_failure.json').exists(),new_CNN_or_training=False)
snapshot['all_restricted_threads']=all_restricted_threads
snapshot['CPU106_all_thread_restricted_owners_free']=True
print(json.dumps(snapshot,allow_nan=False))
