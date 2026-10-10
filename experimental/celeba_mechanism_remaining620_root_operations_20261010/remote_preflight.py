"""Actual read-only Linux/source/resource gate for the finite620 evaluator."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,time
BASE=Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010')
REPO=Path('/workspace/GuardFed-celeba-expanded')
GUIDE='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
SEAL='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
CPUS=set(range(112,120))
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def status(name):
 r=subprocess.run(['supervisorctl','status',name],capture_output=True,text=True,timeout=20)
 a=r.stdout.split()
 assert len(a)>=2 and a[0]==name and r.returncode in (0,3)
 return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,state=a[1])
assert sha('/etc/vast-agents-guide.md')==GUIDE and sha(BASE/'FILES_SHA256.json')==SEAL
assert not (BASE/'attempt1').exists()
for n,v in read(BASE/'FILES_SHA256.json')['files'].items():
 assert (BASE/n).stat().st_size==v['bytes'] and sha(BASE/n)==v['sha256'],n
plan=read(BASE/'PLAN.json')
for k,h in plan['dependency_sha256'].items():assert sha(plan['remote_dependencies'][k])==h,k
manifest=read(plan['remote_dependencies']['manifest'])
assert sha(plan['remote_dependencies']['manifest'])==plan['manifest_sha256']
assert manifest['jobs']==plan['all_manifest_entries'] and len(manifest['jobs'])==800
for e in manifest['jobs']:assert sha(e['job'])==e['job_sha256']
assert sha(plan['remote_dependencies']['protocol'])==manifest['protocol_sha256']
for n,h in manifest['source_hashes'].items():assert sha(REPO/n)==h,n
for n,h in manifest['adapter_hashes'].items():assert sha(n)==h,n
old=status('guardfed_celeba_mechanism_valid_C_after70')
sglang=status('sglang')
assert old['state']=='EXITED' and sglang['state']=='STOPPED'
owners=[];old_workers=[];same_workers=[];processes=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  pid=int(p.parent.name)
  if pid==os.getpid():continue
  a=[x for x in p.read_bytes().decode(errors='replace').split('\0') if x]
  if not a:continue
  if (p.parent/'stat').read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):continue
  tasks=[]
  for t in (p.parent/'task').iterdir():
   tid=int(t.name);cpus=os.sched_getaffinity(tid)
   if len(cpus)<=16 and CPUS & cpus:owners.append(dict(pid=pid,tid=tid,cpus=sorted(cpus)))
   tasks.append(dict(tid=tid,affinity_count=len(cpus),restricted_cpus=sorted(cpus) if len(cpus)<=16 else None))
  scripts=[x for x in a[1:] if x.endswith('.py') and x.startswith('/workspace/')]
  if any('/celeba_mechanism_valid_C_after70_20261010/' in x for x in scripts):old_workers.append(pid)
  if any(str(BASE) in x for x in scripts):same_workers.append(pid)
  if scripts:processes.append(dict(pid=pid,scripts=scripts,tasks=tasks))
 except(OSError,ValueError):pass
assert not owners and not old_workers and not same_workers
quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
cores=int(quota)/int(period)
reserved=96 # conservative ceiling, not a measured utilization/claim of allocation
assert reserved+8<=cores
current=int(Path('/sys/fs/cgroup/memory.current').read_text());maximum=int(Path('/sys/fs/cgroup/memory.max').read_text())
assert maximum-current>=8*1024**3 and shutil.disk_usage('/workspace').free>=40*1024**3
main=status('guardfed_celeba_mechanism_formal');queue=read(plan['remote_training_progress'])
assert not queue['failed'] and len(queue['active'])<=8
assert main['state']=='RUNNING' or (main['state']=='EXITED' and len(queue['completed'])==800 and not queue['active'])
gpu=subprocess.run(['nvidia-smi','-q'],capture_output=True,text=True,timeout=20)
recovery=[x.strip() for x in gpu.stdout.splitlines() if 'GPU Recovery Action' in x]
assert gpu.returncode==0 and len(recovery)==2 and all(x.endswith('None') for x in recovery)
proof=dict(status='ROOT_ACTUAL_REMAINING620_LINUX_SOURCE_RESOURCE_PREFLIGHT_PASS',**{'pass':True},
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),at_unix=time.time(),allowed_cpus=sorted(CPUS),
 source_seal_sha256=SEAL,guide_sha256=GUIDE,prior_C_after70_EXITED_no_workers=True,sglang_STOPPED=True,
 restricted_CPU112_119_owners=owners,other_reserved_cores=reserved,conservative_reservation_not_measured_use=True,
 dependency_sha256=plan['dependency_sha256'],original_source_data_sha256=manifest['source_hashes'],
 adapter_sha256=manifest['adapter_hashes'],original_manifest_sha256=plan['manifest_sha256'],original800_job_hashes_verified=True,
 actual_Linux_measurements=dict(cpu_quota_cores=cores,memory_headroom_bytes=maximum-current,
 disk_free_bytes=shutil.disk_usage('/workspace').free,main_service=main,completed=len(queue['completed']),active=queue['active'],
 recovery_actions=recovery,memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),project_process_all_thread_affinities=processes),
 old_service=old,sglang_service=sglang,no_new_training=True,no_test=True,no_CNN_in_preflight=True)
p=BASE/'ROOT_LINUX_PREFLIGHT.json'
with p.open('x') as f:f.write(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(path=str(p),sha256=sha(p),source_members=len(read(BASE/'FILES_SHA256.json')['files']),quota=cores,reserved=reserved,completed=len(queue['completed']))))
