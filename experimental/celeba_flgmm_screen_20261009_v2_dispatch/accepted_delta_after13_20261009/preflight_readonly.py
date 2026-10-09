import os,json,hashlib,subprocess,shutil
from pathlib import Path
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
blocked=[];processes=[];nominal=0
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
  argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
  for task in (proc/'task').iterdir():
   try:
    cpus=os.sched_getaffinity(int(task.name))
    if len(cpus)<=16 and 106 in cpus:blocked.append(dict(pid=int(proc.name),tid=int(task.name),cpus=sorted(cpus),argv=argv))
   except (FileNotFoundError,ProcessLookupError):pass
  if argv and 'python' in Path(argv[0]).name and any(s.startswith(('/workspace/guardfed_checks/','/workspace/GuardFed-')) for s in argv[1:]):
   env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x);n=max([int(env[k]) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','GUARDFED_CPU_THREADS'] if env.get(k,'').isdigit()]+[1]);nominal+=n;processes.append(dict(pid=int(proc.name),threads=n,argv=argv))
 except (FileNotFoundError,ProcessLookupError):pass
quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();quota=float(quota)/float(period)
assert not blocked and nominal+1<=quota
print(json.dumps(dict(status='READ_ONLY_CPU106_AVAILABLE',cpu_quota=quota,nominal_declared=nominal,restricted_owners=blocked,processes=processes,disk_free=shutil.disk_usage('/workspace').free,memory_current=Path('/sys/fs/cgroup/memory.current').read_text().strip(),memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),services=subprocess.check_output(['supervisorctl','status'],text=True),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,temperature.gpu','--format=csv,noheader'],text=True))))
