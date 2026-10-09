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
status=subprocess.run(['supervisorctl','status'],capture_output=True,text=True)
targets=['guardfed_celeba_mechanism_formal','guardfed_celeba_flgmm_screen','guardfed_celeba_hybrid_screen32']
lines={line.split()[0]:line for line in status.stdout.splitlines() if line.strip()}
assert all(name in lines and lines[name].split()[1]=='RUNNING' for name in targets),lines
fl=Path('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2');hy=Path('/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009')
assert hashlib.sha256((fl/'PACKAGE_SHA256.json').read_bytes()).hexdigest()=='aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'
assert hashlib.sha256((hy/'FILES_SHA256.json').read_bytes()).hexdigest()=='2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f'
assert not list(fl.glob('*FAILURE*.json')) and not (hy/'screen_failure.json').exists()
manifest=json.loads(Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1/manifest.json').read_bytes());active=[]
for row in manifest['jobs']:
 out=Path(row['output'])
 assert not (out/'failure.json').exists()
 if (out/'progress.json').exists() and not (out/'result.json').exists():
  progress=json.loads((out/'progress.json').read_bytes());pid=progress.get('pid')
  if pid and Path('/proc',str(pid)).exists():active.append(dict(id=row['id'],progress=progress))
assert len(active)==8, ('main active',len(active))
print(json.dumps(dict(status='READ_ONLY_TARGET_HEALTH_CPU106_AVAILABLE',cpu_quota=quota,nominal_declared=nominal,restricted_owners=blocked,processes=processes,disk_free=shutil.disk_usage('/workspace').free,memory_current=Path('/sys/fs/cgroup/memory.current').read_text().strip(),memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),supervisor_returncode=status.returncode,supervisor_stderr=status.stderr,targets={name:lines[name] for name in targets},main_active=active,fl_progress=json.loads((fl/'queue_progress.json').read_bytes()),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,temperature.gpu','--format=csv,noheader'],text=True))))
