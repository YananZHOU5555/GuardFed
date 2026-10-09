import datetime,json,os,time,subprocess
from pathlib import Path
rows=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().decode().strip('\0').split('\0')
  if not args or 'python' not in Path(args[0]).name or not any('GuardFed' in a or 'guardfed_checks' in a for a in args):continue
  env=dict(x.split('=',1) for x in (p.parent/'environ').read_bytes().decode().split('\0') if '=' in x)
  rows.append(dict(pid=int(p.parent.name),argv=args,nice=os.getpriority(os.PRIO_PROCESS,int(p.parent.name)),OMP_NUM_THREADS=env.get('OMP_NUM_THREADS'),GUARDFED_CPU_THREADS=env.get('GUARDFED_CPU_THREADS'),affinity=list(os.sched_getaffinity(int(p.parent.name)))))
 except (OSError,UnicodeError):pass
def usage():return int(dict(x.split() for x in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])
t=time.monotonic();u=usage();time.sleep(2)
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),effective_cores=(usage()-u)/1e6/(time.monotonic()-t),memory_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()),memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),tasks=rows,services=subprocess.run(['supervisorctl','status'],capture_output=True,text=True).stdout)
print(json.dumps(r))
