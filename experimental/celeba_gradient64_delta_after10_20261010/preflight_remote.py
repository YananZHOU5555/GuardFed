from pathlib import Path
import os,json,subprocess,hashlib,datetime
G=Path('/etc/vast-agents-guide.md').read_bytes();assert hashlib.sha256(G).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
busy=[];workers=[]
for p in Path('/proc').glob('[0-9]*'):
 if int(p.name)==os.getpid():continue
 try:
  argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  for t in (p/'task').iterdir():
   try:a=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(a)<=16 and 110 in a:busy.append(dict(pid=int(p.name),tid=int(t.name),cpus=sorted(a),argv=argv))
  if any(x.endswith('deployment/celeba_mechanism_20261009/worker.py') for x in argv) and '--job' in argv:workers.append(dict(pid=int(p.name),argv=argv))
 except (FileNotFoundError,ProcessLookupError):continue
s=subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_formal'],capture_output=True,text=True)
b=Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1');q=json.loads((b/'formal_queue_progress.json').read_bytes());fail=[str(x) for x in b.rglob('failure.json')]
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),guide_sha256=hashlib.sha256(G).hexdigest(),restricted_CPU110_owners=busy,main_service=dict(returncode=s.returncode,stdout=s.stdout,stderr=s.stderr),main_workers=workers,main_queue=q,main_failures=fail,cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text(),memory_current=Path('/sys/fs/cgroup/memory.current').read_text(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text())
print(json.dumps(r));assert not busy and s.returncode==0 and 'RUNNING' in s.stdout and 1<=len(workers)<=8 and not fail
