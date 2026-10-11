from pathlib import Path
import os,signal,json,hashlib
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
  if argv and Path(argv[0]).name=='uv' and 'pip' in argv and 'install' in argv and '/workspace/guardfed_envs/logofair-frozen-post-cpu310-20261011/bin/python' in argv and 'netcal==1.3.6' in argv:
   affinity={t.name:sorted(os.sched_getaffinity(int(t.name))) for t in (p/'task').iterdir()};assert all(a==[113] for a in affinity.values());found.append({'pid':int(p.name),'argv':argv,'threads':affinity})
 except (FileNotFoundError,ProcessLookupError):pass
assert len(found)<=1,'Refuse ambiguous owned installation'
for row in found:os.kill(row['pid'],signal.SIGTERM)
print(json.dumps({'reason':'Resolver attempted unapproved torch/CUDA replacement; terminate this owned one-shot install, no retry','targets':found,'signal':'SIGTERM','other_processes_touched':False}))
