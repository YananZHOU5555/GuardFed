from pathlib import Path
import os,json,hashlib,datetime
g=Path('/etc/vast-agents-guide.md').read_bytes();assert hashlib.sha256(g).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
owners=[];collectors=[]
for p in Path('/proc').glob('[0-9]*'):
 if int(p.name)==os.getpid():continue
 try:
  a=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  if any('native_after9_20261011/collect_once.py' in x for x in a):collectors.append({'pid':int(p.name),'argv':a})
  for t in (p/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(aff)<=16 and 108 in aff:owners.append({'pid':int(p.name),'tid':int(t.name),'cpus':sorted(aff),'argv':a})
 except (FileNotFoundError,ProcessLookupError):continue
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU108_owners=owners,matching_collectors=collectors,CPU108_released=not owners and not collectors,guide_sha256=hashlib.sha256(g).hexdigest(),read_only=True)))
assert not owners and not collectors
