from pathlib import Path
import os,json,datetime
owners=[];collectors=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'accepted_delta_after22_20261010/collect_once.py' in argv:collectors.append(dict(pid=int(p.name),argv=argv))
  for t in p.joinpath('task').iterdir():
   try:a=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(a)<=16 and 106 in a:owners.append(dict(pid=int(p.name),tid=int(t.name),affinity=sorted(a)))
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
assert not owners and not collectors
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU106_released=True,restricted_owners=owners,collector_processes=collectors,no_queue_or_acceptance_resample=True)))
