from pathlib import Path
import os,json,datetime
busy=[];collectors=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  a=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any('celeba_flgmm_fullcoverage_delta_after7_20261010/collect_delta.py' in x for x in a):collectors.append(dict(pid=int(p.name),argv=a))
  for t in (p/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(aff)<=16 and 106 in aff:busy.append(dict(pid=int(p.name),tid=int(t.name),affinity=sorted(aff)))
 except (FileNotFoundError,ProcessLookupError):continue
assert not collectors and not busy
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),collector_processes=collectors,restricted_CPU106_threads=busy,CPU106_released=True)))
