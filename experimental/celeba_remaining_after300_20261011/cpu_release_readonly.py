from pathlib import Path
import os,json,datetime
owners=[];exporters=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  if '/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/transport.py' in argv and 'export' in argv:exporters.append(int(p.name))
  for t in (p/'task').iterdir():
   cpus=os.sched_getaffinity(int(t.name))
   if len(cpus)<=16 and 111 in cpus:owners.append({'pid':int(p.name),'tid':int(t.name),'cpus':sorted(cpus)})
 except (OSError,ValueError):pass
latest=json.loads(Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/TRANSPORT_LATEST.json').read_bytes())
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU111_restricted_owners':owners,'exporters':exporters,'transport_latest':latest}))
