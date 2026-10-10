"""Read-only actual queue/worker identity, resource and progress snapshot."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
code=r'''
from pathlib import Path
import hashlib,json,os,subprocess,time
base=Path('/workspace/guardfed_checks/celeba_gradient_screen64_20261010')
out=Path('/workspace/celeba_gradient_screen64_results_20261010')
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
read=lambda p:json.loads(p.read_bytes())
processes=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=[x for x in p.read_bytes().decode(errors='replace').split('\0') if x]
  if any(str(base) in x for x in a):
   pid=int(p.parent.name)
   tasks=[dict(tid=int(t.name),cpus=sorted(os.sched_getaffinity(int(t.name)))) for t in (p.parent/'task').iterdir()]
   processes.append(dict(pid=pid,argv=a,threads=tasks,nice=os.getpriority(os.PRIO_PROCESS,pid),io=subprocess.check_output(['ionice','-p',str(pid)],text=True).strip()))
 except (OSError,ValueError,subprocess.CalledProcessError):pass
proofs=sorted(base.glob('ROOT_RESOURCE_*.json'))
proof=read(proofs[-1]) if proofs else None
manifest=read(base/'jobs/manifest.json')
rows=[]
for e in manifest['jobs']:
 p=out/e['id']
 if p.exists():
  rows.append(dict(id=e['id'],progress=read(p/'progress.json') if (p/'progress.json').exists() else None,
   provenance=read(p/'provenance.json') if (p/'provenance.json').exists() else None,
   diagnostics_rounds=len(read(p/'diagnostics.json')) if (p/'diagnostics.json').exists() else 0,
   result=(p/'result.json').exists(),accepted=(p/'acceptance.json').exists()))
service=subprocess.run(['supervisorctl','status','guardfed_celeba_gradient_screen64'],capture_output=True,text=True)
data=dict(at_unix=time.time(),service=dict(returncode=service.returncode,stdout=service.stdout,stderr=service.stderr),
 processes=processes,resource_proof_path=str(proofs[-1]) if proofs else None,resource_proof=proof,rows=rows,
 failure_paths=[str(p) for p in out.rglob('*FAILURE*')]+[str(p) for p in out.rglob('failure.json')],
 log_tail=(base/'queue.log').read_text(errors='replace').splitlines()[-22:] if (base/'queue.log').exists() else [],
 queue=read(out/'QUEUE_PROGRESS.json') if (out/'QUEUE_PROGRESS.json').exists() else None,
 scientific_offserver_accepted=0)
print(json.dumps(data))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -'],
 input=code.encode(),capture_output=True,timeout=50)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
(HERE/('observe_'+stamp+'.stdout')).write_bytes(r.stdout)
(HERE/('observe_'+stamp+'.stderr')).write_bytes(r.stderr)
r.check_returncode();d=json.loads(r.stdout)
p=HERE/('OBSERVATION_'+stamp+'.json');p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
(HERE/'LATEST_OBSERVATION.json').write_bytes(p.read_bytes())
print(json.dumps(dict(path=str(p),service=d['service'],workers=len(d['processes']),
 rows=[dict(id=x['id'],round=x['progress']['round'] if x['progress'] else None,diagnostics_rounds=x['diagnostics_rounds']) for x in d['rows']],
 actual_resource_preflight=d['resource_proof']['status'] if d['resource_proof'] else None,failures=d['failure_paths'],log_tail=d['log_tail'])))
