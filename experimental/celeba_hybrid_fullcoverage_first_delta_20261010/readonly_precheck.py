from pathlib import Path
import os,json,hashlib,subprocess,datetime
stage=Path('/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/stage');ID='CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage'
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
owners={str(c):[] for c in [107,110]};workers=[]
for p in Path('/proc').glob('[0-9]*'):
 if int(p.name)==os.getpid():continue
 try:
  a=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  if str(stage/'run_one.py') in a:workers.append({'pid':int(p.name),'argv':a})
  for t in (p/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   for c in [107,110]:
    if len(aff)<=16 and c in aff:owners[str(c)].append({'pid':p.name,'tid':t.name,'affinity':sorted(aff),'argv':a})
 except (FileNotFoundError,ProcessLookupError):continue
s=subprocess.run(['supervisorctl','status','guardfed_celeba_hybrid_fullcoverage'],capture_output=True,text=True)
q=json.loads((stage/'queue_progress.json').read_bytes());p=stage/'runs'/ID
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),service=s.stdout,workers=workers,owners=owners,queue=q,selected_progress=json.loads((p/'progress.json').read_bytes()),selected_acceptance=json.loads((p/'acceptance.json').read_bytes()),package_sha256=hashlib.sha256((stage/'PACKAGE_SHA256.json').read_bytes()).hexdigest(),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text(),memory_current=Path('/sys/fs/cgroup/memory.current').read_text(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text(),no_new_acceptance=True)))
