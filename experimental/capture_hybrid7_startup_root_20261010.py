"""One compact read-only actual gate observation; no science or service mutation."""
from pathlib import Path
import datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
script=r'''from pathlib import Path
from collections import deque
import datetime,hashlib,json,os,subprocess
stage=Path('/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/stage')
main=Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
def command(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=20)
 return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
workers=[];coordinators=[]
for proc in Path('/proc').glob('[0-9]*'):
 try:
  argv=[x for x in (proc/'cmdline').read_bytes().decode(errors='replace').split('\0') if x]
  if not argv or 'python' not in Path(argv[0]).name:continue
  if str(stage/'run_canaries.py') in argv:coordinators.append(dict(pid=int(proc.name),argv=argv))
  if str(stage/'run_one.py') not in argv:continue
  ident=argv[argv.index('--job-id')+1];job=read(stage/'jobs'/(ident+'.json'))
  env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x)
  workers.append(dict(pid=int(proc.name),id=ident,argv=argv,job_sha256=sha(stage/'jobs'/(ident+'.json')),
   method=job['method'],attack=job['attack'],seed=job['config']['seed'],alpha=job['config']['client_alpha'],
   progress=read(stage/'gate_runs'/ident/'progress.json') if (stage/'gate_runs'/ident/'progress.json').exists() else None,
   nice=os.getpriority(os.PRIO_PROCESS,int(proc.name)),thread_affinity={t.name:sorted(os.sched_getaffinity(int(t.name))) for t in (proc/'task').iterdir()},
   environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
manifest=read(stage/'manifest.json');terminals=[]
for entry in manifest['preflight_jobs']:
 out=stage/entry['output']
 if (out/'acceptance.json').exists():terminals.append(dict(id=entry['id'],acceptance=read(out/'acceptance.json')))
queue=read(main/'formal_queue_progress.json')
main_active=[]
for row in queue['active']:
 p=main/'runs'/row['id']/'progress.json'
 main_active.append(dict(id=row['id'],pid=row['pid'],progress=read(p) if p.exists() else {}))
result=dict(status='READONLY_ACTUAL_HYBRID7_STARTUP_OBSERVATION_NOT_ACCEPTANCE',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 service=command(['supervisorctl','status','guardfed_celeba_hybrid_fullcoverage_canary']),
 package_sha256=sha(stage/'PACKAGE_SHA256.json'),authorization_sha256=sha(stage/'EXECUTION_AUTHORIZATION.json'),
 workers=workers,coordinators=coordinators,terminal_records=terminals,
 failure_files=[str(p) for p in stage.rglob('*FAILURE*.json')]+[str(p) for p in (stage/'gate_runs').glob('*/failure*.json')],
 gate_acceptance=read(stage/'GATE_ACCEPTANCE.json') if (stage/'GATE_ACCEPTANCE.json').exists() else None,
 main_service=command(['supervisorctl','status','guardfed_celeba_mechanism_formal']),main_completed=len(queue['completed']),main_active=main_active,main_failures=queue.get('failed'),
 gpu=command(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader']),
 stdout_tail=list(deque((stage.parent/'canary_operations/canaries.stdout.log').open(errors='replace'),maxlen=20)) if (stage.parent/'canary_operations/canaries.stdout.log').exists() else [],
 stderr_tail=list(deque((stage.parent/'canary_operations/canaries.stderr.log').open(errors='replace'),maxlen=20)) if (stage.parent/'canary_operations/canaries.stderr.log').exists() else [],
 formal100_started=any((stage/x).exists() for x in ('runs','queue_progress.json')),test=False)
print(json.dumps(result))
'''
def main():
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55',"env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 107 ionice -c 3 nice -n 10 python -B -"],input=script.encode(),capture_output=True,timeout=60)
    now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    with (HERE/('CANARY_OBSERVATION_COMMAND_'+now+'.json')).open('x',encoding='utf8') as f:json.dump(dict(returncode=r.returncode,stderr=r.stderr.decode(errors='replace')),f,indent=2);f.write('\n')
    r.check_returncode();value=json.loads(r.stdout);path=HERE/('CANARY_OBSERVATION_'+now+'.json')
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status=value['status'],workers=[dict(id=w['id'],round=w['progress'].get('round') if w['progress'] else None) for w in value['workers']],terminal_observed=len(value['terminal_records']),failures=value['failure_files'],main_completed=value['main_completed'],formal100_started=value['formal100_started'])))
if __name__=='__main__':main()
