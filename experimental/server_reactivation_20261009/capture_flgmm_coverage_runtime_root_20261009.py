"""Single read-only actual96 queue/worker allocation check; no monitor or recovery."""
from pathlib import Path
import datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
code=r'''
from pathlib import Path
import datetime,hashlib,json,os,subprocess
H=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage')
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert hashlib.sha256((H/'PACKAGE_SHA256.json').read_bytes()).hexdigest()=='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
read=lambda p:json.loads(p.read_bytes())
q=read(H/'queue_progress.json');assert q['failed'] is False and len(q['active'])==2 and q['reused']==4
assert {r['gpu'] for r in q['active']}=={0,1}
workers=[]
for row in q['active']:
 p=Path('/proc')/str(row['pid']);argv=[x for x in (p/'cmdline').read_bytes().decode().split('\0') if x]
 assert str(H/'run_one.py') in argv and row['id'] in argv
 env=dict(x.split('=',1) for x in (p/'environ').read_text().split('\0') if '=' in x)
 expected=dict(CUDA_VISIBLE_DEVICES=str(row['gpu']),GUARDFED_CPU_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 assert all(env.get(k)==v for k,v in expected.items())
 aff={t.name:sorted(os.sched_getaffinity(int(t.name))) for t in (p/'task').iterdir()}
 assert all(v==[102,103] for v in aff.values())
 nice=os.getpriority(os.PRIO_PROCESS,row['pid']);io=subprocess.run(['ionice','-p',str(row['pid'])],capture_output=True,text=True).stdout.strip()
 assert nice>=10 and io=='idle'
 progress=H/'runs'/row['id']/'progress.json'
 workers.append(dict(**row,argv=argv,environment=expected,thread_affinities=aff,nice=nice,IO=io,progress=read(progress) if progress.exists() else None))
status=subprocess.run(['supervisorctl','status','guardfed_celeba_flgmm_fullcoverage'],capture_output=True,text=True).stdout.strip();assert status.split()[1]=='RUNNING'
assert not list(H.glob('QUEUE_FAILURE*.json')) and not list((H/'runs').glob('*/failure*.json'))
print(json.dumps(dict(status='ROOT_ACTUAL_FLGMM96_TWO_GPU_WORKER_ALLOCATION_PASS_NOT_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),service=status,queue=q,workers=workers,new_scientific_acceptance=0,final_test=False,no_remote_writes=True)))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -'],input=code.encode(),capture_output=True,timeout=45)
folder=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
with (folder/('RUNTIME_'+stamp+'.COMMAND.json')).open('x',encoding='utf8') as f:json.dump(dict(returncode=r.returncode,stderr=r.stderr.decode(errors='replace')),f,indent=2)
r.check_returncode();d=json.loads(r.stdout)
out=folder/('RUNTIME_'+stamp+'.json')
with out.open('xb') as f:f.write(r.stdout)
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(r.stdout).hexdigest(),service=d['service'],rounds=[(w['progress'] or {}).get('round') for w in d['workers']])))
