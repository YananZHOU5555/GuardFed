"""Observe the exact37 deployment and real Linux processes; register no result."""
from pathlib import Path
import argparse,datetime,hashlib,json,shlex,subprocess
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
parser=argparse.ArgumentParser();parser.add_argument('--progress',action='store_true')
parser.add_argument('--scope',type=int,choices=(37,11),default=37);args=parser.parse_args()
EX=ROOT/f'tmp/celeba_mechanism_valid_incremental_next{args.scope}_20261009/execution_candidate'
REMOTE=f'/workspace/guardfed_checks/celeba_mechanism_valid_incremental_next{args.scope}_20261009/execution_candidate'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
deployment=read(EX/'deployment_receipt.json');assert deployment['remote_installation']['returncode']==0
code="""from pathlib import Path
import datetime,hashlib,json,os,subprocess
H=Path(%r)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(Path('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(H/'EXECUTION_SOURCE_SHA256.json')==%r and sha(H/'ROOT_APPROVED.json')==%r
for row in read(H/'EXECUTION_SOURCE_SHA256.json')['members']:assert sha(H/row['path'])==row['sha256']
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        argv=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\\0') if s]
        if str(H/'batch.py') not in argv:continue
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':continue
        env=dict(x.split('=',1) for x in (p/'environ').read_text().split('\\0') if '=' in x)
        threads={t.name:sorted(os.sched_getaffinity(int(t.name))) for t in (p/'task').iterdir()}
        processes.append(dict(pid=int(p.name),argv=argv,cpus=sorted(os.sched_getaffinity(int(p.name))),
            nice=os.getpriority(os.PRIO_PROCESS,int(p.name)),io=subprocess.run(['ionice','-p',p.name],capture_output=True,text=True).stdout.strip(),
            thread_affinities=threads,user_seconds=int(stat[11])/os.sysconf('SC_CLK_TCK'),system_seconds=int(stat[12])/os.sysconf('SC_CLK_TCK'),
            environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
files={name:dict(sha256=sha(H/name),data=read(H/name)) for name in ('APPROVED.json','preflight.json','start_receipt.json')}
completed=[read(p) for p in sorted(H.glob('completed_*.json'))]
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    service=subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_valid_next37'],capture_output=True,text=True).stdout.strip(),
    processes=processes,files=files,completed=completed,approval_hash_file_sha256=sha(H/'APPROVED.sha256'),
    batch_failure=read(H/'batch_failure.json') if (H/'batch_failure.json').exists() else None,
    batch_complete=read(H/'batch_complete.json') if (H/'batch_complete.json').exists() else None,
    guide_sha256=sha(Path('/etc/vast-agents-guide.md')),no_registry_update=True)))
"""%(REMOTE,deployment['execution_seal_sha256'],deployment['root_approval_sha256'])
if args.scope==11:code=code.replace('guardfed_celeba_mechanism_valid_next37','guardfed_celeba_mechanism_valid_next11')
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55',
    'python -c '+shlex.quote(code)],capture_output=True,check=True,timeout=45)
data=json.loads(result.stdout)
if args.progress:
    with (EX/f'ROOT_PROGRESS_{stamp}.RAW.json').open('xb') as stream:stream.write(result.stdout)
elif args.scope==11:
    with (EX/'ROOT_STARTUP_OBSERVATION.RAW.json').open('xb') as stream:stream.write(result.stdout)
assert not data['batch_failure']
assert data['files']['APPROVED.json']['data']['root_approval_sha256']==deployment['root_approval_sha256']
assert data['files']['preflight.json']['data']['external_draft_sha256']==deployment['external_draft_sha256']
assert data['files']['preflight.json']['data']['empty_outputs'] and data['files']['preflight.json']['data']['no_duplicate']
budget=data['files']['preflight.json']['data']['resources_after']
assert budget['conservative_total_including_this8']<=budget['actual_quota_cores']
assert data['files']['start_receipt.json']['data']['execution_seal_sha256']==deployment['execution_seal_sha256']
assert len(data['processes'])<=2
for p in data['processes']:
    assert p['cpus']==list(range(112,120)) and p['nice']>=10 and p['io']=='idle'
    assert all(cpus==list(range(112,120)) for cpus in p['thread_affinities'].values())
    assert p['environment']==dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
workers=[p for p in data['processes'] if 'worker' in p['argv']]
assert ('RUNNING' in data['service'] and len(workers)==1) or (data['batch_complete'] and len(data['completed'])==args.scope)
for name,row in data['files'].items():
    source=EX/name;payload=(json.dumps(row['data'],indent=2,allow_nan=False)+'\n').encode()
    assert hashlib.sha256(payload).hexdigest()==row['sha256']
    if source.exists():assert sha(source)==row['sha256']
    else:source.write_bytes(payload)
approved_sha=EX/'APPROVED.sha256'
if args.progress:
    assert approved_sha.exists()
else:
    assert not approved_sha.exists();approved_sha.write_bytes((data['files']['APPROVED.json']['sha256']+'\n').encode())
assert sha(approved_sha)==data['approval_hash_file_sha256']
data.update(status=f'ROOT_NEXT{args.scope}_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS',
    deployment_receipt_sha256=sha(EX/'deployment_receipt.json'),execution_seal_sha256=deployment['execution_seal_sha256'],
    original23_not_rerun=True,new_training=0,new_Full_inference=0,scientific_offserver_new_accepted=0,test_inference=False)
if args.scope==11:data['original60_not_rerun']=True
if args.progress:
    data.update(status=f'ROOT_NEXT{args.scope}_REAL_LINUX_PROGRESS_AND_ALLOCATION_PASS',
        source_startup_proof_sha256=sha(EX/'ROOT_STARTUP_OBSERVATION.json'),offserver_acceptance_not_measured=True)
target=EX/(f'ROOT_PROGRESS_{stamp}.json' if args.progress else 'ROOT_STARTUP_OBSERVATION.json')
with target.open('x',encoding='utf8') as stream:json.dump(data,stream,indent=2);stream.write('\n')
print(json.dumps({'status':data['status'],'service':data['service'],'actual_processes':len(data['processes']),
    'worker_pids':[p['pid'] for p in workers],'remote_terminal_candidates':len(data['completed']),
    'source_sha256':deployment['execution_seal_sha256'],'proof_sha256':sha(target)}))
