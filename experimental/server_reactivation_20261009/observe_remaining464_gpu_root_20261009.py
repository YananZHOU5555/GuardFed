"""One read-only live snapshot of the new source-bound GPU replay queue."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'tmp/celeba_valid_gpu_remaining464_execution_20261009'
code="""from pathlib import Path
import datetime,hashlib,json,os,subprocess
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
base=Path('/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/remaining464_attempt1')
pkg=Path('/workspace/guardfed_checks/celeba_valid_gpu_remaining464_prepared_20261009')
call=lambda argv:subprocess.run(argv,capture_output=True,text=True).stdout.strip()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
        if (str(pkg)+'/remaining.py run' in cmd or ('/recovery.py' in cmd and 'ROOT_REVIEW_REMAINING464.json' in cmd)) and 'python' in cmd and 'python -c' not in cmd:
            processes.append({'pid':int(p.name),'command':cmd,'affinity':sorted(os.sched_getaffinity(int(p.name))),'nice':os.getpriority(os.PRIO_PROCESS,int(p.name))})
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
read=lambda p:json.loads(p.read_bytes())
proofs=[read(p) for p in sorted(base.glob('chunk_*/batch/runs/*.worker.json'))]
pending=[read(p) for p in sorted(base.glob('chunk_*.REMOTE_PENDING_OFFSERVER.json'))]
out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,
     'service':call(['supervisorctl','status','guardfed_celeba_valid_gpu_remaining464_20261009']),
     'source_package_sha256':sha(pkg/'PACKAGE_SHA256.json'),'processes':processes,
     'worker_complete_exit_only':sum(p['status']=='DIAGNOSTIC_NATIVE_MATCH' for p in proofs),
     'worker_failed':[p['id'] for p in proofs if p['status']!='DIAGNOSTIC_NATIVE_MATCH'],
     'remote_closed_n':sum(len(p['remote_closed_ids']) for p in pending),'offserver_accepted_new_n':0,
     'queue_failure':read(base/'queue_failure.json') if (base/'queue_failure.json').exists() else None,
     'queue_exit':read(base/'queue_exit.json') if (base/'queue_exit.json').exists() else None,
     'GPU_compute_apps':call(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'])}
print(json.dumps(out))
"""
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55',
                       'python -c '+shlex.quote(code)],capture_output=True,check=True,timeout=60)
data=json.loads(result.stdout)
path=DEST/('live_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
with path.open('xb') as out:out.write(result.stdout)
assert data['read_only'] and data['source_package_sha256']=='fa5626ad0ab8be12ac501aea531d7b8ad2f2c05b1a18937dd86fb3708acc8d6b'
assert not data['queue_failure'] and not data['worker_failed']
for row in data['processes']:
    assert row['nice']==10 and row['affinity']==([106] if '/remaining.py run' in row['command'] else [105])
print(json.dumps({'path':str(path),'sha256':hashlib.sha256(result.stdout).hexdigest(),
                  'service':data['service'],'processes':len(data['processes']),
                  'worker_complete_exit_only':data['worker_complete_exit_only'],
                  'remote_closed_n':data['remote_closed_n'],'accepted_new_n':0}))
