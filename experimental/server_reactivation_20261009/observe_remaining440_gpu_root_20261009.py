"""Source-bound read-only spawn and progress observation; save before assertions."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'tmp/celeba_valid_gpu_remaining440_resource_gate_v2_execution_20261009'
code="""from pathlib import Path
import datetime,hashlib,json,os,subprocess
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(Path('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
base=Path('/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/remaining440_resource_gate_v2_attempt1')
pkg=Path('/workspace/guardfed_checks/celeba_valid_gpu_remaining440_resource_gate_v2_prepared_20261009')
runtime=Path('/workspace/guardfed_checks/celeba_valid_gpu_resource_gate_fix_20261009/release')
call=lambda argv:subprocess.run(argv,capture_output=True,text=True).stdout.strip()
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
        if (str(pkg)+'/remaining.py run' in cmd or (str(runtime)+'/recovery.py' in cmd and 'ROOT_REVIEW_REMAINING440_RESOURCE_GATE_V2.json' in cmd)) and 'python' in cmd and 'python -c' not in cmd:
            env=dict(e.split('=',1) for e in (p/'environ').read_text().split('\\0') if '=' in e)
            processes.append({'pid':int(p.name),'command':cmd,'exe':str((p/'exe').resolve()),'affinity':sorted(os.sched_getaffinity(int(p.name))),'nice':os.getpriority(os.PRIO_PROCESS,int(p.name)),
                              'thread_env':{k:env.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES')},'ionice':call(['ionice','-p',p.name])})
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
proofs=[read(p) for p in sorted(base.glob('chunk_*/batch/runs/*.worker.json'))]
resources=[{'path':str(p),'sha256':sha(p),'resource':read(p)} for p in sorted(base.glob('chunk_*/batch/runs/*.resource.json'))]
pending=[read(p) for p in sorted(base.glob('chunk_*.REMOTE_PENDING_OFFSERVER.json'))]
out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,
     'service':call(['supervisorctl','status','guardfed_celeba_valid_gpu_remaining440_resource_gate_v2_20261009']),
     'sglang':call(['supervisorctl','status','sglang']),
     'source_package_sha256':sha(pkg/'PACKAGE_SHA256.json'),'runtime_package_sha256':sha(runtime/'PACKAGE_SHA256.json'),
     'review_sha256':sha(base.parent/'ROOT_REVIEW_REMAINING440_RESOURCE_GATE_V2.json'),
     'config_sha256':sha(Path('/etc/supervisor/conf.d/guardfed_celeba_valid_gpu_remaining440_resource_gate_v2_20261009.conf')),
     'processes':processes,'resources':resources,'worker_complete_exit_only':sum(p['status']=='DIAGNOSTIC_NATIVE_MATCH' for p in proofs),
     'worker_failed':[p['id'] for p in proofs if p['status']!='DIAGNOSTIC_NATIVE_MATCH'],
     'remote_closed_n':sum(len(p['remote_closed_ids']) for p in pending),'offserver_accepted_new_n':0,
     'queue_failure':read(base/'queue_failure.json') if (base/'queue_failure.json').exists() else None,
     'queue_exit':read(base/'queue_exit.json') if (base/'queue_exit.json').exists() else None,
     'GPU_compute_apps':call(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'])}
print(json.dumps(out))
"""
if len(sys.argv)==2:
    path=Path(sys.argv[1]);payload=path.read_bytes()
    assert path.resolve().is_relative_to(DEST.resolve())
else:
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -c '+shlex.quote(code)],capture_output=True,check=True,timeout=60)
    payload=result.stdout
    path=DEST/('live_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    with path.open('xb') as out:out.write(payload)
data=json.loads(payload)
assert data['read_only'] and data['source_package_sha256']=='355e22697f213b4b4b9f2509cf5e12000009be0d6e836a5c5cceb1f913906a93'
assert data['runtime_package_sha256']=='7be4686d5117eaab9b4fb57da5ae345f2446b60dfd31292f8a5b931a2ce3b5b8'
assert data['review_sha256']=='f8b337ddabc225ec3e313e686d19efa5999a98fca419adc42113fb4059557d13'
assert data['config_sha256']=='71826d102a628eb9fa0869dfa9f360a71527f2c595d8467d7464f6b720f429f5'
assert not data['queue_failure'] and not data['worker_failed'] and 'STOPPED' in data['sglang']
for row in data['processes']:
    assert row['nice']==10 and row['affinity']==([106] if '/remaining.py run' in row['command'] else [105]) and row['ionice'].startswith('idle')
assert data['processes'] or data['queue_exit']
assert data['resources'], 'No actual worker resource receipt yet; do not call worker runtime verified'
for row in data['resources']:
    resource=row['resource']; guard=resource['main_guard_inputs']
    assert resource['CPU']==105 and resource['gpu_uuid']=='GPU-da357477-30a7-fddc-344b-a20513b9a2d0'
    assert guard['service']['returncode']==0 and not guard['queue_snapshot']['failed'] and 1<=len(guard['queue_snapshot']['active'])<=8
verified={'status':'ROOT_LINUX_SPAWN_AND_V2_RESOURCE_RECEIPTS_PASS','path':str(path),'sha256':hashlib.sha256(payload).hexdigest(),'service':data['service'],'processes':len(data['processes']),'actual_worker_resources':len(data['resources']),'worker_complete_exit_only':data['worker_complete_exit_only'],'remote_closed_n':data['remote_closed_n'],'accepted_new_n':0}
with path.with_suffix('.ROOT.json').open('x',encoding='utf8') as stream:json.dump(verified,stream,indent=2);stream.write('\n')
print(json.dumps(verified))
