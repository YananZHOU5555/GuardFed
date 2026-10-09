"""Bind actual coverage launch, preserved authorization and real two-GPU round evidence."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
BOUND=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009'
A=ROOT/'tmp/celeba_flgmm_fullcoverage_launch_operations_20261009/attempt_20261009T203311939299Z'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
runtime=BOUND/'RUNTIME_20261009T203448Z.json';assert sha(runtime)=='f90230c5eed46ad6c7d496e0cae8302733e443bd056e583d2181686755f7c126'
observed=read(runtime);assert len(observed['workers'])==2 and all(w['progress']['round']>=1 for w in observed['workers'])
start=read(A/'START.json');resource=read(A/'RESOURCE.json');new_auth=read(A/'NEW_EXECUTION_AUTHORIZATION.json')
assert start['fullcoverage_start_command_succeeded'] and not start['scientific_acceptance']
assert start['resource_sha256']==sha(A/'RESOURCE.json') and start['authorization_sha256']==sha(A/'NEW_EXECUTION_AUTHORIZATION.json')
assert sha(A/'PREVIOUS_CANARY_AUTHORIZATION.json')=='5dc144b4fc84ad132ef25e29275aca2f314e343ca33f590d42fa93501bc8496c'
assert new_auth['scope']=='96_new_70round_valid_only' and new_auth['max_workers']==2 and not new_auth['final_test']
assert resource['main_growth_observed'] and resource['protected_main_healthy'] and resource['gpu_recovery_actions']==['None','None']
assert resource['planned_total_cpu_threads']<=resource['cpu_quota_cores']
closure=BOUND/'ROOT_SEVEN_CANARY_CLOSURE.json';assert sha(closure)=='e17c50890129b512dbcf1f426fcf6392e47b1cb9db95a038a8a61b515ca8b958'
assert new_auth['root_closure_sha256']==sha(closure) and new_auth['gate_acceptance_sha256']==read(closure)['gate_sha256']
result=dict(status='ROOT_ACTUAL_FLGMM96_VALID_COVERAGE_STARTUP_AND_ROUNDS_VERIFIED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    startup_path=(A/'START.json').relative_to(ROOT).as_posix(),startup_sha256=sha(A/'START.json'),
    resource_path=(A/'RESOURCE.json').relative_to(ROOT).as_posix(),resource_sha256=sha(A/'RESOURCE.json'),
    runtime_path=runtime.relative_to(ROOT).as_posix(),runtime_sha256=sha(runtime),observed_utc=observed['utc'],
    closure_root_sha256=sha(closure),service=observed['service'],worker_pids=[w['pid'] for w in observed['workers']],
    active_rounds=[w['progress']['round'] for w in observed['workers']],active=2,pending=94,new_completed_observed=0,new_strict_offserver_accepted=0,
    planned_new=96,reused=4,planned_total=100,round_budget=70,evaluation_split='valid',
    source_package_sha256=start['package_sha256'],old_canary_authorization_preserved=True,other_services_changed=False,
    cpus=[102,103],threads_per_worker=1,nice=10,IO='idle',GPU_assignment=[0,1],formal100_started=True,final_test=False,
    limit='Startup and first measured rounds only. Four prior references are explicit;96 new results still require original strict acceptance and incremental offserver backup.')
out=BOUND/'ROOT_COVERAGE_STARTUP.json'
with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out))))
