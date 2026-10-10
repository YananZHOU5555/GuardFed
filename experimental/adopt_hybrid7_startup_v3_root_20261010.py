"""Root adoption of actual launch and first measured round, not gate completion."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
START=ROOT/'tmp/celeba_hybrid_fullcoverage_canary_operations_20261010/v3/attempt_20261010T091540542410Z'
OBS=HERE/'CANARY_OBSERVATION_20261010T091730Z.json'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    measured=read(OBS);start=read(START/'START.json');resource=read(START/'RESOURCE.json')
    assert sha(OBS)=='3e87ab95e7eb275f3147edb883ccfa581ad59f90886d32174743148b0ad27ba5'
    assert start['status']=='START_COMMAND_SUCCEEDED_NOT_GATE_ACCEPTANCE'
    assert start['resource_sha256']==sha(START/'RESOURCE.json')
    assert start['package_sha256']==measured['package_sha256']==read(HERE/'ROOT_BOUND_ADOPTION.json')['package_sha256']
    assert start['authorization_sha256']==measured['authorization_sha256']
    assert resource['main_growth_observed'] and not resource['main_failure_files'] and not resource['main_recent_log_errors']
    assert resource['gpu_recovery_actions']==['None','None'] and resource['planned_total_cpu_threads']<=resource['cpu_quota_cores']
    assert not measured['failure_files'] and not measured['main_failures'] and measured['formal100_started'] is False
    assert len(measured['workers'])==len(measured['coordinators'])==1
    worker=measured['workers'][0]
    assert (worker['method'],worker['attack'],worker['seed'],worker['alpha'],worker['progress']['round'])==('CosineFairnessHybrid','Benign',91002,5.0,2)
    assert worker['nice']>=10 and all(x==[104] for x in worker['thread_affinity'].values())
    assert worker['environment']['CUDA_VISIBLE_DEVICES']=='0'
    assert all(worker['environment'][k]=='1' for k in ('GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    expected=read(HERE/'bound_metadata/stage/jobs'/(worker['id']+'.json'))
    assert expected['config']['rounds']==3 and expected['config']['celeba_evaluation_split']=='valid'
    assert sha(HERE/'bound_metadata/stage/jobs'/(worker['id']+'.json'))==worker['job_sha256']
    result=dict(status='ROOT_ACTUAL_HYBRID7_CANARY_START_AND_FIRST_ROUNDS_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        observed_utc=measured['checked_utc'],observation_path=OBS.relative_to(ROOT).as_posix(),observation_sha256=sha(OBS),
        start_receipt_path=(START/'START.json').relative_to(ROOT).as_posix(),start_receipt_sha256=sha(START/'START.json'),
        resource_sha256=sha(START/'RESOURCE.json'),package_sha256=start['package_sha256'],authorization_sha256=start['authorization_sha256'],
        actual_worker_id=worker['id'],actual_worker_pid=worker['pid'],actual_round=2,cpus=[104],gpu_visible_index=0,
        service=measured['service'],main_observed_completed=measured['main_completed'],main_actual_active=len(measured['main_active']),
        main_growth_verified=True,accepted_canaries=0,planned_canaries=7,scientific70_records=0,formal100_started=False,final_test=False,
        limits='Actual3round gate startup/identity/round2; not gate closure, physical-process GPU mapping, per-round checkpoint completeness or70-round performance.')
    with (HERE/'ROOT_CANARY_STARTUP.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(HERE/'ROOT_CANARY_STARTUP.json'),sha256=sha(HERE/'ROOT_CANARY_STARTUP.json'))))
if __name__=='__main__':main()
