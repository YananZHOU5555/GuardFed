"""Adopt actual frozen Hybrid96 launch and first worker round, not performance."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
START=ROOT/'tmp/celeba_hybrid_fullcoverage_launch_operations_20261010/v3/attempt_20261010T094057685607Z'
OBS=HERE/'FORMAL_OBSERVATION_20261010T094402Z.json'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert sha(OBS)=='72f6b52e71be3819041f9234bf63a2d3fd510dba5d0d00f7eec04809a953a2f7'
    measured=read(OBS);start=read(START/'START.json');resource=read(START/'RESOURCE.json')
    closure=read(HERE/'ROOT_SEVEN_CANARY_CLOSURE.json');auth=read(START/'NEW_EXECUTION_AUTHORIZATION.json')
    assert start['status']=='START_COMMAND_SUCCEEDED_NOT_SCIENTIFIC_ACCEPTANCE' and start['resource_sha256']==sha(START/'RESOURCE.json')
    assert sha(START/'APPROVAL.json')=='85c0ba22167d0ec2cc2463fe033af2afbeade68656e44a8024c2be31dbbd4698'
    assert sha(START/'PREVIOUS_CANARY_AUTHORIZATION.json')==closure['canary_authorization_sha256']
    assert sha(START/'NEW_EXECUTION_AUTHORIZATION.json')==start['authorization_sha256']==measured['authorization_sha256']
    assert auth==measured['new_execution_authorization'] and auth['scope']=='96_new_70round_valid_only'
    assert auth['gate_root_closure_sha256']==sha(HERE/'ROOT_SEVEN_CANARY_CLOSURE.json') and auth['root_approval_sha256']==sha(START/'APPROVAL.json')
    assert start['package_sha256']==measured['package_sha256']==closure['package_sha256']
    assert resource['main_growth_observed'] and not resource['main_failure_files'] and not resource['main_recent_log_errors']
    assert resource['gpu_recovery_actions']==['None','None'] and resource['planned_total_cpu_threads']<=resource['cpu_quota_cores']
    assert not measured['failure_files'] and not measured['main_failures'] and not measured['stderr_tail']
    assert measured['formal100_started'] is True and measured['service']['stdout'].split()[1]=='RUNNING'
    assert len(measured['workers'])==len(measured['coordinators'])==1
    worker=measured['workers'][0];prov=worker['provenance'];progress=worker['progress']
    assert (worker['method'],worker['attack'],worker['seed'],worker['alpha'],progress['round'])==('CosineFairnessHybrid','Benign',91002,5000.,1)
    assert worker['nice']>=10 and all(x==[104] for x in worker['thread_affinity'].values())
    assert worker['environment']['CUDA_VISIBLE_DEVICES']=='0' and all(worker['environment'][k]=='1' for k in ('GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    job=HERE/'bound_metadata/stage/jobs'/(worker['id']+'.json');expected=read(job)
    assert expected['config']['rounds']==70 and expected['config']['celeba_evaluation_split']=='valid' and sha(job)==worker['job_sha256']==prov['job_sha256']
    package=read(HERE/'bound_metadata/stage/PACKAGE_SHA256.json')['files']
    for name,pin in prov['local_hashes'].items():assert package[name]==pin
    assert prov['scope_sha256']==sha(HERE/'bound_metadata/stage/full_scope.json')
    assert prov['source_hashes']==read(HERE/'bound_metadata/stage/full_scope.json')['protected_source_hashes']
    assert prov['pid']==worker['pid']==progress['pid'] and prov['gpu_uuid']=='GPU-da357477-30a7-fddc-344b-a20513b9a2d0'
    assert prov['torch']=='2.11.0+cu128' and prov['cuda_build']=='12.8' and prov['cpu_affinity']==[104] and prov['cpu_threads']==1
    assert measured['physical_gpu_processes']['returncode']==0
    rows=[line.split(',') for line in measured['physical_gpu_processes']['stdout'].splitlines()]
    assert [(int(row[0].strip()),row[1].strip()) for row in rows if int(row[0].strip())==worker['pid']]==[(worker['pid'],prov['gpu_uuid'])]
    queue=measured['queue_progress'];assert queue['completed_new']==0 and queue['reused']==4 and queue['pending']==95 and queue['failed'] is False
    assert queue['active']==[dict(id=worker['id'],pid=worker['pid'],gpu=0)]
    result=dict(status='ROOT_ACTUAL_HYBRID96_VALID_COVERAGE_STARTUP_AND_ROUNDS_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        observed_utc=measured['checked_utc'],observation_path=OBS.relative_to(ROOT).as_posix(),observation_sha256=sha(OBS),
        start_receipt_path=(START/'START.json').relative_to(ROOT).as_posix(),start_receipt_sha256=sha(START/'START.json'),
        resource_sha256=sha(START/'RESOURCE.json'),package_sha256=start['package_sha256'],authorization_sha256=start['authorization_sha256'],
        root_approval_sha256=sha(START/'APPROVAL.json'),canary_closure_sha256=sha(HERE/'ROOT_SEVEN_CANARY_CLOSURE.json'),
        actual_worker_id=worker['id'],actual_worker_pid=worker['pid'],actual_round=1,cpus=[104],gpu_visible_index=0,physical_gpu_uuid=prov['gpu_uuid'],
        service=measured['service'],main_observed_completed=measured['main_completed'],main_actual_active=len(measured['main_active']),
        main_growth_verified=True,new_accepted=0,planned_new=96,reused=4,planned_total=100,canaries_adopted=7,
        scientific70_records=0,formal100_started=True,final_test=False,automatic_retry=False,
        limits='Actual first round and physical process GPU mapping, not a70-round result or universal equivalence. Initial constant-negative round is retained, not a chosen performance result.')
    with (HERE/'ROOT_COVERAGE_STARTUP.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(HERE/'ROOT_COVERAGE_STARTUP.json'),sha256=sha(HERE/'ROOT_COVERAGE_STARTUP.json'))))
if __name__=='__main__':main()
