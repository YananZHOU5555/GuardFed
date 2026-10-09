"""Prepared CUDA-only gate/screen lifecycle; externally frozen scope and approval required."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
import types
import time

HERE=Path(__file__).resolve().parent
CORE_SHA='cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed'
OLD_GATE_SHA='22d6a02b49eb229d46f70bf259a7a3972d34bd392c7ecb28645d9909ba731b40'
KINDS={'gate':('gate_scope.json','FROZEN_BOUNDED_HYBRID_CUDA_GATE_ONLY','APPROVED_FOUR_HYBRID_CUDA_CANARIES_ONLY'),
       'screen':('screen_scope.json','FROZEN_HYBRID_VALID_SCREEN_ONLY','APPROVED_32_HYBRID_VALID_SCREEN_ONLY')}
ARTIFACTS={'result.json','model.pt','diagnostics.json','provenance.json','resource_before.json','resource_after.json','rng_final.json','native_replay.json'}
def require(ok,message):
    if not ok:raise ValueError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def clone(function,**overrides):
    return types.FunctionType(function.__code__,dict(function.__globals__,**overrides),function.__name__,function.__defaults__,function.__closure__)
def finite(item,path=()):
    if isinstance(item,dict):
        for key,value in item.items():finite(value,path+(key,))
    elif isinstance(item,(list,tuple)):
        for i,value in enumerate(item):finite(value,path+(i,))
    elif isinstance(item,float):require(math.isfinite(item),'Unknown nonfinite field refused: '+repr(path))

def validate_entry(entry,job,scope,protocol,worker):
    require(entry['id']==job['id'] and digest(HERE/entry['job'])==entry['job_sha256'],'Job bytes/ID mismatch')
    require(job['runtime_protocol_sha256']==digest(HERE/scope['runtime_protocol']),'Runtime protocol identity changed')
    if scope['kind']=='validation_screen':
        worker.validate_job({k:v for k,v in job.items() if k!='runtime_protocol_sha256'},protocol)
        require(job['method']=='CosineFairnessHybrid' and job['config']['rounds']==70,'Only exact32 method screen jobs')
    else:
        require(job==scope['exact_gate_jobs'][entry['id']] and job['method'] in {'CosineFairnessHybrid','GuardFed'}
                and job['config']['rounds']==3,'Foreign gate method/recipe/horizon')
    require(job['config']['device']=='cuda' and job['config']['celeba_evaluation_split']=='valid'
        and not job['config']['celeba_train_limit'] and not job['config']['celeba_eval_limit'],'Only full-image valid CUDA')

def approve(kind,approval,expected_sha,dispatch=True):
    filename,status,approved_status=KINDS[kind];scope=read(HERE/filename)
    require(scope['status']==status,'PREPARED scope is not frozen; no execution authorized')
    require(digest(approval)==expected_sha,'External approval SHA mismatch');approved=read(approval)
    require(approved['status']==approved_status and approved['scope_sha256']==digest(HERE/filename)
        and approved['execution_seal_sha256']==digest(HERE/'FILES_SHA256.json'),'External exact frozen scope/seal approval required')
    require(approved['selected_ids']==[e['id'] for e in scope['jobs']] and approved['max_processes']==1
        and approved['cpu_threads']==1 and approved['test_authorized'] is False and approved['automatic_retry_authorized'] is False,'Scope/concurrency/test/retry mismatch')
    require(len(approved['allowed_cpus'])==1 and approved['nice']>=10 and approved['cuda_visible_device'] in {'0','1'},'Explicit CPU1/GPU1/nice10 allocation required')
    for name in ('cpu_gate_delivery','cpu_gate_offserver_verification','resource_preflight'):
        artifact=approved[name];require(digest(artifact['path'])==artifact['sha256'],'Required source-bound prerequisite changed: '+name)
    cpu=read(approved['cpu_gate_delivery']['path']);backup=read(approved['cpu_gate_offserver_verification']['path'])
    require(cpu['new_CANARY']==2 and cpu['reused_CANARY']==2 and cpu['scientific_table_records']==0
        and cpu['source9_seal_sha256']=='54c67847b44a26a654b380c6c4d03863279261126a61de29489bfa78572e332b'
        and len(cpu['new_pairs'])==len(cpu['reused_pairs'])==1
        and all(row['all_metrics_model_attacks_diagnostics_rng_exact'] for row in cpu['new_pairs']+cpu['reused_pairs'])
        and backup['pass'] and backup['different_host_observed']
        and backup['accepted_new_ids']==[row['id'] for row in cpu['records']],'All four CPU canaries and actual offserver proof required')
    resource=read(approved['resource_preflight']['path'])
    require((not dispatch or 0<=time.time()-resource['at_unix']<=90) and resource['cpu_allocation']==approved['allowed_cpus']
        and resource['cuda_visible_device']==approved['cuda_visible_device'] and resource['gpu_uuid']==approved['gpu_uuid']
        and resource['no_duplicate_worker'] and resource['no_restricted_CPU_overlap']
        and resource['gpu_free_memory_mib']>=4096
        and resource['existing_nominal_compute_threads']+1<=resource['actual_quota_cores'],'Fresh exact-resource proof required')
    if kind=='screen':
        for name in ('cuda_gate_delivery','cuda_gate_offserver_verification'):
            artifact=approved[name];require(digest(artifact['path'])==artifact['sha256'],'CUDA prerequisite changed')
        cuda=read(approved['cuda_gate_delivery']['path']);cuda_backup=read(approved['cuda_gate_offserver_verification']['path'])
        require(cuda['status']=='FOUR_CUDA_CANARIES_STRICT_ACCEPTED_OFFSERVER'
            and cuda['accepted_ids']==[e['id'] for e in read(HERE/'gate_scope.json')['jobs']]
            and cuda['core_sha256']==CORE_SHA and cuda['body_sha256']==digest(HERE/'body.py')
            and cuda['adapter_sha256']==digest(HERE/'scientific_snapshot/adapters.py')
            and cuda['scientific_worker_sha256']==digest(HERE/'scientific_snapshot/worker.py')
            and cuda['writer_policy_sha256']==digest(HERE/'writer_policy.py')
            and cuda['all_pairs_exact'] is True and cuda_backup['accepted_new_ids']==cuda['accepted_ids']
            and cuda_backup['pass'] and cuda_backup['different_host_observed'],'Four actual CUDA canaries must close before32 screen')
    for name,sha in read(HERE/'FILES_SHA256.json')['files'].items():require(digest(HERE/name)==sha,'Frozen package changed: '+name)
    return scope,approved

def functions(body,scope):
    from writer_policy import sanitize_result,check_sidecar
    entries={entry['id']:entry for entry in scope['jobs']}
    def checked(entry,unused):
        result=body.checked(entry,scope)
        if result is None:return None
        output=HERE/entry['output'];job=read(HERE/entry['job']);receipt=read(output/'acceptance.json');sidecar=read(output/'undefined_diagnostics.json')
        require(set(receipt['artifact_hashes'])==ARTIFACTS|{'undefined_diagnostics.json'},'Incomplete original artifacts/mandatory sidecar')
        require(sidecar['sanitized_result_sha256']==digest(output/'result.json') and sidecar['body_sha256']==digest(HERE/'body.py')
            and sidecar['writer_policy_sha256']==digest(HERE/'writer_policy.py') and sidecar['scope_sha256']==digest(HERE/scope['scope_file']),'Writer/result/source binding failed')
        if job['attack']=='S-DFA':check_sidecar(result,sidecar,job,entry['job_sha256'],OLD_GATE_SHA,CORE_SHA)
        else:
            require(job['attack']=='Benign' and sidecar['undefined_values']==[] and sidecar['job_sha256']==entry['job_sha256'],'Benign must have zero undefined replacements');finite(result)
        return result
    def write(path,value):
        path=Path(path)
        for entry in entries.values():
            output=HERE/entry['output']
            if path==output/'result.json':
                job=read(HERE/entry['job'])
                if job['attack']=='S-DFA':clean,rows=sanitize_result(value,job,digest(HERE/entry['job']),entry['job_sha256'])
                else:finite(value);clean,rows=value,[]
                body.write(path,clean)
                body.write(output/'undefined_diagnostics.json',dict(status='EXPLICIT_UNDEFINED_DIAGNOSTIC_ONLY',job_sha256=entry['job_sha256'],
                    original_gate_sha256=OLD_GATE_SHA,original_core_sha256=CORE_SHA,body_sha256=digest(HERE/'body.py'),writer_policy_sha256=digest(HERE/'writer_policy.py'),
                    scope_sha256=digest(HERE/scope['scope_file']),sanitized_result_sha256=digest(path),undefined_values=rows));return
            if path==output/'acceptance.json':
                require(set(value['artifact_hashes'])==ARTIFACTS,'Original complete artifact contract changed')
                value=dict(value,artifact_hashes=dict(value['artifact_hashes'],**{'undefined_diagnostics.json':digest(output/'undefined_diagnostics.json')}))
                break
        finite(value);body.write(path,value)
    return clone(body.run_one,checked=checked,write=write),checked,clone(body.compare,checked=checked)

def run(kind,approval,expected_sha):
    scope,approved=approve(kind,approval,expected_sha)
    require(sys.platform=='linux' and not sys.flags.optimize,'Linux without -O required')
    require(digest('/etc/vast-agents-guide.md')==scope['guide_sha256'],'Read updated guide before dispatch')
    require(os.getpriority(os.PRIO_PROCESS,0)>=approved['nice'] and set(approved['allowed_cpus'])<=os.sched_getaffinity(0),'CPU/nice allocation unavailable')
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    require(quota!='max' and read(approved['resource_preflight']['path'])['actual_quota_cores']==int(quota)/int(period),'Actual cgroup quota changed')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if not argv or 'python' not in Path(argv[0]).name:continue
            cpus=os.sched_getaffinity(int(proc.name))
            require(not (len(cpus)<=16 and set(approved['allowed_cpus']).intersection(cpus)),'A restricted Python worker occupies proposed CPU')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    import fcntl
    lock=(HERE/'cuda_execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (HERE/scope['run_parent']).exists() and not (HERE/(kind+'_failure.json')).exists(),'Partial outputs/failure are preserved; no retry')
    os.sched_setaffinity(0,approved['allowed_cpus'])
    os.environ.update(CUDA_VISIBLE_DEVICES=approved['cuda_visible_device'],CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    import body
    body.verify_scope(scope)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8' and torch.cuda.device_count()==1,'Expected isolated cu128 single GPU runtime')
    import subprocess
    uuid=subprocess.check_output(['nvidia-smi','--id='+approved['cuda_visible_device'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    require(uuid==approved['gpu_uuid'],'Approved physical GPU identity changed')
    scope=dict(scope,runtime_cuda_visible_device=approved['cuda_visible_device'],runtime_gpu_uuid=uuid)
    worker,core=body.modules();protocol=read(HERE/scope['runtime_protocol'])
    for entry in scope['jobs']:validate_entry(entry,read(HERE/entry['job']),scope,protocol,worker)
    run_one,checked,compare=functions(body,scope)
    (HERE/scope['run_parent']).mkdir();before=body.snapshot();body.write(HERE/(kind+'_dispatch.json'),dict(approval=approved,approval_sha256=expected_sha,resources_before=before))
    accepted=[]
    try:
        for entry in scope['jobs']:
            run_one(entry,scope,worker,core);require(checked(entry,scope) is not None,'Unaccepted terminal')
            accepted.append(entry['id'])
        pairs=compare(scope) if kind=='gate' else []
        body.verify_scope(scope);after=body.snapshot()
        previous={r['id']:r['round'] for r in before['active']}
        grew=after['completed']>before['completed'] or any(r['id'] in previous and r['round'] is not None and previous[r['id']] is not None and r['round']>previous[r['id']] for r in after['active'])
        require(not after['failed'] and grew,'Protected formal queue did not advance or failed')
        body.write(HERE/(kind+'_complete.json'),dict(status='CUDA_GATE_STRICT_ACCEPTED_BACKUP_PENDING' if kind=='gate' else '32_VALID_SCREEN_STRICT_ACCEPTED_BACKUP_PENDING',
            accepted_ids=accepted,pairs=pairs,source_seal_sha256=digest(HERE/'FILES_SHA256.json'),resources_after=after,
            test_evaluated=False,CPU_CUDA_equivalence_claim=False,formal_multi_seed_records=0))
    except BaseException as error:
        body.write(HERE/(kind+'_failure.json'),dict(error=repr(error),traceback=traceback.format_exc(),accepted_ids=accepted,automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inspect','run']);p.add_argument('--kind',choices=KINDS,required=True);p.add_argument('--approved',type=Path);p.add_argument('--approved-sha256');a=p.parse_args()
    if a.action=='inspect':print(read(HERE/KINDS[a.kind][0])['status'],'no CUDA import, training or inference')
    else:run(a.kind,a.approved,a.approved_sha256)
