"""Shared lifecycle/identity bridge; original scientific checker is retained."""
import hashlib,json,os,sys,time
from pathlib import Path
from identity import require,validate_job,validate_grid
HERE=Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write_json(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,p)
def local_identity():
    require(not sys.flags.optimize,'Optimized Python refused')
    require((HERE/'BINDINGS.json').is_file(),'Unbound source template: actual32/recipe/SHA are absent')
    b=read(HERE/'BINDINGS.json');require(b['status']=='BOUND_HYBRID100_NOT_EXECUTION_AUTHORITY','Prepared/null binding cannot execute')
    require(digest(HERE/'PREPARED_SOURCE_SEAL.json')==b['source_seal_sha256'],'Prepared source seal is missing or changed')
    for name,row in read(HERE/'PACKAGE_SHA256.json')['files'].items():require(digest(HERE/name)==row,'Bound stage changed: '+name)
    old=Path(b['original_screen']);require(digest(old/'FILES_SHA256.json')==b['old_screen_seal_sha256'],'Old Hybrid source changed')
    for name,sha in read(old/'FILES_SHA256.json')['files'].items():require(digest(old/name)==sha,'Old Hybrid member changed')
    for name,key in [('SELECTED_SUMMARY.json','actual_summary32_sha256'),('ROOT32_ADOPTION.json','actual_root32_sha256'),('BIND_APPROVAL.json','bind_approval_sha256')]:require(digest(HERE/name)==b[key],'Missing actual32 binding')
    protocol=read(HERE/'ORIGINAL_PROTOCOL.json');manifest=read(HERE/'manifest.json');candidate=b['selected_recipe']
    jobs=[read(HERE/e['job']) for e in manifest['jobs']];gates=[read(HERE/e['job']) for e in manifest['preflight_jobs']]
    for entry,job in zip(manifest['jobs']+manifest['preflight_jobs'],jobs+gates):
        require(digest(HERE/entry['job'])==entry['job_sha256'] and entry['id']==job['id'],'Job bytes changed');validate_job(job,protocol,candidate)
    validate_grid(jobs,gates,manifest['reused_jobs']);return protocol,manifest
def repo_identity(repo,protocol):
    scope=read(HERE/'full_scope.json')
    for name,sha in scope['protected_source_hashes'].items():require(digest(Path(repo)/name)==sha,'Repo/source/data changed: '+name)
    require(digest('/etc/vast-agents-guide.md')==scope['guide_sha256'],'Read new server guide before execution')
def authorized(scope,fresh):
    local_identity();a=read(HERE/'EXECUTION_AUTHORIZATION.json')
    require(sys.platform=='linux','Runtime execution is Linux-only')
    require(os.sched_getaffinity(0)=={104} and os.getpriority(os.PRIO_PROCESS,0)>=10,'Exact dedicated CPU104/nice10 lifecycle required')
    require(a['status']=='AUTHORIZED' and a['scope']==scope and a['package_sha256']==digest(HERE/'PACKAGE_SHA256.json'),'Exact external stage authorization required')
    require(a['max_workers']==a['cpu_threads_per_worker']==1 and a['allowed_cpus']==[104] and a['gpu_index']==0 and a['final_test'] is False and a['automatic_retry'] is False,'Original Hybrid single-slot only')
    require(digest(a['resource_receipt_path'])==a['resource_receipt_sha256'],'Actual resource proof missing')
    r=read(a['resource_receipt_path']);require(not fresh or 0<=time.time()-r['observed_unix']<=120,'Stale startup resource receipt')
    require(r['original_screen_exited'] and r['no_duplicate_workers'] and r['no_restricted_CPU_overlap'] and r['protected_main_healthy'] and r['protected_main_max_workers']==8,'Screen/protected queues or CPU allocation not safe')
    require(r['planned_total_cpu_threads']<=r['cpu_quota_cores'] and r['free_memory_bytes']>=8*1024**3 and r['gpu_free_memory_mib']>=4096 and r['gpu_recovery_action']=='None','Insufficient actual resources')
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    require(quota!='max' and int(quota)/int(period)==r['cpu_quota_cores'],'Actual cgroup quota differs from measured resource proof')
    if fresh:
        old=Path(read(HERE/'BINDINGS.json')['original_screen'])
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            if p.parent.name==str(os.getpid()):continue
            try:argv=p.read_bytes().decode(errors='replace').split('\0')
            except (FileNotFoundError,PermissionError):continue
            require(str(HERE/'run_one.py') not in argv and str(old/'driver.py') not in argv,'Existing old screen or same-stage worker must finish; no restart')
    if scope=='96_new_70round_valid_only':
        require(digest(HERE/'GATE_ACCEPTANCE.json')==a['gate_acceptance_sha256'],'Actual gate missing')
        require(digest(a['gate_root_closure_path'])==a['gate_root_closure_sha256'],'Actual independent gate/offserver root adoption missing')
        g=read(HERE/'GATE_ACCEPTANCE.json');root=read(a['gate_root_closure_path'])
        require(g['status']=='SEVEN_HYBRID_CANARIES_STRICT_PASS_BACKUP_PENDING' and len(g['accepted_ids'])==7 and len(g['pairs'])==2,'Seven real canaries required')
        require(root['status']=='ROOT_SEVEN_HYBRID_CANARIES_OFFSERVER_ADOPTED' and root['package_sha256']==digest(HERE/'PACKAGE_SHA256.json') and root['gate_sha256']==a['gate_acceptance_sha256'],'Root closure does not adopt this exact gate')
        for rel,sha in g['artifact_hashes'].items():require(digest(HERE/rel)==sha,'Gate artifact changed')
    else:require(scope=='seven_same_horizon_3round_canaries','Unknown execution scope')
    return a
def scoped_functions(kind,repo=None):
    from runtime import functions
    b=read(HERE/'BINDINGS.json');scope=read(HERE/('gate_scope.json' if kind=='canary' else 'full_scope.json'))
    scope=dict(scope,runtime_cuda_visible_device='0',runtime_gpu_uuid=b['gpu_uuid'])
    body,factory=functions(HERE,Path(repo or b['repo']),b)
    body.verify_scope=lambda supplied:repo_identity(Path(repo or b['repo']),read(HERE/'ORIGINAL_PROTOCOL.json'))
    run,checked,compare=factory(body,scope);return body,scope,run,checked,compare
def accepted(item,output):
    require(Path(output)==HERE/'runs'/item['id'],'Wrong result namespace')
    _,scope,_,checked,_=scoped_functions('fullcoverage')
    result=checked(item,scope)
    return None if result is None else dict(result,tuning_candidate=read(HERE/item['job'])['tuning_candidate'])
