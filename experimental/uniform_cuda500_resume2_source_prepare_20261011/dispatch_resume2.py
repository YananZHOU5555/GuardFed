"""Root caller for fixed500 CUDA replay; original science/strict loops imported.

No automatic recovery. prepare is metadata-only; queue is one sequential worker.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse,copy,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,time,traceback,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/guardfed_checks/uniform_cuda_formal_20261011_resume2')
MECH=Path('/workspace/guardfed_checks/mechanism_cuda_formal_20261011_resume2')
REPO=Path('/workspace/GuardFed-celeba-expanded')
PY='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
ID='UNIFORM_CUDA500_NO_TARGET_DURING_PREPARE'
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def load(n,p,h):
    need(sha(p)==h,'Source changed: '+str(p))
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
def cmd(argv):return subprocess.run(argv,capture_output=True,text=True,check=True,timeout=20).stdout.strip()
def sources():
    need(HERE==BASE and sys.platform=='linux' and __debug__,'Exact Linux caller required')
    need(sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa','Guide changed')
    seal=read(BASE/'DEPLOYMENT_SEAL.json')
    for name,row in seal['files'].items():
        p=BASE/name;need(p.is_file() and not p.is_symlink() and sha(p)==row['sha256'],'Deployment source changed: '+name)
    for name,h in seal['external_code_pins'].items():need(sha(name)==h,'Original constructor/dependency changed: '+name)
    return seal
def resource_gate(snapshot,a):
    sources()
    disk=os.statvfs(BASE);free=disk.f_bavail*disk.f_frsize
    need(free>=10*1024**3,'Less than10GiB free in formal output partition')
    return dict(resources(a),disk_free_B=free)

def resources(a):
    need(os.getpriority(os.PRIO_PROCESS, 0) >= 10 and 'idle' in cmd(['ionice', '-p', str(os.getpid())]), 'nice10/idleIO required')
    need(set(os.sched_getaffinity(0)) == {107}, 'CPU107 only; never touch NGA CPU105')
    cpus = {t.name: sorted(os.sched_getaffinity(int(t.name))) for t in Path('/proc/self/task').iterdir()}
    need(all(v == [107] for v in cpus.values()), 'A thread escapedCPU107')
    tasks = []; reserved = set(); nominal = 0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid(): continue
        try:
            if (p / 'stat').read_text().rsplit(')', 1)[1].split()[0] in ('Z', 'X'): continue
            argv = [v.decode(errors='replace') for v in (p / 'cmdline').read_bytes().split(b'\0') if v]
            if not argv: continue
            is_python = Path(argv[0]).name.startswith('python')
            process_cwd = str((p / 'cwd').resolve()) if is_python else ''
            is_project = is_python and (process_cwd.startswith(('/workspace/GuardFed-', '/workspace/guardfed_checks/'))
                or any('/workspace/GuardFed-' in v or '/workspace/guardfed_checks/' in v for v in argv[1:]))
            if is_project:
                need(not any(ID in v for v in argv[1:]), 'Existing producer/evaluator for target checkpoint')
                need(not any('gpu_once' in v.lower() or 'mismatch_diagnostic' in v.lower() for v in argv[1:] + [process_cwd]), 'Concurrent diagnostic process')
                env = dict(v.split('=', 1) for v in (p / 'environ').read_text().split('\0') if '=' in v)
                declared = max([int(env[k]) for k in ('GUARDFED_CPU_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS') if env.get(k, '').isdigit()] or [1])
                nominal += declared
                tasks.append({'pid': int(p.name), 'argv': argv, 'declared_threads': declared})
            for t in (p / 'task').iterdir():
                affinity = set(os.sched_getaffinity(int(t.name)))
                if len(affinity) <= 16:
                    need(107 not in affinity, 'Another restricted task ownsCPU107')
                    if is_project: reserved.update(affinity)
        except (FileNotFoundError, ProcessLookupError): continue
        except PermissionError:
            # Only a freshly verified vanished/dead process may leave this scan.
            try:
                state = (p / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError): continue
            if state in ('Z', 'X'): continue
            raise
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    need(quota != 'max', 'Finite actual cgroup quota required')
    cores = int(quota) / int(period)
    need(max(nominal, len(reserved)) + 1 <= cores, 'CPU budget exceeded')
    freeRAM = int(Path('/sys/fs/cgroup/memory.max').read_text()) - int(Path('/sys/fs/cgroup/memory.current').read_text())
    need(freeRAM >= 8 * 1024**3, 'Less than8GiB RAM headroom')
    gpu = cmd(['nvidia-smi', '--id=0', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits']).split(',')
    need(gpu[0].strip() == a['gpu_uuid'] and int(gpu[1]) >= 2048, 'GPU0 identity/headroom changed')
    detail = cmd(['nvidia-smi', '--id=0', '-q'])
    recovery = [v.split(':', 1)[1].strip() for v in detail.splitlines() if 'GPU Recovery Action' in v]
    need(recovery == ['None'], 'GPU recovery action is not None')
    services = {}
    for name, statuses in a['protected_services'].items():
        need(isinstance(statuses, list) and statuses and set(statuses) <= {'RUNNING', 'EXITED', 'STOPPED'}, 'Invalid protected service policy')
        p = subprocess.run(['supervisorctl', 'status', name], capture_output=True, text=True, timeout=15)
        words = p.stdout.split(); need(len(words) >= 2 and words[0] == name and words[1] in statuses, 'Protected service changed: ' + name)
        services[name] = p.stdout.strip()
    return dict(utc=datetime.now(timezone.utc).isoformat(), gpu_uuid=a['gpu_uuid'], GPU_free_MiB=int(gpu[1]), RAM_free_bytes=freeRAM,
                quota_cores=cores, reserved_cpu_count=len(reserved), nominal_project_threads=nominal, project_tasks=tasks,
                thread_affinity=cpus, protected_services=services, no_duplicate=True, CPU107_free_for_this_process=True)

def prepare():
    sources();need(not (BASE/'cohort.json').exists(),'Preserve prepared cohort')
    scope=read(BASE/'scope/EXACT500_INPUTS.json');pins=read(BASE/'scope/SOURCE_PINS.json')
    plan=read('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010/PLAN.json')
    need(sha(BASE/'author.json')==pins['author_decision']['sha256'],'Author decision changed')
    d=plan['remote_dependencies']
    for k,h in plan['dependency_sha256'].items():need(sha(d[k])==h,'Original dependency changed: '+k)
    parent=load('uniform_prepare_parent',d['parent_bridge'],plan['dependency_sha256']['parent_bridge'])
    ctor=load('uniform_original_constructor','/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010/bridge_adapter.py','e65157040ba1b12b7505947a7d1588c7e577782a78e0d06411794dcc0b6dc1e4')
    inspection=read(BASE/'native400_inspection.json');need(sha(BASE/'native400_inspection.json')==pins['native400_inspection']['sha256'],'Native400 authority changed')
    rows={r['id']:r for r in inspection['records']};baseline=read(d['baseline_inventory']);full={r['id']:r for r in baseline['records'] if r['method']=='GuardFed-AD2+'}
    entries={r['id']:r for r in plan['all_manifest_entries']};actual=[]
    projection=load('uniform_projection',BASE/'scope/mechanism_projection.py',sha(BASE/'scope/mechanism_projection.py'))
    target=read(BASE/'scope/TARGET900_PENDING.json')
    for r in scope['records']:
        if r['variant']=='Full':
            record=full[r['id']];need(canonical(record)==r['baseline_record']['canonical_sha256'],'Full original record changed')
        else:
            row=rows[r['id']];need(canonical(row)==r['native_strict_row']['canonical_sha256'],'Original400 accepted row changed')
            e=entries[r['id']];job=read(e['job']);result=read(Path(e['output'])/'result.json');control=full[r['paired_Full']['id']]
            for k,a in r['artifacts'].items():need(sha(a['server_path'])==a['sha256'] and row['files'][a['server_path']]==a['sha256'],'Native accepted artifact changed')
            record=ctor.construct_record(parent,job,e,result,row,control)
            projected=projection.project(parent,record,scope);inv=projection.inventory(plan,record,projected,target);projected.validate_inventory(inv,baseline)
        need(record['id']==r['id'] and record['training_torch']==r['training_torch'],'Materialized identity changed')
        actual.append(record)
    need([r['id'] for r in actual]==scope['ordered_ids'] and len(actual)==500,'Incomplete materialized500')
    author=read(BASE/'scope/AUTHOR_B.normalized.json');author['actual_user_decision_record']=pin(BASE/'author.json');save(BASE/'formal_author.json',author)
    cohort=dict(status='ROOT_FROZEN_FORMAL_CUDA_COHORT',decision='B',scope_kind='INDEPENDENT_COMPLETE_CUDA_COMPARISON',device_policy_outcome_independent=True,ordered_ids=scope['ordered_ids'],records=actual,closed_skip_ids=[],old_CPU_213_preserved=True,paired_Full_replay_required=True,available_phase=500,target_mechanism=900,uniform_CUDA_accepted=0,final_target_complete=False,source_seal=pin(BASE/'DEPLOYMENT_SEAL.json'))
    save(BASE/'cohort.json',cohort)
    save(BASE/'MATERIALIZATION.json',dict(status='ACTUAL500_ORIGINAL_METADATA_MATERIALIZED_NO_INFERENCE',cohort=pin(BASE/'cohort.json'),counts=scope['counts'],original_constructor=pin(Path(ctor.__file__).parent/'record_constructor.py'),new_CUDA_results=0))
    print(json.dumps({'status':'MATERIALIZED500_NO_INFERENCE','cohort_sha256':sha(BASE/'cohort.json')}))

def input_args(v3):
    b=v3.BASE
    a=types.SimpleNamespace(repo=REPO,inventory=Path('/workspace/guardfed_checks/server_reactivation_20261009/model_inventory.json'),storage_map=b/'v3/inputs/storage_map.json',restore_receipt=b/'v3/inputs/restore_bundle.json',restore_acceptance=b/'v3/inputs/restore_acceptance.json')
    for k in ['storage_map','restore_receipt','restore_acceptance']:setattr(a,k+'_sha256',sha(getattr(a,k)))
    return a

def worker(identity):
    global ID
    ID=identity;sources();need(os.sched_getaffinity(0)=={107},'One CPU107 required')
    def stop(_signum,_frame):raise TimeoutError('Formal CUDA worker interrupted; preserve evidence and stop without retry')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop)
    qauth=read(BASE/'ROOT_QUEUE_AUTH.json');need(qauth['execution_authorized'] and time.time()<qauth['deadline_unix'],'No current root authorization')
    need(qauth['cohort']==pin(BASE/'cohort.json') and qauth['deployment_seal']==pin(BASE/'DEPLOYMENT_SEAL.json'),'Queue source/cohort changed')
    cohort=read(BASE/'cohort.json');record=next(r for r in cohort['records'] if r['id']==identity)
    plan=read('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010/PLAN.json');d=plan['remote_dependencies']
    sys.path.insert(0,str(Path(d['v2']).parent));v2=load('replay',d['v2'],plan['dependency_sha256']['v2'])
    need(not v2.torch.cuda.is_initialized() and v2.torch.get_num_interop_threads()==1,'Bootstrap initialized CUDA or duplicate interop')
    os.environ.update(CUDA_VISIBLE_DEVICES=qauth['gpu_uuid'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    t=v2.torch;t.set_num_threads(1);t.use_deterministic_algorithms(True);t.backends.cudnn.deterministic=True;t.backends.cudnn.benchmark=False;t.backends.cudnn.allow_tf32=False;t.backends.cuda.matmul.allow_tf32=False
    for tid in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(tid.name),[107])
    isfull='variant' not in record
    out=(BASE if isfull else MECH)/'runs'/identity
    need(not out.exists() and not (out.parent/(out.name+'.formal_failure.json')).exists() and not (out.parent/(out.name+'.bridge_failure.json')).exists(),'Never overwrite or retry')
    task=BASE/'tasks'/identity;task.mkdir(parents=True,exist_ok=False)
    a=dict(execution_authorized=True,decision='B',author_decision_sha256=sha(BASE/'author.json'),ordered_input_ids=cohort['ordered_ids'],selected_ids=cohort['ordered_ids'],route='ORIGINAL_TRAINING_CUDA_FP32',scope_kind=cohort['scope_kind'],record_canonical_sha256=canonical(record),native_tolerance=1e-12,target_split='valid',device='cuda:0',compute_threads=1,max_processes=1,allowed_cpus=[107],gpu_uuid=qauth['gpu_uuid'],training=False,test=False,automatic_retry=False,diagnostic_reuse=False,deadline_unix=min(qauth['deadline_unix'],time.time()+3600),old_evidence_preserved=True,output_namespace=str(BASE if isfull else MECH),output=str(out),protected_services=qauth['protected_services'],resource_gate_source=pin(__file__))
    fresh=resource_gate(None,a)
    if isfull:
        full=load('uniform_full_adapter',BASE/'scope/full_adapter.py',sha(BASE/'scope/full_adapter.py'))
        pre=dict(fresh,pass_flag=True,observed_unix=time.time(),input_scope_sha256=sha(BASE/'scope/EXACT500_INPUTS.json'),adapter_sha256=sha(full.__file__),allowed_cpus=[107]);pre['pass']=pre.pop('pass_flag')
        save(task/'preflight.json',pre)
        a.update(id=identity,status='ROOT_AUTHORIZED_UNIFORM_CUDA_FULL_FORMAL_V1',exact500_inputs=pin(BASE/'scope/EXACT500_INPUTS.json'),author_decision=pin(BASE/'author.json'),adapter_sha256=sha(full.__file__),source_seal=pin(BASE/'scope/FILES_SHA256.json'),Linux_preflight=pin(task/'preflight.json'),formal_GPU_body=pin(BASE/'common/gpu_body.py'),receipt_version=full.SCOPE)
        save(task/'approval.json',a)
        v3=load('uniform_full_v3',d['v3'],plan['dependency_sha256']['v3']);runtime=full.bind_full(v3,input_args(v3),pin(task/'approval.json'),resource_gate)
        runtime['replay_one'](wall_seconds=1800);acceptance=runtime['accept_saved_predictions']()
    else:
        parent=load('uniform_parent',d['parent_bridge'],plan['dependency_sha256']['parent_bridge']);projection=load('uniform_projector',BASE/'scope/mechanism_projection.py',sha(BASE/'scope/mechanism_projection.py'))
        projected=projection.project(parent,record,read(BASE/'scope/EXACT500_INPUTS.json'))
        inv=projection.inventory(plan,record,projected,read(BASE/'scope/TARGET900_PENDING.json'));save(task/'inventory.json',inv)
        common=load('uniform_common',BASE/'common/adapter.py',sha(BASE/'common/adapter.py'))
        pre=dict(fresh,observed_unix=time.time(),cohort_sha256=sha(BASE/'cohort.json'),inventory_sha256=sha(task/'inventory.json'),adapter_sha256=sha(common.__file__),allowed_cpus=[107]);pre['pass']=True;save(task/'preflight.json',pre)
        a.update(status='ROOT_APPROVED_FORMAL_MECHANISM_CUDA_VERSION1',author_decision=pin(BASE/'formal_author.json'),cohort=pin(BASE/'cohort.json'),inventory=pin(task/'inventory.json'),adapter_sha256=sha(common.__file__),source_seal=pin(BASE/'common/FILES_SHA256.json'),Linux_preflight=pin(task/'preflight.json'),formal_receipt_version=common.SCOPE,cohort_inclusion_authorized=True)
        save(task/'approval.json',a);runtime=common.bind(projected,pin(task/'inventory.json'),pin(task/'approval.json'),d,REPO,identity,resource_gate)
        runtime['replay_one'](out,wall_seconds=1800);acceptance=runtime['accept_saved_predictions'](out)
    resource_gate(None,a);save(task/'ACCEPTED.json',dict(id=identity,cohort=pin(BASE/'cohort.json'),approval=pin(task/'approval.json'),acceptance=acceptance,receipt=pin(out/'receipt.json'),arrays=pin(out/'validation_predictions.npz'),root_offserver_adopted=False))
    print(json.dumps({'id':identity,'status':'ORIGINAL_LINUX_SAVED_ACCEPTED_REMOTE_ONLY','native':acceptance['native_comparison']}))

def queue():
    sources();need(not (BASE/'STARTED.json').exists() and not (BASE/'QUEUE_FAILURE.json').exists(),'No automatic queue retry')
    a=read(BASE/'ROOT_QUEUE_AUTH.json');need(a['execution_authorized'] and a['cohort']==pin(BASE/'cohort.json') and a['deployment_seal']==pin(BASE/'DEPLOYMENT_SEAL.json') and time.time()<a['deadline_unix'],'Actual root source/cohort authorization missing')
    lock=(BASE/'dispatch.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=read(BASE/'cohort.json');ids=c['ordered_ids'];canaries=['GuardFed-AD2+_IID_Benign_seed91001','minus_U_IID_Benign_seed91001','minus_C_IID_Benign_seed91001','minus_A_IID_Benign_seed91001','minus_F_IID_Benign_seed91001',None,'minus_F_non-IID_Sp-DFA_seed91004']
    # Sixth must be the exact matching Sp-DFA Full, derived by immutable cell rather than an assumed path.
    known=next(r for r in c['records'] if r['id']==canaries[-1]);canaries[-2]=known['paired_full']['id']
    order=canaries+[i for i in ids if i not in canaries];need(len(order)==len(set(order))==500 and set(order)==set(ids),'Execution order changed500')
    save(BASE/'STARTED.json',dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),auth=pin(BASE/'ROOT_QUEUE_AUTH.json'),execution_order=order,canaries=canaries,stage='canaries_then_remaining_exact500',automatic_retry=False))
    done=[]
    for identity in order:
        try:
            need(time.time()<a['deadline_unix'],'Root queue deadline expired')
            temp=BASE/'progress.tmp';temp.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),remote_saved_accepted_ids=done,active=identity,pending=499-len(done),failed=0,root_offserver_accepted=0),indent=2)+'\n');temp.replace(BASE/'progress.json')
            argv=['taskset','-c','107','nice','-n','10','ionice','-c','3',PY,str(__file__),'worker','--id',identity]
            with (BASE/'logs'/ (identity+'.log')).open('x') as log:
                p=subprocess.run(argv,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
            need(p.returncode==0,'Worker failed rc='+str(p.returncode))
            accepted=read(BASE/'tasks'/identity/'ACCEPTED.json');need(accepted['id']==identity and accepted['cohort']==pin(BASE/'cohort.json'),'Wrong saved acceptance')
            done.append(identity)
            progress=dict(utc=datetime.now(timezone.utc).isoformat(),remote_saved_accepted_ids=done,active=None,pending=500-len(done),failed=0,root_offserver_accepted=0)
            temp=BASE/'progress.tmp';temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(BASE/'progress.json')
        except BaseException as exc:
            save(BASE/'QUEUE_FAILURE.json',dict(utc=datetime.now(timezone.utc).isoformat(),id=identity,error=repr(exc),traceback=traceback.format_exc(),completed_ids=done,automatic_retry=False));raise
    save(BASE/'REMOTE_COMPLETE.json',dict(remote_saved_accepted=500,root_offserver_accepted=0,cohort=pin(BASE/'cohort.json'),training=False,test=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','queue','worker']);p.add_argument('--id');a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='queue':queue()
    else:worker(a.id)
