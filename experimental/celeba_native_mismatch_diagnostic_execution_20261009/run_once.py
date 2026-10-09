"""Root-authorized one-off GPU diagnostic; frozen prepared18 remains unchanged."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,signal,subprocess,sys,traceback
HERE=Path(__file__).resolve().parent
EXPECTED=Path('/workspace/guardfed_checks/celeba_native_mismatch_diagnostic_execution_20261009')
PREP=HERE/'prepared'
REPO=Path('/workspace/GuardFed-celeba-expanded')
sys.dont_write_bytecode=True
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(x,m):
    if not x:raise ValueError(m)
def save(name,value):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def command(argv):return subprocess.run(argv,capture_output=True,text=True,check=True).stdout.strip()

def run():
    need(HERE==EXPECTED and sys.platform=='linux' and not sys.flags.optimize,'Exact isolated Linux path required')
    need(sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa','Guide changed')
    need(sha(PREP/'PACKAGE_SHA256.json')=='25d6c79f1d70b40f762bd9a372f1e9d2c2f64d8e9e82919c85f1667ce9a0bc61','Wrong prepared18')
    need(sha(PREP/'INPUTS.json')=='3f16a575836789f55e3f7bd48ee1923fe059b1af5921c2707a754583f3fc8c8e','Wrong input scope')
    for name,row in read(HERE/'EXECUTION_FILES_SHA256.json')['members'].items():
        f=HERE/name;need(sha(f)==row['sha256'] and f.stat().st_size==row['bytes'],'Execution source drift')
    for name,row in read(PREP/'PACKAGE_SHA256.json')['members'].items():
        f=PREP/name;need(sha(f)==row['sha256'] and f.stat().st_size==row['bytes'],'Prepared18 drift')
    for name in ['APPROVED.json','resource_receipt.json','runs','execution_started.json','failure.json','execution.lock']:
        need(not (HERE/name).exists(),'Prior evidence forbids retry: '+name)
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and 'idle' in command(['ionice','-p',str(os.getpid())]),'nice10/idleIO required')
    import fcntl
    lock=(HERE/'execution.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inputs=read(PREP/'INPUTS.json');record=read(PREP/'MODEL_RECORD.json')
    pins={Path(v['remote']):v['sha256'] for v in inputs['scientific_sources'].values()}
    pins.update({REPO/k:v for k,v in record['source_hashes'].items()})
    pins.update({Path(p):r['sha256'] for p,r in inputs['original_artifacts'].items()})
    pins[HERE/'original_valid_cache.npz']='39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
    checked={}
    for p,s in pins.items():
        need(sha(p)==s,'Original input changed: '+str(p));checked[str(p)]={'sha256':s,'bytes':p.stat().st_size,'resolved':str(p.resolve())}
    tasks=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            argv=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            if not argv or not Path(argv[0]).name.startswith('python'):continue
            need(str(HERE/'run_once.py') not in argv,'Duplicate diagnostic')
            if not any(a.startswith(('/workspace/GuardFed-','/workspace/guardfed_checks/')) and a.endswith('.py') for a in argv[1:]):continue
            cpus=sorted(os.sched_getaffinity(int(proc.name)))
            need(not(len(cpus)<=16 and 105 in cpus),'Another restricted project process ownsCPU105')
            env=dict(s.split('=',1) for s in (proc/'environ').read_text().split('\0') if '=' in s)
            declarations=[int(env[k]) for k in ['GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'] if env.get(k,'').isdigit()]
            tasks.append(dict(pid=int(proc.name),argv=argv,cpus=cpus,declared_upper_threads=max(declarations,default=1)))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();need(quota!='max','Finite actual CPU quota required')
    nominal=sum(x['declared_upper_threads'] for x in tasks);need(nominal+1<=int(quota)/int(period),'CPU budget exceeded')
    need(105 in os.sched_getaffinity(0),'CPU105 unavailable')
    gpu=command(['nvidia-smi','--id=0','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits']).split(',')
    uuid=gpu[0].strip();free=int(gpu[1].strip())*1024**2
    detail=command(['nvidia-smi','--id=0','-q']);recovery=[x.split(':',1)[1].strip() for x in detail.splitlines() if 'GPU Recovery Action' in x]
    need(recovery==['None'] and free>=2*1024**3,'GPU recovery/headroom failed')
    mem=int(Path('/sys/fs/cgroup/memory.current').read_text());maximum=int(Path('/sys/fs/cgroup/memory.max').read_text());need(maximum-mem>=8*1024**3,'RAM headroom failed')
    main=command(['supervisorctl','status','guardfed_celeba_mechanism_formal']);need('RUNNING' in main,'Protected formal service not healthy')
    queue=read(REPO/'results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json')
    need(not queue['failed'] and len(queue['active'])==8,'Protected formal queue not at expected healthy8')
    utc=datetime.now(timezone.utc).isoformat()
    resources=dict(utc=utc,no_duplicate_diagnostic=True,allowed_cpu105_free=True,gpu_recovery_action='None',gpu_uuid=uuid,
        nominal_threads_before=nominal,cpu_quota_cores=int(quota)/int(period),gpu_memory_free_bytes=free,memory_current=mem,memory_max=maximum,
        project_tasks=tasks,formal_service=main,formal_queue=queue,source_input_members=checked)
    save('resource_receipt.json',resources)
    approval=read(PREP/'APPROVAL_TEMPLATE.json');approval.update(status='APPROVED_EXACT_ONE_GPU_DIAGNOSTIC_ONLY',package_sha256=sha(PREP/'PACKAGE_SHA256.json'),
        gpu_uuid=uuid,resource_receipt=str(HERE/'resource_receipt.json'),resource_receipt_sha256=sha(HERE/'resource_receipt.json'),
        root_authorization_sha256=sha(HERE/'ROOT_AUTHORIZATION.json'),execution_source_seal_sha256=sha(HERE/'EXECUTION_FILES_SHA256.json'),
        authority='Explicit root task approval of prepared18/INPUTS exact one diagnostic',approved_utc=utc)
    save('APPROVED.json',approval);(HERE/'runs').mkdir(exist_ok=False)
    os.sched_setaffinity(0,{105})
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    def timeout(_s,_f):raise TimeoutError('One diagnostic1800sec limit; no automatic retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    sys.path.insert(0,str(Path(inputs['scientific_sources']['v2']['remote']).parent));import replay as v2
    need(sha(v2.__file__)==inputs['scientific_sources']['v2']['sha256'],'Foreign original replay')
    v2.torch.set_num_threads(1);v2.torch.set_num_interop_threads(1)
    v4=v2.load('single_diag_v4',Path(inputs['scientific_sources']['v4']['remote']))
    core=v2.load('single_diag_core',Path(inputs['scientific_sources']['core']['remote']))
    original=v2.load('single_diag_original',REPO/'scripts/run_revision_ablation.py')
    cnn=v2.load('single_diag_cnn',Path(inputs['scientific_sources']['cnn']['remote']))
    evaluator=v2.load('single_diag_evaluator',Path(inputs['scientific_sources']['evaluator']['remote']))
    adapter=v2.load('single_diag_adapter',PREP/'adapter.py')
    save('execution_started.json',dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),cpus=sorted(os.sched_getaffinity(0)),
        approval_sha256=sha(HERE/'APPROVED.json'),runtime_torch=v2.torch.__version__,gpu_uuid=uuid,source_seal_sha256=sha(HERE/'EXECUTION_FILES_SHA256.json')))
    report=adapter.diagnose(v4,core,original,cnn,evaluator,HERE/'APPROVED.json',sha(HERE/'APPROVED.json'))
    signal.alarm(0)
    compare=v2.load('single_diag_compare',PREP/'compare_saved.py')
    compared=compare.compare(Path(approval['output']),HERE/'original_valid_cache.npz',evaluator)
    save('comparison.json',compared)
    save('COMPLETED.json',dict(status='ONE_GPU_DIAGNOSTIC_COMPLETE_NOT_COHORT_ACCEPTED',utc=datetime.now(timezone.utc).isoformat(),id=record['id'],
        native_comparison=report['native_comparison'],comparison_sha256=sha(HERE/'comparison.json'),original_CPU_failure_still_invalid=True,
        new_training=0,test_inference=0,cohort_inclusion=False,retry_count=0))
    print(json.dumps({'status':'ONE_GPU_DIAGNOSTIC_COMPLETE_NOT_COHORT_ACCEPTED','native_comparison':report['native_comparison'],
        'flip_counts':{k:v['prediction_flip_count'] for k,v in compared['views'].items()},'margin_differences':compared['margins']}),flush=True)

if __name__=='__main__':
    try:run()
    except BaseException as error:
        signal.alarm(0)
        if not (HERE/'failure.json').exists():save('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
