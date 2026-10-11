"""Original500 resource guard adapted to exclusive CPU109/GPU1."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time
BASE=Path('/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt003')
HERE=Path(__file__).resolve().parent
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def pin(p):return {'path':str(p),'sha256':sha(p)}
def cmd(argv):return subprocess.run(argv,capture_output=True,text=True,check=True,timeout=20).stdout.strip()
def sources():
    need(HERE==BASE and sys.platform=='linux' and __debug__,'Exact original800 Linux caller required')
    need(sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa','Guide changed')
    seal=read(BASE/'DEPLOYMENT_SEAL.json')
    for rel,row in seal['files'].items():
        p=BASE/rel
        need(p.is_file() and not p.is_symlink() and sha(p)==row['sha256'],'Deployment source changed: '+rel)
    for name,h in seal['external_code_pins'].items():need(sha(name)==h,'Original dependency changed: '+name)
    return seal
def resources(a):
    need(os.getpriority(os.PRIO_PROCESS, 0) >= 10 and 'idle' in cmd(['ionice', '-p', str(os.getpid())]), 'nice10/idleIO required')
    need(set(os.sched_getaffinity(0)) == {109}, 'CPU109 only; never touch NGA CPU105')
    cpus = {t.name: sorted(os.sched_getaffinity(int(t.name))) for t in Path('/proc/self/task').iterdir()}
    need(all(v == [109] for v in cpus.values()), 'A thread escapedCPU109')
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
                need(not any(a['id'] in v for v in argv[1:]), 'Existing producer/evaluator for target checkpoint')
                need(not any('gpu_once' in v.lower() or 'mismatch_diagnostic' in v.lower() for v in argv[1:] + [process_cwd]), 'Concurrent diagnostic process')
                env = dict(v.split('=', 1) for v in (p / 'environ').read_text().split('\0') if '=' in v)
                declared = max([int(env[k]) for k in ('GUARDFED_CPU_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS') if env.get(k, '').isdigit()] or [1])
                nominal += declared
                tasks.append({'pid': int(p.name), 'argv': argv, 'declared_threads': declared})
            for t in (p / 'task').iterdir():
                affinity = set(os.sched_getaffinity(int(t.name)))
                if len(affinity) <= 16:
                    need(109 not in affinity, 'Another restricted task ownsCPU109')
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
    gpu = cmd(['nvidia-smi', '--id=1', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits']).split(',')
    need(gpu[0].strip() == a['gpu_uuid'] and int(gpu[1]) >= 2048, 'GPU0 identity/headroom changed')
    detail = cmd(['nvidia-smi', '--id=1', '-q'])
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
                thread_affinity=cpus, protected_services=services, no_duplicate=True, CPU109_free_for_this_process=True)

def resource_gate(snapshot,a):
    sources()
    free=os.statvfs(BASE);free=free.f_bavail*free.f_frsize
    need(free>=10*1024**3,'Less than10GiB output partition headroom')
    need(time.time()<a['deadline_unix'],'Expired root authorization')
    return dict(resources(a),disk_free_B=free)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);a=p.parse_args()
    q=read(BASE/'ROOT_QUEUE_AUTH.json')
    need(q['execution_authorized'] and a.id in q['ordered_ids'],'Target outside root authorized800')
    for tid in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(tid.name),[109])
    result=resource_gate(None,dict(q,id=a.id))
    result.update({'pass':True,'observed_unix':time.time(),'allowed_cpus':[109],
       'input_scope_sha256':q['exact_cohort']['sha256'],'adapter_sha256':q['adapter']['sha256']})
    print(json.dumps(result,allow_nan=False))
