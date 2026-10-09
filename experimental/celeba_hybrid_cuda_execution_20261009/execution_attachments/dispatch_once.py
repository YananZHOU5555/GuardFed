"""Exact authorized gate4 deployment attachment; no scientific implementation changes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
REPO = Path('/workspace/GuardFed-celeba-expanded')
CHECKS = Path('/workspace/guardfed_checks')
SERVICE = 'guardfed_celeba_hybrid_cuda_four'
ROOT_SHA = '4b9c9b9e1523f0e8d7a28cea7d465c3bc1ce363202e5c0708eec4deea74f8b4b'
GPU_UUID = 'GPU-da357477-30a7-fddc-344b-a20513b9a2d0'
SOURCES = {
    str(REPO/'deployment/celeba_mechanism_20261009/worker.py'): 'e9a5449e0148ad4043dda5eb25c596b0912d6249af6149b69b861385e9668577',
    str(CHECKS/'celeba_final_valid_replay_20261009/v4/replay_v4.py'): '43b16d20d2497b7762cd0f6039f7f4bfc8fddd4e4c32979a16291b26e394ae5e',
    str(CHECKS/'celeba_final_valid_replay_20261009/v4/remaining872_prepared_v2_20261009/bounded_remaining.py'): 'c06d145a1c5b97014bc5d6e7a25c53da1020ec298cae9285f0d8d02da1cd4d00',
    str(CHECKS/'celeba_flgmm_screen_20261009/release_v2/run_one.py'): 'e55fbfc78c82617e8226e700597f9b76eb48ab18006fd9fd11ce8be5433a48c9',
    str(CHECKS/'celeba_mechanism_valid_incremental_v2_execution_20261009/batch.py'): '4237636c524314ff0da34bc6598ecd279dd269972baf9e934a8083b15e88ce7b',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def entry(argv, cwd):
    if not argv or not Path(argv[0]).name.startswith('python'):
        return None
    i = 1
    while i < len(argv) and argv[i].startswith('-'):
        if argv[i] in ('-c', '-m', '-'):
            return None
        i += 2 if argv[i] in ('-W', '-X') else 1
    if i >= len(argv):
        return None
    script = Path(argv[i])
    return str((script if script.is_absolute() else Path(cwd)/script).resolve()), argv[i+1:]


def resource_snapshot():
    tracked, lightweight, unknown, occupied = [], [], [], set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            cwd = str((proc/'cwd').resolve())
            item = entry(argv, cwd)
            if item is None:
                continue
            script, args = item
            cpus = sorted(os.sched_getaffinity(int(proc.name)))
            require(not (len(cpus) <= 16 and 104 in cpus), 'Restricted Python process occupies CPU104')
            require(not (script == str(HERE/'driver.py') and args[:1] == ['run']), 'Duplicate CUDA gate worker')
            if script not in SOURCES:
                if script.startswith(str(CHECKS)):
                    unknown.append(dict(pid=int(proc.name), script=script, action=args[:1], cpus=cpus if len(cpus)<=16 else dict(count=len(cpus))))
                continue
            require(sha(script) == SOURCES[script], 'Known live worker source identity changed')
            env = dict(x.decode(errors='replace').split('=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
            threads = {k:env.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','GUARDFED_CPU_THREADS')}
            if script.endswith('celeba_mechanism_20261009/worker.py'):
                require('--repo' in args and args[args.index('--repo')+1] == str(REPO) and '--job' in args, 'Foreign formal worker')
                role, count = 'formal_GPU_worker', 1
                require(all(threads[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','GUARDFED_CPU_THREADS')), 'Formal thread declaration changed')
            elif script.endswith('release_v2/run_one.py'):
                require(cwd == str(Path(script).parent) and args[:2] == ['--repo',str(REPO)] and '--job-id' in args, 'Foreign FLGMM worker')
                require(all(threads[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','GUARDFED_CPU_THREADS')), 'FLGMM thread declaration changed')
                role, count = 'FLGMM_GPU_worker', 1
            elif script.endswith('/batch.py'):
                require(cwd == str(Path(script).parent) and args[:1] in (['manage'],['worker']), 'Foreign mechanism replay action')
                approval = Path(args[args.index('--approved')+1]); expected = args[args.index('--approved-sha256')+1]
                require(expected == 'ae5a289fffa9d89badfd152f1c0105a764f21e69afa4d011ee5886fe2e210313' and sha(approval)==expected, 'Mechanism replay approval changed')
                require(cpus == list(range(112,120)) and threads['OMP_NUM_THREADS']==threads['MKL_NUM_THREADS']=='8' and threads['OPENBLAS_NUM_THREADS']=='1', 'Mechanism replay CPU allocation changed')
                role, count = ('mechanism_valid_worker',8) if args[0]=='worker' else ('mechanism_replay_coordinator',0)
            elif script.endswith('/bounded_remaining.py'):
                require(args[:1]==['run'] and '--manifest-sha256' in args and args[args.index('--manifest-sha256')+1]=='ad6eebf517f534fb8489acb241c51a9ec5328bb285406e55275f7dd9c0c3ed43', 'Baseline bounded manager identity changed')
                role, count = 'baseline_replay_coordinator', 0
            else:
                require(script.endswith('/replay_v4.py'), 'Unknown known-source role')
                if args[:1]==['worker']:
                    require(len(cpus)==8 and set(cpus)<=set(range(16,104)), 'Baseline worker allocation changed')
                    role, count = 'baseline_valid_worker', 8
                else:
                    require(args[:1] in (['run'],['accept'],['verify']), 'Unknown baseline replay action')
                    role, count = 'baseline_replay_helper_or_manager', 0
            if count==8:
                require(not occupied.intersection(cpus), 'Independent CPU worker allocations overlap')
                occupied.update(cpus)
            row=dict(pid=int(proc.name), role=role, compute_threads=count, argv=argv, cwd=cwd, cpus=cpus if len(cpus)<=16 else dict(count=len(cpus)), source_sha256=SOURCES[script], threads=threads)
            (tracked if count else lightweight).append(row)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    formal=sum(r['role']=='formal_GPU_worker' for r in tracked)
    fl=sum(r['role']=='FLGMM_GPU_worker' for r in tracked)
    baseline=sum(r['role']=='baseline_valid_worker' for r in tracked)
    mechanism=sum(r['role']=='mechanism_valid_worker' for r in tracked)
    require(formal<=8 and fl<=2 and baseline<=11 and mechanism<=1, 'Authorized queue concurrency exceeded')
    # Managers may be between chunks. Reserve their full existing queue maximum,
    # and keep the two already authorized GPU queues' maxima rather than treating
    # momentary gaps as released resources. Coordinator overhead is separate.
    baseline_reserved=88 if baseline or any(r['role']=='baseline_replay_coordinator' for r in lightweight) else 0
    mechanism_reserved=8 if mechanism or any(r['role']=='mechanism_replay_coordinator' for r in lightweight) else 0
    nominal=8+2+baseline_reserved+mechanism_reserved
    q,p=Path('/sys/fs/cgroup/cpu.max').read_text().split();require(q!='max','Finite actual CPU quota required');quota=int(q)/int(p)
    require(nominal+1<=quota,'CPU quota insufficient')
    gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid,memory.free,temperature.gpu,driver_version,gpu_recovery_action','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    require(gpu[0]==GPU_UUID and int(gpu[1])>=4096 and gpu[4]=='None','GPU identity/headroom/recovery failed')
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:')))
    disk=shutil.disk_usage(REPO)
    return dict(at_unix=time.time(),cpu_allocation=[104],cuda_visible_device='0',gpu_uuid=gpu[0],gpu_free_memory_mib=int(gpu[1]),gpu_temperature_c=int(gpu[2]),driver_version=gpu[3],gpu_recovery_action=gpu[4],no_duplicate_worker=True,no_restricted_CPU_overlap=True,existing_nominal_compute_threads=nominal,actual_quota_cores=quota,actual_tracked_compute_threads=sum(r['compute_threads'] for r in tracked),reserved_peak_not_measured_CPU_utilization=True,tracked_compute=tracked,lightweight_coordinators_and_helpers=lightweight,other_check_processes=unknown,memory=mem,disk_free_bytes=disk.free)


def main():
    require(HERE == CHECKS/'celeba_hybrid_cuda_execution_20261009','Wrong execution directory')
    require(sha(HERE/'execution_attachments/ROOT_APPROVED.json')==ROOT_SHA,'Root authorization changed')
    root=read(HERE/'execution_attachments/ROOT_APPROVED.json')
    require(root['execution_authorized_within_existing_user_request'] and not root['screen32_authorized'] and root['allowed_cpus']==[104], 'Root exact scope changed')
    for name in ('gate_runs','screen_runs','gate_failure.json','gate_dispatch.json','gate_complete.json','APPROVED_gate.json','APPROVED_gate.sha256'):
        require(not (HERE/name).exists(),'Existing runtime evidence forbids dispatch: '+name)
    script=Path('/opt/supervisor-scripts')/(SERVICE+'.sh');conf=Path('/etc/supervisor/conf.d')/(SERVICE+'.conf')
    require(not script.exists() and not conf.exists(),'Service already installed')
    sys.path.insert(0,str(HERE))
    import body, driver
    scope=read(HERE/'gate_scope.json');body.verify_scope(scope)
    require(read(HERE/'screen_scope.json')['status']=='PREPARED_NOT_FROZEN','Screen must remain prepared')
    for name,expected in read(HERE/'FILES_SHA256.json')['files'].items():
        require(sha(HERE/name)==expected,'Sealed execution package changed: '+name)
    resource=resource_snapshot();resource_path=HERE/'execution_attachments/resource_preflight.json';save(resource_path,resource)
    approval=dict(status='APPROVED_FOUR_HYBRID_CUDA_CANARIES_ONLY',scope_sha256=sha(HERE/'gate_scope.json'),execution_seal_sha256=sha(HERE/'FILES_SHA256.json'),selected_ids=[e['id'] for e in scope['jobs']],max_processes=1,cpu_threads=1,allowed_cpus=[104],nice=10,idle_io=True,cuda_visible_device='0',gpu_uuid=GPU_UUID,test_authorized=False,automatic_retry_authorized=False,root_authorization=dict(path=str(HERE/'execution_attachments/ROOT_APPROVED.json'),sha256=ROOT_SHA))
    for key,name in [('cpu_gate_delivery','cpu_strict_delivery.json'),('cpu_gate_offserver_verification','cpu_offserver_verification.json'),('resource_preflight','resource_preflight.json')]:
        path=HERE/'execution_attachments'/name;approval[key]=dict(path=str(path),sha256=sha(path))
    approval_path=HERE/'APPROVED_gate.json';save(approval_path,approval);approval_sha=sha(approval_path)
    (HERE/'APPROVED_gate.sha256').write_bytes((approval_sha+'\n').encode())
    driver.approve('gate',approval_path,approval_sha)
    script.write_bytes((HERE/'gate_service.sh').read_bytes());script.chmod(0o755)
    conf.write_bytes((HERE/'gate_supervisor.conf').read_bytes())
    subprocess.run(['supervisorctl','reread'],check=True,capture_output=True,text=True)
    subprocess.run(['supervisorctl','update',SERVICE],check=True,capture_output=True,text=True)
    require(time.time()-resource['at_unix']<=90,'Resource proof expired before start')
    result=subprocess.run(['supervisorctl','start',SERVICE],check=True,capture_output=True,text=True)
    save(HERE/'execution_attachments/start_once.json',dict(status='START_REQUESTED_NOT_SCIENTIFIC_ACCEPTANCE',at_unix=time.time(),service=SERVICE,approval_sha256=approval_sha,source_seal_sha256=sha(HERE/'FILES_SHA256.json'),dispatch_attachment_sha256=sha(__file__),resource_preflight_sha256=sha(resource_path),stdout=result.stdout,screen32_started=False,automatic_retry=False))
    print(json.dumps(dict(status=result.stdout.strip(),existing_nominal_reserved=resource['existing_nominal_compute_threads'],with_this1=resource['existing_nominal_compute_threads']+1,quota=resource['actual_quota_cores'],gpu_free_mib=resource['gpu_free_memory_mib'],approval_sha256=approval_sha)))


if __name__=='__main__':
    main()
