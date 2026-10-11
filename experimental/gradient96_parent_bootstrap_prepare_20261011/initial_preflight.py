"""Root initial NGA96 resource probe; original child measurements, no fake controller."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

parser = __import__('argparse').ArgumentParser()
parser.add_argument('--base', type=Path, required=True)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--queue-out', type=Path, required=True)
parser.add_argument('--proof-dir', type=Path, required=True)
parser.add_argument('--source-approval', type=Path, required=True)
parser.add_argument('--source-approval-sha256', required=True)
parser.add_argument('--gate14-adoption', type=Path, required=True)
parser.add_argument('--gate14-adoption-sha256', required=True)
args = parser.parse_args()
BASE, REPO = args.base.resolve(), args.repo.resolve()
QUEUE_OUT, PROOF_DIR = args.queue_out.resolve(), args.proof_dir.resolve()
assert sys.platform == 'linux' and not sys.flags.optimize
assert BASE == Path('/workspace/guardfed_checks/celeba_gradient_fullcoverage_selected_20261011/Fed-NGA-gradient')
assert REPO == Path('/workspace/GuardFed-celeba-expanded')
assert QUEUE_OUT.is_relative_to(Path('/workspace')) and QUEUE_OUT != Path('/workspace') and not QUEUE_OUT.exists()
assert PROOF_DIR.is_relative_to(Path('/workspace')) and PROOF_DIR.is_dir()
assert os.sched_getaffinity(0) == {105} and os.getpriority(os.PRIO_PROCESS, 0) >= 10
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
assert all(os.environ.get(k) == '1' for k in ('GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
assert hashlib.sha256((BASE/'coverage_contract.py').read_bytes()).hexdigest() == '2ccc116bf67976b33028ff6f4f35a6327056fe8415db98c4d149df810a3edc7c'
sys.path.insert(0, str(BASE))
import coverage_contract
coverage_contract.validate(BASE, args)
manifest = json.loads((BASE/'jobs/manifest.json').read_bytes())
entry = manifest['new_jobs'][0]
OUT = QUEUE_OUT / entry['id']
assert len(manifest['new_jobs']) == 96 and len(manifest['reused_jobs']) == 4
GUIDE = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
SEAL = hashlib.sha256((BASE/'FILES_SHA256.json').read_bytes()).hexdigest()
assert SEAL == '06e3d638aea70123d7fcf73e5a219a0c0d3165c8cf60acb9dee4eace5faad09b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def command(argv):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    return dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)


# Probe the existing shared lock, release only when this probe process exits.
import fcntl
common_lock = (BASE.parent/'coverage_single_worker.lock').open('a')
fcntl.flock(common_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
actual_idle_io = command(['ionice','-p',str(os.getpid())])
assert actual_idle_io['returncode'] == 0 and actual_idle_io['stdout'].strip().startswith('idle')
assert sha('/etc/vast-agents-guide.md') == GUIDE
assert sha(BASE / 'FILES_SHA256.json') == SEAL
assert not OUT.exists(), 'Existing output must be preserved; no relaunch'
members = read(BASE / 'FILES_SHA256.json')['files']
assert all(sha(BASE / n) == h for n, h in members.items())
protocol = read(BASE / 'snapshot/gradient_bridge_fullcoverage/protocol.json')
assert sha(BASE/'jobs'/entry['job']) == entry['sha256']
binding = read(BASE / 'shared_cache_bindings.json')['bindings']
source_rows = []
for name, expected in protocol['source_hashes'].items():
    p = REPO / name
    if name in binding:
        b = binding[name]
        assert str(p.resolve()) == b['resolved_absolute']
        assert p.stat().st_size == b['size_bytes'] and expected == b['expected_sha256']
    else:
        assert p.resolve().is_relative_to(REPO)
    actual = sha(p)
    assert actual == expected, name
    source_rows.append(dict(name=name, resolved=str(p.resolve()), bytes=p.stat().st_size, sha256=actual))

# Inspect every thread of actual /workspace compute processes. Management
# services have unconstrained affinity; their mere eligible CPU sets do not
# reserve scientific cores. No cmdline/environment of those services is saved.
processes = []
overlaps = []
nominal = 0
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv = [x for x in p.read_bytes().decode(errors='replace').split('\0') if x]
        if not argv or not any(x.startswith('/workspace/') for x in argv):
            continue
        if not ('python' in Path(argv[0]).name or 'python' in ' '.join(argv[:2])):
            continue
        pid = int(p.parent.name)
        if pid == os.getpid():
            continue
        tasks = []
        for t in (p.parent / 'task').iterdir():
            try:
                tid = int(t.name)
                allowed = sorted(os.sched_getaffinity(tid))
                tasks.append(dict(tid=tid, cpu_count=len(allowed), cpus=allowed if len(allowed)<=16 else None, broad_cpu_span=[allowed[0],allowed[-1]] if len(allowed)>16 else None))
                if len(allowed) <= 16 and 105 in allowed:
                    overlaps.append(dict(pid=pid, tid=tid, cpus=allowed))
            except (OSError, ValueError):
                pass
        env = {}
        for item in (p.parent / 'environ').read_bytes().split(b'\0'):
            if b'=' in item:
                k, v = item.split(b'=', 1)
                if k in (b'GUARDFED_CPU_THREADS', b'OMP_NUM_THREADS', b'MKL_NUM_THREADS', b'CUDA_VISIBLE_DEVICES'):
                    env[k.decode()] = v.decode(errors='replace')
        compute_threads = int(env.get('GUARDFED_CPU_THREADS', env.get('OMP_NUM_THREADS', '1')))
        nominal += compute_threads
        # Only known project command arguments, never ambient credentials.
        processes.append(dict(pid=pid, argv=argv, tasks=tasks, selected_thread_env=env,
                              nominal_compute_threads=compute_threads))
    except (OSError, ValueError):
        pass
assert not overlaps, 'CPU105 overlaps an existing restricted project compute reservation'
sibling_text = Path('/sys/devices/system/cpu/cpu105/topology/thread_siblings_list').read_text().strip()
siblings = set()
for term in sibling_text.split(','):
    span = [int(x) for x in term.split('-')]
    siblings.update(range(span[0], span[-1]+1))
assert 105 in siblings
smt_overlaps = [dict(pid=r['pid'],tid=t['tid'],cpus=t['cpus']) for r in processes for t in r['tasks']
                if t['cpus'] is not None and siblings.intersection(t['cpus'])]
assert not smt_overlaps, 'CPU105 SMT sibling overlaps a restricted project reservation'
old64_prefix = '/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010/'
assert not any(any(str(BASE.parent)+'/' in a or old64_prefix in a for a in r['argv']) for r in processes), 'Duplicate selected gradient queue/worker or old64 worker'
old64 = command(['supervisorctl','status','guardfed_celeba_gradient_screen64_v2a'])
assert old64['returncode'] == 3 and 'EXITED' in old64['stdout'], 'Old64 must naturally EXITED'
old_output = Path('/workspace/celeba_gradient_screen64_v2_results_20261010')
assert not list(old_output.glob('**/failure.json')) and not list(old_output.glob('QUEUE_FAILURE.json')), 'Old64 failures require review'
quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
cores = int(quota) / int(period) if quota != 'max' else len(os.sched_getaffinity(0))
assert nominal + 1 <= cores
gpu_result = command(['nvidia-smi', '--id=1', '--query-gpu=uuid,memory.free,memory.used,temperature.gpu', '--format=csv,noheader,nounits'])
assert gpu_result['returncode'] == 0
uuid, free, used, temp = [x.strip() for x in gpu_result['stdout'].strip().split(',')]
assert int(free) >= 4096
main = command(['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal'])
assert (main['returncode'] == 0 and 'RUNNING' in main['stdout']) or (main['returncode'] == 3 and 'EXITED' in main['stdout'])
main_queue = read(REPO / 'results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json')
assert not main_queue['failed'] and len(main_queue['active']) <= 8
if main['returncode'] == 3:
    assert len(main_queue['completed']) == 800 and not main_queue['active'] and not main_queue['pending']
assert len({r['id'] for r in main_queue['active']}) == len(main_queue['active'])
main_active_progress = []
for row in main_queue['active']:
    task_dir = REPO/'results/revision_20261009/celeba_mechanism_v1/runs'/row['id']
    progress_path = task_dir/'progress.json'
    failure_path = task_dir/'failure.json'
    assert not failure_path.exists(), 'Current main worker failure requires root review'
    main_active_progress.append(dict(id=row['id'],pid=row.get('pid'),
        progress_path=str(progress_path),progress=read(progress_path) if progress_path.exists() else None,
        failure_path=str(failure_path),failure_present=False))
recovery = command(['nvidia-smi', '-q'])
recovery_lines = [x.strip() for x in recovery['stdout'].splitlines() if 'GPU Recovery Action' in x]
assert recovery['returncode'] == 0 and len(recovery_lines) == 2 and all(x.endswith('None') for x in recovery_lines)
memory_current = int(Path('/sys/fs/cgroup/memory.current').read_text())
memory_max = int(Path('/sys/fs/cgroup/memory.max').read_text())
assert memory_max - memory_current >= 8 * 1024 ** 3
assert shutil.disk_usage('/workspace').free >= 40 * 1024 ** 3
def cpu_sample():
    stat = dict(line.split(maxsplit=1) for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
    cpu = next(line for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu105 '))
    return dict(at=time.monotonic(), usage_usec=int(stat['usage_usec']), cpu105_ticks=[int(x) for x in cpu.split()[1:]])
before=cpu_sample(); time.sleep(1); after=cpu_sample()
usage=(after['usage_usec']-before['usage_usec'])/1e6/(after['at']-before['at'])
proof = dict(status='ROOT_GRADIENT96_RESOURCE_PREFLIGHT_PASS',
    producer_role='ROOT_INITIAL_RESOURCE_PROBE_NOT_QUEUE_CONTROLLER', bootstrap_probe_pid=os.getpid(),
    planned_first_job_id=entry['id'], planned_first_job_sha256=entry['sha256'],
    common_lock_probe_pass=True, cpu105_SMT_siblings=sorted(siblings), restricted_SMT_overlaps=smt_overlaps,
    actual_idle_io_query=actual_idle_io, main_active_progress=main_active_progress,
    source_approval_sha256=args.source_approval_sha256, gate14_adoption_sha256=args.gate14_adoption_sha256,
    selected_manifest_sha256=sha(BASE/'jobs/manifest.json'), old_gradient64_service_status='EXITED',
    old_gradient64_supervisor_rc=old64['returncode'], old_gradient64_live_workers=0, old_gradient64_failures=0,
    new_jobs_authorized=96, reused_jobs_authorized_for_training=0,
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), at_unix=time.time(),
    measured_cgroup_cpu_cores=usage, cpu105_system_samples=[before,after], cpu_ids=[105], cpu_threads=1, max_workers=1, cuda_visible_device='1', nice=os.getpriority(os.PRIO_PROCESS,0), idle_io=True,
    no_restricted_cpu105_owner=True, no_duplicate_worker=True, no_restricted_CPU_overlap=True,
    source_data_hashes_verified=True, existing_nominal_compute_threads=nominal, actual_quota_cores=cores,
    gpu_free_memory_mib=int(free), gpu_uuid=uuid, gpu_used_mib=int(used), gpu_temperature_c=int(temp),
    guide_sha256=GUIDE, package_seal_sha256=SEAL, author_decisions_sha256=sha(BASE / 'AUTHOR_DECISIONS.json'),
    test_authorized=False, automatic_retry_authorized=False, all_project_process_threads=processes,
    source_data_checks=source_rows, main_service=main, memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),
    main_queue=dict(completed=len(main_queue['completed']), active=main_queue['active'], failed=main_queue['failed']),
    recovery_actions=recovery_lines, memory_headroom_bytes=memory_max-memory_current,
    disk_free_bytes=shutil.disk_usage('/workspace').free,
    scope='All project threads inspected; restricted <=16-core masks reserve CPUs, broad masks are eligible scheduling sets; no CPU exclusivity/idle claim')
path = PROOF_DIR/'INITIAL_RESOURCE.json'
with path.open('x',encoding='utf8') as stream: stream.write(json.dumps(proof, indent=2) + '\n')
print(json.dumps(dict(path=str(path), sha256=sha(path), at_unix=proof['at_unix'], nominal=nominal,
    quota=cores, gpu_free_mib=int(free), processes=len(processes), project_threads=sum(len(r['tasks']) for r in processes))))
