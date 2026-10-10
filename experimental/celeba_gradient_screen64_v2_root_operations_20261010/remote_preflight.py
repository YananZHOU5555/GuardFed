"""Actual source/data/resource check before the single gradient64 worker."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

BASE = Path('/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010')
REPO = Path('/workspace/GuardFed-celeba-expanded')
OUT = Path('/workspace/celeba_gradient_screen64_v2_results_20261010')
GUIDE = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
SEAL = '11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'


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


assert sha('/etc/vast-agents-guide.md') == GUIDE
assert sha(BASE / 'FILES_SHA256.json') == SEAL
assert not OUT.exists(), 'Existing output must be preserved; no relaunch'
members = read(BASE / 'FILES_SHA256.json')['files']
assert all(sha(BASE / n) == h for n, h in members.items())
protocol = read(BASE / 'snapshot/gradient_bridge_20261010/protocol.json')
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
assert not any(str(BASE) in ' '.join(r['argv']) for r in processes), 'Duplicate gradient worker'
quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
cores = int(quota) / int(period) if quota != 'max' else len(os.sched_getaffinity(0))
assert nominal + 1 <= cores
gpu_result = command(['nvidia-smi', '--id=1', '--query-gpu=uuid,memory.free,memory.used,temperature.gpu', '--format=csv,noheader,nounits'])
assert gpu_result['returncode'] == 0
uuid, free, used, temp = [x.strip() for x in gpu_result['stdout'].strip().split(',')]
assert int(free) >= 4096
main = command(['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal'])
assert main['returncode'] == 0 and 'RUNNING' in main['stdout']
main_queue = read(REPO / 'results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json')
assert not main_queue['failed'] and len(main_queue['active']) <= 8
assert len({r['id'] for r in main_queue['active']}) == len(main_queue['active'])
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
proof = dict(status='ROOT_GRADIENT64_V2_RESOURCE_PREFLIGHT_PASS',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), at_unix=time.time(),
    measured_cgroup_cpu_cores=usage, cpu105_system_samples=[before,after], cpu_ids=[105], cpu_threads=1, max_workers=1, cuda_visible_device='1', nice=10, idle_io=True,
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
path = BASE / ('ROOT_RESOURCE_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
path.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(dict(path=str(path), sha256=sha(path), at_unix=proof['at_unix'], nominal=nominal,
    quota=cores, gpu_free_mib=int(free), processes=len(processes), project_threads=sum(len(r['tasks']) for r in processes))))
