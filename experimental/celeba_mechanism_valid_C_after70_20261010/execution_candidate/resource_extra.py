"""Known new-role resource additions to the sealed v2 classifier; no science changes."""
import os
from pathlib import Path

from batch import CPUS, digest, load, read, require

CHECKS = Path('/workspace/guardfed_checks')
HYBRID = CHECKS / 'celeba_hybrid_realimage_gate_20261009/execution_repair_v1/repair_execute.py'
FL = CHECKS / 'celeba_flgmm_screen_20261009/release_v2/run_one.py'
RESOURCE_SHA = '5d537f129c6d96f370bbd93b321e86be1e1fef628ba9589485ac9a14486c00c0'


def add_known_roles(base, extra):
    pids = {row['pid'] for row in base['tracked_compute']}
    tasks = set()
    for row in extra:
        require(row['pid'] not in pids and row['task'] not in tasks, 'Duplicate compute PID/task')
        pids.add(row['pid']); tasks.add(row['task'])
        require(row['role'] in {'Hybrid_repair_CPU8','FLscreen_GPU_CPU1'}, 'Unknown additional resource role')
        require(row['compute_threads'] == (8 if row['role'] == 'Hybrid_repair_CPU8' else 1), 'Wrong known compute reservation')
        if row['role'] == 'Hybrid_repair_CPU8':
            require(row['cpus'] == list(range(8, 16)) and not set(row['cpus']).intersection(CPUS), 'Hybrid allocation overlaps new replay')
        else:
            require(not (len(row['cpus']) <= 16 and set(row['cpus']).intersection(CPUS)), 'Restricted FL CPU allocation overlaps replay')
    require(sum(row['role'] == 'Hybrid_repair_CPU8' for row in extra) <= 1
            and sum(row['role'] == 'FLscreen_GPU_CPU1' for row in extra) <= 2, 'Too many new known workers')
    total = base['nominal_compute_threads_including_this8'] + sum(row['compute_threads'] for row in extra)
    require(total <= base['actual_quota_cores'], 'Actual nominal reservations plus this eight exceed quota')
    return dict(base, known_additional_compute=extra, total_effective_nominal=total,
        nominal_is_reservation_not_measured_usage=True,
        worst_case_declared_coexistence=dict(baseline=88, formal=8, Hybrid=8, FLscreen=2, proposed_replay=8, total=114))


def resource_snapshot():
    source = CHECKS / 'celeba_mechanism_valid_replay_20261009/execution_attachments_v2/execute_one_v2.py'
    legacy = load('sealed_v2_resource_for_remaining7', source, RESOURCE_SHA)
    base = legacy.resource_snapshot()
    require('RUNNING' in base['formal_service'], 'Protected formal service is not observable')
    extra = []
    known_pids = {row['pid'] for row in base['tracked_compute']}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [s.decode(errors='replace') for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
            entry = legacy.python_entry(argv)
            if entry is None or (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                continue
            kind, name, args = entry
            cpus = sorted(os.sched_getaffinity(int(proc.name)))
            cwd = str((proc / 'cwd').resolve())
            if kind != 'script':
                continue
            script = legacy.absolute(name, cwd)
            env = dict(s.split('=', 1) for s in (proc / 'environ').read_text().split('\0') if '=' in s)
            if str(script) == str(HYBRID) and args == ['run','--approved',str(HYBRID.parent.parent / 'execution_approved_20261009/APPROVED.json')]:
                require(digest(HYBRID) == '8f57279c08699c655878c1e2eb98338727316c6e0d52b29bddcb5de6a09ce3b2'
                    and digest(HYBRID.parent / 'FILES_SHA256.json') == '54c67847b44a26a654b380c6c4d03863279261126a61de29489bfa78572e332b', 'Hybrid reviewed source identity changed')
                require(env.get('CUDA_VISIBLE_DEVICES') == '' and all(env.get(k) == '8' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')), 'Hybrid eight-thread CPU declaration changed')
                extra.append(dict(pid=int(proc.name), task=str(HYBRID), role='Hybrid_repair_CPU8', compute_threads=8,
                    cpus=cpus, argv=argv, cwd=cwd, source_sha256=digest(HYBRID)))
            elif str(script) == str(FL) and legacy.option(args, '--repo') == '/workspace/GuardFed-celeba-expanded' and legacy.option(args,'--job-id'):
                stage = FL.parent; seal = read(stage / 'PACKAGE_SHA256.json')
                require(seal['status'] == 'FROZEN_PENDING_EXECUTION' and digest(FL) == seal['files']['run_one.py']
                    == 'e55fbfc78c82617e8226e700597f9b76eb48ab18006fd9fd11ce8be5433a48c9' and cwd == str(stage), 'FL screen source/cwd changed')
                identity = legacy.option(args,'--job-id')
                item = next(row for row in read(stage / 'jobs/manifest.json')['jobs'] if row['id'] == identity)
                job = stage / 'jobs' / item['job']
                require(digest(job) == item['job_sha256'] and read(job)['id'] == identity and
                    all(env.get(k) == '1' for k in ('GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')), 'FL actual job/one-thread contract differs')
                extra.append(dict(pid=int(proc.name), task=identity, role='FLscreen_GPU_CPU1', compute_threads=1,
                    cpus=cpus, argv=argv, cwd=cwd, source_sha256=digest(FL), job_sha256=digest(job)))
            elif int(proc.name) not in known_pids and len(cpus) <= 16 and set(cpus).intersection(CPUS):
                require(False, 'An additional restricted Python process occupies CPU112..119: ' + str(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return add_known_roles(base, extra)
