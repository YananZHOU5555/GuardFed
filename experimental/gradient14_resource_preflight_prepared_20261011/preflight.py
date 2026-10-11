"""Root-bound, single-job Linux metadata/resource preflight; never runs an experiment."""
from pathlib import Path
import argparse, ast, datetime, hashlib, json, os, shutil, subprocess, sys, time

GUIDE = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
CPUS = list(range(88, 96))
OLD_BASE = '/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010'
OLD_OUT = Path('/workspace/celeba_gradient_screen64_v2_results_20261010')
OLD_SERVICE = 'guardfed_celeba_gradient_screen64_v2a'
MAIN_SERVICE = 'guardfed_celeba_mechanism_formal'


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


def pin(path, expected):
    assert isinstance(expected, str) and len(expected) == 64 and sha(path) == expected, str(path)
    return read(path)


def cpu_list(value):
    out = set()
    for item in value.strip().split(','):
        ends = item.split('-'); out.update(range(int(ends[0]), int(ends[-1]) + 1))
    return out


def validate_topology(rows, eligible):
    assert [r['cpu'] for r in rows] == CPUS
    assert len({(r['socket'], r['core']) for r in rows}) == 8
    assert len({r['socket'] for r in rows}) == 1
    watched = set()
    for r in rows:
        assert r['cpu'] in r['siblings'] and len(r['siblings']) == 2
        assert not watched.intersection(r['siblings'])
        watched.update(r['siblings'])
    assert watched.issubset(eligible), 'Targets/SMT siblings must be eligible and online'
    return watched


def validate_authorization(a, scope, job_id, scope_sha, self_sha):
    assert a['status'] == 'ROOT_AUTHORIZED_GRADIENT14_PREFLIGHT_SOURCE_ONLY'
    assert a['scope_sha256'] == scope_sha and a['preflight_sha256'] == self_sha
    assert a['root64_sha256'] == scope['root64_sha256']
    assert a['frozen192_approval_sha256'] == scope['frozen192_approval_sha256']
    assert a['ordered_jobs'] == scope['jobs'] and len(scope['jobs']) == 14
    assert len({e['id'] for e in scope['jobs']}) == 14
    assert a['cpu_ids'] == CPUS and a['cpu_threads'] == 8
    assert a['test_authorized'] is False and a['coverage192_authorized'] is False
    assert a['automatic_retry'] is False
    assert scope['status'] == 'PREPARED_EXACT14_NOT_DISPATCHED'
    assert scope['scientific_table_records'] == 0 and scope['test'] is False
    assert (scope['rounds'], scope['seed'], scope['max_workers'], scope['cpu_threads'], scope['device']) == (3, 91002, 1, 8, 'cpu')
    return next(e for e in scope['jobs'] if e['id'] == job_id)


def formal_growth(before, after):
    old = {x['id']: x['round'] for x in before['active']}
    return after['completed'] > before['completed'] or any(
        row['id'] in old and old[row['id']] is not None and row['round'] is not None and row['round'] > old[row['id']]
        for row in after['active'])


def previous_snapshot(record):
    if 'main_snapshot' in record: return record['main_snapshot']
    if 'main_mechanism' not in record: return record
    assert record['status'] == 'SINGLE_FIVE_STAGE_READONLY_HEALTH_NOT_ACCEPTANCE'
    assert record['guide_sha256'] == GUIDE
    m = record['main_mechanism']; q = m['queue']
    workers = {r['pid']: r for r in m['workers']}
    active = []
    for r in q['active']:
        w = workers[r['pid']]; assert w['job'] == r['id']+'.json'
        active.append(dict(id=r['id'],pid=r['pid'],round=w['round']))
    return dict(at_unix=datetime.datetime.fromisoformat(record['utc']).timestamp(),completed=len(q['completed']),failed=q['failed'],active=active,pending=q['pending'])


def progress_ok(before, after):
    assert before['at_unix'] < after['at_unix'] and not before['failed'] and not after['failed']
    assert after['completed'] >= before['completed'] and after['completed'] <= 800
    assert len(after['active']) <= 8 and len({r['id'] for r in after['active']}) == len(after['active'])
    complete = after['completed'] == 800 and not after['active'] and after['pending'] == 0
    assert complete or (after['active'] and formal_growth(before, after)), 'Main800 lacks actual round/completion growth'
    return complete


def main(x):
    assert sys.platform == 'linux' and not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE')
    assert sha('/etc/vast-agents-guide.md') == GUIDE
    stage, repo, package = map(lambda p: Path(p).resolve(), (x.stage, x.repo, x.package))
    assert all(p.is_relative_to(Path('/workspace')) for p in (stage, repo, package))
    assert not Path(x.out).exists(), 'Preserve any prior receipt/refusal; no overwrite'
    a = pin(x.authorization, x.authorization_sha256)
    scope = pin(stage/'scope.json', a['scope_sha256'])
    entry = validate_authorization(a, scope, x.job, sha(stage/'scope.json'), sha(__file__))
    root64 = pin(a['root64_path'], scope['root64_sha256'])
    assert root64['status'] == 'ROOT_GRADIENT64_COMPLETE_STRICT_OFFSERVER_ADOPTED'
    assert root64['accepted_count'] == 64 and root64['all64_offserver_verified'] is True and root64['test'] is False
    frozen = pin(a['frozen192_approval_path'], scope['frozen192_approval_sha256'])
    assert frozen['status'] == 'ROOT_GRADIENT200_FROZEN_SOURCE_ADOPTED' and frozen['test'] is False
    assert sha(package/'adapter.py') == a['adapter_sha256']
    assert sha(package/'metadata.py') == a['metadata_sha256']
    # Only the two original pure metadata functions are executed; no module import.
    text = (package/'metadata.py').read_text(encoding='utf8')
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in ('jobs_for', 'validate_jobs')]
    assert len(nodes) == 2
    ns = {'STAGE': scope['evidence_stage']}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original pure metadata>', 'exec'), ns)
    jobs = []
    for e in scope['jobs']:
        assert e['job'] == 'jobs/'+e['id']+'.json' and e['output'] == 'runs/'+e['id']
        jobs.append(pin(stage/e['job'], e['job_sha256']))
    ns['validate_jobs'](jobs, scope['candidates'], scope['original_protocol'])
    for p, h in scope['external_identities'].items(): assert sha(p) == h, p
    for p, h in scope['local_hashes'].items(): assert sha(stage/p) == h, p
    assert not list(stage.glob('failure*.json'))
    for e in scope['jobs'][:scope['jobs'].index(entry)]:
        out = stage/e['output']; receipt = read(out/'item_receipt.json')
        assert receipt['status'] == 'CPU3_GATE_ORIGINAL_STRICT_AND_EXTERNAL_IDENTITY_PASS'
        assert receipt['scope_sha256'] == a['scope_sha256'] and receipt['scientific_table_records'] == 0
        assert sha(out/'acceptance.json') == receipt['acceptance_sha256']
        accepted = read(out/'acceptance.json')
        assert accepted['status'] == 'PASS'
        assert accepted['job_sha256'] == e['job_sha256']
        assert accepted['scope_sha256'] == a['scope_sha256']
        assert sha(out/'rng_after.json') == receipt['rng_sha256'] and not list(out.glob('failure*.json'))
    for e in scope['jobs'][scope['jobs'].index(entry):]:
        assert not (stage/e['output']).exists(), 'Partial/out-of-order output requires review'
        assert not (stage/('attempt_'+hashlib.sha256(e['id'].encode()).hexdigest()+'.json')).exists()
    binding = read(stage/'shared_cache_bindings.json')['bindings']; source_rows = []
    for name, expected in scope['protected_source_hashes'].items():
        p = repo/name
        if name in binding:
            b = binding[name]
            assert str(p.resolve()) == b['resolved_absolute']
            assert p.stat().st_size == b['size_bytes'] and expected == b['expected_sha256']
        else: assert p.resolve().is_relative_to(repo)
        actual = sha(p); assert actual == expected, name
        source_rows.append(dict(name=name, resolved=str(p.resolve()), bytes=p.stat().st_size, sha256=actual))
    online = cpu_list(Path('/sys/devices/system/cpu/online').read_text())
    topology = []
    for cpu in CPUS:
        t = Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
        topology.append(dict(cpu=cpu, socket=int((t/'physical_package_id').read_text()), core=int((t/'core_id').read_text()), siblings=sorted(cpu_list((t/'thread_siblings_list').read_text()))))
    watched = validate_topology(topology, set(os.sched_getaffinity(0)) & online)
    processes, overlaps, nominal = [], [], 0
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid = int(p.parent.name)
            if pid == os.getpid(): continue
            argv = [s for s in p.read_bytes().decode(errors='replace').split('\0') if s]
            project = any(s.startswith('/workspace/') for s in argv)
            python = bool(argv) and ('python' in Path(argv[0]).name or 'python' in ' '.join(argv[:2]))
            tasks = []
            for t in (p.parent/'task').iterdir():
                try:
                    tid = int(t.name); allowed = sorted(os.sched_getaffinity(tid))
                    if len(allowed) <= 32 and watched.intersection(allowed): overlaps.append(dict(pid=pid, tid=tid, cpus=allowed))
                    if project and python: tasks.append(dict(tid=tid, cpu_count=len(allowed), cpus=allowed if len(allowed)<=32 else None))
                except ProcessLookupError: pass
            if project and python:
                env = {}
                for item in (p.parent/'environ').read_bytes().split(b'\0'):
                    if b'=' in item:
                        k, v = item.split(b'=', 1)
                        if k in (b'GUARDFED_CPU_THREADS', b'OMP_NUM_THREADS', b'MKL_NUM_THREADS', b'CUDA_VISIBLE_DEVICES'): env[k.decode()] = v.decode(errors='replace')
                threads = int(env.get('GUARDFED_CPU_THREADS', env.get('OMP_NUM_THREADS', '1')))
                assert threads >= 1; nominal += threads
                processes.append(dict(pid=pid, argv=argv, tasks=tasks, selected_thread_env=env, nominal_compute_threads=threads))
        except (FileNotFoundError, ProcessLookupError): pass
    assert not overlaps, 'Target/SMT sibling has an existing <=32-CPU reservation'
    assert not any(any(OLD_BASE in s or str(OLD_OUT) in s or s == str(package/'adapter.py') or s == str(stage) for s in r['argv']) for r in processes), 'Old64/gate worker or producer remains'
    old = command(['supervisorctl','status',OLD_SERVICE])
    assert old['returncode'] == 3 and old['stdout'].split()[:2] == [OLD_SERVICE,'EXITED']
    assert not list(OLD_OUT.rglob('failure*.json')) and not list(OLD_OUT.rglob('*FAILURE*.json'))
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    cores = int(quota)/int(period) if quota != 'max' else len(os.sched_getaffinity(0))
    assert nominal + 8 <= cores
    main_service = command(['supervisorctl','status',MAIN_SERVICE])
    base = repo/'results/revision_20261009/celeba_mechanism_v1'; queue = read(base/'formal_queue_progress.json')
    main_manifest = pin(base/'manifest.json', a['main_manifest_sha256'])
    main_ids = {r['id'] for r in main_manifest['jobs']}
    assert len(main_ids) == len(main_manifest['jobs']) == 800
    completed_ids = [r if isinstance(r,str) else r['id'] for r in queue['completed']]
    assert len(completed_ids) == len(set(completed_ids)) and not set(completed_ids).intersection(r['id'] for r in queue['active'])
    assert set(completed_ids).issubset(main_ids) and all(r['id'] in main_ids for r in queue['active'])
    assert len(completed_ids)+len(queue['active'])+queue['pending'] == 800
    active = []
    for r in queue['active']:
        progress = read(base/'runs'/r['id']/'progress.json')
        assert type(progress['round']) is int and 0 <= progress['round'] <= 70
        argv = Path('/proc')/str(r['pid'])/'cmdline'
        args = argv.read_bytes().decode().split('\0')
        assert any(r['id'] in s for s in args) and any(str(repo) in s for s in args)
        active.append(dict(id=r['id'],pid=r['pid'],round=progress['round'],updated_unix=progress.get('updated_unix')))
    snapshot = dict(at_unix=time.time(),completed=len(completed_ids),failed=queue['failed'],active=active,pending=queue['pending'])
    before = previous_snapshot(pin(x.previous_main, x.previous_main_sha256))
    complete = progress_ok(before, snapshot)
    assert (main_service['returncode']==0 and main_service['stdout'].split()[:2]==[MAIN_SERVICE,'RUNNING']) or (complete and main_service['returncode']==3 and main_service['stdout'].split()[:2]==[MAIN_SERVICE,'EXITED'])
    gpu = command(['nvidia-smi','--id=1','--query-gpu=uuid,memory.free,memory.used,temperature.gpu','--format=csv,noheader,nounits'])
    assert gpu['returncode']==0
    uuid, free, used, temp = [s.strip() for s in gpu['stdout'].strip().split(',')]; assert int(free)>=4096
    recovery = command(['nvidia-smi','-q']); lines = [s.strip() for s in recovery['stdout'].splitlines() if 'GPU Recovery Action' in s]
    assert recovery['returncode']==0 and len(lines)==2 and all(s.endswith('None') for s in lines)
    memory_current = int(Path('/sys/fs/cgroup/memory.current').read_text()); memory_max = int(Path('/sys/fs/cgroup/memory.max').read_text())
    ram = memory_max-memory_current; assert ram >= 8*1024**3
    disk = shutil.disk_usage('/workspace').free; assert disk >= 40*1024**3
    def cpu_sample():
        stat = dict(line.split(maxsplit=1) for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
        ticks = {int(line.split()[0][3:]): [int(v) for v in line.split()[1:]] for line in Path('/proc/stat').read_text().splitlines() if line.split()[0] in {'cpu'+str(c) for c in watched}}
        assert set(ticks) == watched
        return dict(at=time.monotonic(), usage_usec=int(stat['usage_usec']), cpu_ticks=ticks)
    cpu_before=cpu_sample(); time.sleep(1); cpu_after=cpu_sample()
    usage=(cpu_after['usage_usec']-cpu_before['usage_usec'])/1e6/(cpu_after['at']-cpu_before['at'])
    proof = dict(status='ROOT_APPROVED_GRADIENT14_CPU3_CANARIES_ONLY', measured_unix=time.time(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        root_authorization_sha256=x.authorization_sha256, preflight_source_sha256=sha(__file__), scope_sha256=a['scope_sha256'],
        jobs={e['id']:e['job_sha256'] for e in scope['jobs']}, job_id=entry['id'], job_sha256=entry['job_sha256'],
        one_job_runtime_scope=dict(job_id=entry['id'],job_sha256=entry['job_sha256'],sequential_index=scope['jobs'].index(entry),fresh_receipt_required_for_next_job=True),
        exclusive_cpu_ids=CPUS, cpu_topology=topology, monitored_logical_cpus=sorted(watched), no_restricted_cpu_overlap=True,
        old_gradient64_exited=True, protected_main_healthy=True, no_duplicate_gate_worker=True, protected_main_growth_or_completed=True,
        nominal_existing_compute_threads=nominal, actual_cpu_quota=cores, ram_available_bytes=ram,
        test_authorized=False, coverage192_authorized=False, automatic_retry=False, cuda_visible_device='', nice=10,idle_io=True,cpu_threads=8,
        measured_cgroup_cpu_cores=usage, cpu_samples=[cpu_before,cpu_after],
        guide_sha256=GUIDE, source_data_checks=source_rows, all_project_process_threads=processes,old64_service=old,
        previous_main_sha256=x.previous_main_sha256,main_snapshot=snapshot,main_service=main_service,
        gpu_uuid=uuid,gpu_free_memory_mib=int(free),gpu_used_mib=int(used),gpu_temperature_c=int(temp),recovery_actions=lines,
        memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),disk_free_bytes=disk,
        scope='Legacy exclusive_cpu_ids means requested affinity only: all /proc threads inspected; <=32 masks reserve targets or SMT siblings; broad masks are eligible scheduling sets, not exclusive ownership or idle guarantees.')
    with Path(x.out).open('x',encoding='utf8') as f: json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=x.out,sha256=sha(x.out),job_id=entry['id'],measured_unix=proof['measured_unix'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('stage','repo','package','authorization','authorization-sha256','job','previous-main','previous-main-sha256','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
