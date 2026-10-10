"""Finite620 CPU validation queue. Source preparation is not deployment permission."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from bridge_adapter import SCOPE, CPUS, canonical, construct_record, digest, inventory, load, project, read, require

HERE = Path(__file__).resolve().parent
REMOTE = Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_20261010')
REPO = Path('/workspace/GuardFed-celeba-expanded')
RUNTIME = REMOTE / 'attempt1'
GUIDE_SHA = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'


def save_new(path, value):
    """Publish immutable complete JSON bytes without an overwrite window."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def source_identity():
    seal = read(HERE / 'FILES_SHA256.json')
    for name, pin in seal['files'].items():
        path = HERE / name
        require(not path.is_symlink() and path.stat().st_size == pin['bytes'] and digest(path) == pin['sha256'], 'Queue source drift: ' + name)
    plan = read(HERE / 'PLAN.json')
    require(plan['scope'] == SCOPE and len(plan['remaining620_ids']) == len(set(plan['remaining620_ids'])) == 620, 'Wrong finite scope')
    require(len(plan['excluded180_ids']) == len(set(plan['excluded180_ids'])) == 180
            and not set(plan['remaining620_ids']) & set(plan['excluded180_ids']), 'Prior180/remaining620 overlap')
    require(all(row['checkpoint_sha256'] is None for row in plan['entries']), 'Prepared source cannot invent future model SHAs')
    require([row['id'] for row in plan['entries']] == plan['remaining620_ids'], 'Ordered entries differ from the frozen complement')
    return plan


def reviewed(path, expected):
    plan = source_identity()
    require(digest(path) == expected, 'External root review bytes changed')
    review = read(path)
    require(review.get('status') == 'ROOT_APPROVED_REMAINING620_MECHANISM_VALID_EVALUATION'
            and review.get('scope') == SCOPE and review.get('execution_authorized') is True, 'No actual root execution approval')
    require(review.get('source_seal_sha256') == digest(HERE / 'FILES_SHA256.json')
            and review.get('plan_sha256') == digest(HERE / 'PLAN.json'), 'Approval source/plan drift')
    require(review.get('selected_ids') == plan['remaining620_ids'] and review.get('excluded_ids') == plan['excluded180_ids']
            and review.get('prior180_adoption_sha256') == plan['prior180_adoption_sha256'], 'Approval cohort/parent drift')
    require(review.get('device') == 'cpu' and review.get('allowed_cpus') == CPUS and review.get('compute_threads') == 8
            and review.get('max_CNN_workers') == 1 and review.get('nice') == 10 and review.get('idle_io') is True, 'Resource contract changed')
    require(review.get('split') == 'valid' and review.get('native_tolerance') == 1e-12 and review.get('new_Full_inference') == 0
            and review.get('test_inference') is False and review.get('automatic_retry') is False, 'Scientific/retry scope changed')
    require(review.get('runtime_namespace') == str(RUNTIME) and review.get('dependency_paths') == plan['remote_dependencies'], 'Wrong new namespace/dependencies')
    require(type(review.get('deadline_unix')) in (int, float) and time.time() < review['deadline_unix'] <= time.time() + 14 * 86400, 'Explicit finite deadline required (at most14days)')
    require(review.get('fresh_Linux_preflight_pass') is True and isinstance(review.get('Linux_preflight_sha256'), str)
            and len(review['Linux_preflight_sha256']) == 64, 'Root must supply actual Linux/source/resource preflight')
    preflight_path = HERE / 'ROOT_LINUX_PREFLIGHT.json'
    require(digest(preflight_path) == review['Linux_preflight_sha256'], 'Actual Linux preflight bytes changed')
    preflight = read(preflight_path)
    require(preflight['pass'] is True and preflight['allowed_cpus'] == CPUS
            and preflight['source_seal_sha256'] == review['source_seal_sha256']
            and preflight['prior_C_after70_EXITED_no_workers'] is True and preflight['sglang_STOPPED'] is True
            and preflight['restricted_CPU112_119_owners'] == [], 'Linux installation boundary not proved')
    require(type(preflight['other_reserved_cores']) in (int, float) and preflight['other_reserved_cores'] >= 0,
            'Explicit conservative other-task reservation required')
    review['_other_reserved_cores'] = preflight['other_reserved_cores']
    return plan, review


def policy():
    require(sys.platform == 'linux' and not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE') and HERE == REMOTE, 'Exact Linux namespace, assertions enabled')
    require(digest('/etc/vast-agents-guide.md') == GUIDE_SHA, 'Read changed instance guide before deployment')
    require(set(CPUS) <= os.sched_getaffinity(0) and os.getpriority(os.PRIO_PROCESS, 0) == 10, 'Exact CPU112..119/nice10 policy required')
    os.sched_setaffinity(0, CPUS)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    require('idle' in subprocess.check_output(['ionice', '-p', str(os.getpid())], text=True).lower(), 'Idle I/O required')


def training_snapshot(plan, evidence):
    path = Path(plan['remote_training_progress'])
    snapshot = read(path)
    observed = subprocess.run(['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal'], capture_output=True, text=True)
    words = observed.stdout.split()
    ids = {row['id'] for row in plan['all_manifest_entries']}
    completed = snapshot['completed']; active = snapshot['active']
    require(not snapshot['failed'] and len(completed) == len(set(completed)) and set(completed) <= ids, 'Producer queue failure/duplicate/foreign ID')
    require(len(active) <= 8 and len({r['id'] for r in active}) == len(active)
            and {r['id'] for r in active} <= ids and not set(completed) & {r['id'] for r in active}, 'Producer active identity invalid')
    require(observed.returncode == 0 and len(words) >= 2 and words[0] == 'guardfed_celeba_mechanism_formal'
            and (words[1] == 'RUNNING' or (words[1] == 'EXITED' and len(completed) == 800 and not active)), 'Producer service/connection error')
    return snapshot, dict(progress_sha256=digest(path), service=observed.stdout.strip(), service_returncode=observed.returncode,
                          completed_n=len(completed), active=active, failed=snapshot['failed'], observed_unix=time.time())


def disposition(entry, snapshot, owner, output_exists, failures):
    require(not failures and not snapshot['failed'], 'Preserved producer failure')
    identity = entry['id']; active = {row['id'] for row in snapshot['active']}
    if identity in snapshot['completed']:
        return 'WAIT_PRODUCER_EXIT' if owner is not None or identity in active else 'READY_FOR_ORIGINAL_STRICT'
    if owner is not None or identity in active:
        return 'WAIT_ORIGINAL_PRODUCER'
    require(not output_exists, 'Orphan partial producer output; preserve and review')
    return 'WAIT_QUEUED_PRODUCER'


def dependencies(plan):
    d = plan['remote_dependencies']
    for key, expected in plan['dependency_sha256'].items():
        require(digest(d[key]) == expected, 'Original dependency drift: ' + key)
    require(digest(d['manifest']) == plan['manifest_sha256'] and read(d['manifest'])['jobs'] == plan['all_manifest_entries'], 'Original manifest changed')
    return d


def resource_gate(review):
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    require(quota != 'max', 'Actual cgroup CPU quota required')
    cores = int(quota) / int(period)
    require(review['_other_reserved_cores'] + 8 <= cores, 'Reserved CPU budget exceeds actual quota')
    current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    require(limit != 'max' and int(limit) - current >= 8 * 1024 ** 3, 'At least8GiB cgroup RAM headroom required')
    occupied = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in (os.getpid(), os.getppid()):
            continue
        try:
            if (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] in ('Z', 'X'):
                continue
            for task in (proc / 'task').iterdir():
                cpus = os.sched_getaffinity(int(task.name))
                if len(cpus) <= 16 and set(CPUS) & cpus:
                    occupied.append(dict(pid=int(proc.name), tid=int(task.name), cpus=sorted(cpus)))
        except (FileNotFoundError, ProcessLookupError):
            continue
    require(not occupied, 'Another restricted task occupies CPU112..119: ' + repr(occupied))
    return dict(quota_cores=cores, other_reserved_cores=review['_other_reserved_cores'],
                RAM_headroom_bytes=int(limit) - current, restricted_owners=occupied, observed_unix=time.time())


def worker(plan, review, identity, review_path, review_sha):
    policy(); d = dependencies(plan)
    import fcntl
    compute_lock = (HERE / 'compute_worker.lock').open('a')
    fcntl.flock(compute_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resources_before = resource_gate(review)
    require(identity in plan['remaining620_ids'] and identity not in plan['excluded180_ids'], 'Old/Full/foreign ID')
    task = RUNTIME / 'tasks' / identity; output = RUNTIME / 'runs' / identity
    require(not (task / 'binding.json').exists() and not output.exists()
            and not output.with_name(output.name + '.bridge_failure.json').exists(), 'Existing binding/partial/failure forbids retry')
    evidence = load('remaining620_original_evidence', d['evidence_v4'], plan['dependency_sha256']['evidence_v4'])
    manifest = read(d['manifest']); entry = next(e for e in manifest['jobs'] if e['id'] == identity)
    adapter_dir = Path(next(iter(manifest['adapter_hashes']))).parent
    snapshot, observation = training_snapshot(plan, evidence)
    owner = evidence.live_worker(entry, REPO, adapter_dir)
    require(disposition(entry, snapshot, owner, Path(entry['output']).exists(), list(Path(entry['output']).glob('failure*.json'))) == 'READY_FOR_ORIGINAL_STRICT', 'Producer not closed')
    parent = load('remaining620_original_bridge', d['parent_bridge'], plan['dependency_sha256']['parent_bridge'])
    baseline = read(d['baseline_inventory'])
    require(digest(entry['job']) == entry['job_sha256'], 'Raw job changed')
    job = read(entry['job'])
    require(job['id'] == entry['id'] and job['output'] == entry['output'] and job['variant'] == entry['variant']
            and job['source_hashes'] == manifest['source_hashes'] and job['adapter_hashes'] == manifest['adapter_hashes']
            and job['protocol_sha256'] == manifest['protocol_sha256'], 'Original job/source identity drift')
    full = next(r for r in baseline['records'] if r['method'] == 'GuardFed-AD2+' and
                (r['distribution'], r['attack'], r['seed']) == (job['distribution'], job['attack'], job['config']['seed']))
    # Original v2 initializes Torch's8/1 thread pools exactly once, before the
    # original strict tensor checks. bind_runtime's later v3 import reuses replay.
    bootstrap = load('remaining620_original_v3_bootstrap', d['v3'], plan['dependency_sha256']['v3'])
    require(digest(bootstrap.v2.__file__) == plan['dependency_sha256']['v2']
            and Path(bootstrap.v2.__file__).resolve() == Path(d['v2']).resolve()
            and bootstrap.v2.torch.__version__ == '2.11.0+cu128'
            and bootstrap.v2.torch.get_num_threads() == 8 and bootstrap.v2.torch.get_num_interop_threads() == 1,
            'Original single-import CPU bootstrap/runtime differs')
    original, checked_worker = evidence.validators(REPO, adapter_dir, manifest)
    result = evidence.accept_new(original, checked_worker, job, entry, manifest, full)
    require(result is not None and result['config'] == job['config'], 'Original terminal70 strict acceptance absent')
    row = evidence.row_from_result(entry, result, 'new')
    require(evidence.live_worker(entry, REPO, adapter_dir) is None, 'Producer reappeared during strict check')
    require(all(digest(p) == expected for p, expected in row['files'].items()), 'Terminal changed during binding')
    record = construct_record(parent, job, entry, result, row, full)
    for relative, expected in manifest['source_hashes'].items():
        require(digest(REPO / relative) == expected, 'Source/data drift before atomic binding: ' + relative)
    for path, expected in manifest['adapter_hashes'].items():
        require(digest(path) == expected, 'Adapter drift before atomic binding: ' + path)
    require(digest(d['protocol']) == manifest['protocol_sha256'], 'Protocol drift before binding')
    native = dict(status='ORIGINAL_V4_TERMINAL_STRICT_PASS_REMOTE_ONLY', id=identity, row=row,
                  producer_exit_observation=observation, source_sha256=plan['dependency_sha256']['evidence_v4'])
    native_sha = canonical(native)
    inv = inventory(plan, record, native_sha); projected = project(parent, plan, identity, native_sha)
    projected.validate_inventory(inv, baseline)
    source_identity()
    source_before = {n: pin['sha256'] for n, pin in read(HERE / 'FILES_SHA256.json')['files'].items()}
    require(digest(review_path) == review_sha and all(digest(p) == expected for p, expected in row['files'].items()),
            'Review/terminal changed immediately before atomic binding')
    save_new(task / 'binding.json', dict(status='IMMUTABLE_TERMINAL_ONCE_BOUND', id=identity, record=record, native_acceptance=native,
             native_acceptance_canonical_sha256=native_sha, plan_sha256=digest(HERE / 'PLAN.json'), parent_review_sha256=review_sha, checkpoint_sha256=record['checkpoint']['sha256']))
    save_new(task / 'inventory.json', inv)
    approval = dict(status='APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY', scope=SCOPE,
        inventory_sha256=digest(task / 'inventory.json'), bridge_sha256=plan['dependency_sha256']['parent_bridge'],
        selected_ids=[identity], selected_id=identity, device='cpu', compute_threads=8, max_processes=1,
        allowed_cpus=CPUS, target_split='valid', native_tolerance=1e-12, final_test_dispatch=False, output=str(output),
        parent_queue_review_sha256=review_sha, binding_sha256=digest(task / 'binding.json'), queue_source_before=source_before)
    save_new(task / 'APPROVED.json', approval)
    require(digest(review_path) == review_sha and digest(task / 'binding.json') == approval['binding_sha256'], 'Root review/binding changed')
    runtime = projected.bind_runtime(task / 'inventory.json', digest(task / 'inventory.json'), d, REPO, identity,
                                     task / 'APPROVED.json', digest(task / 'APPROVED.json'))
    runtime['replay_one'](output, wall_seconds=1800)
    acceptance = runtime['accept_saved_predictions'](output)
    require(acceptance['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and acceptance['native_comparison']['accepted'], 'Original saved-array strict failed')
    require(source_before == {n: digest(HERE / n) for n in source_before}, 'Queue source changed during evaluation')
    require(digest(task / 'binding.json') == approval['binding_sha256'], 'Immutable terminal binding changed')
    resources_after = resource_gate(review)
    save_new(output / 'strict_acceptance.json', acceptance)
    save_new(task / 'REMOTE_COMPLETE.json', dict(status='REMOTE_STRICT_CLOSED_PENDING_OFFSERVER', id=identity,
        checkpoint_sha256=record['checkpoint']['sha256'], binding_sha256=digest(task / 'binding.json'),
        inventory_sha256=digest(task / 'inventory.json'), strict_sha256=digest(output / 'strict_acceptance.json'),
        prediction_arrays_sha256=digest(output / 'validation_predictions.npz'), native_difference=acceptance['native_comparison']['max_abs_difference'],
        queue_source_before=source_before, parent_review_sha256=review_sha, resources_before=resources_before, resources_after=resources_after,
        accepted_offserver=0, new_training=0, new_Full_inference=0, test=False))


def manage(plan, review, review_path, review_sha):
    policy(); d = dependencies(plan)
    require(not RUNTIME.exists(), 'No automatic resume; preserve existing attempt namespace')
    import fcntl
    lock = (HERE / 'queue.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    RUNTIME.mkdir()
    for name in ('runs', 'tasks'):
        (RUNTIME / name).mkdir()
    completed = []
    evidence = load('remaining620_wait_evidence', d['evidence_v4'], plan['dependency_sha256']['evidence_v4'])
    adapter_dir = Path(next(iter(plan['adapter_hashes']))).parent
    try:
        for entry in plan['entries']:
            identity = entry['id']
            while True:
                require(time.time() <= review['deadline_unix'], 'Finite experiment deadline reached')
                snapshot, observation = training_snapshot(plan, evidence)
                owner = evidence.live_worker(entry, REPO, adapter_dir)
                state = disposition(entry, snapshot, owner, Path(entry['output']).exists(), list(Path(entry['output']).glob('failure*.json')))
                temporary = RUNTIME / 'progress.next.json'
                save_new(temporary, dict(status=state, id=identity, remote_closed_ids=completed, remote_closed_n=len(completed),
                         accepted_offserver=0, last_producer_observation=observation, parent_review_sha256=review_sha))
                os.replace(temporary, RUNTIME / 'progress.json')
                if state == 'READY_FOR_ORIGINAL_STRICT':
                    break
                time.sleep(30)
            task = RUNTIME / 'tasks' / identity; task.mkdir()
            with (task / 'worker.log').open('x') as log:
                subprocess.run([sys.executable, '-B', '-u', str(HERE / 'evaluate_remaining.py'), 'worker', '--review', str(review_path),
                                '--review-sha256', review_sha, '--id', identity], stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1900)
            receipt = read(task / 'REMOTE_COMPLETE.json')
            require(receipt['id'] == identity and receipt['status'] == 'REMOTE_STRICT_CLOSED_PENDING_OFFSERVER'
                    and receipt['parent_review_sha256'] == review_sha and receipt['accepted_offserver'] == 0, 'Child closure incomplete')
            out = RUNTIME / 'runs' / identity
            strict = read(out / 'strict_acceptance.json'); bound = read(task / 'binding.json')
            require(digest(out / 'strict_acceptance.json') == receipt['strict_sha256']
                    and digest(out / 'validation_predictions.npz') == receipt['prediction_arrays_sha256']
                    and digest(task / 'binding.json') == receipt['binding_sha256'], 'Closed-child artifact hashes changed')
            require(strict['id'] == bound['id'] == identity and strict['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED'
                    and strict['native_comparison']['accepted'] and strict['native_comparison']['max_abs_difference'] <= 1e-12
                    and strict['checkpoint_sha256'] == bound['checkpoint_sha256'] == receipt['checkpoint_sha256'], 'Child strict/native/checkpoint identity failed')
            completed.append(identity)
        save_new(RUNTIME / 'QUEUE_COMPLETE.json', dict(status='ALL620_REMOTE_STRICT_CLOSED_PENDING_OFFSERVER', ids=completed,
                 accepted_offserver=0, new_training=0, new_Full_inference=0, test=False, original180_unchanged=True))
    except BaseException as exc:
        save_new(RUNTIME / 'QUEUE_FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc(), remote_closed_ids=completed,
                 accepted_offserver=0, automatic_retry=False, last_producer_observation=locals().get('observation')))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'manage', 'worker'])
    parser.add_argument('--review', type=Path); parser.add_argument('--review-sha256'); parser.add_argument('--id')
    args = parser.parse_args()
    try:
        if args.action == 'inspect':
            p = source_identity(); print(json.dumps(dict(status='SOURCE_ONLY_PREPARED', remaining=len(p['remaining620_ids']), CNN=0)))
        else:
            p, a = reviewed(args.review, args.review_sha256)
            if args.action == 'manage': manage(p, a, args.review, args.review_sha256)
            else: worker(p, a, args.id, args.review, args.review_sha256)
    except BaseException as exc:
        # Before external approval, stderr is the only effect. Preserve operational
        # failures inside the approved attempt; never fabricate scientific output.
        if args.action != 'inspect' and locals().get('a'):
            error = RUNTIME / ('COMMAND_FAILURE_' + str(os.getpid()) + '.json')
            save_new(error, dict(command=sys.argv, error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False))
        raise
