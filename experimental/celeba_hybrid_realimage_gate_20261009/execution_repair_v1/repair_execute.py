"""Execution-only undefined-diagnostic repair; two S-DFA canaries need external approval."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import types

from writer_policy import sanitize_result, check_sidecar, require

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
RUNTIME = HERE / 'runtime_overlay'
GATE_SHA = '22d6a02b49eb229d46f70bf259a7a3972d34bd392c7ecb28645d9909ba731b40'
CORE_SHA = 'cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed'
ORIGINAL_ARTIFACTS = {'result.json', 'model.pt', 'diagnostics.json', 'provenance.json',
    'resource_before.json', 'resource_after.json', 'rng_final.json', 'native_replay.json'}


def load_gate():
    import hashlib
    require(hashlib.sha256((ORIGINAL / 'gate.py').read_bytes()).hexdigest() == GATE_SHA, 'Sealed gate changed')
    spec = importlib.util.spec_from_file_location('sealed_hybrid_repair_gate', ORIGINAL / 'gate.py')
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    return gate


def clone(function, **overrides):
    environment = dict(function.__globals__, **overrides)
    return types.FunctionType(function.__code__, environment, function.__name__, function.__defaults__, function.__closure__)


def inspect():
    gate = load_gate()
    seal = gate.read(HERE / 'FILES_SHA256.json')
    for name, expected in seal['files'].items():
        require(gate.digest(HERE / name) == expected, 'Repair attachment changed: ' + name)
    scope = gate.read(HERE / 'REPAIR_SCOPE.json')
    require(scope['status'] == 'PREPARED_NOT_APPROVED' and scope['scientific_table_records'] == 0
            and scope['original_failed_attempt_status'] == 'TERMINAL_FAILURE', 'Wrong stage')
    require(gate.digest(ORIGINAL / 'scope.json') == scope['original_scope_sha256'], 'Original scope changed')
    original = gate.read(ORIGINAL / 'scope.json')
    require(scope['new_jobs'] == original['jobs'][2:] and scope['reused_IID_ids'] == [x['id'] for x in original['jobs'][:2]],
            'Only two original unaccepted S-DFA jobs may be rerun')
    for name, expected in original['local_hashes'].items():
        require(gate.digest(ORIGINAL / name) == expected, 'Original scientific snapshot changed')
    for entry in original['jobs']:
        require(gate.digest(ORIGINAL / entry['job']) == entry['job_sha256'], 'Original job changed')
    require(scope['test_evaluation_authorized'] is False and scope['formal32_authorized'] is False
            and scope['max_concurrent_cpu_processes'] == 1 and scope['cpu_threads'] == 8, 'Scope widened')
    return gate, scope, original


def validate_approval(gate, scope, approved_path):
    require(approved_path is not None and approved_path.is_file(), 'Root review and external APPROVED receipt required')
    approved = gate.read(approved_path)
    require(approved['status'] == 'APPROVED_TWO_UNACCEPTED_HYBRID_CANARIES_DIAGNOSTIC_WRITER_ONLY'
            and approved['repair_scope_sha256'] == gate.digest(HERE / 'REPAIR_SCOPE.json')
            and approved['repair_seal_sha256'] == gate.digest(HERE / 'FILES_SHA256.json'), 'Approval/source identity mismatch')
    require(approved['jobs'] == {x['id']: x['job_sha256'] for x in scope['new_jobs']}
            and approved['exclusive_cpu_ids'] == list(range(8, 16)) and approved['cpu_threads'] == 8,
            'Only exact approved pair and eight CPUs supported')
    require(approved['no_live_compute_overlap_verified'] is True and approved['formal32_authorized'] is False
            and approved['test_authorized'] is False and approved['automatic_retry_authorized'] is False,
            'Bounded approval cannot authorize scientific changes or retry')
    return approved


def functions(gate, scope, original):
    entries = {entry['id']: entry for entry in scope['new_jobs']}
    original_check = clone(gate.checked, HERE=RUNTIME)

    def checked(entry, old_scope):
        require(entry['id'] in entries and entry == entries[entry['id']], 'Unapproved job in repaired checker')
        out = RUNTIME / entry['output']
        if not (out / 'result.json').exists():
            return None
        receipt = gate.read(out / 'acceptance.json')
        require(set(receipt['artifact_hashes']) == ORIGINAL_ARTIFACTS | {'undefined_diagnostics.json'},
                'All original acceptance artifacts plus mandatory sidecar required')
        result = original_check(entry, old_scope)
        sidecar = gate.read(out / 'undefined_diagnostics.json')
        require(sidecar['sanitized_result_sha256'] == gate.digest(out / 'result.json')
                and sidecar['repair_source_sha256'] == gate.digest(__file__)
                and sidecar['writer_policy_sha256'] == gate.digest(HERE / 'writer_policy.py')
                and sidecar['repair_scope_sha256'] == gate.digest(HERE / 'REPAIR_SCOPE.json'), 'Repair identity differs')
        check_sidecar(result, sidecar, gate.read(RUNTIME / entry['job']), entry['job_sha256'], GATE_SHA, CORE_SHA)
        return result

    def writer(path, value):
        path = Path(path)
        for entry in entries.values():
            out = RUNTIME / entry['output']
            if path == out / 'result.json':
                job_path = RUNTIME / entry['job']; job = gate.read(job_path)
                clean, undefined = sanitize_result(value, job, gate.digest(job_path), entry['job_sha256'])
                gate.write(path, clean)  # Original strict allow_nan=False writer remains unchanged.
                gate.write(out / 'undefined_diagnostics.json', dict(status='EXPLICIT_UNDEFINED_DIAGNOSTIC_ONLY',
                    job_sha256=entry['job_sha256'], original_gate_sha256=GATE_SHA, original_core_sha256=CORE_SHA,
                    repair_source_sha256=gate.digest(__file__), writer_policy_sha256=gate.digest(HERE / 'writer_policy.py'),
                    repair_scope_sha256=gate.digest(HERE / 'REPAIR_SCOPE.json'),
                    sanitized_result_sha256=gate.digest(path), undefined_values=undefined))
                return
            if path == out / 'acceptance.json':
                require(set(value['artifact_hashes']) == ORIGINAL_ARTIFACTS, 'Original acceptance contract changed')
                value = dict(value, artifact_hashes=dict(value['artifact_hashes'],
                    **{'undefined_diagnostics.json': gate.digest(out / 'undefined_diagnostics.json')}))
                break
        gate.write(path, value)  # Unknown paths and all other nonfinite fields remain refused.

    return clone(gate.run_one, HERE=RUNTIME, checked=checked, write=writer), checked, clone(gate.compare, HERE=RUNTIME, checked=checked)


def run(approved_path):
    gate, scope, original = inspect()
    approved = validate_approval(gate, scope, approved_path)
    require(sys.platform == 'linux' and not sys.flags.optimize, 'Linux assertions must remain enabled')
    gate.verify_scope(original)
    # Acceptance references are hash checked before any output or Torch import.
    for row in gate.read(HERE / 'reused_IID_references.json')['records']:
        out = ORIGINAL / row['output']
        require(gate.digest(out / 'acceptance.json') == row['acceptance_sha256'], 'Reused IID receipt changed')
        for name, sha in row['artifact_hashes'].items():
            require(gate.digest(out / name) == sha, 'Reused IID artifact changed')
    require(not RUNTIME.exists(), 'Existing/partial repaired attempt blocks automatic retry')
    import fcntl
    lock = (HERE / 'execution.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpus = approved['exclusive_cpu_ids']; require(set(cpus) <= os.sched_getaffinity(0), 'Allocated CPUs unavailable')
    os.sched_setaffinity(0, cpus)
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Launch with nice10 or lower priority')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '8'
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    require(torch.__version__ == '2.11.0+cu128' and torch.version.cuda == '12.8', 'Isolated environment differs')
    RUNTIME.mkdir()
    for name in {'scope.json', *original['local_hashes'], *(x['job'] for x in original['jobs'])}:
        target = RUNTIME / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ORIGINAL / name, target)
        require(gate.digest(target) == gate.digest(ORIGINAL / name), 'Overlay copy differs')
    (RUNTIME / 'runs').mkdir()
    clone(gate.verify_scope, HERE=RUNTIME)(original)
    gate.write(RUNTIME / 'dispatch_receipt.json', approved)
    before = gate.snapshot(); gate.write(RUNTIME / 'protected_before.json', before)
    runner, checker, compare = functions(gate, scope, original)
    worker, core = gate.modules()
    require(gate.read(gate.SEALED / 'protocol.json')['status'] == 'PREPARED_NOT_FROZEN', 'Formal gate widened')
    try:
        worker.validate_job(gate.read(ORIGINAL / 'sealed_formal_job.json'), gate.read(gate.SEALED / 'protocol.json'))
    except ValueError as error:
        require('not frozen' in str(error), 'Unexpected original formal rejection')
    else:
        raise ValueError('Original prepared formal screen must remain rejected')
    try:
        reused = gate.compare(dict(original, jobs=original['jobs'][:2]))  # Read-only saved checkpoint checks; no inference.
        for entry in scope['new_jobs']:
            runner(entry, original, worker, core)
            require(checker(entry, original) is not None, 'Unaccepted repaired canary')
        pairs = compare(dict(original, jobs=scope['new_jobs']))
        gate.verify_scope(original); clone(gate.verify_scope, HERE=RUNTIME)(original)
        after = gate.snapshot(); gate.write(RUNTIME / 'protected_after.json', after)
        previous = {x['id']: x['round'] for x in before['active']}
        grew = after['completed'] > before['completed'] or any(x['id'] in previous and x['round'] is not None
            and previous[x['id']] is not None and x['round'] > previous[x['id']] for x in after['active'])
        require(not after['failed'] and grew, 'Protected formal queue failed or did not advance')
        gate.write(RUNTIME / 'repair_summary.json', dict(status='PASS_TWO_NEW_CANARIES_TWO_REUSED_IID_CANARIES',
            new_actual_runs=2, new_actual_rounds=6, reused_actual_runs=2, scientific_table_records=0,
            original_failed_attempt_status='TERMINAL_FAILURE', original_failed_attempt_reclassified=False,
            reused_pairs=reused, new_pairs=pairs, formal_gpu_progress_grew=True, test_evaluated=False,
            formal_screen_status='PREPARED_NOT_FROZEN'))
    except BaseException as error:
        import traceback
        gate.write(RUNTIME / 'repair_failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'run']); parser.add_argument('--approved', type=Path)
    args = parser.parse_args()
    if args.action == 'inspect':
        _, scope, _ = inspect(); print(scope['status'], len(scope['new_jobs']), 'new canaries; no execution')
    else:
        run(args.approved)
