"""Exact fresh440, sequential failstop; no Torch, automatic resume or cohort writes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REMOTE = Path('/workspace/guardfed_checks/celeba_valid_gpu_remaining440_resource_gate_v2_prepared_20261009')
RECOVERY = Path('/workspace/guardfed_checks/celeba_valid_gpu_resource_gate_fix_20261009/release')
PROPOSAL = Path('/workspace/guardfed_checks/celeba_valid_recovery_prepared_20261009')
INVENTORY = Path('/workspace/guardfed_checks/celeba_final_valid_replay_20261009/inputs/model_inventory.json')
IMPL_SHA = '7be4686d5117eaab9b4fb57da5ae345f2446b60dfd31292f8a5b931a2ce3b5b8'
PROPOSAL_SHA = '125ab344736a0bf306be99fefc71ca701daf7fc0c337f76b3da7dd62e24c5058'
SCOPE = 'EXISTING_BASELINE900_GPU_VALID_RECOVERY_RESOURCE_GUARD_V2'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def need(ok, message):
    if not ok: raise ValueError(message)


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def validate(m, inventory, proposal, prior, parent, proof, guard, a, package_sha):
    all_ids = [r['id'] for r in inventory['records']]; accepted = prior['accepted_ids']
    need(len(all_ids) == len(set(all_ids)) == 900 and len(accepted) == len(set(accepted)) == prior['accepted_n'] == 460 and set(accepted) <= set(all_ids), 'Inventory900/prior460 identity changed')
    exact = [identity for identity in parent['ids'] if identity not in set(accepted)]
    need(len(parent['ids']) == len(set(parent['ids'])) == 464 and len(exact) == 440 and set(exact) == set(all_ids) - set(accepted) and m['ids'] == exact, 'Not the exact inventory900 minus460 in parent464 order')
    need(m['chunks'] == [{'index': n // 11, 'ids': exact[n:n + 11]} for n in range(0, 440, 11)], 'Chunk order/size/duplicate changed')
    need(prior['expected_n'] == 900 and prior['previous_collector_sha256'] == m['historical458_sha256'] and prior['new_proof_sha256'] == m['partial2_proof_sha256'] and prior['added_ids'] == proof['accepted_new_ids'], 'Prior460/458/partial2 chain changed')
    need(proof['status'] == 'ROOT_EXPLICIT_GPU_PARTIAL2_ORIGINAL_STRICT_AND_SAVED_ARRAY_PASS' and proof['new_n'] == 2 and proof['prior458_sha256'] == m['historical458_sha256'] and proof['original_failure_preserved'] and proof['new_CNN_inference'] == 0 and all(r['native_max_abs_difference'] == 0 for r in proof['results']), 'Explicit partial2 acceptance proof changed')
    need(guard['status'] == 'ROOT_RESOURCE_GUARD_V2_DIFF_AND_NO_CNN_REVIEW_PASS_PREPARED' and guard['runtime_seal_sha256'] == IMPL_SHA and guard['accepted'] == 460 and guard['remaining'] == 440 and guard['science_body_unchanged'] and guard['native_tolerance'] == 1e-12, 'GuardV2 source review changed')
    need(a.get('status') == 'ROOT_APPROVED_GPU_VALID_RECOVERY_V1' and a.get('execute_remaining440') is True and a.get('queue_package_sha256') == package_sha and a.get('queue_manifest_sha256') == sha(HERE / 'manifest.json'), 'No SHA-bound fresh440 queue review')
    for key in ('accepted460_collector_sha256', 'partial2_proof_sha256', 'parent464_manifest_sha256', 'parent464_package_sha256', 'resource_guard_root_review_sha256'):
        need(a.get(key) == m[key], 'Queue parent/source binding drift: ' + key)
    need(a.get('implementation_package_sha256') == IMPL_SHA and a.get('proposal_manifest_sha256') == PROPOSAL_SHA and a.get('recovery_scope') == SCOPE, 'Scientific/source version binding drift')
    need(a.get('approved_ids') == exact and a.get('output_parent') == m['output_parent'] and a.get('orchestrator_CPU') == 106, 'Queue ID/output/CPU boundary changed')
    need({r['GPU_uuid'] for r in proof['results']} == {a.get('gpu_uuid')} and a.get('import_cpu_partial10') is False and a.get('import_gpu_diagnostic1') is False, 'Device/import permission changed')
    need(a.get('execute_new465') is True and a.get('inventory_sha256') == m['inventory_sha256'] == proposal['inventory_sha256'] and a.get('accepted424_collector_sha256') == proposal['accepted424_collector_sha256'], 'Original recovery review binding drift')
    need(all(a.get(k) == v for k, v in {'CPU': 105, 'threads': 1, 'interop_threads': 1, 'host_GPU_index': 0, 'max_GPU_workers': 1, 'nice': 10, 'io_class': 'idle', 'native_tolerance': 1e-12, 'target_split': 'valid', 'views': ['native', 'raw', 'shared_calibration']}.items()), 'Original worker/science boundary changed')
    need(all(a.get(k) is False for k in ('test', 'training', 'automatic_retry', 'restart_old872', 'restart_old464', 'change_accepted424', 'change_accepted460', 'final_freeze')), 'Prohibited operation')


def contract(args):
    need(sha(HERE / 'PACKAGE_SHA256.json') == args.package_sha256 and sha(args.review) == args.review_sha256, 'External package/review SHA changed')
    for folder in (HERE, RECOVERY):
        if folder == RECOVERY: need(sha(folder / 'PACKAGE_SHA256.json') == IMPL_SHA, 'Sealed recovery package changed')
        for name, row in read(folder / 'PACKAGE_SHA256.json')['members'].items():
            p = folder / name
            need(p.resolve().is_relative_to(folder) and not p.is_symlink() and sha(p) == row['sha256'] and p.stat().st_size == row['bytes'], 'Source member changed: ' + name)
    m = read(HERE / 'manifest.json'); a = read(args.review)
    need(sha(PROPOSAL / 'manifest.json') == PROPOSAL_SHA and sha(INVENTORY) == m['inventory_sha256'], 'Original proposal/inventory changed')
    paths = {'accepted460_collector_sha256': 'cumulative_460_accepted.json', 'parent464_manifest_sha256': 'parent464_manifest.json', 'partial2_proof_sha256': 'ROOT_OFFSERVER_IMPORT_VERIFICATION.json', 'resource_guard_root_review_sha256': 'GPU_RESOURCE_GUARD_V2_ROOT_REVIEW.json'}
    for key, name in paths.items(): need(sha(HERE / 'prior' / name) == m[key], 'Prior source changed: ' + name)
    validate(m, read(INVENTORY), read(PROPOSAL / 'manifest.json'), *(read(HERE / 'prior' / paths[k]) for k in paths), a, args.package_sha256)
    need(HERE == REMOTE and sys.platform == 'linux' and not sys.flags.optimize and sha('/etc/vast-agents-guide.md') == m['guide_sha256'], 'Exact Linux path/guide required')
    return m, a


def invoke(command, log):
    # A CPU106 parent must not pass its restricted affinity to recovery.bind_cpu().
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, preexec_fn=lambda: os.sched_setaffinity(0, {105}))
    try: return child.wait()
    except BaseException:
        child.terminate()
        try: child.wait(timeout=40)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
        raise


def execute(m, a, args, runner=invoke):
    parent = Path(a['output_parent']); need(parent.parent.resolve() == parent.parent and not parent.exists(), 'Fresh queue output required; no automatic resume')
    parent.mkdir(); closed = []; previous = None; index = None
    try:
        for chunk in m['chunks']:
            need(sha(args.review) == args.review_sha256 and sha(HERE / 'manifest.json') == a['queue_manifest_sha256'] and sha(HERE / 'remaining.py') == m['queue_source_sha256'], 'Queue review/source changed between chunks')
            index, ids = chunk['index'], chunk['ids']; stage = parent / ('chunk_%03d' % index)
            common = [sys.executable, str(RECOVERY / 'recovery.py')]
            auth = ['--review', str(args.review), '--review-sha256', args.review_sha256, '--package-sha256', IMPL_SHA]
            with (parent / ('chunk_%03d.commands.log' % index)).open('xb') as log:
                commands = [('run-chunk', ['--ids', *ids, '--output', str(stage)]), ('accept', ['--batch', str(stage / 'batch'), '--output', str(stage / 'strict_acceptance.json')]), ('backup', ['--stage', str(stage), '--chunk-index', str(index)])]
                for command, extra in commands:
                    if command == 'backup':
                        strict = read(stage / 'strict_acceptance.json')
                        need(strict['status'] == 'SELECTED_VALID_REPLAY_ACCEPTED' and strict['scope'] == SCOPE and strict['accepted_ids'] == ids and not strict['invalid'] and strict['max_abs_native_metric_difference'] <= 1e-12, 'Strict incomplete/mixed IDs/native mismatch')
                        extra += ['--strict-sha256', sha(stage / 'strict_acceptance.json')]
                        save(stage / 'queue_binding.json', {'queue_package_sha256': args.package_sha256, 'queue_manifest_sha256': sha(HERE / 'manifest.json'), 'root_review_sha256': args.review_sha256, 'prior460_sha256': m['accepted460_collector_sha256'], 'implementation_package_sha256': IMPL_SHA, 'recovery_scope': SCOPE, 'cohort_registered': False})
                    need(runner(common + [command] + auth + extra, log) == 0, 'Failstop at chunk%d %s; preserve output, no retry' % (index, command))
            archive = read(stage / 'remote_archive_inventory.json')
            need(archive['status'] == 'REMOTE_STRICT_ACCEPTED_ARCHIVE_VERIFIED_PENDING_OFFSERVER' and archive['accepted_ids'] == ids and archive['chunk_index'] == index and archive['old_model_files_archived'] == 0 and archive['offserver_verified'] is False and sha(stage / archive['archive']) == archive['sha256'], 'Remote archive closure drift')
            closed.extend(ids)
            receipt = {'status': 'REMOTE_CLOSED_PENDING_OFFSERVER', 'chunk_index': index, 'remote_closed_ids': ids, 'remote_closed_cumulative_n': len(closed), 'strict_sha256': sha(stage / 'strict_acceptance.json'), 'archive_sha256': archive['sha256'], 'archive_inventory_sha256': sha(stage / 'remote_archive_inventory.json'), 'previous_receipt_sha256': previous, 'queue_package_sha256': args.package_sha256, 'implementation_package_sha256': IMPL_SHA, 'recovery_scope': SCOPE, 'review_sha256': args.review_sha256, 'offserver_verified': False, 'cohort_registered': False, 'accepted_new_n': 0}
            path = parent / ('chunk_%03d.REMOTE_PENDING_OFFSERVER.json' % index); save(path, receipt); previous = sha(path)
            print(json.dumps(receipt), flush=True)
        save(parent / 'queue_exit.json', {'status': 'ALL440_REMOTE_CLOSED_PENDING_OFFSERVER', 'remote_closed_n': len(closed), 'remote_closed_ids': closed, 'last_receipt_sha256': previous, 'accepted_new_n': 0, 'cohort_registered': False})
    except BaseException as error:
        save(parent / 'queue_failure.json', {'status': 'FAILSTOP_PRESERVED_NO_RETRY', 'error': repr(error), 'traceback': traceback.format_exc(), 'remote_closed_ids': closed, 'failed_chunk': index, 'accepted_new_n': 0, 'cohort_registered': False}); raise


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=('inspect', 'run')); p.add_argument('--review', type=Path, required=True); p.add_argument('--review-sha256', required=True); p.add_argument('--package-sha256', required=True); args = p.parse_args()
    m, a = contract(args)
    if args.command == 'inspect': print(json.dumps({'status': 'EXACT440_SOURCE_AND_REVIEW_VERIFIED_NO_INFERENCE', 'ids': 440, 'chunks': 40})); return
    need({105, 106} <= os.sched_getaffinity(0), 'CPU105/106 unavailable')
    os.sched_setaffinity(0, {106}); priority = os.getpriority(os.PRIO_PROCESS, 0)
    if priority < 10: os.nice(10 - priority)
    need(os.getpriority(os.PRIO_PROCESS, 0) == 10, 'Orchestrator nice must equal10')
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    import fcntl
    with (Path(a['output_parent']).parent / 'remaining440_resource_gate_v2.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(_signal, _frame): raise InterruptedError('Explicit stop; preserve partial without retry')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        execute(m, a, args)


if __name__ == '__main__':
    try: main()
    except BaseException:
        traceback.print_exc(); raise SystemExit(1)
