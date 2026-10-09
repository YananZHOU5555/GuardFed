"""Sequential, failstop wrapper for the exact remaining464; no Torch or cohort writes."""
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
REMOTE = Path('/workspace/guardfed_checks/celeba_valid_gpu_remaining464_prepared_20261009')
RECOVERY = Path('/workspace/guardfed_checks/celeba_valid_gpu_recovery_implementation_20261009')
PROPOSAL = Path('/workspace/guardfed_checks/celeba_valid_recovery_prepared_20261009')
IMPL_SHA = '6ae15988b5d0b1ebe4371166afa99bceca015394cba8ecc6f987773621d55b56'
PROPOSAL_SHA = '125ab344736a0bf306be99fefc71ca701daf7fc0c337f76b3da7dd62e24c5058'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def need(ok, message):
    if not ok: raise ValueError(message)


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def validate(m, proposal, prior, proof, a, package_sha):
    exact = [r['id'] for r in proposal['records'] if r['classification'] == 'UNEXECUTED_465' and r['id'] != prior['new_GPU_id']]
    need(len(exact) == len(set(exact)) == 464 and m['ids'] == exact, 'Not the exact ordered remaining464')
    need(set(prior['accepted_ids']) == set(proposal['accepted424_ids']) | {prior['new_GPU_id']} and len(set(prior['accepted_ids'])) == prior['accepted_n'] == 425, 'Prior425 identity changed')
    need(proof['accepted_new_ids'] == [prior['new_GPU_id']] and proof['status'] == 'ROOT_FIRST_GPU_VALID_REPLAY_OFFSERVER_AND_SAVED_ARRAY_PASS' and proof['source_package_sha256'] == IMPL_SHA and proof['native_max_abs_difference'] == 0 and proof['archive_members_verified'] == 73, 'First1 not strictly offserver accepted')
    need(m['chunks'] == [{'index': n // 11, 'ids': exact[n:n + 11]} for n in range(0, 464, 11)] and not set(exact) & set(prior['accepted_ids']), 'Chunk order/size/duplicate changed')
    need(a.get('status') == 'ROOT_APPROVED_GPU_VALID_RECOVERY_V1' and a.get('execute_remaining464') is True and a.get('queue_package_sha256') == package_sha and a.get('queue_manifest_sha256') == sha(HERE / 'manifest.json'), 'No SHA-bound queue review')
    need(a.get('implementation_package_sha256') == IMPL_SHA and a.get('proposal_manifest_sha256') == PROPOSAL_SHA and a.get('accepted425_collector_sha256') == m['accepted425_sha256'] and a.get('first1_offserver_proof_sha256') == m['first1_proof_sha256'], 'Queue/prior/scientific source binding drift')
    need(a.get('approved_ids') == exact and a.get('output_parent') == m['output_parent'] and a.get('orchestrator_CPU') == 106, 'Queue ID/output/CPU boundary changed')
    need(a.get('gpu_uuid') == prior['new_GPU_uuid'] and a.get('import_cpu_partial10') is False and a.get('import_gpu_diagnostic1') is False, 'Device/import permission changed')
    need(a.get('execute_new465') is True and a.get('inventory_sha256') == proposal['inventory_sha256'] and a.get('accepted424_collector_sha256') == proposal['accepted424_collector_sha256'], 'Original recovery review binding drift')
    need(all(a.get(k) == v for k, v in {'CPU': 105, 'threads': 1, 'interop_threads': 1, 'host_GPU_index': 0, 'max_GPU_workers': 1, 'nice': 10, 'io_class': 'idle', 'native_tolerance': 1e-12, 'target_split': 'valid', 'views': ['native', 'raw', 'shared_calibration']}.items()), 'Original worker/science boundary changed')
    need(all(a.get(k) is False for k in ('test', 'training', 'automatic_retry', 'restart_old872', 'change_accepted424', 'final_freeze')), 'Prohibited operation')


def contract(args):
    need(sha(HERE / 'PACKAGE_SHA256.json') == args.package_sha256 and sha(args.review) == args.review_sha256, 'External package/review SHA changed')
    for folder in (HERE, RECOVERY):
        if folder == RECOVERY: need(sha(folder / 'PACKAGE_SHA256.json') == IMPL_SHA, 'Sealed recovery package changed')
        for name, row in read(folder / 'PACKAGE_SHA256.json')['members'].items():
            p = folder / name
            need(p.resolve().is_relative_to(folder) and not p.is_symlink() and sha(p) == row['sha256'] and p.stat().st_size == row['bytes'], 'Source member changed: ' + name)
    m = read(HERE / 'manifest.json'); a = read(args.review)
    need(sha(PROPOSAL / 'manifest.json') == PROPOSAL_SHA and sha(HERE / 'prior/cumulative_425_accepted.json') == m['accepted425_sha256'] and sha(HERE / 'prior/ROOT_OFFSERVER_VERIFICATION.json') == m['first1_proof_sha256'], 'Input identity changed')
    validate(m, read(PROPOSAL / 'manifest.json'), read(HERE / 'prior/cumulative_425_accepted.json'), read(HERE / 'prior/ROOT_OFFSERVER_VERIFICATION.json'), a, args.package_sha256)
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
                        need(strict['status'] == 'SELECTED_VALID_REPLAY_ACCEPTED' and strict['scope'] == 'EXISTING_BASELINE900_GPU_VALID_RECOVERY_V1' and strict['accepted_ids'] == ids and not strict['invalid'] and strict['max_abs_native_metric_difference'] <= 1e-12, 'Strict incomplete/mixed IDs/native mismatch')
                        extra += ['--strict-sha256', sha(stage / 'strict_acceptance.json')]
                        save(stage / 'queue_binding.json', {'queue_package_sha256': args.package_sha256, 'queue_manifest_sha256': sha(HERE / 'manifest.json'), 'root_review_sha256': args.review_sha256, 'prior425_sha256': m['accepted425_sha256'], 'cohort_registered': False})
                    need(runner(common + [command] + auth + extra, log) == 0, 'Failstop at chunk%d %s; preserve output, no retry' % (index, command))
            archive = read(stage / 'remote_archive_inventory.json')
            need(archive['status'] == 'REMOTE_STRICT_ACCEPTED_ARCHIVE_VERIFIED_PENDING_OFFSERVER' and archive['accepted_ids'] == ids and archive['chunk_index'] == index and archive['old_model_files_archived'] == 0 and archive['offserver_verified'] is False and sha(stage / archive['archive']) == archive['sha256'], 'Remote archive closure drift')
            closed.extend(ids)
            receipt = {'status': 'REMOTE_CLOSED_PENDING_OFFSERVER', 'chunk_index': index, 'remote_closed_ids': ids, 'remote_closed_cumulative_n': len(closed), 'strict_sha256': sha(stage / 'strict_acceptance.json'), 'archive_sha256': archive['sha256'], 'archive_inventory_sha256': sha(stage / 'remote_archive_inventory.json'), 'previous_receipt_sha256': previous, 'queue_package_sha256': args.package_sha256, 'review_sha256': args.review_sha256, 'offserver_verified': False, 'cohort_registered': False, 'accepted_new_n': 0}
            path = parent / ('chunk_%03d.REMOTE_PENDING_OFFSERVER.json' % index); save(path, receipt); previous = sha(path)
            print(json.dumps(receipt), flush=True)
        save(parent / 'queue_exit.json', {'status': 'ALL464_REMOTE_CLOSED_PENDING_OFFSERVER', 'remote_closed_n': len(closed), 'remote_closed_ids': closed, 'last_receipt_sha256': previous, 'accepted_new_n': 0, 'cohort_registered': False})
    except BaseException as error:
        save(parent / 'queue_failure.json', {'status': 'FAILSTOP_PRESERVED_NO_RETRY', 'error': repr(error), 'traceback': traceback.format_exc(), 'remote_closed_ids': closed, 'failed_chunk': index, 'accepted_new_n': 0, 'cohort_registered': False}); raise


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=('inspect', 'run')); p.add_argument('--review', type=Path, required=True); p.add_argument('--review-sha256', required=True); p.add_argument('--package-sha256', required=True); args = p.parse_args()
    m, a = contract(args)
    if args.command == 'inspect': print(json.dumps({'status': 'EXACT464_SOURCE_AND_REVIEW_VERIFIED_NO_INFERENCE', 'ids': 464, 'chunks': 43})); return
    need({105, 106} <= os.sched_getaffinity(0), 'CPU105/106 unavailable')
    os.sched_setaffinity(0, {106}); priority = os.getpriority(os.PRIO_PROCESS, 0)
    if priority < 10: os.nice(10 - priority)
    need(os.getpriority(os.PRIO_PROCESS, 0) == 10, 'Orchestrator nice must equal10')
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    import fcntl
    with (Path(a['output_parent']).parent / 'remaining464.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(_signal, _frame): raise InterruptedError('Explicit stop; preserve partial without retry')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        execute(m, a, args)


if __name__ == '__main__':
    try: main()
    except BaseException:
        traceback.print_exc(); raise SystemExit(1)
