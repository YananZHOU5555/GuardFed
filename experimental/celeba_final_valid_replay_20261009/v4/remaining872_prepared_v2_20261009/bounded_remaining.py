"""Thin, fail-stopping chunks around the unchanged v4 run/accept interfaces."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
REMOTE = PurePosixPath('/workspace/guardfed_checks/celeba_final_valid_replay_20261009')
STORE = PurePosixPath('/workspace/guardfed_checks/celeba_validation900_restore_20261009')
PYTHON = '/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
REPO = PurePosixPath('/workspace/GuardFed-celeba-expanded')
V4_SHA = '43b16d20d2497b7762cd0f6039f7f4bfc8fddd4e4c32979a16291b26e394ae5e'
INVENTORY_SHA = '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
CUMULATIVE_SHA = '11ae999294efb3ed663a8cc77d85bbd930a5fe64f3545380015255d86063852f'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def validate_contract(m, inventory, prior):
    require(m['status'] == 'PREPARED_NOT_DISPATCHED' and m['scope'] == 'NINE_METHOD900_VALID_ONLY_REMAINING872', 'Wrong scope or preparation state')
    require(m['inventory_sha256'] == INVENTORY_SHA and m['prior_cumulative_sha256'] == CUMULATIVE_SHA, 'Original inventory/prior acceptance changed')
    require(prior['accepted_n'] == len(set(prior['accepted_ids'])) == 28 and not prior['all900_native_valid_replayed'], 'Expected the actual28 accepted IDs')
    stock = [r['id'] for r in inventory['records']]
    require(len(stock) == len(set(stock)) == 900, 'Inventory IDs are not exactly900 unique scientific cells')
    expected = [i for i in stock if i not in set(prior['accepted_ids'])]
    require(m['accepted28_ids'] == prior['accepted_ids'] and m['remaining_ids'] == expected and len(expected) == 872, 'Remaining IDs must be the exact ordered complement of actual28')
    require(m['prior_provenance'] == prior['accepted_provenance'], 'Reviewed prior proof/archive/source bindings changed')
    require(m['workers'] == 11 and m['threads_per_worker'] == 8 and m['max_wall_seconds_per_model'] == 1800, 'Frozen worker/thread/time bounds changed')
    require(m['cpu_allowed_list_indices'] == list(range(16, 104)) and m['native_tolerance'] == 1e-12, 'CPU slots or native tolerance changed')
    require(m['target_split'] == 'valid' and m['target_n'] == 19867 and m['clean_root_n'] == 16277 and not m['test_image_inference'] and not m['new_training'], 'Scientific evaluation scope changed')
    chunks = [expected[i:i + 11] for i in range(0, len(expected), 11)]
    require(m['chunks'] == [{'index': n, 'ids': ids} for n, ids in enumerate(chunks)] and len(chunks) == 80, 'Missing, duplicate or reordered chunk/seed')
    require(m['output'] == str(REMOTE / 'v4/remaining872_attempt1'), 'Unreviewed or historical output path')
    require(m['autostart'] is False and m['autorestart'] is False and m['automatic_retry'] is False, 'Automatic restart/retry is forbidden')
    require(m['outer_coordinator_nice'] == 0 and m['cnn_worker_nice'] == 10, 'Coordinator/worker priority contract changed')
    expected_cli = {'inventory': str(REMOTE / 'inputs/model_inventory.json'), 'storage-map': str(STORE / 'storage_map.json'), 'storage-map-sha256': 'e160416101ccb82b42224c9ff1bb337de45ef72fa251197063d1c15e2910f949', 'restore-receipt': str(STORE / 'restore_bundle.json'), 'restore-receipt-sha256': '0cfa391b0d317622551d4c6526f2d14ec33c3dda0f3e9e523066bd3718cbcf3a', 'restore-acceptance': str(STORE / 'restore_acceptance.json'), 'restore-acceptance-sha256': '5114c2cd96e5b8ffaf46e40a341619dd8b3547f89d417263c19c6e7f1f33bf77', 'repo': str(REPO), 'semantic-inspection': str(REMOTE / 'v4/semantic900_inspection.json'), 'semantic-inspection-sha256': 'cc93d94d69478a4ee190abff76da4fb185791ece7e4e1d2ee75360482ead80b4'}
    require(m['scientific_cli'] == expected_cli, 'Scientific source/map/restore paths or hashes changed')


def verify_inputs(manifest_path, expected_sha):
    require(sha(manifest_path) == expected_sha, 'Manifest differs from reviewed external SHA')
    m = read(manifest_path)
    require(sys.platform == 'linux' and str(Path(sys.executable).absolute()) == PYTHON and not sys.flags.optimize, 'Use isolated Linux cu128 Python without -O')
    require(os.getpriority(os.PRIO_PROCESS, 0) == 0, 'Outer coordinator must start nice0; original run applies nice10 exactly once')
    cpus = sorted(os.sched_getaffinity(0))
    require(len(cpus) >= 104, 'Missing coordinated CPU slots16..103')
    os.sched_setaffinity(0, [cpus[16]])
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    for name, expected in m['input_sha256'].items():
        require(sha(name) == expected, 'Pinned source/input bytes changed: ' + name)
    for name, expected in m['scientific_source_sha256'].items():
        require(sha(name) == expected, 'Frozen scientific source/data bytes changed: ' + name)
    require(m['input_sha256'][str(REMOTE / 'v4/replay_v4.py')] == V4_SHA, 'Wrong sealed v4 source')
    require(m['input_sha256'][str(Path(__file__).resolve())] == sha(__file__), 'This thin runner is not source-bound')
    prior = read(REMOTE / 'v4/execution_20261009/cumulative_28_accepted.json')
    inventory = read(REMOTE / 'inputs/model_inventory.json')
    validate_contract(m, inventory, prior)
    require(sha(REMOTE / 'inputs/model_inventory.json') == INVENTORY_SHA and sha(REMOTE / 'v4/execution_20261009/cumulative_28_accepted.json') == CUMULATIVE_SHA, 'Trusted inventory/prior SHA mismatch')
    require(not Path(m['output']).exists(), 'Preserve every prior execution; no implicit resume or overwrite')
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    require(quota != 'max' and float(quota) / float(period) >= m['conservative_total_cpu_budget'], 'Live CPU quota is below the approved budget')
    return m, {'utc_unix': time.time(), 'cpu_max': quota + ' ' + period, 'quota_cores': float(quota) / float(period), 'inherited_allowed_cpus': cpus, 'worker_cpus': cpus[16:104], 'source_before': m['input_sha256']}


def scientific_command(m, action, output, ids=None, batch=None):
    cmd = [PYTHON, str(REMOTE / 'v4/replay_v4.py'), action]
    for key, value in m['scientific_cli'].items():
        cmd += ['--' + key, str(value)]
    cmd += ['--output', str(output)]
    if action == 'run':
        cmd += ['--ids', *ids, '--workers', str(min(11, len(ids))), '--max-wall-seconds', '1800']
    else:
        cmd += ['--batch', str(batch)]
    return cmd


def invoke(command, log, inherited):
    # Restore only while spawning. The stdlib-only outer coordinator then
    # returns to CPU16; scientific children retain the original allowed list.
    os.sched_setaffinity(0, inherited)
    try:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    finally:
        os.sched_setaffinity(0, [inherited[16]])
    try:
        return child.wait()
    except BaseException:
        child.terminate()
        try:
            child.wait(timeout=40)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise


def archive_chunk(m, manifest_path, stage, batch, index, accepted, failed=False):
    """Archive new predictions/evidence only; old900 model bytes are never copied."""
    files = {'execution/manifest.json': Path(manifest_path)}
    for p in batch.rglob('*'):
        if p.is_file():
            require(not p.is_symlink(), 'Symlink in new execution evidence')
            files['batch/' + p.relative_to(batch).as_posix()] = p
    for p in stage.glob('*'):
        if p.is_file() and p.suffix in ('.json', '.log'):
            files['execution/' + p.name] = p
    for member, source in m['source_archive'].items():
        files[member] = Path(source)
    identities = {n: {'sha256': sha(p), 'bytes': p.stat().st_size} for n, p in files.items()}
    prefix = 'failure_' if failed else ''
    archive = stage / (prefix + 'chunk_evidence.tar.gz')
    require(not archive.exists(), 'Preserve existing chunk archive')
    with tarfile.open(archive, 'x:gz') as tf:
        for name, path in sorted(files.items()):
            require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts, 'Unsafe archive member')
            tf.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as tf:
        entries = tf.getmembers()
        require(len(entries) == len({e.name for e in entries}) == len(identities), 'Archive duplicate/missing member')
        for e in entries:
            require(e.isfile() and e.size == identities[e.name]['bytes'] and hashlib.sha256(tf.extractfile(e).read()).hexdigest() == identities[e.name]['sha256'], 'Archive member changed')
    require(all(sha(p) == identities[n]['sha256'] for n, p in files.items()), 'Evidence changed during archive')
    result = {'status': 'PRESERVED_FAILURE_NOT_OFFSERVER_ACCEPTED' if failed else 'REMOTE_STRICT_ACCEPTED_ARCHIVE_VERIFIED_PENDING_OFFSERVER', 'chunk_index': index, 'archive': archive.name, 'sha256': sha(archive), 'bytes': archive.stat().st_size, 'member_n': len(identities), 'members': identities, 'accepted_ids': accepted, 'old_model_files_archived': 0, 'offserver_verified': False, 'manifest_sha256': sha(manifest_path)}
    save(stage / (prefix + 'remote_archive_inventory.json'), result)
    return result


def run(m, manifest_path, resource):
    root = Path(m['output'])
    root.mkdir(parents=True)
    save(root / 'launch_resource_receipt.json', resource)
    # The outer process performs only bookkeeping; children inherit full allowed
    # CPUs so the sealed v4 manager can assign its original indexed slots.
    inherited = resource['inherited_allowed_cpus']
    os.sched_setaffinity(0, [inherited[16]])
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    import fcntl
    with (root.parent / 'remaining872.outer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        accepted_ids, previous_receipt = [], None
        for chunk in m['chunks']:
            index, ids = chunk['index'], chunk['ids']
            require(not set(ids) & (set(accepted_ids) | set(m['accepted28_ids'])), 'Repeated canonical ID across chunks')
            stage = root / ('chunk_%03d' % index)
            stage.mkdir()
            batch = stage / 'batch'
            strict = stage / 'strict_acceptance.json'
            try:
                records = {r['id']: r for r in read(REMOTE / 'inputs/model_inventory.json')['records']}
                save(stage / 'chunk_config_inventory_records.json', [records[i] for i in ids])
                with (stage / 'run.log').open('xb') as log:
                    code = invoke(scientific_command(m, 'run', batch, ids=ids), log, inherited)
                # Even a failed execution is inspected by the unchanged gate;
                # its accepted subset is preserved, never silently rerun.
                if (batch / 'batch_execution.json').exists():
                    with (stage / 'strict_acceptance.log').open('xb') as log:
                        checked = invoke(scientific_command(m, 'accept', strict, batch=batch), log, inherited)
                else:
                    checked = None
                require(code == 0 and checked == 0, 'Execution/strict acceptance failed; stop without retry')
                a = read(strict)
                require(a['status'] == 'SELECTED_VALID_REPLAY_ACCEPTED' and a['accepted_ids'] == ids and a['accepted_n'] == len(ids) and not a['invalid'], 'Chunk did not strictly accept every declared ID')
                require(a['max_abs_native_metric_difference'] <= 1e-12 and not a['all900_native_valid_replayed'], 'Native tolerance or completion scope drift')
                for name, expected in m['input_sha256'].items():
                    require(sha(name) == expected, 'Pinned source/input changed after chunk: ' + name)
                backup = archive_chunk(m, manifest_path, stage, batch, index, ids)
                accepted_ids += ids
                ledger = {'status': 'REMOTE_STRICT_ACCEPTED_PENDING_OFFSERVER', 'chunk_index': index, 'new_accepted_ids': ids, 'remote_strict_cumulative_ids': accepted_ids, 'strict_sha256': sha(strict), 'archive_sha256': backup['sha256'], 'archive_inventory_sha256': sha(stage / 'remote_archive_inventory.json'), 'previous_ledger_sha256': previous_receipt, 'offserver_accepted': False, 'all900_native_valid_replayed': False}
                save(stage / 'accepted_ledger.json', ledger)
                previous_receipt = sha(stage / 'accepted_ledger.json')
                print(json.dumps({'chunk': index, 'strict_n': len(ids), 'remote_strict_total': len(accepted_ids), 'archive_sha256': backup['sha256'], 'ledger_sha256': previous_receipt}), flush=True)
            except BaseException as exc:
                partial = read(strict).get('accepted_ids', []) if strict.exists() else []
                save(stage / 'failure_receipt.json', {'status': 'FAILSTOP_PRESERVE_NO_RETRY', 'error_type': type(exc).__name__, 'error': str(exc), 'requested_ids': ids, 'strict_accepted_subset_not_offserver_registered': partial, 'previous_remote_strict_ids': accepted_ids, 'all900_native_valid_replayed': False})
                archive_chunk(m, manifest_path, stage, batch, index, partial, failed=True)
                raise
        save(root / 'remote_finished.json', {'status': 'ALL872_REMOTE_STRICT_ACCEPTED_PENDING_OFFSERVER_AND_UNIQUE900_COLLECTOR', 'accepted_ids': accepted_ids, 'last_ledger_sha256': previous_receipt, 'all900_native_valid_replayed': False, 'test_image_inference': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['inspect', 'run'])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    a = p.parse_args()
    m, resource = verify_inputs(a.manifest, a.manifest_sha256)
    if a.command == 'inspect':
        print(json.dumps({'status': 'PREPARED_INPUTS_VERIFIED_NO_INFERENCE', 'remaining_n': len(m['remaining_ids']), 'chunks': len(m['chunks']), 'resource': resource}, indent=2))
    else:
        run(m, a.manifest, resource)


if __name__ == '__main__':
    main()
