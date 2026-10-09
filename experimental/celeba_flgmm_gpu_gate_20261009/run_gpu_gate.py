"""Four prepared canaries only, two GPU slots; intended for normal supervisor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpu_worker import HERE, validate_inputs
from verify_gpu import read, sha, verify, check_cpu_receipts


def run(repo, cpu_root):
    scope, freeze = read(HERE / 'SCOPE.json'), read(HERE / 'FREEZE.json')
    prerequisites = freeze['prerequisites']
    assert prerequisites['cpu_offserver_verified'] is True
    for filename, expected in prerequisites['cpu_receipt_hashes'].items():
        assert filename in ('LOCAL_ACCEPTANCE.json', 'BACKUP_SHA256.json')
        assert sha(cpu_root / filename) == expected, filename
    assert set(prerequisites['cpu_receipt_hashes']) == {'LOCAL_ACCEPTANCE.json', 'BACKUP_SHA256.json'}
    receipt = read(cpu_root / 'LOCAL_ACCEPTANCE.json')
    assert receipt['status'] == 'PASS' and receipt['actual_results_checked'] == 2
    check_cpu_receipts(HERE, cpu_root)
    jobs = []
    for name in scope['jobs']:
        _, _, job = validate_inputs(repo, HERE / name)
        assert not (HERE / 'runs' / job['id']).exists(), 'Existing attempts are preserved; no automatic retry'
        jobs.append((name, job))
    assert len(jobs) == 4 and sorted(j['gpu'] for _, j in jobs) == [0, 0, 1, 1]
    active, logs = {}, []
    try:
        while jobs or active:
            for slot in (0, 1):
                if slot in active: continue
                match = next((i for i, (_, job) in enumerate(jobs) if job['gpu'] == slot), None)
                if match is None: continue
                name, job = jobs.pop(match)
                (HERE / 'logs').mkdir(exist_ok=True)
                log = (HERE / 'logs' / (job['id'] + '.log')).open('x')
                logs.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot), OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                command = [sys.executable, '-u', str(HERE / 'gpu_worker.py'), '--repo', str(repo),
                    '--job', str(HERE / name), '--out', str(HERE / 'runs' / job['id'])]
                active[slot] = (subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), job)
            for slot, (process, job) in list(active.items()):
                code = process.poll()
                if code is None: continue
                if code:
                    raise RuntimeError(f"CANARY failed: {job['id']}, exit={code}; no automatic retry")
                del active[slot]
            time.sleep(2)
        report = verify(HERE, cpu_root)
        assert report['status'] == 'PASS', 'Repeat mismatch retained; no screen launch'
    except BaseException as error:
        (HERE / 'QUEUE_FAILURE.json').write_text(json.dumps(dict(error=repr(error), time=time.time()), indent=2))
        for process, _ in active.values():
            if process.poll() is None: process.terminate()
        for process, _ in active.values():
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        raise
    finally:
        for log in logs: log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--cpu-root', type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.cpu_root.resolve())
