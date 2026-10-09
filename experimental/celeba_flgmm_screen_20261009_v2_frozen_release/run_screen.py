"""Two GPU slots; strict completed-job skip; fail-stop without automatic retry."""
import argparse
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from frozen_score import score
from screen_common import HERE, accepted, authorized, digest, local_identity, repo_identity, write_json


def summarize(manifest):
    rows = []
    for item in manifest['jobs']:
        result = accepted(item, HERE / 'runs' / item['id'])
        if result is None:
            raise ValueError('No selection before all 32 terminal results pass')
        row = dict(id=item['id'], candidate=item['tuning_candidate'],
                   distribution=item['distribution'], attack=item['attack'], seed=91001,
                   checkpoint_sha256=digest(HERE / 'runs' / item['id'] / 'model.pt'), **result['metrics'])
        rows.append(dict(row, score=score(row)))
    candidates = []
    for candidate in sorted({row['candidate'] for row in rows}):
        group = [row for row in rows if row['candidate'] == candidate]
        if len(group) != 4:
            raise ValueError('Incomplete candidate')
        candidates.append(dict(candidate=candidate, n_seeds=1, **{
            key: statistics.mean(row[key] for row in group) for key in ['accuracy', 'aeod', 'aspd', 'score']}))
    summary = dict(status='COMPLETE_ACCEPTED_BACKUP_PENDING', accepted=32, records=rows,
        candidates=candidates, selected=min(candidates, key=lambda row: (-row['score'], row['candidate'])),
        package_sha256=digest(HERE / 'PACKAGE_SHA256.json'),
        note='Valid-only n=1; four conditions are not independent seeds. All negative results retained.')
    write_json(HERE / 'summary.json', summary)
    return summary


def inspect_output(item):
    out = HERE / 'runs' / item['id']
    if not out.exists():
        return 'pending'
    if accepted(item, out) is None:
        raise ValueError('Partial output requires diagnosis; no resume: ' + item['id'])
    return 'accepted'


def run(repo):
    import fcntl
    lock = (HERE / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active, logs = {}, []
    try:
        protocol, manifest = local_identity()
        authorized()
        if list(HERE.glob('QUEUE_FAILURE*.json')):
            raise ValueError('Preserved queue failure blocks automatic restart')
        repo_identity(repo, protocol)
        jobs = [item for item in manifest['jobs'] if inspect_output(item) == 'pending']
        while jobs or active:
            # Poll failures before launching further work.
            for gpu, (process, item) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                if code != 0 or inspect_output(item) != 'accepted':
                    raise RuntimeError(f"Job failed: {item['id']} exit={code}; no automatic retry")
                del active[gpu]
            for gpu in (0, 1):
                if gpu in active or not jobs:
                    continue
                item = jobs.pop(0)
                local_identity()
                (HERE / 'logs').mkdir(exist_ok=True)
                log = (HERE / 'logs' / (item['id'] + '.log')).open('x')
                logs.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), GUARDFED_CPU_THREADS='1',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
                process = subprocess.Popen([sys.executable, '-u', str(HERE / 'run_one.py'),
                    '--repo', str(repo), '--job-id', item['id']], env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (process, item)
            write_json(HERE / 'queue_progress.json', dict(time=time.time(),
                active=[dict(id=item['id'], pid=p.pid, gpu=gpu) for gpu, (p, item) in active.items()],
                pending=len(jobs), completed=32-len(jobs)-len(active)))
            time.sleep(2)
        local_identity()
        repo_identity(repo, protocol)
        summarize(manifest)
    except BaseException as error:
        path = HERE / ('QUEUE_FAILURE_' + str(time.time_ns()) + '.json')
        write_json(path, dict(error=repr(error), traceback=traceback.format_exc(), time=time.time()))
        for process, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, _ in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        raise
    finally:
        for log in logs:
            log.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve())
