"""Only lifecycle/identity checks; all training remains in source/worker.py."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from screen_common import HERE, accepted, authorized, digest, local_identity, repo_identity, write_json


def run(repo, job_id):
    protocol, manifest = local_identity()
    authorized()
    item = next(item for item in manifest['jobs'] if item['id'] == job_id)
    out = HERE / 'runs' / job_id
    if out.exists():
        raise ValueError('Existing output preserved; no partial resume or implicit retry')
    before = repo_identity(repo, protocol)
    # Fail before the worker creates output if the runtime differs from the gate.
    import torch
    if (sys.version_info[:3], torch.__version__, torch.version.cuda) != ((3, 12, 3), '2.11.0+cu128', '12.8'):
        raise ValueError('Runtime differs from reviewed cu128 gate')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != protocol['execution']['gpu']:
        raise ValueError('Expected exactly one visible reviewed GPU')
    if os.environ.get('GUARDFED_CPU_THREADS') != '1' or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError('Expected one CPU thread and nice >= 10')
    try:
        from worker import run as original_run
        original_run(repo, HERE / 'jobs' / item['job'], out)
        local_identity()
        after = repo_identity(repo, protocol)
        from accept_result import checked_result
        if checked_result(HERE / 'jobs' / item['job'], out) is None:
            raise ValueError('No accepted terminal result')
        write_json(out / 'screen_identity.json', dict(
            status='PASS', package_sha256=digest(HERE / 'PACKAGE_SHA256.json'),
            before=before, after=after, acceptance_sha256=digest(out / 'acceptance.json'),
            time=time.time(), visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
            nice=os.getpriority(os.PRIO_PROCESS, 0)))
        accepted(item, out)
    except BaseException as error:
        if out.exists():
            path = out / 'failure_screen.json'
            if not path.exists():
                write_json(path, dict(error=repr(error), traceback=traceback.format_exc(), time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.job_id)
