"""One exact new coverage/canary job. Existing source and before/after checks retained."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback
from screen_common import HERE,accepted,authorized,digest,local_identity,repo_identity,write_json


def check_runtime(protocol):
    import torch
    if (sys.version_info[:3],torch.__version__,torch.version.cuda)!=((3,12,3),'2.11.0+cu128','12.8'):
        raise ValueError('Runtime differs from reviewed cu128 gate')
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!=protocol['execution']['gpu']:
        raise ValueError('Require one visible reviewed GPU')
    if any(os.environ.get(k)!='1' for k in ('GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')) or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise ValueError('Require CPU1/nice10')


def run(repo,identity,canary=False):
    protocol,manifest=local_identity();authorized('seven_same_horizon_3round_canaries' if canary else '96_new_70round_valid_only')
    item=next(r for r in manifest['preflight_jobs' if canary else 'jobs'] if r['id']==identity)
    out=HERE/('preflight/runs' if canary else 'runs')/identity
    if out.exists():raise ValueError('Preserve existing output; no partial resume/retry')
    before=repo_identity(repo,protocol);check_runtime(protocol)
    try:
        from worker import run as original_run
        if canary:
            from canary_reference import instrument
            import worker
            instrument(worker,out)
        original_run(repo,HERE/'jobs'/item['job'],out)
        local_identity();after=repo_identity(repo,protocol)
        from accept_result import checked_result
        assert checked_result(HERE/'jobs'/item['job'],out) is not None
        write_json(out/'screen_identity.json',dict(status='PASS',package_sha256=digest(HERE/'PACKAGE_SHA256.json'),
            before=before,after=after,acceptance_sha256=digest(out/'acceptance.json'),time=time.time(),
            visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'],nice=os.getpriority(os.PRIO_PROCESS,0)))
        assert accepted(item,out) is not None
    except BaseException as error:
        if out.exists() and not (out/'failure_screen.json').exists():
            write_json(out/'failure_screen.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--job-id',required=True);p.add_argument('--canary',action='store_true')
    a=p.parse_args();run(a.repo.resolve(),a.job_id,a.canary)
