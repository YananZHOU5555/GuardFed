"""One bound Hybrid job only. Missing actual binding/authorization refuses before torch."""
import argparse,os,sys
from pathlib import Path
from common import HERE,read,require,local_identity,authorized,repo_identity,scoped_functions
def run(repo,jobid,canary):
    protocol,manifest=local_identity();authorized('seven_same_horizon_3round_canaries' if canary else '96_new_70round_valid_only',fresh=False)
    require(sys.platform=='linux' and not sys.flags.optimize,'Linux strict Python required')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and os.sched_getaffinity(0)=={104},'Original Hybrid CPU104/nice10 allocation required')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    repo_identity(repo,protocol)
    entries=manifest['preflight_jobs'] if canary else manifest['jobs'];matches=[e for e in entries if e['id']==jobid];require(len(matches)==1,'Foreign job or stage')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8' and torch.cuda.device_count()==1,'Original isolated cu128 runtime required')
    import subprocess
    require(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==read(HERE/'BINDINGS.json')['gpu_uuid'],'Original physical GPU identity changed')
    body,scope,run_one,checked,_=scoped_functions('canary' if canary else 'fullcoverage',repo)
    worker,core=body.modules();(HERE/('gate_runs' if canary else 'runs')).mkdir(exist_ok=True)
    run_one(matches[0],scope,worker,core);require(checked(matches[0],scope) is not None,'Terminal strict acceptance missing')
    repo_identity(repo,protocol);local_identity()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--job-id',required=True);p.add_argument('--canary',action='store_true');a=p.parse_args();run(a.repo.resolve(),a.job_id,a.canary)
