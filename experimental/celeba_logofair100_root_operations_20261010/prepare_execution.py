"""Explicit one-shot hidden Windows launch of the unchanged96-fit runner."""
import argparse,datetime,subprocess,traceback
from pathlib import Path
from common import *

def run(a):
    sources();require(os.name=='nt','This launcher is Windows-only')
    binding=pinned(a.bind_result,a.bind_result_sha256)
    require(binding['status']=='ROOT_LOGOFAIR100_METADATA_BOUND_NOT_STARTED' and binding['summary_adoption_sha256']==ADOPTION_SHA
        and binding['source_review_sha256']==REVIEW_SHA,'Actual reviewed metadata binding required')
    stage,storage=bulk_path(binding['stage'],256*1024*1024)
    require(stage==(FROOT/'stage001').resolve() and not (stage/'FULLCOVERAGE_STARTED.json').exists(),'Exact unstarted stage001 required')
    require(digest(stage/'manifest.json')==binding['manifest_sha256'] and digest(stage/'SOURCE_SHA256.json')==binding['source_sha256'],'Bound stage changed')
    for name,want in read(stage/'SOURCE_SHA256.json').items():require(digest(stage/name)==want,'Bound source/job drift')
    manifest=read(stage/'manifest.json');jobs=manifest['jobs'];reuse=manifest['reused_jobs']
    require(len(jobs)==96 and len(reuse)==4 and len({r['id'] for r in jobs+reuse})==100
        and manifest['new_CNN']==0 and not manifest['final_test'],'Exact96+4 validation stage required')
    for row in reuse:
        require(digest(row['result'])==row['result_sha256'] and digest(row['acceptance'])==row['acceptance_sha256'],'Original4 accepted bytes changed')
    output=FROOT/'attempt001';logs=FROOT/'launch_logs001';approval=FROOT/'EXECUTION_APPROVAL001.json'
    for p in (output,logs,approval):bulk_path(p,16*1024*1024);require(not p.exists(),'Existing/partial launch evidence preserved; no retry')
    require(a.python.is_file() and a.repo.is_dir(),'Actual Python and checked source repo required')
    require(a.attempt.resolve().parent==HERE.resolve() and not a.attempt.exists(),'New owned control directory required')
    a.attempt.mkdir()
    save(a.attempt/'LAUNCH_ONCE.json',dict(operation='EXACT96_ORIGINAL_POSTPROCESSING',bind_result_sha256=a.bind_result_sha256,storage_preflight=storage,automatic_retry=False))
    try:
        save(approval,dict(status='ROOT_LOGOFAIR96_EXECUTION_APPROVED',manifest_sha256=binding['manifest_sha256'],source_sha256=binding['source_sha256'],runner_sha256=digest(SOURCE/'run96.py'),output=output.resolve().as_posix(),max_workers=1,CPU_threads=1,new_fits=96,reused=4,test=False,automatic_retry=False))
        bulk_path(logs,16*1024*1024);logs.mkdir(parents=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        argv=[str(a.python.resolve()),'-B',str(SOURCE/'run96.py'),'--stage',str(stage),'--repo',str(a.repo.resolve()),'--out',str(output),'--approval',str(approval),'--approval-sha',digest(approval)]
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        bulk_path(logs,16*1024*1024)
        with (logs/'stdout.log').open('xb') as out,(logs/'stderr.log').open('xb') as err:
            process=subprocess.Popen(argv,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS,close_fds=True)
        save(a.attempt/'STARTUP.json',dict(status='WINDOWS_HIDDEN_PROCESS_CREATED_NOT_ACCEPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=process.pid,process_handle=int(process._handle),handle_is_launch_parent_local_not_persistent=True,
            argv=argv,python_sha256=digest(a.python),runner_sha256=digest(SOURCE/'run96.py'),approval_sha256=digest(approval),bind_result_sha256=a.bind_result_sha256,
            environment_overrides={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},
            CPU_policy='One worker; computational thread environment1, idle priority; no claim of exclusive CPU ownership',output=output.as_posix(),logs=logs.as_posix(),initial_poll=process.poll(),accepted_offserver=0,root_adopted=0,automatic_retry=False))
    except BaseException as e:
        save(a.attempt/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bind-result','python','repo','attempt'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bind-result-sha256',required=True);p.add_argument('--launch',action='store_true',required=True)
    run(p.parse_args())
