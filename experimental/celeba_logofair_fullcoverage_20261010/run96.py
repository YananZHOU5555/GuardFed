"""Sequential fresh-process original LoGoFair fits; no CNN, retries or selection."""
import argparse, os, subprocess, sys, traceback
from pathlib import Path
from metadata import HERE, bulk_path, digest, read, require, write, pinned, verify_sources

def run(stage,repo,output,approval,approval_sha):
    verify_sources();stage,_=bulk_path(stage,0);output,storage=bulk_path(output,256*1024*1024)
    require(not output.exists() and not output.is_relative_to(stage) and not stage.is_relative_to(output),'Independent new F output only')
    auth=pinned(approval,approval_sha)
    require(auth['status']=='ROOT_LOGOFAIR96_EXECUTION_APPROVED'
        and auth['manifest_sha256']==digest(stage/'manifest.json')
        and auth['source_sha256']==digest(stage/'SOURCE_SHA256.json')
        and auth['runner_sha256']==digest(Path(__file__))
        and auth['output']==output.as_posix() and auth['max_workers']==auth['CPU_threads']==1
        and auth['new_fits']==96 and auth['reused']==4 and auth['test'] is False
        and auth['automatic_retry'] is False,'Actual exact execution approval required')
    for n,want in read(stage/'SOURCE_SHA256.json').items():require(digest(stage/n)==want,'Bound source/job changed')
    manifest=read(stage/'manifest.json');jobs=manifest['jobs'];reuse=manifest['reused_jobs']
    require(len(jobs)==96 and len(reuse)==4 and len({r['id'] for r in jobs+reuse})==100,'Exact100 unique cells required')
    for r in reuse:
        require(digest(r['result'])==r['result_sha256'] and digest(r['acceptance'])==r['acceptance_sha256'],'Old4 accepted reference changed')
    # Exclusive marker prevents restarting a partial attempt under another output name.
    with (stage/'FULLCOVERAGE_STARTED.json').open('x',encoding='utf8') as f:
        import json
        json.dump(dict(pid=os.getpid(),output=output.as_posix(),approval_sha256=approval_sha),f)
    output.mkdir(parents=True);write(output/'DISPATCH.json',dict(approval_sha256=approval_sha,storage_preflight=storage,new_CNN=0))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    accepted=[]
    try:
        for row in jobs:
            bulk_path(output,8*1024*1024);job=stage/row['job'];require(digest(job)==row['job_sha256'],'Frozen job changed')
            mapping=row['mapping']
            with (output/(row['id']+'.log')).open('x',encoding='utf8') as log:
                subprocess.run([sys.executable,'-B',str(stage/'snapshot/logofair_bridge_20261010/bridge.py'),
                    '--repo',str(repo),'--reference',row['reference'],'--job',str(job),'--mapping',mapping['path'],
                    '--mapping-meta',mapping['metadata'],'--out',str(output/row['id'])],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipt=output/row['id']/'acceptance.json';result=output/row['id']/'result.json';proof=read(receipt)
            require(proof['status']=='PASS' and proof['job_sha256']==row['job_sha256'],'Original bridge strict not complete')
            accepted.append(dict(id=row['id'],cell_id=row['cell_id'],result=result.as_posix(),result_sha256=digest(result),
                                 acceptance=receipt.as_posix(),acceptance_sha256=digest(receipt)))
            write(output/'PROGRESS.json',dict(status='LOCAL_ORIGINAL_STRICT_PROGRESS',records=accepted,offserver_accepted=0,root_adopted=0,final_test=False))
        write(output/'STRICT100_INDEX.json',dict(status='LOCAL_STRICT96_PLUS4_ROOT_REVIEW_PENDING',records=accepted,
            reused_jobs=reuse,manifest_sha256=digest(stage/'manifest.json'),offserver_accepted=0,root_adopted=0,new_CNN=0,final_test=False))
    except BaseException as exc:
        write(output/'QUEUE_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),strict_completed_ids=[r['id'] for r in accepted],automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('stage','repo','out','approval','approval-sha'):p.add_argument('--'+n,required=True)
    a=p.parse_args();run(a.stage,a.repo,a.out,a.approval,a.approval_sha)
