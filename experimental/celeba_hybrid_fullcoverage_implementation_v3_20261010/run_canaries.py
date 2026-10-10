"""Seven fresh-child three-round interface gates; no automatic repeat or70-round claim."""
import argparse,os,subprocess,sys,traceback
from pathlib import Path
from common import HERE,read,write_json,require,local_identity,authorized,repo_identity,scoped_functions,digest
def run(repo):
    protocol,manifest=local_identity();authorized('seven_same_horizon_3round_canaries',fresh=True);repo_identity(repo,protocol)
    require(not (HERE/'gate_runs').exists() and not (HERE/'GATE_FAILURE.json').exists() and not (HERE/'GATE_ACCEPTANCE.json').exists(),'Existing gate evidence preserved; no automatic retry')
    import fcntl
    lock=(HERE/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (HERE/'gate_logs').mkdir(exist_ok=False);accepted=[]
    body,_,_,_,_=scoped_functions('canary',repo);before=body.snapshot()
    try:
        for item in manifest['preflight_jobs']:
            with (HERE/'gate_logs'/(item['id']+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-B',str(HERE/'run_one.py'),'--repo',str(repo),'--job-id',item['id'],'--canary'],stdout=log,stderr=subprocess.STDOUT,check=True)
            accepted.append(item['id'])
        os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        _,scope,_,checked,compare=scoped_functions('canary',repo)
        for item in scope['jobs']:require(checked(item,scope) is not None,'All seven original checks required')
        pairs=compare(dict(scope,jobs=scope['jobs'][:4]))
        require(len(pairs)==2,'Two exact Hybrid/legacy reference pairs required')
        repo_identity(repo,protocol);local_identity();after=body.snapshot()
        previous={r['id']:r['round'] for r in before['active']}
        grew=after['completed']>before['completed'] or any(r['id'] in previous and r['round'] is not None and previous[r['id']] is not None and r['round']>previous[r['id']] for r in after['active'])
        require(not after['failed'] and grew,'Protected formal queue did not advance or failed')
        hashes={p.relative_to(HERE).as_posix():digest(p) for parent in ('gate_runs','gate_logs') for p in (HERE/parent).rglob('*') if p.is_file()}
        write_json(HERE/'GATE_ACCEPTANCE.json',dict(status='SEVEN_HYBRID_CANARIES_STRICT_PASS_BACKUP_PENDING',accepted_ids=accepted,pairs=pairs,artifact_hashes=hashes,package_sha256=digest(HERE/'PACKAGE_SHA256.json'),resources_before=before,resources_after=after,scientific70records=0,test=False))
    except BaseException as e:
        write_json(HERE/'GATE_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),accepted_ids=accepted,automatic_retry=False));raise
    finally:lock.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);run(p.parse_args().repo.resolve())
