"""Existing fail-stop queue pattern: 96 new jobs + four unchanged legacy references."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from screen_common import HERE,accepted,authorized,digest,local_identity,read,repo_identity,write_json


def reused_records(manifest):
    records=[]
    for item in manifest['reused_jobs']:
        completed=subprocess.run([sys.executable,'-B',str(HERE/'legacy_check.py'),'--release',item['legacy_release'],'--id',item['id']],capture_output=True,text=True,check=True)
        row=json.loads(completed.stdout);original=item['accepted_record']
        for key in ('id','candidate','seed','distribution','attack','rounds','metrics','checkpoint_sha256','job_sha256','source_hashes'):
            assert row[key]==original[key],('Original reused record changed',item['id'],key)
        records.append(dict(row,reused=True))
    assert len(records)==4
    return records


def inspect(item):
    output=HERE/'runs'/item['id']
    if not output.exists():return None
    result=accepted(item,output)
    if result is None:raise ValueError('Partial output preserved; no resume: '+item['id'])
    return result


def summarize(manifest):
    local_identity();rows=reused_records(manifest)
    for item in manifest['jobs']:
        result=inspect(item)
        if result is None:raise ValueError('Full100 report requires all96 new terminal results')
        rows.append(dict(id=item['id'],candidate=result['tuning_candidate'],distribution=result['distribution'],attack=result['attack'],
             seed=result['seed'],rounds=70,metrics=result['metrics'],checkpoint_sha256=digest(HERE/'runs'/item['id']/'model.pt'),reused=False))
    assert len(rows)==100 and len({(r['distribution'],r['attack'],r['seed']) for r in rows})==100
    conditions={(d,a) for d in ('IID','non-IID') for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')}
    cohorts={}
    for name,seeds in [('ten_seed',set(range(91001,91011))),('exclude_selection_nine_seed',set(range(91002,91011))),('matching_six_seed',set(range(91005,91011)))]:
        cells=[]
        for distribution,attack in sorted(conditions):
            group=[r for r in rows if (r['distribution'],r['attack'])==(distribution,attack) and r['seed'] in seeds]
            assert {r['seed'] for r in group}==seeds and len(group)==len(seeds)
            cells.append(dict(distribution=distribution,attack=attack,n=len(group),metrics={k:dict(mean=statistics.mean(r['metrics'][k] for r in group),sample_sd=statistics.stdev(r['metrics'][k] for r in group)) for k in ('accuracy','aeod','aspd')}))
        within=[]
        for seed in sorted(seeds):
            group=[r for r in rows if r['seed']==seed];assert {(r['distribution'],r['attack']) for r in group}==conditions
            within.append(dict(seed=seed,**{k:statistics.mean(r['metrics'][k] for r in group) for k in ('accuracy','aeod','aspd')}))
        cohorts[name]=dict(cells=cells,per_seed=within,overall={k:dict(mean=statistics.mean(r[k] for r in within),sample_sd=statistics.stdev(r[k] for r in within)) for k in ('accuracy','aeod','aspd')})
    result=dict(status='100_STRICT_ACCEPTED_BACKUP_PENDING',accepted_new=96,accepted_reused=4,records=rows,cohorts=cohorts,
        manifest_sha256=digest(HERE/'manifest.json'),package_sha256=digest(HERE/'PACKAGE_SHA256.json'),
        note='Fixed selected recipe; all negative results retained; exposed validation; AEOD is TPR gap; no new selection/test/significance. Backup pending.')
    write_json(HERE/'summary.json',result);return result


def run(repo):
    import fcntl
    lock=(HERE/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};logs=[]
    try:
        protocol,manifest=local_identity();authorized('96_new_70round_valid_only',fresh=True)
        if list(HERE.glob('QUEUE_FAILURE*.json')):raise ValueError('Preserved failure blocks automatic restart')
        repo_identity(repo,protocol);reused_records(manifest)
        jobids={i['id'] for i in manifest['jobs']}
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try:argv=proc.read_bytes().decode(errors='replace').split('\0')
            except (FileNotFoundError,PermissionError):continue
            if str(HERE/'run_one.py') in argv and jobids.intersection(argv):raise ValueError('Duplicate worker exists')
        pending=[i for i in manifest['jobs'] if inspect(i) is None]
        completed_new=96-len(pending)  # Only original-checker accepted skips.
        (HERE/'runs').mkdir(exist_ok=True);(HERE/'logs').mkdir(exist_ok=True)
        failed=False
        while pending or active:
            local_identity()
            for gpu,(process,item,log) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close();del active[gpu]
                if code!=0 or inspect(item) is None:
                    failed=True;write_json(HERE/('QUEUE_FAILURE_'+str(time.time_ns())+'.json'),dict(id=item['id'],returncode=code))
                else:
                    completed_new+=1
            for gpu in (0,1):
                if failed or not pending:break
                if gpu in active:continue
                item=pending.pop(0)
                log=(HERE/'logs'/(item['id']+'.log')).open('x');logs.append(log)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),GUARDFED_CPU_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                process=subprocess.Popen([sys.executable,'-B',str(HERE/'run_one.py'),'--repo',str(repo),'--job-id',item['id']],env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(process,item,log)
            write_json(HERE/'queue_progress.json',dict(updated_unix=time.time(),active=[dict(id=i['id'],pid=p.pid,gpu=g) for g,(p,i,l) in active.items()],pending=len(pending),completed_new=completed_new,reused=4,failed=failed))
            if failed and not active:raise RuntimeError('Queue stopped after preserved failure; no automatic retry')
            if pending or active:time.sleep(2)
        repo_identity(repo,protocol);summarize(manifest)
    except BaseException as error:
        write_json(HERE/('QUEUE_FAILURE_'+str(time.time_ns())+'.json'),dict(error=repr(error),traceback=traceback.format_exc()))
        # Existing coverage queue semantics: no new jobs; let authorized peers finish.
        for process,item,log in active.values():process.wait();log.close()
        raise
    finally:
        for log in logs:
            if not log.closed:log.close()
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','summarize']);p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    if a.action=='run':run(a.repo.resolve())
    else:summarize(local_identity()[1])
