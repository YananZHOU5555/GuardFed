#!/usr/bin/env python3
"""Run isolated, resumable revision jobs; never write to historical results."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [123,456,789,1001,2024,3141,4242,5050,6060,7070]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(path)

def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in [
        ROOT/'scripts/reproduce_paper_tables.py', ROOT/'src/data_loader.py',
        Path(__file__).resolve(), ROOT/'data/adult/adult.data', ROOT/'data/adult/adult.test', ROOT/'data/compas/compas-scores-two-years.csv']}

def checked_result(job):
    out=Path(job['output']); path=out/'result.json'
    if not path.exists():return None
    r=json.loads(path.read_text()); provenance=r.get('revision_job',{})
    assert provenance.get('id')==job['id'], ('wrong job',path)
    assert r['config']==job['config'], ('config mismatch',path)
    assert provenance.get('source_hashes')==job['source_hashes'], ('source mismatch',path)
    assert [x['round'] for x in r['trajectory_metrics']]==list(range(1,job['config']['rounds']+1))
    assert len(r['round_summaries'])==job['config']['rounds']
    assert digest(out/'model.pt')==provenance['checkpoint_sha256'], ('checkpoint mismatch',path)
    return r

def prepare(args):
    out = Path(args.output).resolve()
    original = Path('/workspace/GuardFed/results/attack_strength/raw_results.jsonl')
    controls = {}
    for line_no, line in enumerate(original.open(), 1):
        r = json.loads(line)
        if (r.get('dataset')=='adult' and r.get('method')=='GuardFed-AD2+'
            and r.get('attack')=='S-DFA' and r.get('rounds')==70
            and r.get('num_malicious')==4 and r.get('seed') in SEEDS):
            key = (r['distribution'], r['seed'])
            assert key not in controls, ('duplicate control',key)
            controls[key] = (r, line_no, hashlib.sha256(line.encode()).hexdigest())
    assert len(controls)==20, len(controls)
    frozen = source_hashes()
    jobs=[]
    for seed in SEEDS:
        for dist in ['IID','non-IID']:
            old, line_no, line_hash = controls[(dist,seed)]
            c = dict(old['config'])
            assert c['ad2_plus_mode']=='adaptive' and c['server_ratio']==0.1
            assert c['sdfa_foe_mode']=='fedsa' and c['fedsa_gain']==4.5 and c['fedsa_norm_ratio']==3.0
            write_json(out/'reused_full'/f'{dist}_{seed}.json',{
                'status':'reused_historical', 'source':str(original), 'source_line':line_no,
                'source_line_sha256':line_hash, 'result':old,
                'limitation':'Historical 5090 environment; full diagnostics only first/last round. Not newly trained.'})
            for component in ['U','C','A','F','V','N']:
                cfg=dict(c, ablation_component=component, client_alpha=None,
                         full_round_diagnostics=True, experiment_suite='revision_adult_ablation_v1',
                         experiment_tag=f'{dist}_minus{component}_seed{seed}')
                job_id=f'{dist}_minus{component}_seed{seed}'
                job={'id':job_id, 'dataset':'adult','distribution':dist,
                     'method':'GuardFed-AD2+','attack':'S-DFA','config':cfg,
                     'output':str(out/'runs'/job_id),'source_hashes':frozen,'evidence_stage':'formal_supplement'}
                write_json(out/'jobs'/f'{job_id}.json',job)
                jobs.append(str(out/'jobs'/f'{job_id}.json'))
    manifest={'protocol':'revision_adult_ablation_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'jobs':jobs,'reused_full_count':20,'new_run_count':len(jobs),'seeds':SEEDS,
              'statistics':'final round, all metrics at same checkpoint; mean/sample std; no test selection',
              'source_hashes':frozen,'output':str(out),'old_experiments_rerun':False}
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'manifest':str(out/'manifest.json'),'new_jobs':len(jobs),'reused_full':20}),flush=True)

def worker(args):
    job=json.loads(Path(args.job).read_text())
    out=Path(job['output']); out.mkdir(parents=True,exist_ok=True)
    for name, expected in job['source_hashes'].items():
        assert digest(ROOT/name)==expected, ('source changed after freeze',name)
    if checked_result(job) is not None:return
    import torch
    import reproduce_paper_tables as core
    torch.set_num_threads(int(os.environ.get('GUARDFED_CPU_THREADS','1')))
    torch.set_num_interop_threads(1)
    cfg=core.ExperimentConfig(**job['config'])
    device=torch.device(cfg.device)
    started=time.time()
    def progress(item):
        write_json(out/'progress.json',dict(item,job_id=job['id'],pid=os.getpid(),
                   elapsed_sec=time.time()-started,updated_unix=time.time()))
    result=core.run_experiment(job['dataset'],job['distribution'],job['method'],job['attack'],
               cfg,'revision',device,progress_callback=progress,checkpoint_path=out/'model.pt')
    assert len(result['trajectory_metrics'])==cfg.rounds
    assert len(result['round_summaries'])==cfg.rounds
    assert all(x['round']==i+1 for i,x in enumerate(result['trajectory_metrics']))
    if job['method']=='GuardFed-AD2+':
        for entry in result['round_summaries']:
            info=entry['aggregate']; weights=info['client_weights']
            assert len(weights)==cfg.num_clients and abs(sum(weights)-1)<1e-6
            assert all(math.isfinite(w) and w>=0 for w in weights)
            if cfg.ablation_component in 'UCAFV' and cfg.ablation_component!='none':
                assert all(t[cfg.ablation_component]==0 for t in info['component_contributions'])
            if cfg.ablation_component=='N':assert all(s==1 for s in info['norm_clip_scales'])
    assert (out/'model.pt').is_file()
    result['revision_job']=dict(job,visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cpu_threads=torch.get_num_threads(),torch_version=torch.__version__,python_version=sys.version,
        checkpoint_sha256=digest(out/'model.pt'),finished_unix=time.time())
    write_json(out/'result.json',result)
    print(json.dumps({'complete':job['id'],'metrics':result['metrics'],'seconds':result['duration_sec']}),flush=True)

def summarize(manifest):
    root=Path(manifest['output']); rows=[]
    for path in sorted((root/'reused_full').glob('*.json')):
        r=json.loads(path.read_text())['result']
        rows.append(dict(dataset=r['dataset'],distribution=r['distribution'],alpha=r['alpha'],method=r['method'],attack=r['attack'],profile='full_historical',root_label_noise=r['config'].get('root_label_noise',0.0),root_sensitive_noise=r['config'].get('root_sensitive_noise',0.0),root_protected_share=r['config'].get('root_protected_share',None),seed=r['seed'],
                         source='historical',**r['metrics']))
    completed=failed=0
    for path in manifest['jobs']:
        job=json.loads(Path(path).read_text()); out=Path(job['output'])
        r=checked_result(job)
        if r is not None:
            completed+=1
            rows.append(dict(dataset=r['dataset'],distribution=r['distribution'],alpha=r['alpha'],method=r['method'],attack=r['attack'],profile='full' if r['config']['ablation_component']=='none' else 'minus_'+r['config']['ablation_component'],
                             root_label_noise=r['config'].get('root_label_noise',0.0),root_sensitive_noise=r['config'].get('root_sensitive_noise',0.0),root_protected_share=r['config'].get('root_protected_share',None),seed=r['seed'],source='new',**r['metrics']))
        elif (out/'failure.json').exists(): failed+=1
    group_fields=['dataset','distribution','alpha','method','attack','profile','root_label_noise','root_sensitive_noise','root_protected_share']
    fields=group_fields+['seed','source','accuracy','aeod','aspd']
    with (root/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary=[]
    for key in sorted({tuple(r[f] for f in group_fields) for r in rows}):
        group=[r for r in rows if tuple(r[f] for f in group_fields)==key]
        assert len({r['seed'] for r in group})==len(group)
        for metric in ['accuracy','aeod','aspd']:
            vals=[r[metric] for r in group if math.isfinite(r[metric])]
            summary.append(dict(zip(group_fields,key),metric=metric,n_runs=len(group),
                n_defined=len(vals),mean=statistics.mean(vals) if vals else None,
                sample_std=statistics.stdev(vals) if len(vals)>1 else None))
    write_json(root/'summary.json',summary)
    status={'new_complete':completed,'new_total':len(manifest['jobs']),'failed':failed,
            'historical_controls':manifest.get('reused_full_count',0),'updated_unix':time.time()}
    write_json(root/'status.json',status)
    return status

def run(args):
    manifest=json.loads(Path(args.manifest).read_text()); pending=[]
    for path in manifest['jobs']:
        j=json.loads(Path(path).read_text())
        if args.first_seeds and j['config']['seed'] not in SEEDS[:args.first_seeds]:continue
        out=Path(j['output'])
        if checked_result(j) is None:
            if (out/'failure.json').exists():raise RuntimeError(f'Unresolved failure: {out}')
            pending.append((path,j))
    slots=list(range(args.concurrency)); active={}; failed=False;start=time.time();last_monitor=0
    monitor=Path(manifest['output'])/'resource_samples.jsonl'
    try:
        while pending or active:
            while pending and slots and not failed:
                slot=slots.pop(0); path,job=pending.pop(0); out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
                env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                    GUARDFED_CPU_THREADS='1',CUDA_VISIBLE_DEVICES=str(slot%2),PYTHONUNBUFFERED='1')
                log=(out/'worker.log').open('a')
                proc=subprocess.Popen([sys.executable,__file__,'worker','--job',path],env=env,stdout=log,stderr=subprocess.STDOUT)
                active[slot]=(proc,log,job)
                print(json.dumps({'started':job['id'],'pid':proc.pid,'gpu':slot%2}),flush=True)
            for slot,(proc,log,job) in list(active.items()):
                ret=proc.poll()
                if ret is None:continue
                log.close(); del active[slot];slots.append(slot)
                if ret or not (Path(job['output'])/'result.json').exists():
                    failed=True;write_json(Path(job['output'])/'failure.json',{'exitcode':ret,'time':time.time()})
                print(json.dumps({'finished':job['id'],'exitcode':ret,'status':summarize(manifest)}),flush=True)
            if time.time()-last_monitor>=10:
                g=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
                sample={'time':time.time(),'active':len(active),'pending':len(pending),'gpu':g.stdout.strip(),
                        'cgroup_cpu_stat':Path('/sys/fs/cgroup/cpu.stat').read_text()}
                with monitor.open('a') as f:f.write(json.dumps(sample)+'\n')
                last_monitor=time.time()
            if failed and not active:break
            time.sleep(1)
    finally:
        for proc,log,job in active.values():
            proc.terminate();proc.wait();log.close()
    print(json.dumps({'elapsed_sec':time.time()-start,'status':summarize(manifest),'failed':failed}),flush=True)
    if failed:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','run','summarize'])
    p.add_argument('--output');p.add_argument('--job');p.add_argument('--manifest');p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--first-seeds',type=int,default=0);args=p.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='worker':worker(args)
    elif args.action=='run':run(args)
    else:print(json.dumps(summarize(json.loads(Path(args.manifest).read_text()))))

if __name__=='__main__':main()
