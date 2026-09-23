"""Matched performance-only replicas; invokes frozen workers and never the scientific summarizer."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path('/workspace/GuardFed-image-deterministic')
BASE=ROOT/'results/revision_20260923/celeba_deterministic_acceptance_v1'
sys.path.insert(0,str(ROOT/'scripts'))
import run_revision_ablation as runner
def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')
def cpu():
    return {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())}
def sample():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return {'time':time.time(),'gpu':[dict(zip(['index','utilization_percent','memory_used_mib'],map(int,line.split(',')))) for line in p.stdout.strip().splitlines()],'cpu_stat':cpu()}
def batch(concurrency, name=None):
    m=json.loads((BASE/(name or f'concurrency{concurrency}')/'manifest.json').read_text());out=Path(m['output'])
    jobs=[(p,json.loads(Path(p).read_text())) for p in m['jobs']]
    assert not any((Path(j['output'])/'result.json').exists() or (Path(j['output'])/'failure.json').exists() for p,j in jobs), 'Never silently rerun previous benchmark'
    pending=list(jobs);active=[];finished=[];samples=[];failed=False
    start=time.monotonic();first_cpu=cpu();last_sample=0
    try:
        while pending or active:
            for p,j in list(pending):
                gpu=j['benchmark_gpu']
                if failed or sum(a['gpu']==gpu for a in active)>=concurrency//2:continue
                dest=Path(j['output']);dest.mkdir(parents=True,exist_ok=True)
                env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',GUARDFED_CPU_THREADS='1',CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
                log=(dest/'worker.log').open('a')
                process=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_revision_ablation.py'),'worker','--job',p],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active.append(dict(process=process,log=log,job=j,gpu=gpu,start=time.monotonic()))
                pending.remove((p,j));print(json.dumps({'concurrency':concurrency,'started':j['id'],'gpu':gpu}),flush=True)
            for a in list(active):
                ret=a['process'].poll()
                if ret is None:continue
                a['log'].close();active.remove(a);j=a['job']
                try:r=runner.checked_result(j) if ret==0 else None
                except Exception as exc:r=None;print(repr(exc),flush=True)
                good=r is not None and r['config']['rounds']==j['config']['rounds'] and len(r.get('trajectory_metrics',[]))==j['config']['rounds']
                # checked_result verifies artifact completeness; trajectory is saved in result.
                entry=dict(rounds=j['config']['rounds'],id=j['id'],gpu=a['gpu'],exitcode=ret,complete=good,wall_seconds=time.monotonic()-a['start'])
                finished.append(entry)
                if not good:
                    failed=True;write(Path(j['output'])/'benchmark_failure.json',entry)
                print(json.dumps(entry),flush=True)
            if time.monotonic()-last_sample>=2:
                s=sample();s.update(active=len(active),pending=len(pending));samples.append(s)
                with (out/'resource_samples.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
                last_sample=time.monotonic()
            if failed and not active:break
            time.sleep(.2)
    finally:
        for a in active:
            a['process'].terminate();a['process'].wait();a['log'].close()
    elapsed=time.monotonic()-start;last_cpu=cpu();delta={k:last_cpu[k]-first_cpu.get(k,0) for k in last_cpu}
    report=dict(concurrency=concurrency,complete=sum(x['complete'] for x in finished),failed=failed,wall_seconds=elapsed,
      total_completed_rounds=sum(x['rounds'] for x in finished if x['complete']),rounds_per_second=sum(x['rounds'] for x in finished if x['complete'])/elapsed,
      cgroup_cpu_delta=delta,cgroup_average_cpu_cores=delta['usage_usec']/1e6/elapsed,
      gpu_peak_memory_mib={str(g):max(sg['memory_used_mib'] for s in samples for sg in s['gpu'] if sg['index']==g) for g in (0,1)},
      gpu_mean_utilization_percent={str(g):sum(sg['utilization_percent'] for s in samples for sg in s['gpu'] if sg['index']==g)/len(samples) for g in (0,1)},workers=finished)
    write(out/'performance.json',report)
    assert not failed and len(finished)==len(jobs), 'Benchmark failure retained; later stages stopped'
    return report

def equivalent(first,second):
    import torch
    ma=json.loads((BASE/first/'manifest.json').read_text());mb=json.loads((BASE/second/'manifest.json').read_text());rows=[]
    for pa,pb in zip(ma['jobs'],mb['jobs']):
        a=json.loads(Path(pa).read_text());b=json.loads(Path(pb).read_text());assert {k:v for k,v in a['config'].items() if k!='experiment_tag'}=={k:v for k,v in b['config'].items() if k!='experiment_tag'}
        ra=json.loads((Path(a['output'])/'result.json').read_text());rb=json.loads((Path(b['output'])/'result.json').read_text())
        sa=torch.load(Path(a['output'])/'model.pt',map_location='cpu',weights_only=True);sb=torch.load(Path(b['output'])/'model.pt',map_location='cpu',weights_only=True)
        exact=all(torch.equal(sa[k],sb[k]) for k in sa)
        close=all(torch.allclose(sa[k],sb[k],rtol=1e-5,atol=1e-7) for k in sa)
        diff=max(float((sa[k]-sb[k]).abs().max()) for k in sa)
        trajectory_exact=ra['trajectory_metrics']==rb['trajectory_metrics']
        diagnostics_exact=json.dumps(ra['round_summaries'],sort_keys=True)==json.dumps(rb['round_summaries'],sort_keys=True)
        metrics_close=all(abs(x['metrics'][key]-y['metrics'][key])<=1e-7 for x,y in zip(ra['trajectory_metrics'],rb['trajectory_metrics']) for key in x['metrics'])
        rows.append(dict(id=a['id'],checkpoint_bitwise_equal=exact,checkpoint_allclose=close,max_abs_weight_difference=diff,all_round_diagnostics_exact=diagnostics_exact,all_round_metrics_exact=trajectory_exact,all_round_metrics_close=metrics_close))
    report=dict(first=first,second=second,all_passed=all(r['checkpoint_bitwise_equal'] and r['all_round_metrics_exact'] and r['all_round_diagnostics_exact'] for r in rows),tolerance=dict(weight_rtol=1e-5,weight_atol=1e-7,metric_atol=1e-7),rows=rows)
    write(BASE/f'equivalence_{first}_{second}.json',report);assert report['all_passed'],'Concurrent execution changed matched outputs beyond tolerance'
    return report


def main():
    first=batch(2,'first_round')
    m=json.loads((BASE/'first_round/manifest.json').read_text())
    for gpu in [0,1]:
        paths=[p for p in m['jobs'] if json.loads(Path(p).read_text())['benchmark_gpu']==gpu]
        d=BASE/f'first_round_gpu{gpu}';d.mkdir(exist_ok=True)
        write(d/'manifest.json',dict(m,jobs=paths))
    acceptance=equivalent('first_round_gpu0','first_round_gpu1')
    write(BASE/'first_round_acceptance.json',acceptance)
    print('FIRST_ROUND_EXACT_PASS',flush=True)
    two=batch(2);four=batch(4)
    exact=equivalent('concurrency2','concurrency4')
    report=dict(kind='deterministic_acceptance_performance_only',first_round=first,first_round_equivalence=acceptance,runs=[two,four],
      throughput_ratio_4_over_2=four['rounds_per_second']/two['rounds_per_second'],equivalence=exact,test_used=False,
      caveats=['Single ordered comparison; repeated seed123 is performance only, never independent scientific evidence.',
               'Whole-container CPU includes unrelated root/ACS jobs; GPU memory is sampled rather than allocator peak.',
               'Three-round end-to-end throughput includes startup and is not an exact 70-round forecast.'])
    if report['throughput_ratio_4_over_2']>=1.10:
        extended4=batch(4,'eight_jobs_concurrency4');extended8=batch(8,'eight_jobs_concurrency8')
        report['extension']=dict(runs=[extended4,extended8],throughput_ratio_8_over_4=extended8['rounds_per_second']/extended4['rounds_per_second'],
               equivalence=equivalent('eight_jobs_concurrency4','eight_jobs_concurrency8'))
    write(BASE/'comparison.json',report)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
