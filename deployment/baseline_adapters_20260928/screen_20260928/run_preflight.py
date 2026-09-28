"""Four bounded real-image regression gates; no formal dispatch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/GuardFed-celeba-expanded')
BASE=ROOT/'deployment/baseline_adapters_20260928/screen_20260928'
OUT=ROOT/'results/revision_20260928/celeba_baseline_screen_v1/preflight'
OLD=ROOT/'results/revision_20260928/baseline_image_gates_v1'

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    live=[]
    for algorithm,pattern,worker in [('fedaa','gate_jobs/jobs/*.json','run_fedaa_screen.py'),('lasa','jobs/preflight/*.json','worker.py')]:
        for job_path in sorted((BASE/algorithm).glob(pattern)):
            job=json.loads(job_path.read_text());attack=job['attack'];name=algorithm+'_'+attack
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(['Benign','S-DFA'].index(attack)),GUARDFED_CPU_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            log=(OUT/(name+'.log')).open('x')
            proc=subprocess.Popen([sys.executable,str(BASE/algorithm/worker),'--repo',str(ROOT),'--job',str(job_path),'--out',str(OUT/name)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            live.append((name,proc,log))
    assert len(live)==4
    codes={name:proc.wait() for name,proc,_ in live}
    for _,_,log in live:log.close()
    (OUT/'exit.json').write_text(json.dumps(codes,indent=2));assert not any(codes.values()),codes
    import torch
    for attack in ['Benign','S-DFA']:
        subprocess.run([sys.executable,str(BASE/'fedaa/compare_pilot_equivalence.py'),'--pilot',str(OLD/('fedaa_'+attack)/'training_state.pt'),'--screen',str(OUT/('fedaa_'+attack)/'training_state.pt'),'--out',str(OUT/('fedaa_'+attack+'_equality.json'))],check=True)
        a=OLD/('lasa_'+attack);b=OUT/('lasa_'+attack)
        ma=torch.load(a/'model.pt',map_location='cpu',weights_only=True);mb=torch.load(b/'model.pt',map_location='cpu',weights_only=True)
        assert ma.keys()==mb.keys() and all(torch.equal(ma[k],mb[k]) for k in ma)
        ra=json.loads((a/'result.json').read_text());rb=json.loads((b/'result.json').read_text())
        for k in ['metrics','trajectory_metrics','round_summaries','attack_audit']:
            assert ra[k]==rb[k],(attack,k)
        (OUT/('lasa_'+attack+'_equality.json')).write_text(json.dumps({'status':'passed','model_tensors_exact':True,'three_round_metrics_diagnostics_attack_exact':True}))
    (OUT/'acceptance.json').write_text(json.dumps({'status':'PASS','real_image_gates':4,'default_parameter_exact_regressions':4,'formal_results':False,'time':time.time()},indent=2))
