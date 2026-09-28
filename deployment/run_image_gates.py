"""One bounded supervised pilot batch; no retries or formal training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/GuardFed-celeba-expanded')
BASE=ROOT/'deployment/baseline_adapters_20260928'
OUT=ROOT/'results/revision_20260928/baseline_image_gates_v1'

def commands(resume=False):
    result=[]
    for gpu,attack in enumerate(['Benign','S-DFA']):
        name='fedaa_'+attack
        cmd=[sys.executable,str(BASE/'integration_20260928/run_fedaa_pilot.py'),'--repo',str(ROOT),
             '--job',str(BASE/'integration_20260928/jobs'/f'FedAA-DDPG_non-IID_{attack}_seed91001_gate3.json'),
             '--out',str(OUT/(name+('_resume' if resume else '')))]
        if resume:cmd+=['--resume',str(OUT/name/'checkpoint_round1.pt')]
        result.append((name+('_resume' if resume else ''),gpu,cmd))
        if not resume:
            result.append(('lasa_'+attack,gpu,[sys.executable,str(BASE/'lasa_20260928/run_pilot.py'),'--repo',str(ROOT),
                '--job',str(BASE/'lasa_20260928'/f'job_{attack}.json'),'--out',str(OUT/('lasa_'+attack))]))
    return result

def batch(resume):
    live=[]
    for name,gpu,cmd in commands(resume):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),GUARDFED_CPU_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        log=(OUT/(name+'.log')).open('x')
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        live.append((name,p,log))
    codes={name:p.wait() for name,p,log in live}
    for _,_,log in live:log.close()
    (OUT/('resume_exit.json' if resume else 'pilot_exit.json')).write_text(json.dumps({'codes':codes,'time':time.time()},indent=2))
    assert not any(codes.values()),codes

if __name__=='__main__':
    batch(False)
    batch(True)
    for attack in ['Benign','S-DFA']:
        name='fedaa_'+attack
        subprocess.run([sys.executable,str(BASE/'integration_20260928/compare_pilot_resume.py'),
            '--continuous',str(OUT/name/'training_state.pt'),'--resumed',str(OUT/(name+'_resume')/'training_state.pt'),
            '--out',str(OUT/(name+'_recovery_check.json'))],check=True)
    (OUT/'queue_complete.json').write_text(json.dumps({'complete':True,'pipeline_gates':4,'exact_recovery_checks':2,'formal_results':False,'time':time.time()},indent=2))
