"""Seven sequential fresh-child canaries: 2 original references, 5 revised-worker conditions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from screen_common import HERE,accepted,authorized,digest,local_identity,read,repo_identity,write_json


def equal(a,b):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);args=parser.parse_args()
    protocol,manifest=local_identity();authorized('seven_same_horizon_3round_canaries',fresh=True)
    assert not (HERE/'GATE_ACCEPTANCE.json').exists() and not (HERE/'preflight').exists(),'Existing gate evidence: no automatic retry'
    (HERE/'preflight/logs').mkdir(parents=True)
    before=repo_identity(args.repo,protocol)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',GUARDFED_CPU_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    try:
        for attack in ('Benign','S-DFA'):
            with (HERE/'preflight/logs'/('reference_'+attack+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-B',str(HERE/'canary_reference.py'),'--repo',str(args.repo),'--attack',attack],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        (HERE/'preflight/runs').mkdir()
        for item in manifest['preflight_jobs']:
            with (HERE/'preflight/logs'/(item['id']+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-B',str(HERE/'run_one.py'),'--repo',str(args.repo),'--job-id',item['id'],'--canary'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        import torch
        pairs=[]
        for item in manifest['preflight_jobs']:
            output=HERE/'preflight/runs'/item['id'];result=accepted(item,output);assert result is not None and result['rounds']==3
            if item['attack'] not in ('Benign','S-DFA'):continue
            original=next(x for x in manifest['reused_jobs'] if x['distribution']=='non-IID' and x['attack']==item['attack'])
            ref=HERE/'preflight/references'/original['id'];identity=read(ref/'REFERENCE_IDENTITY.json')
            assert identity['status']=='PASS_CANARY_NOT_SCIENTIFIC_SCREEN' and identity['package_sha256']==digest(HERE/'PACKAGE_SHA256.json')
            old=read(ref/'result.json')
            evidence=read(ref/'acceptance.json')
            assert evidence['status']=='PASS' and evidence['rounds']==old['rounds']==3
            assert old['seed']==91001 and old['distribution']=='non-IID' and old['attack']==item['attack']
            for name,expected in evidence['artifact_hashes'].items():assert digest(ref/name)==expected
            for key in ('metrics','trajectory_metrics','round_summaries','evaluation_stats','data_contract'):
                assert equal(result[key],old[key]),('same-horizon mismatch',key,item['id'])
            for name in ('state.json','diagnostics.json'):
                assert equal(read(output/name),read(ref/name)),name
            assert equal(torch.load(output/'model.pt',map_location='cpu',weights_only=True),torch.load(ref/'model.pt',map_location='cpu',weights_only=True))
            for prefix in ['round_001','round_002','round_003','final']:
                assert equal(read(output/(prefix+'_rng.json')),read(ref/(prefix+'_rng.json')))
                assert equal(torch.load(output/(prefix+'_rng.pt'),map_location='cpu',weights_only=True),torch.load(ref/(prefix+'_rng.pt'),map_location='cpu',weights_only=True))
                # Preserve all imported state, compare only the measured training registry.
                a,b=read(output/(prefix+'_rng_scope.json')),read(ref/(prefix+'_rng_scope.json'))
                assert equal(a['training_states'],b['training_states']) and a['import_advanced_indices']==b['import_advanced_indices']
            pairs.append(item['id'])
        assert repo_identity(args.repo,protocol)==before;local_identity()
        files={p.relative_to(HERE).as_posix():digest(p) for p in (HERE/'preflight').rglob('*') if p.is_file()}
        write_json(HERE/'GATE_ACCEPTANCE.json',dict(status='PASS',package_sha256=digest(HERE/'PACKAGE_SHA256.json'),
            accepted_new_canaries=5,same_horizon_pairs=len(pairs),pairs=pairs,artifact_hashes=files,
            new_attack_canaries=['F Flip','FedSA','Sp-DFA'],horizon=3,formal_table_samples=0,
            limitation='Selected Tg10/20 retained; canaries cover GMM but not later UCL/monitor. No universal70-round equivalence; no comparison to later70-round prefix.'))
    except BaseException as error:
        write_json(HERE/'preflight/failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
