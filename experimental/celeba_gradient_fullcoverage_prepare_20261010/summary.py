"""Future original-checker 96+4 summary. No CNN, no peak/seed selection or refitting."""
from pathlib import Path
import argparse,copy,importlib,json,statistics,sys
sys.dont_write_bytecode=True
from prepare import METHODS,SEEDS,DIST,ATTACKS,OLD,BRIDGE,H,read,inputs,grid,render,SEAL

def checked(stage,job,out):
    # Both checkers retain the actual job/source identity and terminal tensor checks.
    for n in ('worker','accept_result'):sys.modules.pop(n,None)
    sys.path.insert(0,str(stage))
    try:return importlib.import_module('accept_result').checked_result(job,out)
    finally:sys.path.pop(0)

def stats(records):
    metrics=('accuracy','aeod','aspd');answer=[]
    for label,seeds in [('10seed',SEEDS),('9seed',SEEDS[1:]),('6seed',SEEDS[4:])]:
        for method in METHODS:
            rows=[r for r in records if r['method']==method and r['seed'] in seeds]
            assert len(rows)==10*len(seeds)
            for d,a in [(d,a) for d in DIST for a in ATTACKS]+[('balanced','all10')]:
                for metric in metrics:
                    if d=='balanced':
                        values=[statistics.mean(r['metrics'][metric] for r in rows if r['seed']==s) for s in seeds]
                    else:
                        selected=[r for r in rows if (r['distribution'],r['attack'])==(d,a)]
                        assert sorted(r['seed'] for r in selected)==list(seeds)
                        values=[r['metrics'][metric] for r in sorted(selected,key=lambda r:r['seed'])]
                    answer.append(dict(panel=label,method=method,distribution=d,attack=a,metric=metric,
                        n=len(seeds),mean=statistics.mean(values),sample_sd=statistics.stdev(values)))
    return answer

def summarize(bound,outputs,out,approval_path,approval_sha):
    original,original_manifest,_,_=inputs();bound,outputs,out=map(Path,(bound,outputs,out))
    approval_path=Path(approval_path);assert H(approval_path.read_bytes())==approval_sha
    approval=read(approval_path)
    assert approval['status']=='ROOT_GRADIENT200_FROZEN_SOURCE_ADOPTED' and approval['test'] is False
    assert approval['bound_inputs_sha256']==H((bound/'BOUND_INPUTS.json').read_bytes())
    assert not out.exists()
    binding=read(bound/'BOUND_INPUTS.json')
    assert H(Path(binding['root_path']).read_bytes())==binding['root_sha256']
    assert H(Path(binding['summary_path']).read_bytes())==binding['summary_sha256']
    root=read(binding['root_path']);assert root['status']=='ROOT_GRADIENT64_COMPLETE_STRICT_OFFSERVER_ADOPTED'
    assert root['summary_sha256']==binding['summary_sha256'] and root['screen_source_seal_sha256']==SEAL
    assert root['accepted_count']==64 and root['all64_offserver_verified'] is True and root['test'] is False
    assert root['selected_candidates']==binding['winners']
    derived,_=render()
    records=[]
    for method in METHODS:
        base=bound/method;manifest=read(base/'jobs/manifest.json');stage=base/'snapshot/gradient_bridge_fullcoverage'
        protocol=read(stage/'protocol.json');assert protocol['status']=='FROZEN'
        assert approval['manifest_sha256'][method]==H((base/'jobs/manifest.json').read_bytes())
        assert all((stage/n).read_text(encoding='utf8')==text for n,text in derived.items())
        candidates=protocol['candidates'];assert len(candidates)==1 and candidates[0]['id']==binding['winners'][method]
        assert candidates==[c for c in original['candidates'] if c['id']==binding['winners'][method]]
        assert candidates[0]['method']==method and protocol['attacks']==list(ATTACKS) and protocol['distributions']==DIST
        expected_protocol=copy.deepcopy(original)
        expected_protocol.update(status='FROZEN',version='celeba_gradient_fullcoverage_20261010',attacks=list(ATTACKS),
            candidates=candidates,component_hashes=read(OLD/'jobs'/original_manifest['jobs'][0]['job'])['component_hashes'])
        assert protocol==expected_protocol
        hashes={n:H((stage/n).read_bytes()) for n in ('worker.py','accept_result.py','protocol.json')}
        expected={j['id']:j for j in grid(candidates[0],protocol,hashes)}
        assert len(manifest['new_jobs'])==96 and {j['id'] for j in manifest['new_jobs']}==set(expected)
        for entry in manifest['new_jobs']:
            path=base/'jobs'/entry['job'];assert H(path.read_bytes())==entry['sha256'] and read(path)==expected[entry['id']]
            result=checked(stage,path,outputs/method/entry['id']);assert result is not None
            records.append(result)
        reused=manifest['reused_jobs'];assert len(reused)==4
        original_rows=read(binding['summary_path'])['records']
        assert reused==[r for r in original_rows if r['candidate']==binding['winners'][method]]
        for r in reused:
            oldjob=OLD/'jobs'/(r['id']+'.json');path=Path(r['output'])
            assert H(oldjob.read_bytes())==r['job_sha256']
            for name,key in [('model.pt','model_sha256'),('result.json','result_sha256'),('acceptance.json','acceptance_sha256')]:
                assert H((path/name).read_bytes())==r[key]
            result=checked(OLD/BRIDGE,oldjob,path);assert result is not None
            assert result['tuning_candidate']==binding['winners'][method] and result['metrics']==r['metrics']
            records.append(result)
    keys=[(r['method'],r['distribution'],r['attack'],r['seed']) for r in records]
    assert len(keys)==len(set(keys))==200 and set(keys)=={(m,d,a,s) for m in METHODS for d in DIST for a in ATTACKS for s in SEEDS}
    compact=[{k:r[k] for k in ('method','distribution','attack','seed','metrics','provenance')} for r in records]
    result=dict(status='ORIGINAL_CHECKED_200_VALID_RECORDS_SUMMARY_OFFSERVER_ADOPTION_SEPARATE',records=compact,statistics=stats(compact),
        limits=['valid-only; seed91001 selected recipes','Huber identity-Rp CNN adaptation, no theorem inheritance',
               'scenario averages formed within seed; sampleSD ddof1','all constant/negative outcomes retained','not final test; not root adoption'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--bound',required=True);a.add_argument('--outputs',required=True);a.add_argument('--out',required=True)
    a.add_argument('--stage-approval',required=True);a.add_argument('--approval-sha256',required=True)
    x=a.parse_args();summarize(x.bound,x.outputs,x.out,x.stage_approval,x.approval_sha256)
