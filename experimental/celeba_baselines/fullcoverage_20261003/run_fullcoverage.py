"""Reuse the existing fail-stop queue; add the fixed 192+8 coverage contract."""
import collections
import copy
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path('/workspace/GuardFed-celeba-expanded')
BASE=ROOT/'deployment/baseline_adapters_20260928/fullcoverage_20261003'
STAGE=ROOT/'results/revision_20261003/celeba_baseline_fullcoverage_v1'
OLD=BASE.parent/'screen_20260928'

def module(name,path):
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return sys.modules[name]

legacy=module('coverage_original_screen',OLD/'run_screen.py')
engine=module('coverage_queue_engine',OLD/'run_screen.py')
digest,save=legacy.digest,legacy.save
METRICS=legacy.METRICS
CONDITIONS={(d,a) for d in ['IID','non-IID'] for a in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA']}
SELECTED={'FedAA':'policy0.001_keep16_local0.001','LASA':'LASA_s0.3_l2_lr0.001'}

def frozen_manifest(m):
    assert not sys.flags.optimize
    assert digest(STAGE/'PROTOCOL.md')==m['protocol_sha256'],'Protocol drift'
    assert m['selected_recipes']==SELECTED and m['planned_new']==192 and m['planned_total']==200
    assert m['seeds']==list(range(91001,91011)) and m['distributions']=={'IID':5000.,'non-IID':5.}
    assert m['concurrency']==8 and m['evaluation_split']=='valid' and len(m['jobs'])==192 and len(m['reused_jobs'])==8
    all_items=m['jobs']+m['reused_jobs'];assert len({i['id'] for i in all_items})==200
    assert len({i['job'] for i in all_items})==len({i['output'] for i in all_items})==200
    observed=set();definitions={}
    for item in all_items:
        assert digest(item['job'])==item['job_sha256'],('Job drift',item['id'])
        j=json.loads(Path(item['job']).read_text());cfg=j['config'];definitions[item['id']]=j
        assert j['id']==item['id'] and j['tuning_candidate']==SELECTED[item['algorithm']]
        assert all(item[k]==j[k] for k in ['method','distribution','attack','tuning_candidate'])
        assert cfg['rounds']==70 and cfg['celeba_evaluation_split']=='valid'
        assert cfg['client_alpha']==m['distributions'][j['distribution']]
        key=(item['algorithm'],j['distribution'],j['attack'],cfg['seed']);assert key not in observed;observed.add(key)
        assert (j['distribution'],j['attack']) in CONDITIONS and cfg['seed'] in m['seeds']
        if item in m['jobs']:
            folder='fedaa' if item['algorithm']=='FedAA' else 'lasa'
            worker=module('coverage_'+folder+'_definition',BASE/folder/('run_fedaa_screen.py' if folder=='fedaa' else 'worker.py'))
            worker.validate_job(j)
            assert j['evidence_stage']=='multi_seed_validation_coverage'
        else:
            assert cfg['seed']==91001 and j['attack'] in ['Benign','S-DFA']
    expected={(a,d,t,s) for a in SELECTED for d,t in CONDITIONS for s in m['seeds']}
    assert observed==expected,'Incomplete or duplicate matrix'
    assert len(m['preflight_jobs'])==10
    gate_keys=set()
    for item in m['preflight_jobs']:
        assert digest(item['job'])==item['job_sha256']
        j=json.loads(Path(item['job']).read_text())
        assert j['id']==item['id'] and j['distribution']=='non-IID' and j['config']['seed']==91001 and j['config']['rounds']==3
        assert j['evidence_stage']=='pipeline_canary_only' and j['tuning_candidate']==SELECTED[item['algorithm']]
        key=(item['algorithm'],j['attack']);assert key not in gate_keys;gate_keys.add(key)
        folder='fedaa' if item['algorithm']=='FedAA' else 'lasa'
        module('coverage_'+folder+'_definition',BASE/folder/('run_fedaa_screen.py' if folder=='fedaa' else 'worker.py')).validate_job(j)
        if j['attack'] in ['Benign','S-DFA']:
            assert item['regression_source'] in m['reused_jobs']
            prior=item['regression_source']
            assert (prior['algorithm'],prior['distribution'],prior['attack'])==(item['algorithm'],'non-IID',j['attack'])
        else:assert item['regression_source'] is None
    assert gate_keys=={(a,t) for a in SELECTED for t in m['attacks']}
    assert len({i['job'] for i in m['preflight_jobs']})==len({i['output'] for i in m['preflight_jobs']})==10
    assert len(m.get('reference_gates',[]))==2 and {i['attack'] for i in m['reference_gates']}=={'Benign','S-DFA'}
    for item in m['reference_gates']:
        assert digest(item['job'])==item['job_sha256']
        j=json.loads(Path(item['job']).read_text())
        assert j['id']==item['id'] and j['attack']==item['attack']
        module('coverage_lasa_reference',BASE/'run_lasa_reference.py').validate(j)
    for name,h in m['source_hashes'].items():assert digest(ROOT/name)==h,('Source/data drift',name)
    assert digest(ROOT/'results/revision_20260928/celeba_baseline_screen_v1/manifest.json')==m['screen_manifest_sha256']
    assert digest(ROOT/'results/revision_20260928/celeba_baseline_screen_v1/summary.json')==m['screen_summary_sha256']
    return definitions

def fedaa_acceptance():
    worker=module('coverage_fedaa_definition',BASE/'fedaa/run_fedaa_screen.py')
    # The historical and extended acceptors import the same legacy module name.
    # Bind it only while loading; each acceptor keeps its own imported functions.
    previous=sys.modules.get('run_fedaa_screen')
    sys.modules['run_fedaa_screen']=worker
    try:return module('coverage_fedaa_acceptance',BASE/'fedaa/accept_result.py')
    finally:
        if previous is None:sys.modules.pop('run_fedaa_screen',None)
        else:sys.modules['run_fedaa_screen']=previous

def checked_new(item):
    out=Path(item['output']);p=out/'result.json'
    assert not list(out.glob('failure*.json')),('Preserved failure',str(out))
    assert digest(item['job'])==item['job_sha256']
    if not p.exists():return None
    j=json.loads(Path(item['job']).read_text());rounds=j['config']['rounds']
    if item['algorithm']=='FedAA':
        r=fedaa_acceptance().checked_result(Path(item['job']),out)
        stats=r['trajectory_metrics'][-1]['evaluation_stats']
    else:
        r=json.loads(p.read_text());prov=r['revision_job']
        assert all(prov[k]==v for k,v in j.items())
        assert Path(prov['output']).resolve()==out.resolve()
        assert prov['checkpoint_sha256']==digest(out/'model.pt')
        evidence=json.loads((out/'acceptance.json').read_text())
        assert r['status']==('coverage_complete' if j['phase']=='fullcoverage' else 'pilot_complete')
        assert r['evidence_stage']==evidence['evidence_stage']==j['evidence_stage']
        assert r['stage_version']==evidence['stage_version']=='celeba_baseline_fullcoverage_20261003_v1'
        assert evidence['completion_status']==r['status']
        assert evidence['status']=='PASS' and evidence['phase']==j['phase'] and evidence['rounds']==rounds
        assert evidence['tuning_candidate']==j['tuning_candidate'] and evidence['evaluation_split']=='valid'
        assert evidence['result_sha256']==digest(p) and evidence['checkpoint_sha256']==digest(out/'model.pt')
        assert evidence['diagnostics_sha256']==digest(out/'diagnostics.json')
        assert json.loads((out/'diagnostics.json').read_text())==r['round_summaries']
        provenance=json.loads((out/'provenance.json').read_text())
        assert provenance['job_id']==j['id'] and provenance['job_sha256']==item['job_sha256']
        for k in ['source_hashes','adapter_source_hashes']:assert evidence[k]==provenance[k]==j[k]
        dc=r['data_contract']['image_data_contract']
        assert (dc['evaluation_split'],dc['actual_train_rows'],dc['actual_evaluation_rows'])==('valid',162770,19867)
        assert dc['train_eval_disjoint'] and dc['root_client_disjoint']
        import torch
        model=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in model.values())
        stats=r['evaluation_stats']
    for k in ['dataset','method','distribution','attack']:assert r[k]==j[k]
    assert r['rounds']==rounds and r['seed']==j['config']['seed'] and r['config']==j['config']
    assert r['metrics']==r['trajectory_metrics'][-1]['metrics']
    assert [x['round'] for x in r['trajectory_metrics']]==list(range(1,rounds+1))
    assert [x['round'] for x in r['round_summaries']]==list(range(1,rounds+1))
    assert all(math.isfinite(x['metrics'][k]) and 0<=x['metrics'][k]<=1 for x in r['trajectory_metrics'] for k in METRICS)
    assert stats['prediction_count']==19867
    assert all(math.isfinite(stats[k]) and 0<=stats[k]<=1 for k in ['positive_rate','majority_accuracy'])
    return dict(id=j['id'],algorithm=item['algorithm'],method=j['method'],distribution=j['distribution'],attack=j['attack'],seed=j['config']['seed'],candidate=j['tuning_candidate'],output=str(out),checkpoint_sha256=digest(out/'model.pt'),evaluation_stats=stats,reused=False,**r['metrics'])

def summarize(m):
    frozen_manifest(m);rows=[];invalid=[]
    for item in m['jobs']+m['reused_jobs']:
        try:
            row=legacy.checked(item) if item in m['reused_jobs'] else checked_new(item)
            if row:
                row['reused']=item in m['reused_jobs'];rows.append(row)
        except Exception as e:invalid.append(dict(id=item['id'],error=repr(e)))
    failures=engine.preserved_failures(m)
    cohorts={}
    for label,seeds in [('ten_seed',set(range(91001,91011))),('exclude_selection_nine_seed',set(range(91002,91011))),('matching_six_seed',set(range(91005,91011)))]:
        cells=[];overall=[]
        for algorithm in SELECTED:
            for distribution,attack in sorted(CONDITIONS):
                group=[r for r in rows if r['algorithm']==algorithm and r['distribution']==distribution and r['attack']==attack and r['seed'] in seeds]
                assert len(group)==len({r['seed'] for r in group})
                cell=dict(algorithm=algorithm,distribution=distribution,attack=attack,n=len(group),expected_n=len(seeds),complete={r['seed'] for r in group}==seeds)
                if cell['complete']:
                    cell['metrics']={k:dict(mean=statistics.mean(r[k] for r in group),sample_sd=statistics.stdev(r[k] for r in group)) for k in METRICS}
                cells.append(cell)
            within=[]
            for seed in sorted(seeds):
                group=[r for r in rows if r['algorithm']==algorithm and r['seed']==seed]
                if {(r['distribution'],r['attack']) for r in group}==CONDITIONS:
                    within.append(dict(seed=seed,**{k:statistics.mean(r[k] for r in group) for k in METRICS}))
            if len(within)==len(seeds):
                overall.append(dict(algorithm=algorithm,n=len(seeds),metrics={k:dict(mean=statistics.mean(r[k] for r in within),sample_sd=statistics.stdev(r[k] for r in within)) for k in METRICS},per_seed=within))
        cohorts[label]=dict(cells=cells,overall=overall)
    summary=dict(accepted=len(rows),accepted_new=sum(not r['reused'] for r in rows),accepted_reused=sum(r['reused'] for r in rows),planned_new=192,planned_total=200,complete=len(rows)==200 and not invalid and not failures,records=rows,invalid_records=invalid,preserved_failures=failures,historical_failures=[str(p) for p in (STAGE/'failed_attempts').rglob('*failure*.json')],cohorts=cohorts,manifest_sha256=digest(STAGE/'manifest.json'),summarizer_sha256=digest(__file__),note='Validation coverage; fixed selected recipes; sampleSD across seeds; no new recipe or seed selection')
    save(STAGE/'summary.json',summary)
    fields=['id','algorithm','method','distribution','attack','seed','candidate','reused',*METRICS,'checkpoint_sha256','output']
    with (STAGE/'per_seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return summary

def run_gates(m):
    frozen_manifest(m)
    for item in m['reused_jobs']:assert legacy.checked(item) is not None
    path=STAGE/'preflight/acceptance.json'
    if path.exists():
        g=json.loads(path.read_text());assert g['status']=='PASS' and g['planned_gates']==10
        assert all(digest(p)==h for p,h in g['evidence_hashes'].items())
        return g
    results=[]
    # A three-round core run performs its last-ten reporting evaluation every
    # round; the 70-round core starts that extra RNG-consuming evaluation later.
    # LASA therefore compares against an unchanged-worker three-round fixture,
    # plus the original 70-round first boundary, rather than a different horizon.
    references={}
    for item in m.get('reference_gates',[]):
        out=Path(item['output'])
        assert digest(item['job'])==item['job_sha256']
        if not (out/'result.json').exists():
            assert not out.exists(),'Partial reference requires review'
            logpath=STAGE/'preflight/logs'/f"{item['id']}_reference.log";logpath.parent.mkdir(parents=True,exist_ok=True)
            with logpath.open('x') as log:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',GUARDFED_CPU_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                subprocess.run([sys.executable,str(BASE/'run_lasa_reference.py'),'--job',item['job'],'--out',item['output']],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert not list(out.glob('failure*.json'))
        result=json.loads((out/'result.json').read_text())
        assert result['config']['rounds']==3 and len(result['trajectory_metrics'])==3
        assert result['revision_job']['checkpoint_sha256']==digest(out/'model.pt')
        assert digest(out/'result.json')==json.loads((out/'acceptance.json').read_text())['result_sha256']
        references[result['attack']]=result
    for start in range(0,10,8):
        live=[]
        for slot,item in enumerate(m['preflight_jobs'][start:start+8]):
            assert digest(item['job'])==item['job_sha256']
            if (Path(item['output'])/'result.json').exists():
                assert checked_new(item) is not None
                live.append((item,None,None));continue
            assert not Path(item['output']).exists(),'Prior partial gate needs review'
            folder='fedaa' if item['algorithm']=='FedAA' else 'lasa'
            script=BASE/folder/('run_fedaa_screen.py' if folder=='fedaa' else 'worker.py')
            logpath=STAGE/'preflight/logs'/f"{item['id']}.log";logpath.parent.mkdir(parents=True,exist_ok=True);log=logpath.open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(slot%2),GUARDFED_CPU_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            proc=subprocess.Popen([sys.executable,str(script),'--repo',str(ROOT),'--job',item['job'],'--out',item['output']],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            live.append((item,proc,log))
        errors=[]
        for item,proc,log in live:
            if proc:proc.wait();log.close()
            try:
                assert proc is None or proc.returncode==0,(item['id'],proc.returncode if proc else None)
                row=checked_new(item);assert row is not None
                result=json.loads((Path(item['output'])/'result.json').read_text())
                prior=item['regression_source']
                if prior:
                    old_result=json.loads((Path(prior['output'])/'result.json').read_text())
                    if item['algorithm']=='LASA':
                        assert result['trajectory_metrics'][0]==old_result['trajectory_metrics'][0]
                        assert result['round_summaries'][0]==old_result['round_summaries'][0]
                        reference=references[result['attack']]
                        assert result['trajectory_metrics']==reference['trajectory_metrics']
                        assert result['round_summaries']==reference['round_summaries']
                        import torch
                        ref_item=next(x for x in m['reference_gates'] if x['attack']==result['attack'])
                        before=torch.load(Path(ref_item['output'])/'model.pt',map_location='cpu',weights_only=True)
                        after=torch.load(Path(item['output'])/'model.pt',map_location='cpu',weights_only=True)
                        assert before.keys()==after.keys() and all(torch.equal(v,after[k]) for k,v in before.items())
                    else:
                        assert result['trajectory_metrics']==old_result['trajectory_metrics'][:3],('Trajectory regression',item['id'])
                        assert result['round_summaries']==old_result['round_summaries'][:3],('Diagnostic regression',item['id'])
                    if item['algorithm']=='FedAA':
                        import torch
                        import numpy as np
                        before=torch.load(Path(prior['output'])/'checkpoint_round1.pt',map_location='cpu',weights_only=False)
                        after=torch.load(Path(item['output'])/'checkpoint_round1.pt',map_location='cpu',weights_only=False)
                        def equal(a,b):
                            if isinstance(a,torch.Tensor):return torch.equal(a,b)
                            if isinstance(a,np.ndarray):return np.array_equal(a,b,equal_nan=True)
                            if isinstance(a,(float,np.floating)) and isinstance(b,(float,np.floating)) and math.isnan(a) and math.isnan(b):return True
                            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
                            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
                            return a==b
                        for k in ['model','controller','rng','trajectory_metrics','round_summaries','attack_audit','warnings']:assert equal(before[k],after[k]),('First boundary regression',item['id'],k)
                results.append(dict(id=item['id'],algorithm=item['algorithm'],attack=row['attack'],rounds=3,regression=bool(prior),status='PASS'))
            except Exception as exc:errors.append(dict(id=item['id'],error=repr(exc)))
        if errors:
            save(STAGE/'preflight/failure.json',dict(status='FAIL',errors=errors,successful_gates=results));raise RuntimeError('Preflight failed; no formal dispatch')
    evidence={}
    for item in m['preflight_jobs']:
        for name in ['result.json','model.pt','progress.json']:
            p=Path(item['output'])/name;evidence[str(p)]=digest(p)
    for item in m.get('reference_gates',[]):
        for name in ['result.json','model.pt','acceptance.json']:
            p=Path(item['output'])/name;evidence[str(p)]=digest(p)
    g=dict(status='PASS',planned_gates=10,real_image_gates=4,default_parameter_exact_regressions=4,fedaa_three_round_prefix_regressions=2,lasa_same_horizon_model_regressions=2,lasa_first_round_prefix_regressions=2,new_attack_support_gates=6,formal_results=False,evidence_hashes=evidence,results=results,checked_unix=time.time(),limit='FedAA3round prefix/first-state;LASA same3round horizon exactmodel+diagnostics and70round first-boundary;not full70round universal equivalence')
    save(path,g);return g

engine.BASE=BASE;engine.STAGE=STAGE
engine.frozen_manifest=frozen_manifest;engine.checked=checked_new;engine.summarize=summarize

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','summarize','preflight']);a=p.parse_args()
    m=json.loads((STAGE/'manifest.json').read_text())
    if a.action=='summarize':
        result=summarize(m);print(json.dumps({k:v for k,v in result.items() if k not in ['records','cohorts']}))
    else:
        try:
            gate=run_gates(m)
            if a.action=='run':
                if m['status']=='PREPARED_PENDING_GATES':
                    m.update(status='FROZEN_READY',gate_acceptance_sha256=digest(STAGE/'preflight/acceptance.json'),frozen_unix=time.time())
                    save(STAGE/'manifest.json',m)
                assert m['status']=='FROZEN_READY' and digest(STAGE/'preflight/acceptance.json')==m['gate_acceptance_sha256']
                engine.run(m)
        except BaseException:
            import traceback
            save(STAGE/'launch_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
