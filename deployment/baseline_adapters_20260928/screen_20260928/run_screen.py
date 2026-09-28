"""Bounded, fail-stop 64-job validation queue; strictly accept before skipping."""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path('/workspace/GuardFed-celeba-expanded')
BASE=ROOT/'deployment/baseline_adapters_20260928/screen_20260928'
STAGE=ROOT/'results/revision_20260928/celeba_baseline_screen_v1'
CONDITIONS={(d,a) for d in ['IID','non-IID'] for a in ['Benign','S-DFA']}
METRICS=('accuracy','aeod','aspd')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def load_module(name,path):
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]

def frozen_manifest(manifest):
    """Both run and summarize validate the same immutable 2 x 8 x 4 design."""
    assert not sys.flags.optimize,'Assertions must remain enabled'
    assert digest(STAGE/'PROTOCOL.md')==manifest['protocol_sha256'],'protocol drift'
    jobs=manifest['jobs'];assert len(jobs)==64
    assert len({j['id'] for j in jobs})==len({str(Path(j['job']).resolve()) for j in jobs})==len({str(Path(j['output']).resolve()) for j in jobs})==64
    assert {j['algorithm'] for j in jobs}=={'FedAA','LASA'}
    assert isinstance(manifest['concurrency'],int) and 1<=manifest['concurrency']<=8
    hashes=dict(manifest['source_hashes']);definitions={};coverage=collections.defaultdict(list)
    sys.path.insert(0,str(BASE/'fedaa'))
    fedaa=load_module('screen_fedaa_definition',BASE/'fedaa/run_fedaa_screen.py')
    lasa=load_module('screen_lasa_definition',BASE/'lasa/worker.py')
    for item in jobs:
        assert digest(item['job'])==item['job_sha256'],('job drift',item['id'])
        job=json.loads(Path(item['job']).read_text());definitions[item['id']]=job
        assert item['id']==job['id'] and job['dataset']=='celeba'
        for key in ['method','distribution','attack','tuning_candidate']:
            if key in item:assert item[key]==job[key],('manifest/job mismatch',key)
        cfg=job['config'];assert cfg['rounds']==70 and cfg['seed']==91001 and cfg['celeba_evaluation_split']=='valid'
        assert cfg['client_alpha']=={'IID':5000.,'non-IID':5.}[job['distribution']]
        assert (job['distribution'],job['attack']) in CONDITIONS
        if item['algorithm']=='FedAA':
            fedaa.validate_job(job);assert job['evidence_stage']=='validation_screen'
            expected=f"policy{job['policy_config']['actor_lr']:g}_keep{job['aggre_num']}_local{cfg['learning_rate']:g}"
            assert job['tuning_candidate']==expected
            assert job['id']==f"FedAA-DDPG_{expected}_{job['distribution']}_{job['attack']}_seed91001"
            adapter_paths={'screen_runner':BASE/'fedaa/run_fedaa_screen.py',
                'round_adapter':BASE/'fedaa/fedaa_round_adapter.py',
                'official_wrapper':BASE.parent/'fedaa/fedaa_official_adapter.py',
                'official_ddpg':BASE.parent/'fedaa/official/DDPG/DDPG.py'}
            adapter_hashes={str(adapter_paths[k]):v for k,v in job['adapter_hashes'].items()}
        else:
            lasa.validate_job(job);assert job['phase']=='screen'
            adapter_hashes={str(BASE/'lasa'/k):v for k,v in job['adapter_source_hashes'].items()}
        for name,h in list(job['source_hashes'].items())+list(adapter_hashes.items()):
            assert name not in hashes or hashes[name]==h,('conflicting frozen hash',name)
            hashes[name]=h
        coverage[(item['algorithm'],job['tuning_candidate'])].append((job['distribution'],job['attack']))
    expected_fedaa={f'policy{p:g}_keep{k}_local{lr:g}' for p in [.001,.01] for k in [10,16] for lr in [.0005,.001]}
    expected_lasa={f'LASA_s{s:g}_l{v:g}_lr{lr:g}' for s in [.3,.5] for v in [1.,2.] for lr in [.0005,.001]}
    for algorithm,candidates in [('FedAA',expected_fedaa),('LASA',expected_lasa)]:
        assert {c for a,c in coverage if a==algorithm}==candidates,('candidate grid changed',algorithm)
        for candidate in candidates:
            observed=coverage[(algorithm,candidate)]
            assert len(observed)==4 and set(observed)==CONDITIONS,('incomplete/duplicate candidate coverage',algorithm,candidate)
    for name,h in hashes.items():assert digest(ROOT/name)==h,('source/data/adapter drift',name)
    return definitions

def preserved_failures(manifest):
    paths=list(STAGE.glob('queue_failure*.json'))
    for item in manifest['jobs']:paths.extend(Path(item['output']).glob('failure*.json'))
    return sorted({str(p) for p in paths})

def record_failure(value):
    path=STAGE/'queue_failure.json'
    history=json.loads(path.read_text()) if path.exists() else {'failures':[]}
    if 'failures' not in history:history={'failures':[history]}
    history['failures'].append(dict(value,time=time.time()));save(path,history)

def checked(item):
    out=Path(item['output']);p=out/'result.json'
    assert not list(out.glob('failure*.json')),('preserved failure',out)
    assert digest(item['job'])==item['job_sha256'],('job drift',item['id'])
    job=json.loads(Path(item['job']).read_text());assert item['id']==job['id']
    if not p.exists():return None
    if item['algorithm']=='FedAA':
        sys.path.insert(0,str(BASE/'fedaa'))
        accept=load_module('screen_fedaa_acceptance',BASE/'fedaa/accept_result.py')
        r=accept.checked_result(Path(item['job']),out)
        eval_stats=r['trajectory_metrics'][-1]['evaluation_stats']
    elif item['algorithm']=='LASA':
        r=json.loads(p.read_text());prov=r['revision_job']
        for k,v in job.items():assert prov[k]==v,(item['id'],k)
        assert Path(prov['output']).resolve()==out.resolve()
        assert prov['checkpoint_sha256']==digest(out/'model.pt')
        assert r['config']==job['config'] and r['metrics']==r['trajectory_metrics'][-1]['metrics']
        assert [v['round'] for v in r['trajectory_metrics']]==list(range(1,71))
        assert [v['round'] for v in r['round_summaries']]==list(range(1,71))
        assert all(math.isfinite(v['metrics'][k]) and 0<=v['metrics'][k]<=1 for v in r['trajectory_metrics'] for k in METRICS)
        dc=r['data_contract']['image_data_contract']
        assert (dc['evaluation_split'],dc['actual_train_rows'],dc['actual_evaluation_rows'])==('valid',162770,19867)
        assert dc['train_eval_disjoint'] and dc['root_client_disjoint']
        evidence=json.loads((out/'acceptance.json').read_text())
        assert evidence['status']=='PASS' and evidence['phase']=='screen' and evidence['rounds']==70
        assert evidence['tuning_candidate']==job['tuning_candidate']
        assert evidence['result_sha256']==digest(p) and evidence['checkpoint_sha256']==digest(out/'model.pt')
        assert evidence['diagnostics_sha256']==digest(out/'diagnostics.json')
        assert json.loads((out/'diagnostics.json').read_text())==r['round_summaries']
        provenance=json.loads((out/'provenance.json').read_text())
        assert provenance['job_id']==job['id'] and provenance['job_sha256']==item['job_sha256']
        for k in ['source_hashes','adapter_source_hashes']:
            assert evidence[k]==provenance[k]==job[k]
        eval_stats=r['evaluation_stats']
    else:raise ValueError('Unknown algorithm: '+item['algorithm'])
    for k in ['dataset','method','distribution','attack']:assert r[k]==job[k]
    assert r['rounds']==70 and r['seed']==91001 and r['config']['celeba_evaluation_split']=='valid'
    assert all(math.isfinite(r['metrics'][k]) and 0<=r['metrics'][k]<=1 for k in METRICS)
    assert eval_stats['prediction_count']==19867
    assert all(math.isfinite(eval_stats[k]) and 0<=eval_stats[k]<=1 for k in ['positive_rate','majority_accuracy'])
    return dict(id=item['id'],algorithm=item['algorithm'],method=r['method'],distribution=r['distribution'],attack=r['attack'],
        seed=91001,candidate=job['tuning_candidate'],output=str(out),checkpoint_sha256=digest(out/'model.pt'),
        evaluation_stats=eval_stats,**r['metrics'])

def summarize(manifest):
    definitions=frozen_manifest(manifest);manifest_hash=digest(STAGE/'manifest.json');rows=[];invalid=[]
    for item in manifest['jobs']:
        try:row=checked(item)
        except Exception as exc:
            invalid.append(dict(id=item['id'],error=repr(exc)));continue
        if row:
            g=max(row['aeod'],row['aspd']);row['score']=row['accuracy']-.35*(.45*row['aeod']+.45*row['aspd']+.10*g)-.10*max(0,g-.06);rows.append(row)
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['algorithm'],row['candidate'])].append(row)
    candidates=[];all_candidates=[]
    planned=sorted({(i['algorithm'],definitions[i['id']]['tuning_candidate']) for i in manifest['jobs']})
    for algorithm,candidate in planned:
        group=groups[(algorithm,candidate)]
        assert len(group)==len({(x['distribution'],x['attack']) for x in group})
        entry=dict(algorithm=algorithm,candidate=candidate,n_seeds=1,seed=91001,
            completed_conditions=len(group),expected_conditions=4,complete=False,
            missing_conditions=[list(c) for c in sorted(CONDITIONS-{(x['distribution'],x['attack']) for x in group})])
        if {(x['distribution'],x['attack']) for x in group}==CONDITIONS:
            entry.update(complete=True,**{k:statistics.mean(x[k] for x in group) for k in [*METRICS,'score']})
            candidates.append(entry)
        all_candidates.append(entry)
    selections={}
    for algorithm in ['FedAA','LASA']:
        available=[c for c in candidates if c['algorithm']==algorithm]
        selection={'complete':len(available)==8,'completed_candidates':len(available),'expected_candidates':8}
        if len(available)==8:
            selection['accuracy_champion']=min(available,key=lambda c:(-c['accuracy'],c['candidate']))
            selection['score_champion']=min(available,key=lambda c:(-c['score'],c['candidate']))
            selection['pareto']=sorted([c for c in available if not any(
                other['accuracy']>=c['accuracy'] and other['aeod']<=c['aeod'] and other['aspd']<=c['aspd'] and
                (other['accuracy']>c['accuracy'] or other['aeod']<c['aeod'] or other['aspd']<c['aspd']) for other in available)],key=lambda c:c['candidate'])
        selections[algorithm]=selection
    failures=preserved_failures(manifest)
    complete=len(rows)==64 and len(candidates)==16 and not failures and not invalid
    assert digest(STAGE/'manifest.json')==manifest_hash,'manifest changed during summary'
    summary={'accepted':len(rows),'planned':64,'complete':complete,'records':rows,'completed_candidates':candidates,
        'all_candidates':all_candidates,'selections':selections,'invalid_records':invalid,'preserved_failures':failures,
        'manifest_sha256':manifest_hash,'summarizer_sha256':digest(__file__),
        'selection_rule':'Each candidate uses mean of all4conditions; accuracy/score descending, exact ties by candidate lexicographic order. Pareto maximizes ACC, minimizes AEOD/ASPD. No selection until all8candidates complete.',
        'note':'Valid-only n=1 bounded search; candidate means over4scenarios are not independent seed samples. Preserve every candidate and negative result.'}
    save(STAGE/'summary.json',summary)
    return summary

def run(manifest):
    import fcntl
    # OS lock releases on coordinator death, but partial attempts still require review.
    lock=(STAGE/'coordinator.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise RuntimeError('An existing coordinator owns this queue')
    try:return run_locked(manifest)
    finally:lock.close()

def run_locked(manifest):
    live={};failed=False
    try:
        return dispatch(manifest,live)
    except BaseException:
        record_failure({'coordinator_error':traceback.format_exc()})
        # Do not orphan or kill an in-flight authorized experiment on coordinator error.
        for proc,log,_item in live.values():
            proc.wait();log.close()
        raise

def dispatch(manifest,live):
    frozen_manifest(manifest);manifest_hash=digest(STAGE/'manifest.json')
    gate_path=Path(manifest['gate_acceptance'])
    assert gate_path.is_absolute(),'Preflight gate must use an explicit absolute path'
    assert digest(gate_path)==manifest['gate_acceptance_sha256'],'preflight gate changed'
    gate=json.loads(gate_path.read_text())
    assert gate['status']=='PASS' and gate['real_image_gates']==4 and gate['default_parameter_exact_regressions']==4,'All four real-image exact preflight gates must pass before dispatch'
    assert not preserved_failures(manifest),'Resolve preserved failure evidence before restart'
    scripts={str(BASE/'fedaa/run_fedaa_screen.py'),str(BASE/'lasa/worker.py')}
    jobpaths={str(Path(i['job'])) for i in manifest['jobs']}
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=entry.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError):continue
        assert not (scripts.intersection(args) and jobpaths.intersection(args)),('Existing worker requires recovery review',entry.parent.name)
    pending=[]
    for item in manifest['jobs']:
        assert digest(item['job'])==item['job_sha256']
        if checked(item) is None:
            assert not Path(item['output']).exists(),('Partial attempt requires preserved recovery review',item['id'])
            pending.append(item)
    failed=False
    (STAGE/'logs').mkdir(parents=True,exist_ok=True)
    while pending or live:
        assert digest(STAGE/'manifest.json')==manifest_hash,'manifest changed during execution'
        if preserved_failures(manifest):failed=True
        # Reap failures before filling a free slot, so a known failure cannot dispatch more work.
        for slot,(proc,log,item) in list(live.items()):
            if proc.poll() is None:continue
            log.close();del live[slot]
            if proc.returncode:
                failed=True;record_failure({'id':item['id'],'exit_code':proc.returncode})
            else:
                try: assert checked(item) is not None
                except Exception as exc:
                    failed=True;record_failure({'id':item['id'],'acceptance_error':repr(exc)})
        for slot in range(manifest['concurrency']):
            if failed or not pending:break
            if slot in live:continue
            if preserved_failures(manifest):failed=True;break
            item=pending.pop(0)
            assert digest(item['job'])==item['job_sha256']
            log=(STAGE/'logs'/f"{item['id']}.log").open('a')
            script=BASE/('fedaa/run_fedaa_screen.py' if item['algorithm']=='FedAA' else 'lasa/worker.py')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(slot%2),GUARDFED_CPU_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            try:proc=subprocess.Popen([sys.executable,str(script),'--repo',str(ROOT),'--job',item['job'],'--out',item['output']],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            except BaseException:log.close();raise
            live[slot]=(proc,log,item)
        save(STAGE/'queue_progress.json',{'pending':len(pending),'active':[dict(id=x[2]['id'],pid=x[0].pid) for x in live.values()],'failed':failed,'updated_unix':time.time()})
        if failed and not live:raise RuntimeError('Queue stopped after preserved failure; no retry')
        if pending or live:time.sleep(3)
    result=summarize(manifest);assert result['complete']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','summarize']);a=p.parse_args()
    m=json.loads((STAGE/'manifest.json').read_text())
    if a.action=='run':run(m)
    else:print(json.dumps({k:v for k,v in summarize(m).items() if k not in ['records','completed_candidates']}))
