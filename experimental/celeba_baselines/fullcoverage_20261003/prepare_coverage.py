"""Prepare immutable selected-recipe jobs; run on the server, never train."""
import copy
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path('/workspace/GuardFed-celeba-expanded')
BASE=ROOT/'deployment/baseline_adapters_20260928/fullcoverage_20261003'
OLD=BASE.parent/'screen_20260928'
STAGE=ROOT/'results/revision_20261003/celeba_baseline_fullcoverage_v1'
SCREEN=ROOT/'results/revision_20260928/celeba_baseline_screen_v1'
ATTACKS=['Benign','F Flip','FedSA','S-DFA','Sp-DFA']
SELECTED={'FedAA':'policy0.001_keep16_local0.001','LASA':'LASA_s0.3_l2_lr0.001'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    data=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if path.exists():assert path.read_text()==data,('Existing frozen file differs',str(path))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data)

def main():
    old=json.loads((SCREEN/'manifest.json').read_text())
    summary=json.loads((SCREEN/'summary.json').read_text())
    assert summary['complete'] and summary['accepted']==64 and not summary['invalid_records'] and not summary['preserved_failures']
    assert summary['manifest_sha256']==sha(SCREEN/'manifest.json')
    assert all(summary['selections'][a]['score_champion']['candidate']==c for a,c in SELECTED.items())
    assert STAGE.is_dir() and not (STAGE/'manifest.json').exists()
    sources={}
    for item in old['jobs']:
        j=json.loads(Path(item['job']).read_text())
        for name,h in j['source_hashes'].items():
            assert name not in sources or sources[name]==h
            sources[name]=h
        if item['algorithm']=='FedAA':
            for key,relative in [('official_wrapper','deployment/baseline_adapters_20260928/fedaa/fedaa_official_adapter.py'),('official_ddpg','deployment/baseline_adapters_20260928/fedaa/official/DDPG/DDPG.py')]:
                h=j['adapter_hashes'][key]
                assert relative not in sources or sources[relative]==h
                sources[relative]=h
    for path in BASE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['run_screen.py','fedaa/run_fedaa_screen.py','fedaa/accept_result.py','fedaa/fedaa_round_adapter.py','lasa/worker.py','lasa/adapter.py','lasa/protocol.json','lasa/prepare_jobs.py']:
        p=OLD/name;sources[p.relative_to(ROOT).as_posix()]=sha(p)
    sources.update(old['source_hashes'])
    reused=[];jobs=[];gates=[];references=[]
    for distribution,attack,seed,algorithm in itertools.product(['IID','non-IID'],ATTACKS,range(91001,91011),['FedAA','LASA']):
        selected=SELECTED[algorithm]
        prototype=next(i for i in old['jobs'] if i['algorithm']==algorithm and i['tuning_candidate']==selected and i['distribution']==distribution and i['attack']==(attack if attack in ['Benign','S-DFA'] else 'Benign'))
        if seed==91001 and attack in ['Benign','S-DFA']:
            reused.append(copy.deepcopy(prototype));continue
        j=copy.deepcopy(json.loads(Path(prototype['job']).read_text()))
        j.update(attack=attack,evidence_stage='multi_seed_validation_coverage')
        cfg=j['config'];cfg.update(seed=seed,client_alpha={'IID':5000.,'non-IID':5.}[distribution],rounds=70)
        if algorithm=='FedAA':
            j['id']=f'FedAA-DDPG_{selected}_{distribution}_{attack}_seed{seed}'
            j['policy_seed']=seed
            cfg.update(experiment_suite='celeba_baseline_fullcoverage_20261003_v1',experiment_tag=j['id'])
            j['adapter_hashes']['screen_runner']=sha(BASE/'fedaa/run_fedaa_screen.py')
            j['adapter_hashes']['round_adapter']=sha(BASE/'fedaa/fedaa_round_adapter.py')
        else:
            j.update(phase='fullcoverage')
            j['id']=f'{selected}_{distribution}_{attack}_seed{seed}_fullcoverage'
            cfg.update(experiment_suite='celeba_baseline_fullcoverage_20261003_v1_fullcoverage',experiment_tag=j['id'])
            j['adapter_source_hashes']={name:sha(BASE/'lasa'/name) for name in ['adapter.py','worker.py','protocol.json','prepare_jobs.py']}
            j['reporting']='raw fixed-final-round multi-seed validation coverage; no additional calibration or test labels'
        j['derivation_fullcoverage']=dict(original_job=prototype['job'],original_job_sha256=prototype['job_sha256'],selected_candidate=selected)
        path=STAGE/'jobs'/f"{j['id']}.json";write(path,j)
        jobs.append(dict(id=j['id'],algorithm=algorithm,job=str(path),job_sha256=sha(path),output=str(STAGE/'runs'/j['id']),**{k:j[k] for k in ['method','distribution','attack','tuning_candidate']}))
    for algorithm,attack in itertools.product(['FedAA','LASA'],ATTACKS):
        selected=SELECTED[algorithm]
        prototype=next(i for i in old['jobs'] if i['algorithm']==algorithm and i['tuning_candidate']==selected and i['distribution']=='non-IID' and i['attack']==(attack if attack in ['Benign','S-DFA'] else 'Benign'))
        j=copy.deepcopy(json.loads(Path(prototype['job']).read_text()));j['attack']=attack;j['config']['rounds']=3
        if algorithm=='FedAA':
            j.update(id=f'FedAA-DDPG_{selected}_non-IID_{attack}_seed91001_canary',evidence_stage='pipeline_canary_only')
            j['config'].update(experiment_suite='celeba_baseline_fullcoverage_20261003_v1',experiment_tag=j['id'])
            j['adapter_hashes']['screen_runner']=sha(BASE/'fedaa/run_fedaa_screen.py')
            j['adapter_hashes']['round_adapter']=sha(BASE/'fedaa/fedaa_round_adapter.py')
        else:
            j.update(id=f'{selected}_non-IID_{attack}_seed91001_preflight',phase='preflight',evidence_stage='pipeline_canary_only')
            j['config'].update(experiment_suite='celeba_baseline_fullcoverage_20261003_v1_preflight',experiment_tag=j['id'])
            j['adapter_source_hashes']={name:sha(BASE/'lasa'/name) for name in ['adapter.py','worker.py','protocol.json','prepare_jobs.py']}
        path=STAGE/'preflight/jobs'/f"{j['id']}.json";write(path,j)
        gates.append(dict(id=j['id'],algorithm=algorithm,job=str(path),job_sha256=sha(path),output=str(STAGE/'preflight/runs'/j['id']),regression_source=prototype if attack in ['Benign','S-DFA'] else None))
    for attack in ['Benign','S-DFA']:
        prototype=next(i for i in reused if i['algorithm']=='LASA' and i['distribution']=='non-IID' and i['attack']==attack)
        j=copy.deepcopy(json.loads(Path(prototype['job']).read_text()))
        j['config']['rounds']=3;j['pipeline_reference_only']=True
        path=STAGE/'preflight/reference_jobs'/f"{j['id']}.json";write(path,j)
        references.append(dict(id=j['id'],attack=attack,job=str(path),job_sha256=sha(path),output=str(STAGE/'preflight/reference_runs'/j['id'])))
    assert len(jobs)==192 and len(reused)==8 and len(gates)==10
    assert len({i['id'] for i in jobs+reused})==200
    for algorithm in ['FedAA','LASA']:
        for f in ['fedaa_round_adapter.py'] if algorithm=='FedAA' else ['adapter.py']:
            folder='fedaa' if algorithm=='FedAA' else 'lasa'
            assert sha(BASE/folder/f)==sha(OLD/folder/f),'Numerical adapter changed'
    m=dict(stage='celeba_baseline_fullcoverage_v1',status='PREPARED_PENDING_GATES',authorization='User continue running2026-10-03',planned_new=192,planned_total=200,concurrency=8,rounds=70,seeds=list(range(91001,91011)),distributions={'IID':5000.,'non-IID':5.},attacks=ATTACKS,evaluation_split='valid',selected_recipes=SELECTED,source_hashes=sources,protocol_sha256=sha(STAGE/'PROTOCOL.md'),jobs=jobs,reused_jobs=reused,preflight_jobs=gates,reference_gates=references,screen_manifest_sha256=sha(SCREEN/'manifest.json'),screen_summary_sha256=sha(SCREEN/'summary.json'),gate_acceptance=str(STAGE/'preflight/acceptance.json'),gate_acceptance_sha256=None,reporting='10seed mean/sampleSD;9seed excluding selection;6seed matching StageA;validation not untouched test')
    write(STAGE/'manifest.json',m)
    print(json.dumps({'planned_new':192,'reused':8,'gates':10,'manifest_sha256':sha(STAGE/'manifest.json')}))

if __name__=='__main__':main()
