"""Independent portable acceptance of the two CANARY backups; never trains."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def check(root, source_only=False):
    freeze=json.loads((root/'FREEZE.json').read_text())
    assert freeze['status']=='FROZEN' and freeze['scope']=='two_3round_real_image_CANARY_only'
    for name,value in freeze['local_hashes'].items():assert sha(root/name)==value, name
    scope=json.loads((root/'SCOPE.json').read_text())
    assert not scope['formal_screen_authorized'] and len(scope['jobs'])==2
    sealed=json.loads((root/'sealed_group_b/protocol.json').read_text())
    assert sealed['status']=='PREPARED_NOT_FROZEN'
    if source_only:return dict(status='SOURCE_PASS',frozen_files=len(freeze['local_hashes']),results_checked=0)
    import torch
    reports=[]
    conditions=set()
    artifacts={'model.pt','state.json','diagnostics.json','result.json','provenance.json','job.json',
               'round_001_state.json','round_002_state.json','round_003_state.json'}
    for name in scope['jobs']:
        job=json.loads((root/name).read_text());out=root/'runs'/job['id']
        assert not list(out.glob('failure*.json'))
        conditions.add((job['distribution'],job['attack']))
        accept=json.loads((out/'acceptance.json').read_text())
        assert accept['status']=='PASS' and not accept['formal_table_eligible']
        assert set(accept['artifact_hashes'])==artifacts
        for filename,value in accept['artifact_hashes'].items():assert sha(out/filename)==value, filename
        r=json.loads((out/'result.json').read_text());p=json.loads((out/'provenance.json').read_text())
        assert r['revision_job']==job and r['config']==job['config'] and r['provenance']==p
        assert r['evidence_stage']=='CANARY_REAL_IMAGE_ONLY' and r['rounds']==3 and r['seed']==91001
        assert p['job_sha256']==sha(root/name) and p['freeze_sha256']==sha(root/'FREEZE.json')
        assert p['source_hashes']==scope['source_hashes'] and p['local_hashes']==freeze['local_hashes']
        assert p['threads']==8 and p['device']=='cpu' and p['real_data'] and not p['formal_table_eligible']
        expected=dict(scope['base_config'],rounds=3,seed=91001,device='cpu',learning_rate=.001,
            client_alpha={'IID':5000.,'non-IID':5.}[job['distribution']],experiment_suite=scope['version'],experiment_tag=job['id'])
        assert job['config']==expected and job['adapter']==dict(warmup_rounds=1,control_width=3.)
        for key in ['method','dataset','distribution','attack']:assert r[key]==job[key],key
        for field in ['trajectory_metrics','round_summaries']:assert [row['round'] for row in r[field]]==[1,2,3]
        assert r['metrics']==r['trajectory_metrics'][-1]['metrics']
        assert all(math.isfinite(v) and 0<=v<=1 for row in r['trajectory_metrics'] for v in row['metrics'].values())
        c=r['data_contract']['image_data_contract']
        assert (c['evaluation_split'],c['actual_train_rows'],c['actual_evaluation_rows'],r['evaluation_stats']['prediction_count'])==('valid',162770,19867,19867)
        assert c['train_eval_disjoint'] and c['root_client_disjoint']
        diag=json.loads((out/'diagnostics.json').read_text())
        assert [row['aggregate']['stage'] for row in diag]==['per_round_gmm','fit_control_limit','monitor']
        for i,row in enumerate(diag,1):
            assert row['aggregate']==r['round_summaries'][i-1]['aggregate']
            state=json.loads((out/f'round_{i:03d}_state.json').read_text())
            assert state['round_index']==i and state['warmup_rounds']==1 and state['control_width']==3
            assert state['client_ids']==list(range(20)) and len(state['history'])==20
            assert all(len(h)==i and all(math.isfinite(z) for z in h) for h in state['history'])
            assert state['ucl']==row['aggregate']['ucl']
        model=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
        assert model and all(torch.isfinite(v).all() for v in model.values())
        reports.append(dict(id=job['id'],metrics=r['metrics'],elapsed_sec=accept['elapsed_sec'],
                            cpu_user_seconds=accept['cpu_user_seconds'],cpu_system_seconds=accept['cpu_system_seconds'],
                            model_sha256=accept['artifact_hashes']['model.pt'],states_verified=3))
    assert conditions=={('IID','Benign'),('non-IID','S-DFA')}
    return dict(status='PASS',evidence_stage='CANARY_REAL_IMAGE_ONLY',formal_table_eligible=False,
                actual_results_checked=2,results=reports,limitations=['CPU, not GPU equivalence','Tg1 stage coverage, not formal candidate evidence'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--source-only',action='store_true');a=p.parse_args()
    r=check(a.root,a.source_only)
    print(json.dumps(r,indent=2))
    if not a.source_only:(a.root/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(r,indent=2),encoding='utf8')
