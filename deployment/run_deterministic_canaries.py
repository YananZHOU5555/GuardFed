import json,os,subprocess,sys,time
from pathlib import Path
R=Path('/workspace/GuardFed-image-deterministic')
A=R/'results/revision_20260923/celeba_deterministic_acceptance_v2'
O=R/'results/revision_20260923/celeba_formal_v1'
sys.path.insert(0,str(R/'scripts'))
import run_revision_ablation as runner
while not (A/'comparison.json').exists():
    pid=int((A/'controller.pid').read_text())
    if not Path(f'/proc/{pid}').exists():raise RuntimeError('Numerical acceptance failed; formal draft/canaries not prepared')
    time.sleep(10)
subprocess.run([sys.executable,str(R/'deployment/prepare_celeba_formal.py'),'--acceptance-dir',str(A)],cwd=R,check=True)
C=O/'canary'
with (C/'runner.log').open('w') as log:
    subprocess.run([sys.executable,str(R/'scripts/run_revision_ablation.py'),'run','--manifest',str(C/'manifest.json'),'--concurrency','2'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
import reproduce_paper_tables as core
manifest=json.loads((C/'manifest.json').read_text());checks=[]
for p in manifest['jobs']:
    j=json.loads(Path(p).read_text());r=runner.checked_result(j)
    assert r and r['rounds']==2 and len(r['round_summaries'])==2
    cfg=j['config'];assert cfg['celeba_evaluation_split']=='valid' and cfg['celeba_train_limit']==8192 and cfg['celeba_eval_limit']==2048
    num=r['data_contract']['image_data_contract']['numerical_execution']
    assert num['deterministic_algorithms'] and num['cudnn_deterministic'] and not num['cudnn_benchmark']
    audits=r['attack_audit'];bad=core.validate_attack_audit(j['attack'],list(range(4)),audits);assert not bad,bad
    mal=[a for a in audits if a['is_malicious']];assert len(mal)==4
    if j['attack']=='S-DFA':assert all(set(a['attack_types'])=={'fflip','foe'} for a in mal)
    else:assert sum(a['attack_types']==['fflip'] for a in mal)==2 and sum(a['attack_types']==['foe'] for a in mal)==2
    for a in mal:
        if 'fflip' in a['attack_types']:assert a['fflip_changed']>0 and a['label_changed_count']==0
        if 'foe' in a['attack_types']:assert a['foe_mode']=='fedsa' and a['foe_post_update_norm']>0
    checks.append(dict(job=j['id'],all_passed=True,metrics=r['metrics'],checkpoint_sha256=r['revision_job']['checkpoint_sha256'],numerical_execution=num,malicious_attack_audit=mal))
runner.write_json(C/'validation.json',dict(all_passed=True,completed=len(checks),failed=0,pipeline_source_hashes=runner.source_hashes(),checks=checks,
    limitation='Strict pipeline/attack wiring only; 2 rounds do not establish learning or robustness.',test_used=False))
with (C/'checkpoint_analysis.log').open('w') as log:
    subprocess.run([sys.executable,str(R/'deployment/celeba_checks/analyze_validation_pilot.py'),'--manifest',str(C/'manifest.json'),'--output',str(C/'checkpoint_analysis')],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Six strict attack canaries validated with raw/cal same-checkpoint analysis; formal remains draft.',flush=True)
