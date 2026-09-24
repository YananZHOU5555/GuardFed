#!/usr/bin/env python3
"""Stage 2a: bounded, validation-only expansion; historical evidence immutable."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_revision_ablation as run
OLD=Path('/workspace/GuardFed-celeba-tuning/results/revision_20260924/celeba_tuning_screen_v1')
OUT=ROOT/'results/revision_20260924/celeba_expanded_screen_v2'
old=json.loads((OLD/'manifest.json').read_text())
base=json.loads(Path(old['jobs'][0]).read_text())['config']
hashes=run.source_hashes()
hashes['deployment/prepare_celeba_expanded.py']=run.digest(__file__)
assert hashes['scripts/reproduce_paper_tables.py']=='cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed'
existing=set()
for jp in old['jobs']:
 j=json.loads(Path(jp).read_text())
 assert run.checked_result(j) is not None
 existing.add((j['method'],j['attack'],j['config']['learning_rate'],j['config']['ad2_calibration_max_acc_drop']))
recipes=[]
for method in ['Median','FairGuard','FLTrust+FairGuard']:
 for lr in [.00025,.000375,.0005,.00075,.001,.002,.003]:recipes.append((method,lr,.03))
for method in ['FedAvg','FairFed','FLTrust']:
 for lr in [.00025,.000375,.00075]:recipes.append((method,lr,.03))
for lr in [.00025,.000375,.0005,.00075,.001]:
 for drop in [0.,.0025,.005]:recipes.append(('GuardFed-AD2+',lr,drop))
labels={'FairGuard':'FairGuard-root adaptation (project implementation)','FLTrust+FairGuard':'FairGuard-root then FLTrust (project hybrid)','FairFed':'FairFed-style root-fairness weighting (project implementation)'}
jobs=[]
for attack in ['Benign','S-DFA']:
 for method,lr,drop in recipes:
  if (method,attack,lr,drop) in existing:continue
  cand=f'lr{lr:g}'+(f'_drop{drop:g}' if method=='GuardFed-AD2+' else '')
  ident=f'{method}_{cand}_{attack}_seed91001'
  cfg=dict(base,learning_rate=lr,ad2_calibration_max_acc_drop=drop,seed=91001,experiment_suite='celeba_expanded_validation_v2',experiment_tag=ident)
  j=dict(id=ident,dataset='celeba',distribution='non-IID',method=method,attack=attack,config=cfg,output=str(OUT/'runs'/ident),source_hashes=hashes,evidence_stage='post_initial_test_validation_exploration',tuning_candidate=cand,reporting_label=labels.get(method,method))
  p=OUT/'jobs'/f'{ident}.json';assert not p.exists();run.write_json(p,j);jobs.append(str(p))
assert len(jobs)==82
manifest=dict(old,protocol='celeba_expanded_validation_v2',jobs=jobs,output=str(OUT),new_run_count=len(jobs),source_hashes=hashes,previous_verified_manifest=str(OLD/'manifest.json'),
 reporting_labels=labels,original_methods_pending_faithful_adapters=['LoGoFair','FedAA','Fed-NGA'],
 selection_note='Combine with immutable prior40 validation jobs by method/recipe/condition. Same predeclared score; retain accuracy champions, all 3metric Pareto candidates and all negative results. Shared7LR grid for baselines; GuardFed extra selector-drop search disclosed.',
 next_stage='Complete baseline fidelity work; do not equate adapted implementations with original papers. After all eligible baselines tuned, freeze selected configs and multi-seed IID/nonIID x Benign/S-DFA/Sp-DFA confirmation. No automatic test queue from this manifest.')
run.write_json(OUT/'manifest.json',manifest)
canary=OUT/'canary';paths=[]
for method in ['Median','FairGuard','FLTrust+FairGuard']:
 j=json.loads(Path(next(p for p in jobs if json.loads(Path(p).read_text())['method']==method)).read_text())
 j['output']=str(canary/'runs'/j['id']);j['config'].update(rounds=2,celeba_train_limit=8192,celeba_eval_limit=2048,experiment_suite='expanded_canary_v2');j['evidence_stage']='pipeline_canary_only'
 p=canary/'jobs'/f"{j['id']}.json";run.write_json(p,j);paths.append(str(p))
run.write_json(canary/'manifest.json',dict(manifest,jobs=paths,output=str(canary),new_run_count=3))
print(json.dumps({'manifest':str(OUT/'manifest.json'),'new_jobs':82,'reuse_without_retraining':40,'canary':str(canary/'manifest.json')}))

