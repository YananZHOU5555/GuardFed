#!/usr/bin/env python3
"""Bounded validation-only CelebA hyperparameter screen; no test jobs."""
import json,sys,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_revision_ablation as run
OLD=Path('/workspace/GuardFed-image-deterministic/results/revision_20260923/celeba_formal_v1')
OUT=ROOT/'results/revision_20260924/celeba_tuning_screen_v1'
base=json.loads((OLD/'jobs/non-IID_GuardFed-AD2+_Benign_seed123.json').read_text())['config']
frozen=run.source_hashes();frozen['deployment/prepare_celeba_tuning.py']=run.digest(__file__)
# Training core and deterministic execution are unchanged; only summaries group candidates.
assert frozen['scripts/reproduce_paper_tables.py']=='cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed'
jobs=[]
for attack in ['Benign','S-DFA']:
 for lr in [.0005,.001,.002,.003]:
  for method,drop in [('GuardFed-AD2+',0.),('GuardFed-AD2+',.005),('FedAvg',.03),('FairFed',.03),('FLTrust',.03)]:
   candidate=f'lr{lr:g}'+(f'_drop{drop:g}' if method=='GuardFed-AD2+' else '')
   ident=f'{method}_{candidate}_{attack}_seed91001'
   cfg=dict(base,learning_rate=lr,ad2_calibration_max_acc_drop=drop,seed=91001,celeba_evaluation_split='valid',experiment_suite='celeba_validation_tuning_v1',experiment_tag=ident)
   job=dict(id=ident,dataset='celeba',distribution='non-IID',method=method,attack=attack,config=cfg,output=str(OUT/'runs'/ident),source_hashes=frozen,evidence_stage='post_initial_test_validation_exploration',tuning_candidate=candidate)
   path=OUT/'jobs'/f'{ident}.json';assert not path.exists();run.write_json(path,job);jobs.append(str(path))
assert len(jobs)==40
manifest=dict(protocol='celeba_validation_tuning_v1',jobs=jobs,output=str(OUT),new_run_count=40,reused_full_count=0,tuning_search=True,source_hashes=frozen,
 stage='exploratory_single_development_seed',development_seed=91001,evaluation_split='valid',prior_test_results_seen=True,test_used_for_candidate_ranking=False,
 candidate_selection='Rank mean over Benign/S-DFA of ACC-.35*(.45*AEOD+.45*ASPD+.10*max(AEOD,ASPD))-.10*max(0,max(AEOD,ASPD)-.06); retain Pareto frontier and accuracy champion separately. No guaranteed dominance and no metric-wise seed splicing.',
 baseline_tuning='Same four learning rates for all four methods. GuardFed additionally screens selector/calibration drop 0/.005; unequal additional method-specific dimension disclosed.',
 limitations=['Single seed screening is not formal multiseed evidence.','Learning rate changes clients, root and root-dependent attack jointly.','Validation targets future confirmation; previously reported test has already been observed.','No test jobs or COMPAS selection-table changes authorized by this script.'])
run.write_json(OUT/'manifest.json',manifest)
canary=OUT/'canary';cjobs=[]
for jp in jobs[:2]:
 j=json.loads(Path(jp).read_text());j['output']=str(canary/'runs'/j['id']);j['config'].update(rounds=2,celeba_train_limit=8192,celeba_eval_limit=2048,experiment_suite='celeba_tuning_canary_v1');j['evidence_stage']='pipeline_canary_not_formal';p=canary/'jobs'/f"{j['id']}.json";run.write_json(p,j);cjobs.append(str(p))
run.write_json(canary/'manifest.json',dict(manifest,jobs=cjobs,output=str(canary),new_run_count=2,stage='pipeline_canary'))
print(json.dumps(dict(jobs=len(jobs),guardfed=16,baselines=24,manifest=str(OUT/'manifest.json'),canary=str(canary/'manifest.json'))))
