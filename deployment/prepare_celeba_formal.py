#!/usr/bin/env python3
"""Prepare a reviewable deterministic CelebA queue; this never launches training."""
import argparse,json,subprocess,time,sys
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import reproduce_paper_tables as core
import run_revision_ablation as runner
LEGACY=Path('/workspace/GuardFed-image')
OUT=ROOT/'results/revision_20260923/celeba_formal_v1'
PILOT=LEGACY/'results/revision_20260923/celeba_full_validation_pilot_v1/manifest.json'
def validated_acceptance(path):
    d=Path(path)
    first=json.loads((d/'first_round_acceptance.json').read_text())
    comp=json.loads((d/'comparison.json').read_text())
    assert first['all_passed'] and comp['equivalence']['all_passed']
    if 'extension' in comp:assert comp['extension']['equivalence']['all_passed']
    manifest=json.loads((d/'concurrency2/manifest.json').read_text())
    assert manifest['source_hashes']==runner.source_hashes(),'Acceptance used another source version'
    return dict(directory=str(d),first_round_sha256=runner.digest(d/'first_round_acceptance.json'),
                comparison_sha256=runner.digest(d/'comparison.json'),source_hashes=manifest['source_hashes'])
def prepare(acceptance_dir):
    acceptance=validated_acceptance(acceptance_dir)
    pilot=json.loads(PILOT.read_text())
    reference=next(json.loads(Path(p).read_text()) for p in pilot['jobs'] if json.loads(Path(p).read_text())['method']=='GuardFed-AD2+')
    base=reference['config']
    assert base['rounds']==70 and base['learning_rate']==.001 and base['batch_size']==64
    assert base['celeba_train_limit']==base['celeba_eval_limit']==0 and base['celeba_evaluation_split']=='valid'
    frozen=runner.source_hashes();frozen['deployment/prepare_celeba_formal.py']=runner.digest(__file__)
    jobs=[]
    for seed in runner.SEEDS:
        for distribution,alpha in [('IID',5000.),('non-IID',5.)]:
            for attack in ['Benign','S-DFA','Sp-DFA']:
                for method in ['GuardFed-AD2+','FedAvg','FairFed','FLTrust']:
                    name=f'{distribution}_{method}_{attack}_seed{seed}'
                    cfg=asdict(core.ExperimentConfig(**dict(base,seed=seed,client_alpha=alpha,
                        celeba_evaluation_split='test',experiment_suite='revision_celeba_deterministic_formal_v1',experiment_tag=name)))
                    j=dict(id=name,dataset='celeba',distribution=distribution,method=method,attack=attack,config=cfg,
                      output=str(OUT/'runs'/name),source_hashes=frozen,evidence_stage='formal_supplement',
                      configuration_status='draft_pending_parent_review')
                    path=OUT/'jobs'/f'{name}.json';runner.write_json(path,j);jobs.append(str(path))
    assert len(jobs)==240
    draft=dict(protocol='revision_celeba_deterministic_formal_v1',jobs=jobs,output=str(OUT),new_run_count=240,reused_full_count=0,
       source_hashes=frozen,source_commit_at_prepare=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       seeds=runner.SEEDS,protocol_status='draft_pending_parent_review',formal_launch_authorized=False,
       numerical_acceptance=acceptance,legacy_learning_pilot=str(PILOT),
       legacy_canary_evidence=str(LEGACY/'results/revision_20260923/celeba_formal_v1/canary/validation.json'),
       legacy_evidence_limitation='Exploratory earlier numerical execution; retained, not relabelled as new deterministic evidence.',
       test_results_used_for_protocol=False,statistics='Common terminal70 checkpoint,10 independent seeds,mean/sample std; no test selection.',
       primary_reporting='Existing method reporting: calibrated GuardFed; raw FedAvg/FairFed/FLTrust.',
       additional_diagnostic='Any raw+cal evaluation uses same final checkpoint, separate from primary method results.')
    runner.write_json(OUT/'manifest.draft.json',draft)

    canaries=[]
    conditions=[('FairFed','S-DFA','IID',5000.),('FLTrust','Sp-DFA','IID',5000.),
      ('GuardFed-AD2+','S-DFA','non-IID',5.),('FairFed','Sp-DFA','non-IID',5.),
      ('FLTrust','S-DFA','non-IID',5.),('GuardFed-AD2+','Sp-DFA','IID',5000.)]
    croot=OUT/'canary'
    for method,attack,distribution,alpha in conditions:
        name=f'{distribution}_{method}_{attack}_seed123'
        cfg=asdict(core.ExperimentConfig(**dict(base,seed=123,client_alpha=alpha,rounds=2,
          celeba_train_limit=8192,celeba_eval_limit=2048,celeba_evaluation_split='valid',
          experiment_suite='revision_celeba_deterministic_canary_v1',experiment_tag=name)))
        j=dict(id=name,dataset='celeba',distribution=distribution,method=method,attack=attack,config=cfg,
          output=str(croot/'runs'/name),source_hashes=frozen,evidence_stage='strict_two_round_validation_canary')
        path=croot/'jobs'/f'{name}.json';runner.write_json(path,j);canaries.append(str(path))
    runner.write_json(croot/'manifest.json',dict(output=str(croot),jobs=canaries,new_run_count=6,reused_full_count=0,
      source_hashes=frozen,evidence_stage='strict_two_round_validation_canary'))
    print(json.dumps(dict(draft=str(OUT/'manifest.draft.json'),jobs=240,canaries=6,training_started=False)))
def freeze(learning_manifest,review_note):
    assert review_note,'Explicit parent review note required; no automatic freeze'
    draft=json.loads((OUT/'manifest.draft.json').read_text())
    acceptance=validated_acceptance(draft['numerical_acceptance']['directory'])
    assert acceptance==draft['numerical_acceptance']
    for name,value in draft['source_hashes'].items():assert runner.digest(ROOT/name)==value,('source changed',name)
    canary=json.loads((OUT/'canary/validation.json').read_text())
    assert canary['all_passed'] and canary['completed']==6
    assert canary['pipeline_source_hashes']==runner.source_hashes()
    learning=json.loads(Path(learning_manifest).read_text());evidence=[]
    assert len(learning['jobs'])==2
    for p in learning['jobs']:
        j=json.loads(Path(p).read_text());r=runner.checked_result(j)
        assert r and r['rounds']==10 and j['config']['celeba_evaluation_split']=='valid'
        assert j['config']['celeba_train_limit']==j['config']['celeba_eval_limit']==0
        assert j['config']['learning_rate']==.001 and j['config']['batch_size']==64
        assert j['source_hashes']==runner.source_hashes()
        evidence.append(dict(method=j['method'],metrics=r['metrics'],checkpoint_sha256=r['revision_job']['checkpoint_sha256']))
    assert {e['method'] for e in evidence}=={'GuardFed-AD2+','FedAvg'}
    for p in draft['jobs']:
        j=json.loads(Path(p).read_text());assert not (Path(j['output'])/'result.json').exists(),'No retrofit after test results'
        j['configuration_status']='frozen_after_parent_review';runner.write_json(p,j)
    draft.update(protocol_status='frozen_after_parent_review',formal_launch_authorized=True,
        deterministic_learning10_manifest=str(learning_manifest),deterministic_learning10_terminal=evidence,
        review_note=review_note,frozen_unix=time.time())
    runner.write_json(OUT/'manifest.json',draft)
    print(json.dumps(dict(manifest=str(OUT/'manifest.json'),jobs=240,training_started=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--acceptance-dir');p.add_argument('--freeze',action='store_true')
    p.add_argument('--learning-manifest');p.add_argument('--review-note');a=p.parse_args()
    if a.freeze:freeze(a.learning_manifest,a.review_note)
    else:
        assert a.acceptance_dir,'--acceptance-dir is required'
        prepare(a.acceptance_dir)
