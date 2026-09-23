#!/usr/bin/env python3
"""Prepare the isolated E5 queue. This command never launches training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from run_revision_ablation import ROOT, SEEDS, digest, write_json
from reproduce_paper_tables import ExperimentConfig

RAW = 'data/acs_income/2018/1-Year/psam_p06.csv'
RAW_SHA = 'dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0'

def prepare(output, device='cpu'):
    output=Path(output).resolve()
    assert not (output/'manifest.json').exists(), 'Refuse to overwrite an existing frozen queue'
    assert digest(ROOT/RAW)==RAW_SHA, 'ACS source checksum mismatch'
    frozen={name:digest(ROOT/name) for name in [RAW,'src/data_loader.py',
        'scripts/reproduce_paper_tables.py','scripts/run_revision_ablation.py','scripts/prepare_revision_acs.py']}
    base=asdict(ExperimentConfig(device=device,server_ratio=.1,synthetic_method='none',
        fflip_mode='all_unprivileged',sdfa_foe_mode='fedsa',spdfa_foe_mode='fedsa',fedsa_gain=4.5,
        fedsa_norm_ratio=3.,full_round_diagnostics=True,experiment_suite='revision_acs_income_v1'))
    jobs=[]
    for seed in SEEDS:
        for dist in ['IID','non-IID']:
            for method in ['GuardFed-AD2+','FedAvg','FLTrust']:
                for attack in ['Benign','S-DFA']:
                    job_id=f'{dist}_{method}_{attack}_seed{seed}'
                    job=dict(id=job_id,dataset='acs_income',distribution=dist,method=method,attack=attack,
                        config=dict(base,seed=seed,experiment_tag=job_id),source_hashes=frozen,
                        output=str(output/'runs'/job_id),evidence_stage='formal_supplement')
                    path=output/'jobs'/f'{job_id}.json';write_json(path,job);jobs.append(str(path))
    manifest=dict(protocol='revision_acs_income_v1',created_unix=time.time(),jobs=jobs,output=str(output),
        new_run_count=len(jobs),reused_full_count=0,seeds=SEEDS,source_hashes=frozen,
        launch_status='prepared_only_pending_protocol_review',old_experiments_rerun=False,
        preprocessing='acs_sex_train_only_v1',statistics='final round 70; mean/sample std only all 10 prespecified seeds; no checkpoint selection')
    write_json(output/'manifest.json',manifest)
    pilots=[]
    for method,attack,dist,rounds in [('FedAvg','Benign','IID',5),('FLTrust','Benign','IID',5),
        ('GuardFed-AD2+','Benign','IID',5),('GuardFed-AD2+','S-DFA','non-IID',2)]:
        job_id=f'{dist}_{method}_{attack}_seed123_rounds{rounds}'
        job=dict(id=job_id,dataset='acs_income',distribution=dist,method=method,attack=attack,
            config=dict(base,device='cpu',seed=123,rounds=rounds,experiment_suite='revision_acs_pilot',experiment_tag=job_id),
            source_hashes=frozen,output=str(output/'pilot_runs'/job_id),evidence_stage='pipeline_and_learning_pilot_not_formal')
        path=output/'pilot_jobs'/f'{job_id}.json';write_json(path,job);pilots.append(str(path))
    write_json(output/'pilot_manifest.json',dict(manifest,jobs=pilots,new_run_count=len(pilots),
        protocol='revision_acs_pilot',launch_status='cpu_only_max_four',output=str(output)))
    print(json.dumps(dict(manifest=str(output/'manifest.json'),formal_jobs=len(jobs),pilot_jobs=len(pilots))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();prepare(a.output,a.device)
