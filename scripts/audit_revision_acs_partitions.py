#!/usr/bin/env python3
"""Offline identity audit: no optimizer, model, or test-metric selection."""
import json
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import reproduce_paper_tables as core
from run_revision_ablation import SEEDS, write_json

def audit(output):
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
    jobs=[json.loads(Path(p).read_text()) for p in manifest['jobs']]
    rows=[]
    for seed in SEEDS:
        for distribution,alpha in [('IID',5000.),('non-IID',5.)]:
            group=[j for j in jobs if j['config']['seed']==seed and j['distribution']==distribution]
            assert len(group)==6
            configs=[{k:v for k,v in j['config'].items() if k!='experiment_tag'} for j in group]
            assert all(c==configs[0] for c in configs), 'Method/attack configs not paired'
            config=core.ExperimentConfig(**configs[0]);config.device='cpu'
            captured={};original=core.create_client_data_dict
            def capture(client_df,*args,**kwargs):
                captured['frame']=client_df.copy();return original(client_df,*args,**kwargs)
            with patch.object(core,'create_client_data_dict',side_effect=capture):
                b=core.load_bundle('acs_income',alpha,config,torch.device('cpu'))
            loader=b['loader'];root,_=core.sample_server_dataframe(loader.train_df,'income','sex',config)
            pool=loader.train_df.drop(root.index)
            frame=captured['frame'];frame['_row_id']=loader.train_original_row_ids[pool.index]
            with_ids=original(frame,b['feature_cols']+['_row_id'],'income','sex',config.num_clients,alpha,torch.device('cpu'),seed)
            ids=[];hashes=[]
            for cid, c in with_ids.items():
                assert torch.equal(c['X'][:,:-1],b['clients'][cid]['X'])
                assert torch.equal(c['y'],b['clients'][cid]['y'])
                seq=c['X'][:,-1].numpy().astype('<i8');ids.extend(seq.tolist())
                hashes.append(hashlib.sha256(seq.tobytes()).hexdigest())
            roots=set(loader.train_original_row_ids[root.index]);test=set(loader.test_original_row_ids)
            assert len(ids)==len(set(ids)), 'Duplicate client row'
            assert not set(ids)&roots and not set(ids)&test and not roots&test
            assert set(ids)|roots==set(loader.train_original_row_ids)
            assert loader.scaler.n_samples_seen_==len(loader.train_df)
            row=dict(seed=seed,distribution=distribution,alpha=alpha,paired_method_attack_jobs=len(group),
                client_rows=len(ids),root_rows=len(roots),test_rows=len(test),all_pairwise_overlap=0,
                client_row_ids_sha256=hashes,train_row_ids_sha256=loader.preprocessing_audit['train_row_ids_sha256'],
                test_row_ids_sha256=loader.preprocessing_audit['test_row_ids_sha256'],
                root_row_ids_sha256=loader.preprocessing_audit['partition']['root_original_row_ids_sha256'])
            rows.append(row);print(seed,distribution,'PASS',flush=True)
    for seed in SEEDS:
        pair=[r for r in rows if r['seed']==seed]
        for key in ['train_row_ids_sha256','test_row_ids_sha256','root_row_ids_sha256']:
            assert pair[0][key]==pair[1][key]
    assert len({r['test_row_ids_sha256'] for r in rows})==len(SEEDS)
    write_json(output/'partition_identity_audit.json',dict(status='passed',conditions=rows,
        evidence_stage='offline_pipeline_check_not_scientific_evidence',
        checks='Original CSV row identities; root/client/test pairwise disjoint; each client-pool row assigned exactly once; all six method/attack jobs paired; same seed shares split/root across alphas; different seeds differ; train-only scaler',
        no_training=True,no_test_metrics=True))

if __name__=='__main__':
    torch.set_num_threads(1);audit(sys.argv[1])
