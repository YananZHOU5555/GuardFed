"""All32 candidate terminals under one frozen rule; no training or image inference."""
import argparse
import json
import os
from pathlib import Path
import sys
from driver import HERE,approve,functions,digest,read,require,validate_entry

def score(metrics):
    maximum=max(metrics['aeod'],metrics['aspd'])
    return metrics['accuracy']-.35*(.45*metrics['aeod']+.45*metrics['aspd']+.10*maximum)-.10*max(0,maximum-.06)

def rank(records,candidates):
    require(len(records)==32 and len({r['id'] for r in records})==32,'All32 unique terminals required')
    conditions={(d,a) for d in ('IID','non-IID') for a in ('Benign','S-DFA')}
    aggregate=[]
    for candidate in candidates:
        rows=[r for r in records if r['candidate']==candidate['id']]
        require(len(rows)==4 and {(r['distribution'],r['attack']) for r in rows}==conditions,'Allfour conditions per candidate')
        aggregate.append(dict(candidate=candidate['id'],mean_four_condition_score=sum(score(r['metrics']) for r in rows)/4,
            mean_four_condition_metrics={key:sum(r['metrics'][key] for r in rows)/4 for key in ('accuracy','aeod','aspd')},records=rows))
    require(len(aggregate)==8,'Originaleight candidates only')
    winner=sorted(aggregate,key=lambda r:(-r['mean_four_condition_score'],r['candidate']))[0]['candidate']
    accuracy=sorted(aggregate,key=lambda r:(-r['mean_four_condition_metrics']['accuracy'],r['candidate']))[0]['candidate']
    def dominates(a,b):
        x,y=a['mean_four_condition_metrics'],b['mean_four_condition_metrics']
        return x['accuracy']>=y['accuracy'] and x['aeod']<=y['aeod'] and x['aspd']<=y['aspd'] and x!=y
    pareto=sorted(row['candidate'] for row in aggregate if not any(dominates(other,row) for other in aggregate))
    return dict(selected_recipe=winner,accuracy_champion=accuracy,three_metric_Pareto=pareto,
        all_candidates=sorted(aggregate,key=lambda r:(-r['mean_four_condition_score'],r['candidate'])))

def main(approval,approval_sha):
    require(not (HERE/'screen_summary.json').exists(),'Existing summary preserved; no overwrite')
    scope,approved=approve('screen',approval,approval_sha,dispatch=False)
    require((HERE/'screen_complete.json').is_file() and not (HERE/'screen_failure.json').exists(),'Queue completion alone insufficient; no failures')
    os.environ.update(CUDA_VISIBLE_DEVICES=approved['cuda_visible_device'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import body
    scope=dict(scope,runtime_cuda_visible_device=approved['cuda_visible_device'],runtime_gpu_uuid=approved['gpu_uuid'])
    body.verify_scope(scope);worker,_=body.modules();protocol=read(HERE/scope['runtime_protocol']);_,checked,_=functions(body,scope)
    records=[]
    for entry in scope['jobs']:
        job=read(HERE/entry['job']);validate_entry(entry,job,scope,protocol,worker);result=checked(entry,scope)
        require(result is not None,'Missing/partial terminal cannot enter selection')
        records.append(dict(id=entry['id'],candidate=job['tuning_candidate'],distribution=job['distribution'],attack=job['attack'],
            seed=91001,metrics=result['metrics'],checkpoint_sha256=digest(HERE/entry['output']/'model.pt'),acceptance_sha256=digest(HERE/entry['output']/'acceptance.json')))
    require(protocol['score']=='accuracy - .35*(.45*aeod + .45*aspd + .10*max(aeod,aspd)) - .10*max(0,max(aeod,aspd)-.06)','Frozen selection score changed')
    summary=dict(status='32_VALID_ONLY_TERMINALS_STRICT_ACCEPTED_SUMMARIZED_BACKUP_PENDING',**rank(records,protocol['candidates']),
        seed_n=1,no_sample_std_or_significance=True,four_conditions_not_four_seeds=True,all_negative_results_preserved=True,
        final_test_evaluated=False,formal_multi_seed_records=0,source_seal_sha256=digest(HERE/'FILES_SHA256.json'))
    body.write(HERE/'screen_summary.json',summary);print(summary['status'],summary['selected_recipe'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--approved',type=Path,required=True);parser.add_argument('--approved-sha256',required=True);args=parser.parse_args();main(args.approved,args.approved_sha256)
