"""Future saved-checkpoint comparison only; no CNN inference or metric refitting."""
from pathlib import Path
import argparse,json,traceback
from adapter import load_context
from metadata import H,read


def compare(stage,repo):
    stage=Path(stage).resolve();gate,scope=load_context(stage,repo)
    destination=stage/'GATE_COMPARISON.json';assert not destination.exists(),'Preserve existing comparison evidence'
    assert not list(stage.glob('failure*.json')),'Prior failure must be reviewed, not retried'
    import torch
    rows={};receipts={}
    for entry in scope['jobs']:
        output=stage/entry['output'];accepted=read(output/'acceptance.json');outer=read(output/'item_receipt.json')
        assert outer['status']=='CPU3_GATE_ORIGINAL_STRICT_AND_EXTERNAL_IDENTITY_PASS'
        assert outer['scope_sha256']==H((stage/'scope.json').read_bytes()) and outer['scientific_table_records']==0
        assert outer['acceptance_sha256']==H((output/'acceptance.json').read_bytes())
        assert outer['rng_sha256']==H((output/'rng_after.json').read_bytes())
        result=gate.checked(entry,scope,accepted['dispatch_receipt']);assert result is not None
        job=read(stage/entry['job']);key=(job['method'],job['attack'],job['implementation'])
        assert key not in rows;rows[key]=(output,result)
        receipts[entry['id']]=H((output/'item_receipt.json').read_bytes())

    def exact(left,right,fflip_null=False):
        lp,lr=rows[left];rp,rr=rows[right]
        a=torch.load(lp/'model.pt',map_location='cpu',weights_only=True)
        b=torch.load(rp/'model.pt',map_location='cpu',weights_only=True)
        assert list(a)==list(b) and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and torch.equal(a[k],b[k]) for k in a)
        for key in ('metrics','trajectory_metrics','round_summaries','evaluation_stats','data_contract','gradient_recipe'):
            assert lr[key]==rr[key],(left,right,key)
        for name in ('native_replay.json','rng_after.json'):assert read(lp/name)==read(rp/name),(left,right,name)
        la,ra=read(lp/'gradient_audit.json'),read(rp/'gradient_audit.json')
        if fflip_null:
            # Only metadata assignment differs; check every numerical/audit field after removing these two labels.
            def numerical(audit):return [{k:v for k,v in row.items() if k not in ('actual_attack_types','actual_foe_mode')} for row in audit]
            assert numerical(la)==numerical(ra)
            assert all(x['raw_gradient_sha256']==x['uploaded_gradient_sha256'] for x in la+ra)
        else:
            assert la==ra and lr['attack_audit']==rr['attack_audit']
        return dict(left=list(left),right=list(right),model_tensor_exact=True,metrics_rounds_audit_rng_exact=True,
                    metadata_only_fflip_null=fflip_null)

    comparisons=[]
    for method in scope['methods']:
        for attack in ('F Flip','FedSA','Sp-DFA'):comparisons.append(exact((method,attack,'screen'),(method,attack,'coverage')))
        comparisons.append(exact((method,'F Flip','coverage'),(method,'Benign','screen'),True))
    before=read(stage/scope['jobs'][0]['output']/'resource_before.json')
    after=read(stage/scope['jobs'][-1]['output']/'resource_after.json')
    assert not before['failed'] and not after['failed']
    observed_growth=gate.formal_growth(before,after)
    assert observed_growth or (before['completed']==after['completed']==800 and not after['active'] and not after['pending'])
    payload=dict(status='EXACT14_CPU3_GATE_SAVED_COMPARISON_PASS',scope_sha256=H((stage/'scope.json').read_bytes()),
        checked_jobs=14,rounds_per_job=3,total_rounds=42,comparisons=comparisons,strict_receipts_sha256=receipts,
        protected_main_observed_growth=observed_growth,scientific_table_records=0,coverage_authorized=False,
        limitation='Same-horizon CPU interface check only; no CPU/GPU equivalence, 70-round claim or selected-recipe performance evidence.')
    gate.write(destination,payload);return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--repo',required=True);a=p.parse_args()
    try:print(json.dumps(compare(a.stage,a.repo)))
    except BaseException as error:
        failure=Path(a.stage)/'failure_comparison.json'
        if failure.parent.is_dir() and not failure.exists():
            with failure.open('x',encoding='utf8') as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc()),f,indent=2)
        raise
