"""Independent common-valid-label counts from seven saved arrays; no image/model inference."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def verify(saved, inventory, cache):
    assert sha(inventory)=='a14f43e8ef2ab513f56315272748de897c6616bc005d10f0ebbe9f41fd406c43'
    assert sha(cache)=='39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
    with np.load(cache,allow_pickle=False) as z:y,sensitive=z['valid_y'],z['valid_sensitive']
    records={r['id']:r for r in read(inventory)['records']}
    selected=read(saved.parent/'backup_inventory.json')['accepted_new_ids']
    allowed=read(inventory)['selected_replay_ids']
    assert selected and len(selected)==len(set(selected)) and set(selected)<=set(allowed)
    assert sorted(p.name for p in saved.iterdir())==sorted(selected)
    accepted=[]
    for identity in selected:
        path=saved/identity;record=records[identity]
        receipt,bridge,acceptance=(read(path/n) for n in ('receipt.json','bridge_receipt.json','strict_acceptance.json'))
        assert receipt['id']==bridge['id']==acceptance['id']==identity
        assert acceptance['status']=='MECHANISM_VALID_THREE_VIEWS_ACCEPTED'
        assert bridge['source_before']==bridge['source_after'] and bridge['artifact_before']==bridge['artifact_after']
        assert len(bridge['source_before'])==35 and len(bridge['artifact_before'])==7
        assert receipt['checkpoint_sha256']==acceptance['checkpoint_sha256']==record['checkpoint']['sha256']
        assert receipt['weights_before']==receipt['weights_after'] and not receipt['optimizer_created'] and not receipt['gradients_created']
        assert receipt['runtime']['device']=='cpu' and receipt['runtime']['cuda_device_count']==0
        assert receipt['runtime']['torch_threads']==8 and receipt['runtime']['interop_threads']==1 and receipt['runtime']['nice']==10
        assert receipt['valid_n']==19867 and receipt['root_reconstruction']['root_n']==16277
        assert sha(path/'validation_predictions.npz')==receipt['prediction_arrays_sha256']
        assert sha(path/'receipt.json')==bridge['scientific_body_receipt_sha256']
        assert sha(path/'bridge_receipt.json')==acceptance['bridge_receipt_sha256']
        for resource in (receipt['before_resources'],receipt['after_resources']):
            assert all(cpus==list(range(112,120)) for cpus in resource['thread_cpu_affinities'].values())
        differences={};counts_checked=0
        with np.load(path/'validation_predictions.npz',allow_pickle=False) as data:
            assert len(data['valid_margins'])==len(y)==19867 and len(data['root_margins'])==16277
            for split,contract in [('valid','evaluation'),('root','root')]:
                assert hashlib.sha256(data[split+'_image_ids'].tobytes()).hexdigest()==record['data_contract'][contract+'_image_ids_sha256']
            for view in ('native','raw','shared_calibration'):
                prediction,fit=data['prediction_'+view],receipt['fits'][view]
                if fit['rule']=='argmax_margin_strictly_positive':
                    assert fit['thresholds'] is None;expected=data['valid_margins']>0
                else:
                    assert fit['rule']=='group_margin_greater_equal' and fit['fit_data']=='clean_train_root_only'
                    expected=np.where(sensitive==0,data['valid_margins']>=fit['thresholds']['0'],data['valid_margins']>=fit['thresholds']['1'])
                assert np.array_equal(prediction,expected)
                tpr=[int(((sensitive==g)&(y==1)&(prediction==1)).sum())/int(((sensitive==g)&(y==1)).sum()) for g in (0,1)]
                rates=[int(((sensitive==g)&(prediction==1)).sum())/int((sensitive==g).sum()) for g in (0,1)]
                direct=dict(accuracy=int((prediction==y).sum())/len(y),aeod=abs(tpr[0]-tpr[1]),aspd=abs(rates[0]-rates[1]))
                differences[view]={k:direct[k]-receipt['views'][view][k] for k in direct}
                assert all(d==0 for d in differences[view].values()) and receipt['views'][view]==acceptance['views'][view]
                for g in (0,1):
                    for name,yy,pp in [('tp',1,1),('fp',0,1),('tn',0,0),('fn',1,0)]:
                        assert receipt['views'][view]['group_confusion_counts'][str(g)][name]==int(((sensitive==g)&(y==yy)&(prediction==pp)).sum());counts_checked+=1
        native=max(abs(receipt['views']['native'][k]-record['prior_validation_metrics'][k]) for k in ('accuracy','aeod','aspd'))
        assert native==receipt['native_comparison']['max_abs_difference']<=1e-12
        accepted.append(dict(id=identity,checkpoint_sha256=receipt['checkpoint_sha256'],views=receipt['views'],
            independent_metric_checks=9,independent_confusion_count_checks=counts_checked,prediction_rule_checks=3,
            differences=differences,native_max_abs_difference=native,prediction_arrays_sha256=sha(path/'validation_predictions.npz')))
    return dict(status='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS',records=accepted,accepted_n=len(accepted),
        independent_metric_checks=9*len(accepted),independent_confusion_count_checks=24*len(accepted),prediction_rule_checks=3*len(accepted),
        new_training=0,new_Full_inference=0,test_inference=False,torch_imported='torch' in sys.modules,
        root_fit_recomputed_by_original_strict_server_bridge=True,local_recompute_uses_only_common_valid_labels=True,
        verifier_sha256=sha(__file__),claim_limit='This exact10-scope subset of validation terminal replays; not final test or complete900 mechanism evidence.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--saved',type=Path,required=True);parser.add_argument('--inventory',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();proof=verify(args.saved,args.inventory,args.cache)
    with args.out.open('x',encoding='utf-8') as out:out.write(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=proof['status'],accepted_n=proof['accepted_n'],metrics=proof['independent_metric_checks'],confusion_counts=proof['independent_confusion_count_checks'],rules=proof['prediction_rule_checks'])),flush=True)
