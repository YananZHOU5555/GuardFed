"""Compare frozen CPU and future single-GPU arrays; no model import/inference."""
from pathlib import Path
import hashlib,json

HERE=Path(__file__).resolve().parent.parent/'prepared'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def restore_fit_keys(fit,evaluator):
    restored=dict(fit)
    thresholds=fit.get('thresholds')
    if thresholds is not None:
        assert isinstance(thresholds,dict) and set(thresholds)=={'0','1'}, 'Only exact JSON threshold keys 0/1 accepted'
        restored['thresholds']={int(k):v for k,v in thresholds.items()}
    assert evaluator.canonical_sha({k:v for k,v in restored.items() if k!='fit_sha256'})==fit['fit_sha256']
    assert json.dumps(restored,sort_keys=True)==json.dumps(fit,sort_keys=True)
    return restored

def compare(gpu_output,valid_cache,evaluator):
    import numpy as np
    inputs=read(HERE/'INPUTS.json');record=read(HERE/'MODEL_RECORD.json')
    assert sha(evaluator.__file__)==inputs['scientific_sources']['evaluator']['sha256']
    assert sha(valid_cache)=='39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
    for rel,row in inputs['cpu_evidence'].items():assert sha(HERE/rel)==row['sha256']
    gpu_output=Path(gpu_output);cpu=read(HERE/'cpu_evidence/receipt.json');gpu=read(gpu_output/'receipt.json')
    provenance=read(gpu_output.with_name(gpu_output.name+'.diagnostic.json'))
    assert provenance['source_before']==provenance['source_after'] and provenance['accepted_for_cohort'] is False
    assert gpu['scope']=='SINGLE_MODEL_GPU_DIAGNOSTIC_NOT_ACCEPTED_COHORT' and gpu['status'] in ['DIAGNOSTIC_NATIVE_MATCH','DIAGNOSTIC_NATIVE_MISMATCH']
    assert gpu['id']==cpu['id']==record['id'] and gpu['checkpoint_sha256']==cpu['checkpoint_sha256']==record['checkpoint']['sha256']
    assert gpu['weights_before']==gpu['weights_after']==cpu['weights_before']==cpu['weights_after']
    assert gpu['original_job_sha256']==record['raw_job']['sha256'] and gpu['original_result_sha256']==record['result']['sha256']
    assert gpu['native_comparison']['tolerance']==1e-12 and gpu['runtime']['device']=='cuda:0'
    assert gpu['native_comparison']['expected']==cpu['native_comparison']['expected']==record['prior_validation_metrics']
    assert gpu['valid_n']==cpu['valid_n']==19867 and gpu['root_reconstruction']==cpu['root_reconstruction']
    assert sha(gpu_output/'validation_predictions.npz')==gpu['prediction_arrays_sha256']
    with np.load(valid_cache,allow_pickle=False) as z:y,s=z['valid_y'],z['valid_sensitive']
    views={};margins={}
    with np.load(HERE/'cpu_evidence/validation_predictions.npz',allow_pickle=False) as c, np.load(gpu_output/'validation_predictions.npz',allow_pickle=False) as g:
        for split,size in [('valid',19867),('root',16277)]:
            assert np.array_equal(c[split+'_image_ids'],g[split+'_image_ids']) and len(c[split+'_margins'])==len(g[split+'_margins'])==size
            cm,gm=c[split+'_margins'],g[split+'_margins'];assert np.isfinite(cm).all() and np.isfinite(gm).all()
            delta=gm.astype(np.float64)-cm.astype(np.float64)
            margins[split]=dict(n=size,exact_array_equal=bool(np.array_equal(cm,gm)),changed_count=int(np.sum(cm!=gm)),
                max_abs_difference=float(np.max(np.abs(delta))),mean_abs_difference=float(np.mean(np.abs(delta))),
                CPU_min_abs=float(np.min(np.abs(cm))),GPU_min_abs=float(np.min(np.abs(gm))))
        for view in ['native','raw','shared_calibration']:
            cp,gp=c['prediction_'+view],g['prediction_'+view]
            cm=evaluator.group_metrics(y,cp,s);gm=evaluator.group_metrics(y,gp,s)
            assert cm==cpu['views'][view] and gm==gpu['views'][view]
            assert np.array_equal(evaluator.predict_views(c['valid_margins'],s,{view:restore_fit_keys(cpu['fits'][view],evaluator)})[view],cp)
            assert np.array_equal(evaluator.predict_views(g['valid_margins'],s,{view:restore_fit_keys(gpu['fits'][view],evaluator)})[view],gp)
            changed=np.flatnonzero(cp!=gp)
            views[view]=dict(CPU_metrics=cm,GPU_metrics=gm,CPU_fit=cpu['fits'][view],GPU_fit=gpu['fits'][view],prediction_flip_count=len(changed),
                flips=[dict(position=int(i),image_id=int(c['valid_image_ids'][i]),y=int(y[i]),sensitive=int(s[i]),CPU_prediction=int(cp[i]),GPU_prediction=int(gp[i]),
                    CPU_margin=float(c['valid_margins'][i]),GPU_margin=float(g['valid_margins'][i])) for i in changed])
    return dict(status='DIAGNOSTIC_SAVED_ARRAY_COMPARISON_ONLY',id=record['id'],historical_GPU_aggregate=record['prior_validation_metrics'],
        historical_GPU_array_available=False,CPU_native_comparison=cpu['native_comparison'],GPU_native_comparison=gpu['native_comparison'],
        margins=margins,views=views,accepted_for_cohort=False,old_CPU_failure_remains=True,
        limitation='Measured flips are between saved CPU and this new GPU execution only. Historical aggregate equality cannot identify historical per-sample predictions or establish unique causality.')
