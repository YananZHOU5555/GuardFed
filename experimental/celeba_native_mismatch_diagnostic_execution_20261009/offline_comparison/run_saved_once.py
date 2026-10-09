from pathlib import Path
import os,sys,importlib.util,json,hashlib,datetime
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
p=Path(__file__).resolve().parent;root=p.parent
def load(name,f):
 spec=importlib.util.spec_from_file_location(name,f);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
e=load('original_evaluator',root.parent/'celeba_final_valid_replay_20261009/v4/remaining872_attempt1/chunk_036/failure_source_original/evaluator.py')
c=load('offline_compare',p/'compare_saved_v2.py');e.torch.set_num_threads(1)
a=json.loads((p/'APPROVED.json').read_text());assert hashlib.sha256((p/'compare_saved_v2.py').read_bytes()).hexdigest()==a['source_sha256']
gpu=root/'attempt2_backup/verified_extract/runs/FairGuard_IID_F-Flip_seed91009'
result=c.compare(gpu,root/'original_valid_cache.npz',e)
assert not e.torch.cuda.is_initialized()
with (p/'comparison.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
# Serialization boundary rejects unknown/missing keys and changed fitted values.
fit=json.loads((gpu/'receipt.json').read_text())['fits']['shared_calibration'];checks=[]
for bad in [{'0':0.1},{'0':0.1,'1':0.2,'2':0.3},{0:0.1,1:0.2},{'0':fit['thresholds']['0']+1,'1':fit['thresholds']['1']}]:
 try:c.restore_fit_keys(dict(fit,thresholds=bad),e)
 except (AssertionError,ValueError,TypeError):checks.append('REJECTED')
 else:raise AssertionError('Bad threshold map accepted')
with (p/'OFFLINE_ACCEPTANCE.json').open('x') as f:json.dump(dict(status='SAVED_ARRAY_COMPARISON_PASS_NOT_COHORT_ACCEPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rejections=checks,CUDA_initialized=False,training_or_inference=False,metrics_counts_and_predictions_recomputed_three_views=True,comparison_sha256=hashlib.sha256((p/'comparison.json').read_bytes()).hexdigest()),f,indent=2)
print(json.dumps({'native':result['GPU_native_comparison'],'margins':result['margins'],'flips':{k:v['flips'] for k,v in result['views'].items()}},indent=2))
