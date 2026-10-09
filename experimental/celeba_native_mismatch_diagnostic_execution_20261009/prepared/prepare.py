"""Prepare one diagnostic without importing Torch/CNN or loading model tensors."""
from pathlib import Path
import ast,difflib,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ID='FairGuard_IID_F-Flip_seed91009'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,value):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
base=ROOT/'tmp/celeba_final_valid_replay_20261009'
inventory=base/'inputs/model_inventory.json'
assert sha(inventory)=='3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
record=next(r for r in read(inventory)['records'] if r['id']==ID)
cpu=read(HERE/'cpu_evidence/receipt.json');worker=read(HERE/'cpu_evidence'/f'{ID}.worker.json')
assert cpu['id']==worker['id']==ID and cpu['status']=='NATIVE_VALID_REPLAY_MISMATCH'
assert cpu['checkpoint_sha256']==record['checkpoint']['sha256'] and cpu['original_result_sha256']==record['result']['sha256'] and cpu['original_job_sha256']==record['raw_job']['sha256']
assert cpu['weights_before']==cpu['weights_after'] and worker['artifacts_unchanged'] and worker['artifact_before']==worker['artifact_after']
assert cpu['prediction_arrays_sha256']==sha(HERE/'cpu_evidence/validation_predictions.npz')
assert cpu['native_comparison']['expected']==record['prior_validation_metrics'] and cpu['native_comparison']['tolerance']==1e-12 and not cpu['native_comparison']['accepted']
save('MODEL_RECORD.json',record)
scientific={'v2':base/'replay.py','v3':base/'v3/replay_v3.py','v4':base/'v4/replay_v4.py','evaluator':base/'inputs/evaluator.py',
    'core':ROOT/'tmp/celeba_shared_calibration_20260928/reproduce_paper_tables.py','cnn':ROOT/'tmp/celeba_shared_calibration_20260928/celeba_data.py'}
expected={'v2':'8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803','v3':'abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e',
 'v4':'43b16d20d2497b7762cd0f6039f7f4bfc8fddd4e4c32979a16291b26e394ae5e','evaluator':'805eedf1fb08137cd86a543a80c83b9527e5c8937be02f7d2dca83a33b86e04c',
 'core':'cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed','cnn':'0f48fbc6d241e7d0cde692839659e817feab407991795a6f92a33052a9cc07ce'}
assert all(sha(p)==expected[k] for k,p in scientific.items())
text=scientific['v2'].read_text();tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay_one')
body=ast.get_source_segment(text,node)+'\n'
replacements=[("model = cnn.CelebACNN(seed=cfg.seed).cpu()","model = cnn.CelebACNN(seed=cfg.seed).to('cuda:0')"),
 ("'scope': 'VALID_ONLY_IMPLEMENTATION_PREFLIGHT'","'scope': 'SINGLE_MODEL_GPU_DIAGNOSTIC_NOT_ACCEPTED_COHORT'"),
 ("'NATIVE_VALID_REPLAY_PASS' if comparison['accepted'] else 'NATIVE_VALID_REPLAY_MISMATCH'","'DIAGNOSTIC_NATIVE_MATCH' if comparison['accepted'] else 'DIAGNOSTIC_NATIVE_MISMATCH'"),
 ("'device': 'cpu', 'original_config_device'","'device': 'cuda:0', 'original_config_device'"),
 ("Two bounded valid-only canaries are implementation evidence, not all900 replay or a new final performance table","Single-model CUDA diagnostic only; never authorizes cohort inclusion, CPU failure invalidation, training, test or queue restart")]
gpu=body
for before,after in replacements:
    assert gpu.count(before)==1,before
    gpu=gpu.replace(before,after)
ast.parse(gpu)
with (HERE/'gpu_replay_body.py').open('x',encoding='utf-8') as f:f.write(gpu)
with (HERE/'SCIENCE_BODY_DIFF.patch').open('x',encoding='utf-8') as f:f.write(''.join(difflib.unified_diff(body.splitlines(True),gpu.splitlines(True),fromfile='sealed_v2/replay_one',tofile='diagnostic_gpu/replay_one')))
reverted=gpu
for before,after in reversed(replacements):reverted=reverted.replace(after,before)
assert reverted==body
functions={}
for key,names in {'v2':['metadata','rebuild_root','check_native','validate_original'],'core':['model_margins','fit_group_thresholds'],'evaluator':['extract_and_predict','fit_views','predict_views','evaluate_frozen_predictions','thresholds_from_root','weights_identity']}.items():
    source=scientific[key].read_text();t=ast.parse(source)
    for name in names:
        n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
        functions[key+'.'+name]=hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest()
remote_base='/workspace/guardfed_checks/celeba_final_valid_replay_20261009'
remote_sources={'v2':remote_base+'/replay.py','v3':remote_base+'/v3/replay_v3.py','v4':remote_base+'/v4/replay_v4.py','evaluator':remote_base+'/inputs/evaluator.py','core':'/workspace/GuardFed-celeba-expanded/scripts/reproduce_paper_tables.py','cnn':'/workspace/GuardFed-celeba-expanded/src/celeba_data.py'}
cpu_origin=remote_base+'/v4/remaining872_attempt1/chunk_036/batch/runs/'+ID
cpu_members={p.relative_to(HERE).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size,'original_remote_path':cpu_origin+'/'+p.name if p.name in ['receipt.json','validation_predictions.npz'] else cpu_origin+('.worker.json' if p.suffix=='.json' else '.log')} for p in (HERE/'cpu_evidence').iterdir() if p.is_file()}
save('INPUTS.json',dict(status='PREPARED_NOT_AUTHORIZED',id=ID,record_sha256=sha(HERE/'MODEL_RECORD.json'),inventory_sha256=sha(inventory),cpu_evidence=cpu_members,
    scientific_sources={k:dict(local=str(p),remote=remote_sources[k],sha256=sha(p)) for k,p in scientific.items()},
    original_artifacts=worker['artifact_before'],source_data_pins=record['source_hashes'],adapter_pins=record['adapter_source_hashes'],
    historical_torch=record['training_torch'],proposed_runtime_torch='2.11.0+cu128',batch_size=record['config']['batch_size'],root_n=16277,valid_n=19867,
    original_validation_metrics=record['prior_validation_metrics'],cpu_validation_metrics=cpu['native_comparison']['observed'],
    native_tolerance=1e-12,views=['native','raw','shared_calibration'],GPU_run_performed=False,CNN_run_performed=False,
    training_authorized=False,test_authorized=False,automatic_retry_authorized=False,cohort_inclusion_authorized=False,
    historical_GPU_prediction_array_available=False,failed_batch_receipt_sha256=worker['batch_receipt_sha256'],semantic_inspection_sha256=worker['semantic_inspection_sha256'],storage_map_sha256=worker['storage_map_sha256']))
save('SOURCE_REUSE.json',dict(original_sources=expected,unchanged_function_source_hashes=functions,original_body_sha256=hashlib.sha256(body.encode()).hexdigest(),
    derived_body_sha256=sha(HERE/'gpu_replay_body.py'),exact_inverse_diff_equal=True,changes=[dict(before=x,after=y) for x,y in replacements],
    extra_dependency_change='Independent strict GPU resource_gate replaces CPU-only resource_gate in private function globals; no metric/threshold/native gate change',
    cnn_forward_note='Existing forward transfers each uint8 batch to classifier.weight.device and FP32; tensor ordering/batch size/normalization unchanged',
    no_torch_or_CNN_imported=True))
print(json.dumps({'status':'PREPARED','id':ID,'cpu_array_sha256':cpu['prediction_arrays_sha256'],'batch_size':record['config']['batch_size'],'sources_verified':len(expected)}))
