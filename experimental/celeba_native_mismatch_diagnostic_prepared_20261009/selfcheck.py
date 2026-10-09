"""Pure-stdlib contract/AST checks; never import Torch/NumPy or instantiate CNN."""
from pathlib import Path
import ast,copy,hashlib,json,sys
import adapter
HERE=Path(__file__).resolve().parent
inputs=adapter.read(HERE/'INPUTS.json')
fixture=dict(status='APPROVED_EXACT_ONE_GPU_DIAGNOSTIC_ONLY',inputs_sha256='input',package_sha256='package',id=adapter.ID,
    output=adapter.OUTPUT,allowed_cpus=[105],compute_threads=1,max_processes=1,host_gpu_index=0,logical_device='cuda:0',
    native_tolerance=1e-12,target_split='valid',views=['native','raw','shared_calibration'],training_authorized=False,
    test_authorized=False,retry_authorized=False,cohort_inclusion_authorized=False,restart_old872_authorized=False,gpu_uuid='GPU-fixture',resource_receipt_sha256='a'*64)
adapter.validate_contract(fixture,'input','package')
cases=[]
for key,value in [('status','PREPARED_NOT_AUTHORIZED'),('inputs_sha256','wrong'),('package_sha256','wrong'),('id','other_seed'),
    ('output','/old872/overwrite'),('allowed_cpus',[104]),('compute_threads',8),('max_processes',2),('host_gpu_index',1),
    ('logical_device','cpu'),('native_tolerance',1e-3),('target_split','test'),('views',['native']),('training_authorized',True),
    ('test_authorized',True),('retry_authorized',True),('cohort_inclusion_authorized',True),('restart_old872_authorized',True),('gpu_uuid',''),('resource_receipt_sha256','')]:
    item=copy.deepcopy(fixture);item[key]=value
    try:adapter.validate_contract(item,'input','package')
    except ValueError:cases.append(key)
    else:raise AssertionError('Failed to reject '+key)
try:adapter.authorized(None,None)
except ValueError:cases.append('no_external_approval')
else:raise AssertionError('Missing approval passed')
proof=adapter.read(HERE/'SOURCE_REUSE.json');source=Path(inputs['scientific_sources']['v2']['local']).read_text()
fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='replay_one')
original=ast.get_source_segment(source,fn)+'\n';gpu=(HERE/'gpu_replay_body.py').read_text();reverted=gpu
for row in reversed(proof['changes']):
    assert reverted.count(row['after'])==1;reverted=reverted.replace(row['after'],row['before'])
assert reverted==original
assert "require(comparison['accepted'], 'Native metrics exceed fixed tolerance; preserve evidence and stop without retry')" in gpu
assert 'extract_and_predict' in gpu and 'evaluate_frozen_predictions' in gpu and 'weights_after == weights_before' in gpu
for path in HERE.glob('*.py'):ast.parse(path.read_text())
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
result=dict(status='PASS_PREPARED_CONTRACT_AND_SOURCE_DIFF_ONLY',rejections=cases,rejection_count=len(cases),exact_inverse_scientific_diff=True,
    native_failure_guard_unchanged=True,Torch_imported=False,NumPy_imported=False,CNN_or_GPU_or_training_executed=False,
    diagnostic_array_comparison_not_executed=True,limited_to_one_model=True)
with (HERE/'selfcheck.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
template=copy.deepcopy(fixture);template.update(status='PREPARED_NOT_AUTHORIZED',inputs_sha256=adapter.sha(HERE/'INPUTS.json'),
    package_sha256=None,gpu_uuid=None,resource_receipt=None,resource_receipt_sha256=None)
with (HERE/'APPROVAL_TEMPLATE.json').open('x',encoding='utf-8') as f:json.dump(template,f,indent=2);f.write('\n')
print(json.dumps(result))
