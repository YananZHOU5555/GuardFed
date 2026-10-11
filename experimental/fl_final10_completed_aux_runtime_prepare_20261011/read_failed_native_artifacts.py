"""Read frozen failure artifacts only; no NumPy/Torch, prediction or refit."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert __debug__ and sorted(os.sched_getaffinity(0))==[110] and os.getpriority(os.PRIO_PROCESS,0)>=10
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
assert 'idle' in subprocess.check_output(['ionice','-p',str(os.getpid())],text=True).lower()
base=Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010');run=base/'attempt1';identity='minus_F_non-IID_Sp-DFA_seed91004'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bp=run/'tasks'/identity/'binding.json';assert sha(bp)=='489e9bb0beed5e9977d5a8e81b62fbe02a9cabe73d08ce72df0f73b8f1e712cd'
b=read(bp);r=b['record'];assert r['id']==identity
out=run/'runs'/identity;receipt=read(out/'receipt.json')
original=Path(r['runtime_paths']['result']);result=read(original)
compact={k:result[k] for k in result if k not in ('round_summaries','config','data_contract','source_hashes','adapter_source_hashes') and not isinstance(result[k],list)}
# Keep all metric/evaluation/calibration objects; omit per-round diagnostic arrays.
last=result['round_summaries'][-1]
terminal={k:v for k,v in last.items() if k=='round' or any(x in k.lower() for x in ('metric','calibr','threshold','evaluat','predict','device','torch'))}
artifacts={}
for key,p in [('binding',bp),('receipt',out/'receipt.json'),('array',out/'validation_predictions.npz'),('original_result',original),('original_checkpoint',Path(r['runtime_paths']['checkpoint'])),('original_job',Path(r['runtime_paths']['raw_job'])),('queue_failure',run/'QUEUE_FAILURE.json')]:
 artifacts[key]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
assert artifacts['original_result']['sha256']==r['result']['sha256'] and artifacts['original_checkpoint']['sha256']==r['checkpoint']['sha256'] and artifacts['original_job']['sha256']==r['raw_job']['sha256']
assert receipt['prediction_arrays_sha256']==artifacts['array']['sha256']
failures={p.name:{'sha256':sha(p),'record':read(p)} for p in run.glob('*FAILURE*.json')}
related={}
for p in sorted(out.glob('*.json')):related[p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
for p in [out.with_name(out.name+'.bridge_failure.json')]:
 if p.exists():related[p.name]={'sha256':sha(p),'bytes':p.stat().st_size,'record':read(p)}
selected={k:receipt[k] for k in ('status','id','checkpoint_sha256','original_result_sha256','original_job_sha256','config_canonical_sha256','original_training_torch','runtime','root_reconstruction','valid_n','valid_image_ids_sha256','fits','views','native_comparison','prediction_arrays_sha256','zero_margin_count','all_input_sha_unchanged') if k in receipt}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'id':identity,'artifacts':artifacts,'receipt_keys':list(receipt),'receipt':selected,'original_result_keys':list(result),'original_result_compact':compact,'original_round70_keys':list(last),'original_round70':terminal,'original_config':result.get('config'),'bound_training_torch':r['training_torch'],'bound_native_metrics':r['prior_validation_metrics'],'bound_prediction_support':r['accepted_v4_row']['prediction_support'],'bound_group_support':r['accepted_v4_row']['group_denominator_support'],'paired_full_reference_only':r['paired_full'],'output_json_files':related,'failures':failures,'no_original_training_predictions_available_claim':True,'read_only':True,'new_CNN':0,'new_fit':0,'training':0,'retry':False,'test':False}))
