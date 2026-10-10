"""One bounded original saved-check on Linux; preserves Windows audit mismatch."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess
R=Path(__file__).resolve().parents[1]
O=R/'tmp/celeba_added_cnn_exact3_root_execution_20261010'
P=R/'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009/chunk_039/verified_extract/sourcefreeze/062_saved_science.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P)=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
assert not (O/'LINUX_EXACT_ROOT_EXIT.json').exists()
payload=json.dumps({'saved_check_source':P.read_text(encoding='utf-8'),'source_sha256':sha(P)}).encode()
remote=r'''
import ast,copy,datetime,hashlib,importlib.util,json,os,pathlib,sys,types
payload=json.load(sys.stdin)
assert hashlib.sha256(payload['saved_check_source'].encode()).hexdigest()==payload['source_sha256']
assert payload['source_sha256']=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
base=pathlib.Path('/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010'); output=base/'LINUX_EXACT_ROOT_CHECK.json'
assert not output.exists() and not (base/'LINUX_EXACT_ROOT_FAILURE.json').exists()
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.getpriority(os.PRIO_PROCESS,0)>=10 and len(os.sched_getaffinity(0))==1
quota,period=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split();assert quota=='max' or int(quota)/int(period)>=32
spec=importlib.util.spec_from_file_location('root_exact3_source',base/'source/candidate.py');c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
c.package_check('49c42b90214eae57d456f8ded1d2a4ff2de2762a78d6bcf5c4ae1a5766de9434')
m=c.read(base/'source/MANIFEST.json');c.validate_manifest(m)
import numpy as np,pandas as pd,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.cuda.device_count()==0
bridge,ev=c.bound_bridge(m,runtime=True,torch_module=torch,pandas_module=pd)
rt=c.runtime_originals(ev);repo=pathlib.Path(m['server_repo']);sys.path.insert(0,str(repo))
for rel,h in m['runtime_repo_hashes'].items():
 if not rel.endswith('images.npy'):assert c.sha(repo/rel)==h,rel
core=c.load('_exact3_root_core',repo/'scripts/reproduce_paper_tables.py');ids,y,s,metadata=rt.metadata(repo)
gate=c.read(base/'outputs/attempt001/GATE_RESULT.json');assert gate['status']=='EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED'
node=next(n for n in ast.parse(payload['saved_check_source']).body if isinstance(n,ast.FunctionDef) and n.name=='check_saved')
assert hashlib.sha256(ast.get_source_segment(payload['saved_check_source'],node).encode()).hexdigest()=='ed8dc5eb2332ef8df176166087a91fb5f2b5043cc0e4669f7c985ca5b7452cfc'
results=[]
try:
 for row,r in zip(m['records'],gate['receipts']):
  assert row['id']==r['id'] and c.read(base/'outputs/attempt001'/row['id']/'receipt.json')==r
  actual=bridge.identity_record(row['method'],row['id']);assert actual==row['identity']
  record=copy.deepcopy(actual);record.update(distribution=row['distribution'],attack=row['attack'],seed=row['seed'],result=actual['original_artifact_pins']['result'],raw_job=actual['original_artifact_pins']['job'],config_canonical_sha256=rt.canonical(actual['config']),training_torch=row['original_training_torch'])
  assert rt.canonical(record)==r['external_identity_record_sha256']
  paths={k:pathlib.Path(row['runtime_artifacts'][j]['server_path']) for k,j in [('checkpoint','model'),('result','result'),('raw_job','job')]}
  def measure(_pins=None):
   out={}
   for k,p in paths.items():
    h=c.sha(p);assert h==record[k]['sha256'];out[str(p)]={'sha256':h,'bytes':p.stat().st_size}
   return out
  before=measure();proof={'artifact_before':before,'artifact_after':measure()}
  def validate(_original,rec,_repo):
   assert bridge.identity_record(row['method'],row['id'])==row['identity'];return c.read(paths['result'])
  v2=types.SimpleNamespace(np=np,require=c.require,rebuild_root=ev.rebuild_root,digest=c.sha,canonical=rt.canonical,VIEWS=ev.VIEWS,check_native=ev.check_native)
  ns={'v2':v2,'mapping_paths':lambda rec,bind:paths,'full_hashes':measure,'mapped_functions':lambda rec,p,orig:(validate,None),'KINDS':('checkpoint','result','raw_job')}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged original check_saved>','exec'),ns)
  comp=ns['check_saved'](record,base/'outputs/attempt001'/row['id'],r,proof,types.SimpleNamespace(repo=repo),None,core,None,ev,ids,y,s)
  assert measure()==before
  results.append({'id':row['id'],'native_comparison':comp,'root_receipt_exact':True,'cached_root_fit_exact':True,'saved_predictions_metrics_counts_exact':True,'checkpoint_sha256':r['checkpoint_sha256'],'array_sha256':r['prediction_arrays_sha256'],'receipt_sha256':c.sha(base/'outputs/attempt001'/row['id']/'receipt.json')})
 report={'status':'LINUX_ORIGINAL_EXACT3_SAVED_ROOT_AND_ARRAY_CHECK_PASS_NOT_ROOT_ADOPTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':results,'original_check_saved_sha256':payload['source_sha256'],'cached_root_refits':3,'new_CNN':0,'new_training':0,'test':False,'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'device':'cpu','cpu_affinity':list(os.sched_getaffinity(0))},'Windows_group_KL_difference_remains_preserved':True,'native_tolerance_unchanged':1e-12}
 output.write_text(json.dumps(report,indent=2)+'\n');sys.stdout.write(output.read_text())
except BaseException:
 import traceback
 report={'status':'LINUX_EXACT_ROOT_FAILED_PRESERVED_NO_RETRY','traceback':traceback.format_exc(),'completed':len(results)}
 (base/'LINUX_EXACT_ROOT_FAILURE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));sys.exit(1)
'''
(O/'linux_saved_root_remote.py').write_text(remote,encoding='utf-8')
cmd='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -c '+shlex.quote(remote)
cp=subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',cmd],input=payload,capture_output=True,timeout=100)
(O/'LINUX_EXACT_ROOT_STDOUT.txt').write_bytes(cp.stdout);(O/'LINUX_EXACT_ROOT_STDERR.txt').write_bytes(cp.stderr)
(O/'LINUX_EXACT_ROOT_EXIT.json').write_text(json.dumps({'exit':cp.returncode,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stdout_sha256':sha(O/'LINUX_EXACT_ROOT_STDOUT.txt'),'stderr_sha256':sha(O/'LINUX_EXACT_ROOT_STDERR.txt')},indent=2)+'\n',encoding='utf-8')
if cp.returncode:print(cp.stdout.decode(errors='replace'));print(cp.stderr.decode(errors='replace'));raise SystemExit(cp.returncode)
report=json.loads(cp.stdout);assert report['status']=='LINUX_ORIGINAL_EXACT3_SAVED_ROOT_AND_ARRAY_CHECK_PASS_NOT_ROOT_ADOPTED'
(O/'LINUX_EXACT_ROOT_CHECK.json').write_bytes(cp.stdout)
print(json.dumps({'status':report['status'],'records':len(report['records']),'native_differences':[r['native_comparison']['max_abs_difference'] for r in report['records']],'proof_sha256':sha(O/'LINUX_EXACT_ROOT_CHECK.json')}))
