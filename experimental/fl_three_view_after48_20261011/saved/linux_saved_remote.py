
import ast,copy,datetime,hashlib,importlib.util,json,os,pathlib,sys,types
import argparse,subprocess
ap=argparse.ArgumentParser(description='Root-only unchanged original whole saved checker; no CNN')
ap.add_argument('--gate-result-sha256',required=True)
ap.add_argument('--allow-original-cached-root-refit',action='store_true',required=True)
a=ap.parse_args()
saved=pathlib.Path('/workspace/guardfed_checks/fl_three_view_after48_20261011/source/originals/saved_science.py')
payload={'saved_check_source':saved.read_text(encoding='utf-8'),'source_sha256':hashlib.sha256(saved.read_bytes()).hexdigest()}
assert hashlib.sha256(payload['saved_check_source'].encode()).hexdigest()==payload['source_sha256']
assert payload['source_sha256']=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
base=pathlib.Path('/workspace/guardfed_checks/fl_three_view_after48_20261011'); output=base/'LINUX_SAVED_CHECK.json'
assert not output.exists() and not (base/'LINUX_SAVED_FAILURE.json').exists()
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.getpriority(os.PRIO_PROCESS,0)>=10 and len(os.sched_getaffinity(0))==1
quota,period=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split();assert quota=='max' or int(quota)/int(period)>=32
spec=importlib.util.spec_from_file_location('root_exact3_source',base/'source/candidate.py');c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
c.package_check('fb1e22aa70ad7f8d3f16c9f0a7c0a152d0e354f77177650d82da984b50ae6734')
m=c.read(base/'source/MANIFEST.json');c.validate_manifest(m)
out=base/'outputs/attempt001'
assert hashlib.sha256((out/'GATE_RESULT.json').read_bytes()).hexdigest()==a.gate_result_sha256
gate=c.read(out/'GATE_RESULT.json')
assert not (out/'FAILURE.json').exists() and not list(out.rglob('FAILURE.json'))
assert gate['status']=='FLGMM_EXACT13_THREE_VIEW_PASS_NOT_ROOT_ADOPTED'
assert gate['package_sha256']=='fb1e22aa70ad7f8d3f16c9f0a7c0a152d0e354f77177650d82da984b50ae6734'
assert len(gate['receipts'])==13 and [r['id'] for r in gate['receipts']]==m['exact_ids']
status=subprocess.run(['supervisorctl','status','guardfed_flgmm_after48_exact13_valid'],capture_output=True,text=True)
assert status.returncode==3 and status.stdout.split()[:2]==['guardfed_flgmm_after48_exact13_valid','EXITED'],status.stdout
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 try:
  cmd=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  assert not any(pathlib.Path(token).name=='candidate.py' and 'fl_three_view_after48_20261011' in token for token in cmd),'Live replay worker'
 except (FileNotFoundError,ProcessLookupError):pass
 for task in (proc/'task').glob('*'):
  try:
   affinity=set(os.sched_getaffinity(int(task.name)))
   assert not (len(affinity)<=8 and affinity.intersection(os.sched_getaffinity(0))),'Restricted-thread CPU overlap'
  except ProcessLookupError:pass
import numpy as np,pandas as pd,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);assert torch.cuda.device_count()==0
bridge,ev=c.bound_bridge(m,runtime=True,torch_module=torch,pandas_module=pd)
rt=c.runtime_originals(ev);repo=pathlib.Path(m['server_repo']);sys.path.insert(0,str(repo))
for rel,h in m['runtime_repo_hashes'].items():
 if not rel.endswith('images.npy'):assert c.sha(repo/rel)==h,rel
core=c.load('_exact3_root_core',repo/'scripts/reproduce_paper_tables.py');ids,y,s,metadata=rt.metadata(repo)
assert gate['status']=='FLGMM_EXACT13_THREE_VIEW_PASS_NOT_ROOT_ADOPTED'
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
 report={'status':'LINUX_ORIGINAL_FLGMM13_WHOLE_SAVED_CHECK_PASS_NOT_ROOT_ADOPTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':results,'original_check_saved_sha256':payload['source_sha256'],'gate_result_sha256':a.gate_result_sha256,'package_sha256':'fb1e22aa70ad7f8d3f16c9f0a7c0a152d0e354f77177650d82da984b50ae6734','cached_root_refits':13,'new_CNN':0,'new_training':0,'test':False,'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'device':'cpu','cpu_affinity':list(os.sched_getaffinity(0))},'Windows_group_KL_difference_remains_preserved':True,'native_tolerance_unchanged':1e-12}
 output.write_text(json.dumps(report,indent=2)+'\n');sys.stdout.write(output.read_text())
except BaseException:
 import traceback
 report={'status':'LINUX_FLGMM13_WHOLE_SAVED_FAILED_PRESERVED_NO_RETRY','traceback':traceback.format_exc(),'completed':len(results)}
 (base/'LINUX_SAVED_FAILURE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));sys.exit(1)
