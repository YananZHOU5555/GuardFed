"""Reuse the root-accepted record bridge, including five wrong-runtime refusals."""
from pathlib import Path
import copy,hashlib,json,sys,traceback
sys.dont_write_bytecode=True
B=Path('F:/YananResearchStorage/GuardFed/celeba_hybrid_delta_after23_20261010');bridge_dir=Path(__file__).resolve().parent/'local_record_bridge_v2';sys.path.insert(0,str(bridge_dir))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
try:
 for row in json.loads((bridge_dir/'FILES_SHA256.json').read_bytes())['members']:assert sha(bridge_dir/row['path'])==row['sha256']
 import bridge
 proof=bridge.replay();server=json.loads((B/'PARTIAL_ACCEPTANCE.json').read_bytes())
 expected={r['id']:r for r in server['records']}
 assert len(proof['records'])==len(expected)==4 and {r['id'] for r in proof['records']}==set(server['accepted_new_ids'])
 for row in proof['records']:
  assert row['metrics']==expected[row['id']]['metrics'] and row['checkpoint_sha256']==expected[row['id']]['model_sha256']
 refused=[]
 for key,value in [('CPU',[105]),('torch','2.8.0+cpu'),('cuda_build',None),('threads',2),('gpu_name_for_original_checker','WRONG_GPU')]:
  changed=copy.deepcopy(server);changed['runtime'][key]=value
  try:bridge.check_runtime(changed)
  except AssertionError:refused.append(key)
  else:raise AssertionError('Wrong original runtime was accepted: '+key)
 proof['runtime_refusals']=refused
 save('LOCAL_RECORD_CHECKS.json',proof)
 print(json.dumps(dict(status=proof['status'],accepted_new=len(proof['records']),proof_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),runtime_refusals=refused)))
except BaseException as error:
 save('LOCAL_RECORD_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise
