"""Source and in-memory metadata fixtures only; no scientific or remote execution."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
for f in HERE.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
spec=importlib.util.spec_from_file_location('remaining_gate_fixture',HERE/'bind_gate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
memory={}
class MemoryPath:
 def __init__(self,p):self.p=p.p if isinstance(p,MemoryPath) else p
 def read_bytes(self):return memory[self.p]
m.Path=MemoryPath
blob=lambda x:json.dumps(x,sort_keys=True).encode()
audit=blob({'status':'SYNTHETIC_NOT_AN_ACTUAL_PROOF'});memory['synthetic_audit']=audit
root={'root_accepted':True,'ids':m.CANARIES,'cohort':{'path':m.BASE+'/adapter/EXACT100.json','sha256':'52d6a15e077eb50b1d81d2aa0e481c6e2e5fb5dfeea24e84ba6036535dffed9b'},'FedAvg_CUDA_root':{'path':'/workspace/guardfed_checks/FedAvg100_cuda_adopted_cache_index_20261011_actual002/ROOT_SCIENTIFIC_ADOPTION.json','sha256':'83a047fdd4d9c997c3ca31ff713de134575f1bcc6c75f6da579031f604efb083'},'records':{i:{role:{'path':m.BASE+'/runs/'+i+'/'+name,'sha256':'a'*64} for role,name in [('acceptance','formal_saved_acceptance.json'),('binding','formal_binding.json')]} for i in m.CANARIES},'offserver_audit':{'path':'synthetic_audit','sha256':hashlib.sha256(audit).hexdigest()}}
def invoke(r,allowed=True):
 memory['synthetic_root']=blob(r)
 return m.bind({'execution_authorized':allowed},{'path':'synthetic_root','sha256':hashlib.sha256(memory['synthetic_root']).hexdigest()})
out=invoke(root);assert out['canary_gate']['records']==root['records']
failures=0
for kind in ['not_adopted','wrong_order','wrong_cache','missing_audit','wrong_record']:
 bad=copy.deepcopy(root)
 if kind=='not_adopted':bad['root_accepted']=False
 elif kind=='wrong_order':bad['ids']=list(reversed(bad['ids']))
 elif kind=='wrong_cache':bad['FedAvg_CUDA_root']['sha256']='0'*64
 elif kind=='missing_audit':bad['offserver_audit']['sha256']='0'*64
 else:bad['records'][m.CANARIES[0]]['acceptance']['path']='wrong'
 try:invoke(bad)
 except ValueError:failures+=1
 else:raise AssertionError(kind+' incorrectly accepted')
try:invoke(root,False)
except ValueError:failures+=1
else:raise AssertionError('Unapproved input passed')
start=(HERE/'start_remote.py').read_text();launcher=(HERE/'run_root.py').read_bytes();parent=HERE.parent/'LoGo100_cuda_actual_root_20261011'
assert launcher==(parent/'run_root.py').read_bytes()
assert "'remaining','--root-auth-sha256'" in start and "'--canary-gate-sha256'" in start
assert "status.get('guardfed_uniform_cuda_LoGo100_canary_valid')=='EXITED'" in start
assert start.index("pre=run(")<start.index("save(BASE/'ROOT_CANARY_ACCEPTANCE.json'")<start.index("conf.open('x'")
assert 'autorestart=false' in start and 'startretries=0' in start
report={'status':'PASS_SOURCE_AND_IN_MEMORY_METADATA_FIXTURES','positive_bind':1,'rejected_fixtures':failures,'launcher_parent_raw_exact':True,'original_guard_queue_science_unchanged':True,'future_root_audit_unbound':True,'scientific_or_remote_execution':False,'fixture_objects':'In-memory synthetic only; never written as root/audit artifacts'}
(HERE/'SOURCE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
