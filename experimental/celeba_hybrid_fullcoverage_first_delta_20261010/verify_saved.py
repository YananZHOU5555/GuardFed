"""Saved CPU tensor/record checks; original scientific predicates and pair comparison."""
from pathlib import Path
import ast,hashlib,json,sys
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
PACKAGE='a87a050b497a184efbe18b4649ad6bde40b9ea29a16f9e3efddd0ef1156e3b04'
CANARIES='722c83d5ebb4c211815480cde267a9250e0216c3cbaddbbf6d7a51d4736df29e'
DRIVER='1aedc2dc67c7b0fd3d515e16e778a00b784ae119fec5a3b7b3a5df3467bcf47f'
REPLACEMENTS=[("torch.__version__ == '2.11.0+cu128'","server_runtime['torch'] == '2.11.0+cu128'"),("torch.version.cuda == '12.8'","server_runtime['cuda_build'] == '12.8'"),("torch.cuda.get_device_name(0)","server_runtime['gpu_name_for_original_checker']")]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def record_checker(source):
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='checked')
 original=ast.get_source_segment(source,node);text=original
 changes=[("result['seed'] == 91001","result['seed'] == job['config']['seed']")]+REPLACEMENTS
 for old,new in changes:assert text.count(old)==1;text=text.replace(old,new)
 restored=text
 for old,new in reversed(changes):assert restored.count(new)==1;restored=restored.replace(new,old)
 assert restored==original
 return text

DELTA=read(Path(__file__).resolve().parent/'DELTA_SCOPE.json')
IDS=DELTA['ids']
assert IDS and len(IDS)==len(set(IDS)) and DELTA['no_later_completions'] is True
assert all(isinstance(x,str) and '/' not in x and '\\' not in x for x in IDS)

def verify(stage,server_runtime,legacy_source=None,offserver=False):
 stage=Path(stage).resolve();assert sha(stage/'PACKAGE_SHA256.json')==PACKAGE and sha(stage/'run_canaries.py')==CANARIES
 package=read(stage/'PACKAGE_SHA256.json')
 for n,h in package['files'].items():assert sha(stage/n)==h
 binding=read(stage/'BINDINGS.json');manifest=read(stage/'manifest.json');scope=read(stage/'full_scope.json')
 assert scope['rounds']==70 and scope['jobs']==manifest['jobs'] and len(scope['jobs'])==96
 selected=[e for e in scope['jobs'] if e['id'] in IDS]
 assert [e['id'] for e in selected]==IDS and not set(IDS)&{e['id'] for e in manifest['reused_jobs']}
 prior=[]
 if DELTA['accepted_before']:
  assert DELTA['prior_offserver_path']=='PRIOR_OFFSERVER.json'
  prior_path=Path(__file__).resolve().parent/DELTA['prior_offserver_path']
  assert sha(prior_path)==DELTA['prior_offserver_sha256']
  previous=read(prior_path);assert previous['status']=='PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON'
  prior=previous['accepted_ids'];assert len(prior)==DELTA['accepted_before'] and not set(prior)&set(IDS)
 else:assert DELTA['accepted_before']==0 and DELTA['prior_offserver_path'] is None and DELTA['prior_offserver_sha256'] is None
 assert all(not list((stage/e['output']).rglob('failure*.json')) for e in selected)
 gate=read(stage/'GATE_ACCEPTANCE.json');assert gate['package_sha256']==PACKAGE
 assert gate['status']=='SEVEN_HYBRID_CANARIES_STRICT_PASS_BACKUP_PENDING'
 assert server_runtime['torch']=='2.11.0+cu128' and server_runtime['cuda_build']=='12.8' and server_runtime['threads']==1
 assert server_runtime['gpu_uuid']==binding['gpu_uuid'] and server_runtime['gpu_index']==0 and server_runtime['gpu_name_for_original_checker']=='NVIDIA GeForce RTX 5090'
 sys.path.insert(0,str(stage))
 import identity,runtime,torch
 assert Path(identity.__file__).resolve()==stage/'identity.py' and Path(runtime.__file__).resolve()==stage/'runtime.py'
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 if not offserver:assert torch.__version__==server_runtime['torch'] and torch.version.cuda==server_runtime['cuda_build']
 protocol=read(stage/'ORIGINAL_PROTOCOL.json')
 identity.validate_grid([read(stage/e['job']) for e in manifest['jobs']],[read(stage/e['job']) for e in manifest['preflight_jobs']],manifest['reused_jobs'])
 for e in selected:assert sha(stage/e['job'])==e['job_sha256'];identity.validate_job(read(stage/e['job']),protocol,binding['selected_recipe'])
 bindings=dict(binding)
 if legacy_source is not None:bindings['original_screen']=str(Path(legacy_source).resolve())
 old=Path(bindings['original_screen']);assert sha(old/'driver.py')==DRIVER
 body,factory=runtime.functions(stage,Path(binding['repo']),bindings)
 # Existing seed identity bridge plus the same three host-record queries as the accepted screen bridge.
 text=record_checker((stage/'body.py').read_text('utf8'))
 ns=dict(body.__dict__,server_runtime=server_runtime);exec(compile(text,str(stage/'body.py')+':SAVED_RECORD_HOST_IDENTITY','exec'),ns);body.checked=ns['checked']
 scope=dict(scope,runtime_cuda_visible_device='0',runtime_gpu_uuid=binding['gpu_uuid'])
 _,checked,compare=factory(body,scope)
 records=[]
 for entry in selected:
  result=checked(entry,scope);assert result is not None
  prov=read(stage/entry['output']/'provenance.json');assert prov['gpu_uuid']==server_runtime['gpu_uuid'] and prov['gpu_name']==server_runtime['gpu_name_for_original_checker']
  records.append(dict(id=entry['id'],metrics=result['metrics'],checkpoint_sha256=sha(stage/entry['output']/'model.pt'),acceptance_sha256=sha(stage/entry['output']/'acceptance.json')))
 assert not torch.cuda.is_initialized()
 return dict(status='PASS_SAVED_ORIGINAL_FULLCOVERAGE_RECORD',package_sha256=PACKAGE,original_checked_source_sha256=hashlib.sha256(ast.get_source_segment((stage/'body.py').read_text('utf8'),next(n for n in ast.parse((stage/'body.py').read_text('utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='checked')).encode()).hexdigest(),gate_sha256=sha(stage/'GATE_ACCEPTANCE.json'),accepted_new=len(IDS),accepted_new_ids=IDS,accepted_before=len(prior),accepted_ids=prior+IDS,rounds=70,records=records,formal_table_samples=len(IDS),reused_models_repacked=0,canary_models_repacked=0,CNN_calls=0,server_runtime=server_runtime,local_verification_runtime=dict(python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda),local_CUDA_initialized=False,runtime_statement='Original saved scientific/writer checker; three host queries bound to actual server receipt, no runtime equivalence claim.')
