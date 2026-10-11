"""Compile and exercise metadata bundle construction in memory; no remote/file deployment."""
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,copy,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;src=HERE/'deploy_caller_once.py'
compile(src.read_bytes(),str(src),'exec')
spec=importlib.util.spec_from_file_location('logo_metadata_installer_fixture',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
compile(m.REMOTE,'<remote metadata only>','exec')
realread=m.read;realsha=m.sha
fresh=realread(HERE.parent/'REPAIR_FRESH.stdout');fresh['utc']=datetime.now(timezone.utc).isoformat()
review={'source_adoptable':True,'installer_source_sha256':realsha(src)}
m.read=lambda p: copy.deepcopy(fresh) if str(p)=='SYNTHETIC_FRESH' else copy.deepcopy(review) if str(p)=='SYNTHETIC_REVIEW' else realread(p)
m.sha=lambda p:'0'*64 if str(p).startswith('SYNTHETIC_') else realsha(p)
names=['guardfed_celeba_fednga_fullcoverage_v1','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_mechanism_formal','guardfed_uniform_cuda500_resume2_valid','guardfed_uniform_cuda800_remaining_v3_valid','guardfed_uniform_cuda_FL100_remaining_valid','sglang']
seen={line.split()[0]:line.split()[1] for line in fresh['services']['stdout'].splitlines() if len(line.split())>1}
inputs={'execution_authorized':True,'root_restoration_authorized':True,'authorize_fixed2_queue':False,'installer_source_sha256':realsha(src),'installer_review':{'path':'SYNTHETIC_REVIEW','sha256':'0'*64},'fresh_resource':{'path':'SYNTHETIC_FRESH','sha256':'0'*64},'protected_services':{n:['STOPPED'] if n=='sglang' else ['RUNNING','EXITED'] for n in names},'protected_services_observed':{n:seen[n] for n in names},'deadline_unix':datetime.now(timezone.utc).timestamp()+3600}
raw=m.blob(inputs);bundle=m.prepare(inputs,hashlib.sha256(raw).hexdigest(),raw)
restore=json.loads(base64.b64decode(bundle['files']['RESTORE_INDEX.json']))
assert len(restore['records'])==100 and all(len(r['artifacts'])==7 and len(r['runtime_dependencies'])==4 and len(r['original_source_members'])==2 for r in restore['records'])
assert bundle['queue']['execution_authorized'] is False
assert bundle['queue']['protected_services_observed']=={n:seen[n] for n in names}
# Build a deterministic synthetic service snapshot instead of depending on column spacing.
fresh['services']['stdout']='\n'.join(n+' '+('EXITED' if n=='guardfed_uniform_cuda_FL100_remaining_valid' else seen[n]) for n in names)
life=copy.deepcopy(inputs);life['protected_services_observed']['guardfed_uniform_cuda_FL100_remaining_valid']='EXITED'
life_raw=m.blob(life);life_bundle=m.prepare(life,hashlib.sha256(life_raw).hexdigest(),life_raw)
assert life_bundle['queue']['protected_services']['guardfed_uniform_cuda_FL100_remaining_valid']==['RUNNING','EXITED']
assert life_bundle['queue']['protected_services_observed']['guardfed_uniform_cuda_FL100_remaining_valid']=='EXITED'
fresh['services']['stdout']='\n'.join(n+' '+seen[n] for n in names)
for label,change in [('unobserved',lambda x:x['protected_services_observed'].update(sglang='RUNNING')),('sglang_relaxed',lambda x:x['protected_services'].update(sglang=['STOPPED','RUNNING'])),('project_relaxed',lambda x:x['protected_services'].update(guardfed_uniform_cuda_FL100_remaining_valid=['RUNNING','EXITED','FATAL']))]:
    negative=copy.deepcopy(inputs);change(negative);nr=m.blob(negative)
    try:m.prepare(negative,hashlib.sha256(nr).hexdigest(),nr)
    except ValueError:pass
    else:raise AssertionError('Rejected lifecycle fixture accepted: '+label)
assert base64.b64decode(bundle['files']['ROOT_INSTALL_INPUTS.json'])==raw
assert bundle['queue']['python']['sha256']=='2e66193c1a7b09e0d8fa1b72dcff95132dc0e85ac16c51347dc7d294a63758d9'
assert bundle['queue']['FedAvg_CUDA_index']['sha256']=='6a21265f08fb26da130d95c67bf4130039b25916f44a543cd695e15665a25b64'
assert not any(Path(p).suffix in {'.pt','.pth','.safetensors','.jpg','.png'} for p in bundle['external_pins'])
for n in ['guard.py','worker.py','run_queue.py','bindings.py']:
    assert base64.b64decode(bundle['files'][n])==(m.ROOT/'tmp/LoGo100_cuda_cache_runtime_prepare_20261011'/n).read_bytes()
for n in ['adapter.py','original_functions.py.txt']:
    assert base64.b64decode(bundle['files']['adapter/'+n])==(m.ROOT/'tmp/LoGo100_uniform_cuda_cache_adapter_prepare_20261011'/n).read_bytes()
bad=dict(inputs,execution_authorized=False);b=m.blob(bad)
try:m.prepare(bad,hashlib.sha256(b).hexdigest(),b)
except ValueError:pass
else:raise AssertionError('Template incorrectly allowed')
tree=ast.parse(m.REMOTE)
literalwrites=[ast.literal_eval(n.args[0]) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write' and n.args and isinstance(n.args[0],ast.Constant)]
assert '\n' in literalwrites and all(v!='\\n' for v in literalwrites)
report={'status':'PASS_METADATA_SOURCE_ONLY_SYNTHETIC_FIXTURE','source_sha256':realsha(src),'remote_source_compiles':True,'source_members_in_memory':len(bundle['files']),'needed_external_pins_in_memory':len(bundle['external_pins']),'original_guard_worker_queue_adapter_bytes_exact':True,'original_order_restoration_records':100,'each_record_artifacts':7,'each_record_runtime_roles':4,'each_record_official_sources':2,'template_rejected':True,'normal_project_EXITED_accepted':True,'observed_and_policy_separate':True,'lifecycle_negative_fixtures':3,'queue_not_authorized_fixture':True,'fresh_and_review_fixture':'Synthetic in memory only; not saved as runtime proofs','SSH_F_arrays_models_scientific_execution':False,'actual_deployment':None}
(HERE/'SOURCE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))
