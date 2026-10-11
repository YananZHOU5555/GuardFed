"""Install the reviewed saved-model CUDA800 caller and start only three canaries."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,subprocess
R=Path(__file__).resolve().parents[1]
D=R/'tmp/baseline_old800_runtime_root_20261011_attempt003'
P=R/'tmp/baseline_old800_cuda_formal_adapter_prepare_20261011'
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert not (D/'INSTALL_COMMAND.json').exists(),'Preserve first install attempt'
review_path=R/'tmp/baseline_old800_cuda_formal_independent_review_20261011/SOURCE_REVIEW.json'
caller_review_path=R/'tmp/baseline_old800_cuda_formal_independent_review_20261011/ROOT_CALLER_ATTEMPT003_REVIEW.json'
review=read(review_path)
caller_review=read(caller_review_path)
assert H(review_path.read_bytes())=='1522064167140cb6c8fb4a70f99fc33645ad00f891e62f58d8c7275f738a2e4c'
assert not review['blocking_api_findings'] and not caller_review['blocking_source']
assert review['status'].startswith('PASS_') and caller_review['status'].startswith('PASS_')
seal=read(P/'FILES_SHA256.json');prepared=read(D/'ROOT_PREPARATION.json')
payload={}
for n,pin in seal['files'].items():
    b=(P/n).read_bytes();assert H(b)==pin['sha256'] and len(b)==pin['bytes'],n
    payload['adapter/'+n]=base64.b64encode(b).decode()
payload['adapter/FILES_SHA256.json']=base64.b64encode((P/'FILES_SHA256.json').read_bytes()).decode()
for n,pin in prepared['caller_sources'].items():
    b=(D/n).read_bytes();assert H(b)==pin['sha256'] and len(b)==pin['bytes'],n
    payload[n]=base64.b64encode(b).decode()
assert review['adapter_sha256']==H((P/'adapter.py').read_bytes())
assert review['package_seal_sha256']==H((P/'FILES_SHA256.json').read_bytes())
assert caller_review['guard_sha256']==H((D/'guard.py').read_bytes())
assert caller_review['queue_sha256']==H((D/'run_queue.py').read_bytes())
assert caller_review['worker_sha256']==H((D/'worker.py').read_bytes())
assert caller_review['reuse_validator_sha256']==H((D/'reuse_fedavg.py').read_bytes())
remote=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess,sys,time
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def pin(p):return {'path':str(p),'sha256':sha(p)}
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def run(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=70)
 assert p.returncode==0,(argv,p.returncode,p.stdout,p.stderr)
 return p.stdout
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
payload=json.loads(base64.b64decode('__PAYLOAD__'))
prepared=json.loads(base64.b64decode('__PREPARED__'))
base=Path(prepared['namespace']);old=Path('/workspace/guardfed_checks/celeba_final_valid_replay_20261009')
repo=Path('/workspace/GuardFed-celeba-expanded')
py='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
service='guardfed_uniform_cuda800_canary_v3_valid'
conf=Path('/etc/supervisor/conf.d/'+service+'.conf')
assert not base.exists() and not conf.exists(),'Preserve prior namespace/service'
failed_base=Path('/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011')
assert sha(failed_base/'canary/QUEUE_FAILURE.json')=='58dc7930537b301135caa204021a3d05fac162d2bf2129b7a25f4f08efdfc27e'
failed_status=subprocess.run(['supervisorctl','status','guardfed_uniform_cuda800_canary_valid'],capture_output=True,text=True,timeout=20)
assert failed_status.stdout.split()[1]=='EXITED','No duplicate or restart of first failed queue'
assert not list((failed_base/'tasks').glob('*/ACCEPTED.json')) and not list((failed_base/'runs').glob('*/receipt.json'))
previous=Path('/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002')
assert sha(previous/'canary/QUEUE_FAILURE.json')=='6ecd9f3e90adc44713d976772ac6d05bb29d70ee4325a10b10356b1d03f94358'
assert sha(previous/'runs/FedAA-DDPG_policy0.001_keep16_local0.001_IID_Benign_seed91001.formal_failure.json')=='32500f3565b680041b1d0d8f7f479973b79c2a7e4446d2e4ad6680f46e47fbec'
previous_status=subprocess.run(['supervisorctl','status','guardfed_uniform_cuda800_canary_v2_valid'],capture_output=True,text=True,timeout=20)
assert previous_status.stdout.split()[1]=='EXITED','No duplicate producer from previous attempt'
assert len(list((previous/'tasks').glob('*/ACCEPTED.json')))==1
assert not (previous/'runs/FedAA-DDPG_policy0.001_keep16_local0.001_IID_Benign_seed91001/receipt.json').exists()
reused_cuda={'FedAvg_IID_Benign_seed91001':{
 'task_accepted':{'path':str(previous/'tasks/FedAvg_IID_Benign_seed91001/ACCEPTED.json'),'sha256':'a0507114f5c5de34fcfcff6527df6cf3032f5126755771ac8fe410d346d0f2b1'},
 'deployment_seal':{'path':str(previous/'DEPLOYMENT_SEAL.json'),'sha256':'c411816b1f73c47e400f3cbc1adbd6e52e89975f360f131429f5279cddf70a4f'},
 'queue_auth':{'path':str(previous/'ROOT_QUEUE_AUTH.json'),'sha256':'c5684f028a11f59d8d7eab3dd229c91e22a0aa329f70441e453d526720b46270'}}}

assert not ({Path(n).stem for n in payload if '/' not in n and n.endswith('.py')} & sys.stdlib_module_names),'No stdlib module shadowing'
pins=json.loads(base64.b64decode(payload['adapter/SOURCE_PINS.json']))
runtime={k:{'path':str(old/suffix),'sha256':pins[k]['sha256']} for k,suffix in {
 'v2':'replay.py','v3':'v3/replay_v3.py','v4':'v4/replay_v4.py',
 'evaluator':'inputs/evaluator.py','semantic_inspection':'v4/semantic900_inspection.json'}.items()}
inputs={k:{'path':str(old/suffix),'sha256':pins[role]['sha256']} for k,suffix,role in [
 ('inventory','inputs/model_inventory.json','baseline900_inventory'),
 ('storage_map','v3/inputs/storage_map.json','baseline_storage_map'),
 ('restore_receipt','v3/inputs/restore_bundle.json','baseline_restore_receipt'),
 ('restore_acceptance','v3/inputs/restore_acceptance.json','baseline_restore_acceptance')]}
for item in list(runtime.values())+list(inputs.values()):assert sha(item['path'])==item['sha256'],item['path']
original=read(inputs['inventory']['path'])['records'];rows=[r for r in original if r['method']!='GuardFed-AD2+']
assert len(original)==900 and len(rows)==800 and len({r['id'] for r in rows})==800
cohort={'status':'ROOT_FROZEN_ORIGINAL800_INPUT_OBJECTS_NOT_CUDA_RESULTS',
 'selection_basis':'all_original900_records_except_GuardFed-AD2+','outcome_independent':True,
 'CUDA_results':None,'Full_excluded_and_reused_separately':True,
 'original_inventory_sha256':inputs['inventory']['sha256'],
 'ordered_ids':[r['id'] for r in rows],'records':rows}
cohort_bytes=(json.dumps(cohort,ensure_ascii=False,indent=2)+'\n').encode()
assert hashlib.sha256(cohort_bytes).hexdigest()==prepared['cohort_sha256'] and len(cohort_bytes)==prepared['cohort_bytes']
author=Path('/workspace/guardfed_checks/uniform_cuda_formal_20261011_resume1/author.json')
assert sha(author)==pins['author_decision']['sha256']
assert read(author)['approved']['uniform_CUDA_evaluation']
base.mkdir()
for rel,b64 in payload.items():
 p=base/rel;assert '..' not in Path(rel).parts and p.is_relative_to(base)
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(b64))
with (base/'EXACT800_INPUTS.json').open('xb') as f:f.write(cohort_bytes)
with (base/'author.json').open('xb') as f:f.write(author.read_bytes())
deadline=time.time()+48*3600
gpu='GPU-9a424316-0f5c-5e05-0291-a4ee24bed9ae'
protected={'guardfed_celeba_mechanism_formal':['RUNNING','EXITED'],
 'guardfed_celeba_hybrid_fullcoverage':['RUNNING','EXITED'],
 'guardfed_celeba_fednga_fullcoverage_v1':['RUNNING','EXITED'],
 'guardfed_uniform_cuda500_resume1_valid':['RUNNING','EXITED'],'sglang':['STOPPED']}
approval=read(base/'adapter/ROOT_APPROVAL.template.json')
approval.update(status='ROOT_AUTHORIZED_UNIFORM_CUDA_ORIGINAL800_FORMAL_V1',execution_authorized=True,
 author_decision=pin(base/'author.json'),exact_cohort=pin(base/'EXACT800_INPUTS.json'),
 adapter=pin(base/'adapter/adapter.py'),adapter_sha256=sha(base/'adapter/adapter.py'),
 source_seal=pin(base/'adapter/FILES_SHA256.json'),runtime_dependencies=runtime,
 original_inputs=dict(inputs,repo=str(repo)),formal_GPU_body=pin(base/'adapter/gpu_body.py'),
 resource_gate_source=pin(base/'guard.py'),gpu_uuid=gpu,allowed_cpus=[109],
 output_namespace=str(base),protected_services=protected)
auth={'execution_authorized':True,'stage':'ORIGINAL800_CUDA_REEVALUATION_CANARY_FIRST',
 'ordered_ids':cohort['ordered_ids'],'canary_ids':prepared['canary_ids'],
 'exact_cohort':pin(base/'EXACT800_INPUTS.json'),'adapter':pin(base/'adapter/adapter.py'),
 'gpu_uuid':gpu,'allowed_cpus':[109],'deadline_unix':deadline,'protected_services':protected,
 'approval_base':approval,'training':False,'test':False,'automatic_retry':False,
 'source_review_sha256':'__API_REVIEW_SHA__','caller_review_sha256':'__CALLER_REVIEW_SHA__',
 'reused_cuda':reused_cuda,'recovery_from':str(previous),'recovery_new_ids':prepared['new_canary_ids']}
save(base/'ROOT_QUEUE_AUTH.json',auth)
deployment={'status':'ROOT_DEPLOYED_CODE_AND_COHORT_NOT_SCIENTIFIC_RESULTS',
 'files':{rel:{'sha256':sha(base/rel),'bytes':(base/rel).stat().st_size} for rel in
          list(payload)+['EXACT800_INPUTS.json','author.json','ROOT_QUEUE_AUTH.json']},
 'external_code_pins':{item['path']:item['sha256'] for item in list(runtime.values())+list(inputs.values())}}
save(base/'DEPLOYMENT_SEAL.json',deployment)
sys.path.insert(0,str(base))
from reuse_fedavg import verify_reused
verified=verify_reused(auth,cohort)
assert list(verified)==prepared['reused_canary_ids']
first=prepared['new_canary_ids'][0]
preflight=run(['nice','-n','10','ionice','-c','3','taskset','-c','109',py,str(base/'guard.py'),'--id',first])
fresh=json.loads(preflight);assert fresh['pass'] and fresh['gpu_uuid']==gpu and fresh['allowed_cpus']==[109]
save(base/'ROOT_INSTALL_PREFLIGHT.json',fresh)
text='[program:'+service+']\ncommand=/usr/bin/nice -n 10 /usr/bin/ionice -c 3 '+py+' '+str(base/'run_queue.py')+' canary\ndirectory='+str(base)+'\nautostart=false\nautorestart=false\nstartsecs=1\nstartretries=0\nstopasgroup=true\nkillasgroup=true\nstdout_logfile='+str(base/'canary.supervisor.stdout')+'\nstderr_logfile='+str(base/'canary.supervisor.stderr')+'\n'
with conf.open('x',encoding='utf-8') as f:f.write(text)
commands=[]
for argv in [['supervisorctl','reread'],['supervisorctl','update',service],['supervisorctl','start',service],['supervisorctl','status',service]]:
 commands.append({'command':argv,'stdout':run(argv)})
report={'status':'ROOT_INSTALLED_RECOVERY_NEW2_REUSED1_NOT800_COMPLETE',
 'utc':datetime.now(timezone.utc).isoformat(),'namespace':str(base),'service':service,
 'canary_ids':prepared['canary_ids'],'new_ids':prepared['new_canary_ids'],'reused_ids':list(verified),
 'cohort':pin(base/'EXACT800_INPUTS.json'),
 'auth':pin(base/'ROOT_QUEUE_AUTH.json'),'seal':pin(base/'DEPLOYMENT_SEAL.json'),
 'preflight':pin(base/'ROOT_INSTALL_PREFLIGHT.json'),'commands':commands,
 'scientific_results_adopted':0,'Full_rerun':False,'remaining797_started':False}
save(base/'ROOT_START.json',report)
print(json.dumps(report,allow_nan=False))
'''
remote=remote.replace('__PAYLOAD__',base64.b64encode(json.dumps(payload).encode()).decode())
remote=remote.replace('__PREPARED__',base64.b64encode(json.dumps(prepared).encode()).decode())
remote=remote.replace('__API_REVIEW_SHA__',H(review_path.read_bytes()))
remote=remote.replace('__CALLER_REVIEW_SHA__',H(caller_review_path.read_bytes()))
compile(remote,'<old800-fixed3-install>','exec')
result=subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20',
 'root@89.22.197.55','python3 -'],input=remote.encode(),capture_output=True,timeout=120)
(D/'INSTALL.stdout').write_bytes(result.stdout);(D/'INSTALL.stderr').write_bytes(result.stderr)
record={'utc':datetime.now(timezone.utc).isoformat(),'exit_code':result.returncode,
 'remote_script_sha256':H(remote.encode()),'installer_sha256':H(Path(__file__).read_bytes()),
 'stdout_sha256':H(result.stdout),'stderr_sha256':H(result.stderr)}
(D/'INSTALL_COMMAND.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
assert result.returncode==0,result.stderr.decode(errors='replace')
print(result.stdout.decode())
