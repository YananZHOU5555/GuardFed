"""Bind actual metadata prerequisites and deploy the unchanged LoGo caller once."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shlex, subprocess, sys, time
R=Path(__file__).resolve().parents[1]
D=R/'tmp/LoGo100_cuda_actual_root_20261011'
S=R/'tmp/LoGo100_cuda_actual_runtime_prerequisites_20261011/caller_installer_v2_source'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def pin(p):return {'path':str(p),'sha256':sha(p)}
def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
p=argparse.ArgumentParser();p.add_argument('--source-seal',required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--review-sha256',required=True);a=p.parse_args()
installer=S/'deploy_caller_once.py'
assert sha(installer)=='7325c376aa97f03ac728a3a6fece1c508b1c7ed58bb7858fbb5fbe22eef4fe7a'
assert sha(S/'FILES_SHA256.json')==a.source_seal
for n,row in read(S/'FILES_SHA256.json')['files'].items():
    assert Path(n).name==n and sha(S/n)==row['sha256'] and (S/n).stat().st_size==row['bytes']
assert a.review.resolve().is_relative_to(R.resolve()) and sha(a.review)==a.review_sha256
review=read(a.review)
assert review['source_adoptable'] is True and review['installer_source_sha256']==sha(installer)
sys.path.insert(0,str(R/'tmp'))
from guardfed_local_storage import check_bulk_storage
volume=check_bulk_storage(250*1024**2)
probe=R/'tmp/LoGo100_cuda_actual_prerequisites_20261011/read_remote_prerequisites.py'
assert sha(probe)=='f9489fe9511ba415c4c1b1a4379719ccdf42a5382d60ecf657d52fd967cb4d9f'
D.mkdir(exist_ok=True)
assert not (D/'ROOT_INSTALL_INPUTS.json').exists()
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',
     'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ionice -c 3 python3 -B -']
r=subprocess.run(cmd,input=probe.read_bytes(),capture_output=True,timeout=120)
for n,b in [('RESOURCE_STDOUT',r.stdout),('RESOURCE_STDERR',r.stderr)]:
    with (D/n).open('xb') as f:f.write(b)
write(D/'RESOURCE_COMMAND.json',{'command':cmd,'exit_code':r.returncode,'source':pin(probe),'utc':datetime.now(timezone.utc).isoformat()})
assert r.returncode==0,r.stderr.decode(errors='replace')
fresh=json.loads(r.stdout);write(D/'ROOT_READONLY_RESOURCE_OBSERVATION.json',fresh)
cpu=fresh['cpu113']
assert cpu['candidate_available_observation_only'] and not cpu['unreadable_live_processes'] and not cpu['conflicting_restricted_tasks']
names=['guardfed_celeba_fednga_fullcoverage_v1','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_mechanism_formal','guardfed_uniform_cuda500_resume2_valid','guardfed_uniform_cuda800_remaining_v3_valid','guardfed_uniform_cuda_FL100_remaining_valid','sglang']
observed={line.split()[0]:line.split()[1] for line in fresh['services']['stdout'].splitlines() if len(line.split())>1}
observed={n:observed[n] for n in names}
assert observed['sglang']=='STOPPED' and all(observed[n] in ('RUNNING','EXITED') for n in names[:-1])
allowed={n:['RUNNING','EXITED'] for n in names[:-1]};allowed['sglang']=['STOPPED']
inputs=read(S/'ROOT_INPUTS.template.json')
inputs.update(execution_authorized=True,root_restoration_authorized=True,authorize_fixed2_queue=True,
    installer_source_sha256=sha(installer),installer_review=pin(a.review),
    fresh_resource=pin(D/'ROOT_READONLY_RESOURCE_OBSERVATION.json'),
    protected_services=allowed,protected_services_observed=observed,deadline_unix=time.time()+86400,
    root_decision='User authorizes complete CUDA reevaluation; restore exact artifacts and original code, then fixed2 only. Remaining98 requires separate adopted canary gate.',
    F_volume=volume,actual_deployment=None)
input_path=D/'ROOT_INSTALL_INPUTS.json';write(input_path,inputs)
cmd=[sys.executable,'-B',str(installer),'--root-inputs',str(input_path),'--root-inputs-sha256',sha(input_path),'--execute']
r=subprocess.run(cmd,cwd=R,capture_output=True,timeout=650)
for n,b in [('INSTALL_STDOUT',r.stdout),('INSTALL_STDERR',r.stderr)]:
    with (D/n).open('xb') as f:f.write(b)
write(D/'INSTALL_COMMAND.json',{'command':cmd,'exit_code':r.returncode,'utc':datetime.now(timezone.utc).isoformat(),'stdout':pin(D/'INSTALL_STDOUT'),'stderr':pin(D/'INSTALL_STDERR')})
assert r.returncode==0,r.stderr.decode(errors='replace')
deployed=json.loads((S/'DEPLOY.stdout').read_bytes())
assert deployed['status']=='LOGO_CALLER_SOURCE_AND_NEEDED_DEPENDENCIES_DEPLOYED_NO_START' and deployed['science_started'] is False
write(D/'ROOT_DEPLOYMENT_RECORD.json',{'status':'ROOT_EXACT_LOGO100_CALLER_DEPLOYMENT_REGISTERED_NO_SCIENCE_START',
    'deployment':deployed,'installer':pin(installer),'review':pin(a.review),'root_inputs':pin(input_path),
    'restore_registration_only':True,'scientific_adopted':0,'canary_started':False,'test':False})
print(json.dumps({'record':pin(D/'ROOT_DEPLOYMENT_RECORD.json'),'deployment':deployed},ensure_ascii=False))
