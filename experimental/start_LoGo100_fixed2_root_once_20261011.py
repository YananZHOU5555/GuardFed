from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,subprocess,sys,time
R=Path(__file__).resolve().parents[1];D=R/'tmp/LoGo100_cuda_actual_root_20261011'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
assert sha(D/'FILES_SHA256.json')=='940291f7860463c2484b5e813693eee1f621b97ce117e6d122777d72f32e3263'
for n,row in read(D/'FILES_SHA256.json')['files'].items():
    assert Path(n).name==n and sha(D/n)==row['sha256'] and (D/n).stat().st_size==row['bytes']
    if n.endswith('.py'):compile((D/n).read_bytes(),str(D/n),'exec')
deploy=read(D/'ROOT_DEPLOYMENT_RECORD.json')['deployment']
assert deploy['science_started'] is False and deploy['hashed_needed_dependencies']==1366
code="""from pathlib import Path
import hashlib,json,subprocess
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
r=subprocess.run(['supervisorctl','status'],capture_output=True,text=True,timeout=20)
assert r.returncode in (0,3)
print(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
"""
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55','python3 -B -']
r=subprocess.run(cmd,input=code.encode(),capture_output=True,timeout=45)
write(D/'START_FRESH_PROTECTED.json',{'utc':datetime.now(timezone.utc).isoformat(),'command':cmd,
    'exit_code':r.returncode,'stdout':r.stdout.decode(errors='replace'),'stderr':r.stderr.decode(errors='replace'),
    'source_sha256':hashlib.sha256(code.encode()).hexdigest()})
assert r.returncode==0,r.stderr.decode(errors='replace')
observed={line.split()[0]:line.split()[1] for line in json.loads(r.stdout)['stdout'].splitlines() if len(line.split())>1}
names=list(read(D/'ROOT_INSTALL_INPUTS.json')['protected_services'])
observed={n:observed[n] for n in names}
assert observed['sglang']=='STOPPED' and all(s in ('RUNNING','EXITED') for n,s in observed.items() if n!='sglang')
binding=read(D/'ROOT_START_INPUTS.template.json')
binding.update(execution_authorized=True,issued_unix=time.time(),
    deployment_seal_sha256=deploy['deployment_seal_sha256'],queue_auth_sha256=deploy['queue_auth_sha256'],
    protected_services_observed=observed,
    root_review='Original metadata guard runs before supervisor writes; no duplicate namespace, no autorestart, fixed2 only, broad manager and CPU113 hidden-GPU child. Root inspected complete sources and verified frozen source seal.')
input_path=D/'ROOT_START_INPUTS.json';write(input_path,binding)
cmd=[sys.executable,'-B',str(D/'run_root.py'),'start','--output',str(D/'start001'),
    '--root-inputs',str(input_path),'--root-inputs-sha256',sha(input_path)]
r=subprocess.run(cmd,cwd=R,capture_output=True,timeout=280)
write(D/'ROOT_START_COMMAND.json',{'command':cmd,'exit_code':r.returncode,'utc':datetime.now(timezone.utc).isoformat(),
    'stdout':r.stdout.decode(errors='replace'),'stderr':r.stderr.decode(errors='replace'),
    'original_source_seal_sha256':sha(D/'FILES_SHA256.json'),'root_start_inputs_sha256':sha(input_path),
    'scientific_acceptance_created':0,'canary_scope':2,'remaining98_authorized_here':False})
assert r.returncode==0,r.stderr.decode(errors='replace')
print(json.dumps({'status':'ORIGINAL_FIXED2_CANARY_START_COMMAND_SUCCEEDED_NOT_ACCEPTANCE',
    'command_record':str(D/'ROOT_START_COMMAND.json'),'sha256':sha(D/'ROOT_START_COMMAND.json')}))
