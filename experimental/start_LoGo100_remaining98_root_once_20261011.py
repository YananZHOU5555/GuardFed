"""Bind actual fixed2 root adoption, then start the unchanged original remaining98 once."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,subprocess,sys,time
R=Path(__file__).resolve().parents[1]
D=R/'tmp/LoGo100_remaining98_root_start_prepare_20261011'
A=R/'tmp/LoGo_cuda_fixed2_saved_audit_prepare_v2_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
assert sha(D/'FILES_SHA256.json')=='1425ad02a6f024d02e5e5bfa3f904119076a8ceb6254290bf4f77439e2c76015'
assert sha(D/'ROOT_SOURCE_REVIEW.json')=='fb0d99f87d011317c10a7bd95c8f50f0cbb7c0f524954ea71ca39ef47b7eed78'
for n,row in read(D/'FILES_SHA256.json')['files'].items():
    assert Path(n).name==n and sha(D/n)==row['sha256'] and (D/n).stat().st_size==row['bytes']
    if n.endswith('.py'):compile((D/n).read_bytes(),str(D/n),'exec')
adopt=read(A)
assert adopt['root_accepted'] is True and adopt['accepted']==2 and adopt['archive_all_members_rehashed']
assert (adopt['metric_values_checked'],adopt['integer_confusion_counts_recomputed'],adopt['prediction_items_checked'])==(6,16,39734)
assert adopt['new_fit']==adopt['new_CNN']==0 and not adopt['test'] and adopt['max_metric_error']<=1e-12
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
assert observed['guardfed_uniform_cuda_LoGo100_canary_valid']=='EXITED'
names=list(read(R/'tmp/LoGo100_cuda_actual_root_20261011/ROOT_INSTALL_INPUTS.json')['protected_services'])
observed={n:observed[n] for n in names}
assert observed['sglang']=='STOPPED' and all(s in ('RUNNING','EXITED') for n,s in observed.items() if n!='sglang')
binding=read(D/'ROOT_START_INPUTS.template.json')
binding.update(execution_authorized=True,issued_unix=time.time(),protected_services_observed=observed,
    root_review='Actual fixed2 original Linux whole plus offserver zero-fit audit and F hashes adopted; original metadata guard before supervisor writes; original remaining98 order and frozen worker unchanged.')
input_path=D/'ROOT_START_INPUTS.json';write(input_path,binding)
bound=D/'ROOT_BOUND_START_INPUTS.json'
cmd=[sys.executable,'-B',str(D/'bind_gate.py'),'--root-inputs',str(input_path),'--root-inputs-sha256',sha(input_path),
    '--root-adoption',str(A),'--root-adoption-sha256',sha(A),'--output',str(bound)]
r=subprocess.run(cmd,cwd=R,capture_output=True,timeout=30)
write(D/'ROOT_BIND_COMMAND.json',{'command':cmd,'exit_code':r.returncode,'stdout':r.stdout.decode(errors='replace'),'stderr':r.stderr.decode(errors='replace')})
assert r.returncode==0,r.stderr.decode(errors='replace')
cmd=[sys.executable,'-B',str(D/'run_root.py'),'start','--output',str(D/'start001'),
    '--root-inputs',str(bound),'--root-inputs-sha256',sha(bound)]
r=subprocess.run(cmd,cwd=R,capture_output=True,timeout=280)
write(D/'ROOT_REMAINING98_START_COMMAND.json',{'command':cmd,'exit_code':r.returncode,'utc':datetime.now(timezone.utc).isoformat(),
    'stdout':r.stdout.decode(errors='replace'),'stderr':r.stderr.decode(errors='replace'),
    'original_source_seal_sha256':sha(D/'FILES_SHA256.json'),'root_bound_inputs_sha256':sha(bound),
    'fixed2_root':{'path':str(A),'sha256':sha(A)},'scientific_acceptance_created':0,'remaining_scope':98})
assert r.returncode==0,r.stderr.decode(errors='replace')
print(json.dumps({'status':'ORIGINAL_REMAINING98_START_COMMAND_SUCCEEDED_NOT_ACCEPTANCE',
    'command_record':str(D/'ROOT_REMAINING98_START_COMMAND.json'),'sha256':sha(D/'ROOT_REMAINING98_START_COMMAND.json')}))
