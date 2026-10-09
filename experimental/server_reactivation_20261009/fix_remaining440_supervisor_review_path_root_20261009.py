"""Repair a pre-contract supervisor filename mismatch; preserve original bytes."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'tmp/celeba_valid_gpu_remaining440_resource_gate_v2_execution_20261009'
PROGRAM='guardfed_celeba_valid_gpu_remaining440_resource_gate_v2_20261009'
config=LOCAL/(PROGRAM+'.conf')
original=config.read_bytes()
old=b'ROOT_REVIEW_REMAINING440_RESOURCE_GUARD_V2.json'
new=b'ROOT_REVIEW_REMAINING440_RESOURCE_GATE_V2.json'
assert original.count(old)==1 and new not in original
corrected=original.replace(old,new)
sha=lambda b:hashlib.sha256(b).hexdigest()
values=dict(PROGRAM=PROGRAM,ORIGINAL=original,CORRECTED=corrected,
            REVIEW_SHA='f8b337ddabc225ec3e313e686d19efa5999a98fca419adc42113fb4059557d13',
            GUIDE='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa')
code="""from pathlib import Path
import datetime,hashlib,json,subprocess
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009')
config=Path('/etc/supervisor/conf.d')/(PROGRAM+'.conf')
out=base/'remaining440_resource_gate_v2_attempt1'
log=base/'remaining440_resource_gate_v2_supervisor.log'
assert sha(Path('/etc/vast-agents-guide.md'))==GUIDE
assert config.read_bytes()==ORIGINAL and not out.exists()
assert sha(base/'ROOT_REVIEW_REMAINING440_RESOURCE_GATE_V2.json')==REVIEW_SHA
status=subprocess.run(['supervisorctl','status',PROGRAM],capture_output=True,text=True)
original_log=log.read_bytes()
assert status.returncode==3 and ('FATAL' in status.stdout or 'EXITED' in status.stdout)
assert b'FileNotFoundError' in original_log and b'ROOT_REVIEW_REMAINING440_RESOURCE_GUARD_V2.json' in original_log
assert b'/remaining.py' in original_log and b'recovery.py' not in original_log
saved=base/'remaining440_v2_precontract_config_mismatch'
assert not saved.exists();saved.mkdir()
for name,data in [('supervisor.conf.original',ORIGINAL),('supervisor.log.original',original_log)]:
    with (saved/name).open('xb') as stream:stream.write(data)
before={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'service':status.stdout,'original_log':original_log.decode(errors='replace'),'output_absent':True,'CNN_started':False,'original_config_sha256':hashlib.sha256(ORIGINAL).hexdigest(),'corrected_config_sha256':hashlib.sha256(CORRECTED).hexdigest()}
with (saved/'failure.json').open('x') as stream:json.dump(before,stream,indent=2)
config.write_bytes(CORRECTED)
logs=[]
for command in [['supervisorctl','reread'],['supervisorctl','update',PROGRAM],['supervisorctl','start',PROGRAM],['supervisorctl','status',PROGRAM]]:
    result=subprocess.run(command,capture_output=True,text=True)
    logs.append({'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    if result.returncode:break
print(json.dumps({'status':'PRECONTRACT_FILENAME_FIX_TARGETED_START_CHECK','before':before,'commands':logs,'review_sha256':REVIEW_SHA,'scientific_changes':False,'old464_restarted':False,'new_accepted':0}))
"""
source='\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+code
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -c '+shlex.quote(source)],capture_output=True,timeout=60)
dest=LOCAL/'ROOT_PRECONTRACT_CONFIG_FIX_AND_LAUNCH.json'
with dest.open('xb') as stream:stream.write(result.stdout)
(LOCAL/'ROOT_PRECONTRACT_CONFIG_FIX_SSH_STDERR.log').write_bytes(result.stderr)
result.check_returncode();proof=json.loads(result.stdout)
assert len(proof['commands'])==4 and all(row['returncode']==0 for row in proof['commands']) and 'RUNNING' in proof['commands'][-1]['stdout']
with (LOCAL/(PROGRAM+'.conf.corrected')).open('xb') as stream:stream.write(corrected)
print(json.dumps({'status':'CORRECTED_CONFIG_TARGETED_START_OBSERVED','proof':str(dest),'sha256':sha(result.stdout),'config_sha256':sha(corrected),'new_accepted':0}))
