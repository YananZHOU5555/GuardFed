"""Root-pinned canary gate then the unchanged remaining97 finite queue."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,subprocess
R=Path(__file__).resolve().parents[1]
D=R/'tmp/FL100_cuda_actual_root_20261011'
proof=R/'tmp/FL100_cuda3_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(proof)=='e6db56a24eda9ea9840caba54958e79ee77bb427a4b11804e0b03b2462f8c905'
v=json.loads(proof.read_bytes());assert v['accepted']==3 and v['archive_all_members_rehashed']
gate={'root_accepted':True,'ids':v['ids'],'cohort':{'path':'/workspace/guardfed_checks/uniform_cuda_FL100_20261011_canary1/adapter/EXACT100.json','sha256':v['cohort_sha256']},
    'records':v['records'],'root_offserver_adoption':{'path':str(proof),'sha256':sha(proof)},
    'archive_sha256':v['archive']['sha256'],'scientific_scope':'Fixed3 only; continue exact remaining97, no new training/test'}
gate_bytes=(json.dumps(gate,indent=2,allow_nan=False)+'\n').encode()
gate_sha=hashlib.sha256(gate_bytes).hexdigest()
assert not (D/'REMAINING97_START_COMMAND.json').exists()
remote=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,subprocess,sys
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
B=Path('/workspace/guardfed_checks/uniform_cuda_FL100_20261011_canary1')
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(B/'DEPLOYMENT_SEAL.json')=='1e2384f2e87caafb56cf047a2dc6f8037621a246472d86d8d265195bde016a7d'
old=subprocess.run(['supervisorctl','status','guardfed_uniform_cuda_FL100_canary_valid'],capture_output=True,text=True,timeout=20)
assert old.stdout.split()[:2]==['guardfed_uniform_cuda_FL100_canary_valid','EXITED']
assert not (B/'canary/QUEUE_FAILURE.json').exists() and not (B/'remaining').exists()
raw=base64.b64decode('__GATE__');assert hashlib.sha256(raw).hexdigest()=='__GATE_SHA__'
gate=json.loads(raw);complete=json.loads((B/'canary/COMPLETE.json').read_bytes())
auth=json.loads((B/'ROOT_QUEUE_AUTH.json').read_bytes())
assert gate['ids']==complete['ids']==auth['canary_ids'] and gate['cohort']==auth['exact_cohort']
for identity,row in gate['records'].items():assert sha(row['acceptance']['path'])==row['acceptance']['sha256']
service='guardfed_uniform_cuda_FL100_remaining_valid';conf=Path('/etc/supervisor/conf.d/'+service+'.conf')
assert not conf.exists() and not (B/'ROOT_CANARY_ACCEPTANCE.json').exists()
py='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
remaining=[i for i in auth['ordered_ids'] if i not in auth['canary_ids']];assert len(remaining)==97
pre=subprocess.run(['nice','-n','10','ionice','-c','3','taskset','-c','111',py,str(B/'guard.py'),'--id',remaining[0]],capture_output=True,text=True,timeout=60)
assert pre.returncode==0,pre.stderr
with (B/'ROOT_CANARY_ACCEPTANCE.json').open('xb') as f:f.write(raw)
with conf.open('x',encoding='utf-8') as f:f.write('[program:'+service+']\ncommand=/usr/bin/nice -n 10 /usr/bin/ionice -c 3 '+py+' '+str(B/'run_queue.py')+' remaining --canary-gate-sha256 __GATE_SHA__\ndirectory='+str(B)+'\nautostart=false\nautorestart=false\nstartsecs=1\nstartretries=0\nstopasgroup=true\nkillasgroup=true\nstdout_logfile='+str(B/'remaining.supervisor.stdout')+'\nstderr_logfile='+str(B/'remaining.supervisor.stderr')+'\n')
commands=[]
for argv in [['supervisorctl','reread'],['supervisorctl','update',service],['supervisorctl','start',service],['supervisorctl','status',service]]:
 p=subprocess.run(argv,capture_output=True,text=True,timeout=30);assert p.returncode==0,(argv,p.stdout,p.stderr)
 commands.append({'command':argv,'stdout':p.stdout})
report={'status':'ROOT_PINNED_CANARY3_THEN_EXACT_REMAINING97_ACTUALLY_STARTED','utc':datetime.now(timezone.utc).isoformat(),
 'service':service,'canary_gate_sha256':sha(B/'ROOT_CANARY_ACCEPTANCE.json'),'cohort':auth['exact_cohort'],
 'remaining':97,'root_adopted_canaries':3,'preflight':json.loads(pre.stdout),'commands':commands,'training':False,'test':False}
with (B/'ROOT_REMAINING97_START.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
'''
remote=remote.replace('__GATE__',base64.b64encode(gate_bytes).decode()).replace('__GATE_SHA__',gate_sha)
compile(remote,'<start-remaining97>','exec')
p=subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','python3 -'],input=remote.encode(),capture_output=True,timeout=100)
for suffix,data in [('stdout',p.stdout),('stderr',p.stderr)]:
    with (D/('REMAINING97_START.'+suffix)).open('xb') as f:f.write(data)
record={'utc':datetime.now(timezone.utc).isoformat(),'exit_code':p.returncode,'source_sha256':sha(__file__),
    'remote_script_sha256':hashlib.sha256(remote.encode()).hexdigest(),'gate_sha256':gate_sha,
    'stdout_sha256':sha(D/'REMAINING97_START.stdout'),'stderr_sha256':sha(D/'REMAINING97_START.stderr')}
with (D/'REMAINING97_START_COMMAND.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
assert p.returncode==0,p.stderr.decode(errors='replace')
result=json.loads(p.stdout)
print(json.dumps({k:result[k] for k in ['status','utc','service','canary_gate_sha256','remaining','root_adopted_canaries','commands']}))
