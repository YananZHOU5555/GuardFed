"""Diagnosed wrapper-only repair. No queue/science or frozen approval changes."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
old = json.loads((HERE / 'DISPATCH_RECEIPT.json').read_bytes())
assert old['commands'][-1]['returncode'] == 7 and len(old['commands']) == 3
assert sha(HERE / 'ROOT_APPROVED.json') == old['root_approval_sha256']
assert not (HERE / 'V2A_DISPATCH_RECEIPT.json').exists()
service = 'guardfed_celeba_mechanism_remaining620_valid_v2a'
wrapper_old = (HERE / 'service.sh').read_text()
assert wrapper_old.count('taskset -c112-119') == 1
wrapper = wrapper_old.replace('taskset -c112-119', 'taskset -c 112-119').replace('/queue.log', '/queue_v2a.log')
config = (HERE / 'guardfed_celeba_mechanism_remaining620_valid_v2.conf').read_text().replace(
    'guardfed_celeba_mechanism_remaining620_valid_v2', service).replace('/service.sh', '/service_v2a.sh')
for name, value in [('service_v2a.sh', wrapper), (service + '.conf', config)]:
    with (HERE / name).open('x', encoding='utf8', newline='\n') as f:
        f.write(value)
payload = {'root_operations/service_v2a.sh': wrapper.encode().hex(), 'root_operations/' + service + '.conf': config.encode().hex()}
code = r'''from pathlib import Path
import datetime,hashlib,json,os,subprocess
base=Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010')
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert not (base/'attempt1').exists() and hashlib.sha256((base/'ROOT_APPROVED.json').read_bytes()).hexdigest()==APPROVAL_SHA
assert hashlib.sha256((base/'root_operations/service.sh').read_bytes()).hexdigest()==OLD_WRAPPER_SHA
assert hashlib.sha256((base/'FILES_SHA256.json').read_bytes()).hexdigest()=='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
for name,pin in json.loads((base/'FILES_SHA256.json').read_bytes())['files'].items():
 assert hashlib.sha256((base/name).read_bytes()).hexdigest()==pin['sha256'],name
old_service=subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_remaining620_valid_v2'],capture_output=True,text=True,timeout=20)
assert old_service.stdout.split()[:2]==['guardfed_celeba_mechanism_remaining620_valid_v2','FATAL']
log=(base/'queue.log').read_text()
assert "taskset: invalid option -- '1'" in log
occupied=[]
for p in Path('/proc').glob('[0-9]*/stat'):
 try:
  if p.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):continue
  for t in (p.parent/'task').iterdir():
   cpus=os.sched_getaffinity(int(t.name))
   if len(cpus)<=16 and set(range(112,120)) & cpus:occupied.append(dict(pid=int(p.parent.name),tid=int(t.name),cpus=sorted(cpus)))
 except (OSError,ValueError):pass
assert not occupied,occupied
syntax=subprocess.run(['/usr/bin/taskset','-c','112-119','/bin/true'],capture_output=True,text=True,timeout=10)
assert syntax.returncode==0
first=dict(status='PRESERVED_WRAPPER_TASKSET_SYNTAX_FAILURE_BEFORE_PYTHON',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 service=old_service.stdout.strip(),returncode=old_service.returncode,log=log,attempt_namespace_absent=True,
 new_training=0,CNN=0,queue_source_unchanged=True,approval_sha256=APPROVAL_SHA,old_wrapper_sha256=OLD_WRAPPER_SHA,
 repair='Separate taskset option -c from CPU-list argument; new wrapper and service only',fixed_taskset_syntax_exit=syntax.returncode)
with (base/'FIRST_WRAPPER_START_FAILURE.json').open('x') as f:f.write(json.dumps(first,indent=2)+'\n')
for name,h in PAYLOAD.items():
 with (base/name).open('xb') as f:f.write(bytes.fromhex(h))
os.chmod(base/'root_operations/service_v2a.sh',0o755)
with (Path('/etc/supervisor/conf.d')/(SERVICE+'.conf')).open('xb') as f:f.write(bytes.fromhex(PAYLOAD['root_operations/'+SERVICE+'.conf']))
commands=[]
for argv in [['supervisorctl','reread'],['supervisorctl','update',SERVICE],['supervisorctl','start',SERVICE]]:
 r=subprocess.run(argv,capture_output=True,text=True,timeout=35);commands.append(dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 if r.returncode:break
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=commands,preserved_first_failure=first,
 actual_service=SERVICE,root_approval_sha256=APPROVAL_SHA,restricted_owners_before_start=occupied,science_source_changed=False,automatic_retry=False)))
'''
for token, value in [('APPROVAL_SHA', old['root_approval_sha256']), ('OLD_WRAPPER_SHA', sha(HERE / 'service.sh')),
                     ('SERVICE', service), ('PAYLOAD', payload)]:
    code = code.replace(token, repr(value))
r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350', 'root@89.22.197.55', 'python -B -'],
                   input=code.encode(), capture_output=True, timeout=120)
(HERE / 'v2a_dispatch.stdout').write_bytes(r.stdout)
(HERE / 'v2a_dispatch.stderr').write_bytes(r.stderr)
with (HERE / 'V2A_DISPATCH_COMMAND_EXIT.json').open('x') as f:
    f.write(json.dumps(dict(returncode=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False),indent=2)+'\n')
r.check_returncode()
receipt = json.loads(r.stdout)
with (HERE / 'V2A_DISPATCH_RECEIPT.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
assert len(receipt['commands']) == 3 and all(x['returncode'] == 0 for x in receipt['commands'])
print(json.dumps(dict(actual_service=service,commands=receipt['commands'],first_failure_preserved=True,science_source_changed=False)))
