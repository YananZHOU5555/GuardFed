"""One root-reviewed finite evaluation dispatch; preserves existing attempts."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'tmp/celeba_mechanism_remaining_evaluation_v2_20261010'
REMOTE = '/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010'
SERVICE = 'guardfed_celeba_mechanism_remaining620_valid_v2'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE / 'DISPATCH_RECEIPT.json').exists(), 'Never blindly redispatch'
assert sha(SOURCE / 'FILES_SHA256.json') == 'a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
independent = ROOT / 'tmp/celeba_mechanism_remaining620_v2_independent_review_20261010/REVIEW.json'
assert sha(independent) == 'e54a0c2f9e0182cc6f45f4757513fb3932706d55e30039a5fe1e79450b5afd4c'
preflight = json.loads((HERE / 'SOURCE_AND_PREFLIGHT.json').read_bytes())
actual = json.loads(preflight['stdout'])
assert preflight['returncode'] == 0 and actual['sha256'] == 'ff10c165e2b338094cd4af7b057c89e566fab47e0f235538dafb664ee7a79c20'
plan = json.loads((SOURCE / 'PLAN.json').read_bytes())
review = json.loads((SOURCE / 'ROOT_APPROVAL_TEMPLATE.json').read_bytes())
review.update(status='ROOT_APPROVED_REMAINING620_MECHANISM_VALID_EVALUATION', execution_authorized=True,
    source_seal_sha256=sha(SOURCE / 'FILES_SHA256.json'), selected_ids=plan['remaining620_ids'],
    excluded_ids=plan['excluded180_ids'], deadline_unix=time.time() + 13 * 86400,
    dependency_paths=plan['remote_dependencies'], fresh_Linux_preflight_pass=True,
    Linux_preflight_sha256=actual['sha256'], independent_source_review_sha256=sha(independent),
    root_review_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    supervision='finite approved experiment, no automatic retry/resume, no recurring monitor',
    scientific_boundary='valid-only; no new training or Full inference; original native tolerance and root-only fits unchanged')
review.pop('root_must_fill_actual_status', None)
review_bytes = (json.dumps(review, indent=2, allow_nan=False) + '\n').encode()
with (HERE / 'ROOT_APPROVED.json').open('xb') as f:
    f.write(review_bytes)
review_sha = hashlib.sha256(review_bytes).hexdigest()
wrapper = '''#!/bin/bash
set -e
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" REMOTE/queue.log
. "${utils}/environment.sh"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
exec /usr/bin/taskset -c112-119 /usr/bin/nice -n10 /usr/bin/ionice -c3 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -u REMOTE/evaluate_remaining.py manage --review REMOTE/ROOT_APPROVED.json --review-sha256 REVIEW_SHA
'''.replace('REMOTE', REMOTE).replace('REVIEW_SHA', review_sha)
config = '''[program:SERVICE]
environment=PROC_NAME="%(program_name)s"
command=REMOTE/root_operations/service.sh
directory=REMOTE
autostart=false
autorestart=false
startretries=0
startsecs=1
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
'''.replace('SERVICE', SERVICE).replace('REMOTE', REMOTE)
for name, value in [('service.sh', wrapper), (SERVICE + '.conf', config)]:
    with (HERE / name).open('x', encoding='utf8', newline='\n') as f:
        f.write(value)
payload = {'ROOT_APPROVED.json': review_bytes.hex(), 'root_operations/service.sh': wrapper.encode().hex(),
           'root_operations/' + SERVICE + '.conf': config.encode().hex()}
code = r'''from pathlib import Path
import datetime,hashlib,json,os,subprocess
base=Path(REMOTE)
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert hashlib.sha256((base/'ROOT_LINUX_PREFLIGHT.json').read_bytes()).hexdigest()==PREFLIGHT_SHA
assert not (base/'attempt1').exists() and not (base/'ROOT_APPROVED.json').exists()
seal=json.loads((base/'FILES_SHA256.json').read_bytes())
assert hashlib.sha256((base/'FILES_SHA256.json').read_bytes()).hexdigest()==SOURCE_SHA
for name,pin in seal['files'].items():
 p=base/name
 assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'],name
occupied=[]
for p in Path('/proc').glob('[0-9]*/stat'):
 try:
  if p.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):continue
  for t in (p.parent/'task').iterdir():
   cpus=os.sched_getaffinity(int(t.name))
   if len(cpus)<=16 and set(range(112,120)) & cpus:occupied.append(dict(pid=int(p.parent.name),tid=int(t.name),cpus=sorted(cpus)))
 except (OSError,ValueError):pass
assert not occupied,occupied
sg=subprocess.run(['supervisorctl','status','sglang'],capture_output=True,text=True,timeout=20)
assert sg.stdout.split()[:2]==['sglang','STOPPED']
target=Path('/etc/supervisor/conf.d')/(SERVICE+'.conf')
assert not target.exists()
for name,hexdata in PAYLOAD.items():
 p=base/name;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('xb') as f:f.write(bytes.fromhex(hexdata))
os.chmod(base/'root_operations/service.sh',0o755)
with target.open('xb') as f:f.write(bytes.fromhex(PAYLOAD['root_operations/'+SERVICE+'.conf']))
commands=[]
for argv in [['supervisorctl','reread'],['supervisorctl','update',SERVICE],['supervisorctl','start',SERVICE]]:
 r=subprocess.run(argv,capture_output=True,text=True,timeout=35)
 commands.append(dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 if r.returncode:break
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=commands,
 source_seal_sha256=SOURCE_SHA,root_approval_sha256=REVIEW_SHA,guide_sha256='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa',
 restricted_CPU112_119_owners_before_install=occupied,queue_service=SERVICE,automatic_retry=False,new_training=0,test=False)))
'''
for key, value in [('PREFLIGHT_SHA', actual['sha256']), ('SOURCE_SHA', review['source_seal_sha256']),
                   ('REVIEW_SHA', review_sha), ('SERVICE', SERVICE), ('REMOTE', REMOTE), ('PAYLOAD', payload)]:
    code = code.replace(key, repr(value))
r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350',
                    'root@89.22.197.55', 'python -B -'], input=code.encode(), capture_output=True, timeout=120)
(HERE / 'dispatch.stdout').write_bytes(r.stdout)
(HERE / 'dispatch.stderr').write_bytes(r.stderr)
with (HERE / 'DISPATCH_COMMAND_EXIT.json').open('x') as f:
    f.write(json.dumps(dict(returncode=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           root_approval_sha256=review_sha,automatic_retry=False),indent=2)+'\n')
r.check_returncode()
receipt = json.loads(r.stdout)
with (HERE / 'DISPATCH_RECEIPT.json').open('x') as f:
    f.write(json.dumps(receipt,indent=2)+'\n')
assert len(receipt['commands']) == 3 and all(x['returncode'] == 0 for x in receipt['commands'])
print(json.dumps(receipt))
