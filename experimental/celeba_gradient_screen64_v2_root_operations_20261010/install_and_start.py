"""Root's explicit one-time dispatch after independent source review."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
review = HERE / 'ROOT_SOURCE_REVIEW.json'
assert sha(review) == '58c30abe8f61298fcd2af379a8aec79d377ff5901df7cceec0e933aa7172a1db'
assert not (HERE / 'DISPATCH_RECEIPT.json').exists(), 'No blind repeated start'
assert json.loads((HERE / 'SOURCE_UPLOAD.json').read_bytes())['actual_dispatch'] is False
authority = ROOT / 'docs/server_deployment_20260923/training_20260923/AUTHOR_DECISIONS_20261010.json'
assert sha(authority) == 'aee89e8210b5aa83d8ee814d5afde4bb655f3ba559bb53ae6ebcd1ca0d46851d'
payload = {n: (HERE / n).read_text(encoding='utf-8') for n in
           ('root_launch.py', 'service.sh', 'guardfed_celeba_gradient_screen64_v2.conf')}
approval = dict(status='ROOT_SOURCE_REVIEW_APPROVED_FRESH_RESOURCE_GATE_REQUIRED_BEFORE_DISPATCH',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_seal_sha256='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced',
    independent_review_sha256=sha(review), author_decisions_sha256=sha(authority), jobs=64, worker_count=1,
    no_retry=True, no_test=True, model_output='/workspace/celeba_gradient_screen64_v2_results_20261010',
    operation_hashes={n: sha(HERE/n) for n in payload}, preflight_code_sha256=sha(HERE/'remote_preflight.py'))
(HERE / 'ROOT_SOURCE_APPROVAL.json').write_text(json.dumps(approval, indent=2)+'\n', encoding='utf8')
code = """from pathlib import Path
import hashlib,json,subprocess
base=Path('/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010')
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
payload=PAYLOAD
assert hashlib.sha256((base/'root_operations/remote_preflight.py').read_bytes()).hexdigest()==PREFLIGHT
config=Path('/etc/supervisor/conf.d/guardfed_celeba_gradient_screen64_v2.conf')
assert not config.exists() and not Path('/workspace/celeba_gradient_screen64_v2_results_20261010').exists()
for name,value in payload.items():
 p=base/'root_operations'/name
 with p.open('x') as f:f.write(value)
 if name=='service.sh':p.chmod(0o755)
(base/'ROOT_SOURCE_APPROVAL.json').write_text(APPROVAL)
config.write_text(payload['guardfed_celeba_gradient_screen64_v2.conf'])
records=[]
for argv in [['supervisorctl','reread'],['supervisorctl','update','guardfed_celeba_gradient_screen64_v2'],['supervisorctl','start','guardfed_celeba_gradient_screen64_v2']]:
 r=subprocess.run(argv,capture_output=True,text=True,timeout=40)
 records.append(dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 if r.returncode:break
print(json.dumps(dict(status='ACTUAL_ONE_TIME_START_COMMAND_RECORDED_CHECK_RESOURCE_AND_PROGRESS_NEXT',commands=records)))
""".replace('PAYLOAD', repr(payload)).replace('PREFLIGHT', repr(approval['preflight_code_sha256'])).replace('APPROVAL)', repr(json.dumps(approval,indent=2)+'\n')+')')
r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -'],
                   input=code.encode(), capture_output=True, timeout=100)
(HERE/'dispatch.stdout').write_bytes(r.stdout)
(HERE/'dispatch.stderr').write_bytes(r.stderr)
r.check_returncode()
d=json.loads(r.stdout)
(HERE/'DISPATCH_RECEIPT.json').write_text(json.dumps(dict(**d,source_approval_sha256=sha(HERE/'ROOT_SOURCE_APPROVAL.json'),
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n',encoding='utf8')
assert all(c['returncode']==0 for c in d['commands']), 'Preserve first deployment failure; do not restart'
print(json.dumps(d))
