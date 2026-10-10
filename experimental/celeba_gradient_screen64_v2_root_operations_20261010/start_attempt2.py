"""Start the diagnosed engineering revision only after actual preflight passed."""
from pathlib import Path
import datetime,hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
assert json.loads((HERE/'ATTEMPT2_PREFLIGHT.json').read_bytes())['returncode']==0
assert not (HERE/'ATTEMPT2_DISPATCH.json').exists()
expected={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('remote_preflight_attempt2.py','root_launch_attempt2.py','service_attempt2.sh','guardfed_celeba_gradient_screen64_v2a.conf')}
code="""from pathlib import Path
import hashlib,json,subprocess
base=Path('/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010')
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert not Path('/workspace/celeba_gradient_screen64_v2_results_20261010').exists()
assert all(hashlib.sha256((base/'root_operations'/n).read_bytes()).hexdigest()==h for n,h in EXPECTED.items())
p=Path('/etc/supervisor/conf.d/guardfed_celeba_gradient_screen64_v2a.conf')
with p.open('x') as f:f.write((base/'root_operations/guardfed_celeba_gradient_screen64_v2a.conf').read_text())
records=[]
for argv in [['supervisorctl','reread'],['supervisorctl','update','guardfed_celeba_gradient_screen64_v2a'],['supervisorctl','start','guardfed_celeba_gradient_screen64_v2a']]:
 r=subprocess.run(argv,capture_output=True,text=True,timeout=35);records.append(dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 if r.returncode:break
print(json.dumps(dict(commands=records)))
""".replace('EXPECTED',repr(expected))
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -'],input=code.encode(),capture_output=True,timeout=90)
(HERE/'attempt2_dispatch.stdout').write_bytes(r.stdout);(HERE/'attempt2_dispatch.stderr').write_bytes(r.stderr)
r.check_returncode();d=json.loads(r.stdout)
(HERE/'ATTEMPT2_DISPATCH.json').write_text(json.dumps(dict(**d,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_failed_services_restarted=False,source_or_jobs_changed=False,root_operation_hashes=expected),indent=2)+'\n',encoding='utf8')
assert len(d['commands'])==3 and all(x['returncode']==0 for x in d['commands'])
print(json.dumps(d))
