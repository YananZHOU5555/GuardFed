"""Preserve the no-training failure; fix the launcher/log-tee identity check."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=(HERE/'remote_preflight.py').read_text()
before="assert not any(str(BASE) in ' '.join(r['argv']) for r in processes), 'Duplicate gradient worker'"
after="assert not any(any(Path(a).name == 'run_queue.py' for a in r['argv']) for r in processes), 'Duplicate gradient queue/worker'"
assert original.count(before)==1
updated=original.replace(before,after)
(HERE/'remote_preflight_attempt2.py').write_bytes(updated.encode())
launch=(HERE/'root_launch.py').read_text().replace('root_operations/remote_preflight.py','root_operations/remote_preflight_attempt2.py')
(HERE/'root_launch_attempt2.py').write_bytes(launch.encode())
service=(HERE/'service.sh').read_text().replace('queue.log','queue_attempt2.log').replace('root_operations/root_launch.py','root_operations/root_launch_attempt2.py')
(HERE/'service_attempt2.sh').write_bytes(service.encode())
config=(HERE/'guardfed_celeba_gradient_screen64_v2.conf').read_text().replace('guardfed_celeba_gradient_screen64_v2]','guardfed_celeba_gradient_screen64_v2a]').replace('root_operations/service.sh','root_operations/service_attempt2.sh')
(HERE/'guardfed_celeba_gradient_screen64_v2a.conf').write_bytes(config.encode())
failure=json.loads((HERE/'LATEST_OBSERVATION.json').read_bytes())
assert not failure['processes'] and not failure['rows'] and failure['resource_proof'] is None
payload={n:(HERE/n).read_text() for n in ('remote_preflight_attempt2.py','root_launch_attempt2.py','service_attempt2.sh','guardfed_celeba_gradient_screen64_v2a.conf')}
code="""from pathlib import Path
import hashlib,json,subprocess
base=Path('/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010')
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert not Path('/workspace/celeba_gradient_screen64_v2_results_20261010').exists()
for name,value in PAYLOAD.items():
 p=base/'root_operations'/name
 with p.open('x') as f:f.write(value)
 if name.endswith('.sh'):p.chmod(0o755)
r=subprocess.run(['/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(base/'root_operations/remote_preflight_attempt2.py')],capture_output=True,text=True,timeout=50)
print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
""".replace('PAYLOAD',repr(payload))
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -'],input=code.encode(),capture_output=True,timeout=75)
(HERE/'attempt2_preflight.stdout').write_bytes(r.stdout);(HERE/'attempt2_preflight.stderr').write_bytes(r.stderr)
r.check_returncode();d=json.loads(r.stdout)
(HERE/'ATTEMPT2_PREFLIGHT.json').write_text(json.dumps(dict(**d,preserved_failure_sha256=sha(HERE/'LATEST_OBSERVATION.json'),
 root_operation_hashes={n:sha(HERE/n) for n in payload},science_source_changed=False,new_training=0),indent=2)+'\n',encoding='utf8')
assert d['returncode']==0, 'Preserve diagnostic failure; never continue blindly'
print(d['stdout'])
