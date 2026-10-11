
from pathlib import Path,PurePosixPath
import hashlib,io,json,subprocess,sys,tarfile
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
stage=Path('/workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011')
assert H('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert not (stage/'closure_source/RUNTIME_MEMBERS.json').exists()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 ms=t.getmembers();assert all(m.isfile() for m in ms) and len(ms)==len({m.name for m in ms})
 pins=json.load(t.extractfile('TRANSFER.json'));assert set(pins)|{'TRANSFER.json'}=={m.name for m in ms}
 for name,pin in pins.items():
  q=PurePosixPath(name);assert name.startswith(('runtime/','closure_source/')) and '..' not in q.parts and ':' not in name and '\\' not in name
  p=stage/name;assert p.resolve().is_relative_to(stage.resolve()) and not p.is_symlink()
  v=t.extractfile(name).read();assert len(v)==pin['bytes']<2_000_000 and hashlib.sha256(v).hexdigest()==pin['sha256']
  if p.exists():assert p.read_bytes()==v,name
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
contract=stage/'runtime/closure_role_binding/ROLE_CONTRACT.json';binding=stage/'runtime/closure_role_binding/ROOT_BINDING.json'
cmd=['taskset','-c','110','nice','-n','10','ionice','-c','3','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(stage/'runtime/local_evidence/runtime_role_binder_source/bind_runtime_roles_once.py'),'--stage',str(stage),'--contract',str(contract),'--contract-sha256',H(contract),'--root-binding',str(binding),'--root-binding-sha256',H(binding)]
r=subprocess.run(cmd,capture_output=True);print(json.dumps({'installed_files':len(pins),'returncode':r.returncode,'binder_stdout':r.stdout.decode(),'binder_stderr':r.stderr.decode(),'contract_sha256':H(contract),'binding_sha256':H(binding)}));sys.exit(r.returncode)
