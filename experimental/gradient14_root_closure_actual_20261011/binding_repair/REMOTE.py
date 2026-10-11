
from pathlib import Path,PurePosixPath
import hashlib,io,json,subprocess,sys,tarfile
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
stage=Path('/workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011')
assert H('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert not (stage/'closure_source/RUNTIME_MEMBERS.json').exists()
old={'ROLE_CONTRACT.json': {'sha256': '24f84afe55b29c442e93258bd342eadeb94982d5b81ae3d22bca6b78d78b3888', 'bytes': 69272}, 'ROOT_BINDING.json': {'sha256': '1ab04de81cd19344a979680a2ba276ae773d2cc02526496b6b2a66dac5cd19ff', 'bytes': 65560}}
for n,pin in old.items():assert H(stage/'runtime/closure_role_binding'/n)==pin['sha256']
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 ms=t.getmembers();assert all(m.isfile() for m in ms) and len(ms)==len({m.name for m in ms})
 pins=json.load(t.extractfile('TRANSFER.json'));assert set(pins)|{'TRANSFER.json'}=={m.name for m in ms}
 for name,pin in pins.items():
  q=PurePosixPath(name);assert '..' not in q.parts and ':' not in name and '\\' not in name
  assert name.startswith('runtime/local_evidence/runtime_role_binding_duplicate_failure/') or name in {'runtime/closure_role_binding/ROLE_CONTRACT.json','runtime/closure_role_binding/ROOT_BINDING.json'}
  p=stage/name;assert p.resolve().is_relative_to(stage.resolve()) and not p.is_symlink()
  v=t.extractfile(name).read();assert len(v)==pin['bytes']<2_000_000 and hashlib.sha256(v).hexdigest()==pin['sha256']
  if name.startswith('runtime/local_evidence/'):
   assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
 # Preserve actual refused inputs before changing only the two current metadata pointers.
 for n in old:
  assert H(stage/'runtime/local_evidence/runtime_role_binding_duplicate_failure'/n)==old[n]['sha256']
 for n in old:(stage/'runtime/closure_role_binding'/n).write_bytes(t.extractfile('runtime/closure_role_binding/'+n).read())
contract=stage/'runtime/closure_role_binding/ROLE_CONTRACT.json';binding=stage/'runtime/closure_role_binding/ROOT_BINDING.json'
cmd=['taskset','-c','110','nice','-n','10','ionice','-c','3','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(stage/'runtime/local_evidence/runtime_role_binder_source/bind_runtime_roles_once.py'),'--stage',str(stage),'--contract',str(contract),'--contract-sha256',H(contract),'--root-binding',str(binding),'--root-binding-sha256',H(binding)]
r=subprocess.run(cmd,capture_output=True);print(json.dumps({'returncode':r.returncode,'binder_stdout':r.stdout.decode(),'binder_stderr':r.stderr.decode(),'contract_sha256':H(contract),'binding_sha256':H(binding),'prior_refusal_preserved':True}));sys.exit(r.returncode)
