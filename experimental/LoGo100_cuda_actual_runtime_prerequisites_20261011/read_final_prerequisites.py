"""Read actual transfer receipt and failed environment package metadata; no scientific imports."""
from pathlib import Path
import base64,hashlib,json,subprocess
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
p=Path('/workspace/guardfed_checks/uniform_cuda_LoGo100_artifacts_20261011_attempt001/TRANSPORT_VERIFICATION.json')
assert sha(p)=='b86b959f8737479ce3917921dfc9aa59f50890ea19051d5dfe73a407a0060712'
environment=Path('/workspace/guardfed_envs/logofair-frozen-post-cpu310-20261011')
query="""import sys,json,importlib.metadata as m
from pathlib import Path
import hashlib
distributions=[]
for d in m.distributions():
 distributions.append({'name':d.metadata['Name'],'version':d.version,'requires':d.requires or []})
p=Path(sys.executable)
print(json.dumps({'python':sys.version,'executable':str(p),'resolved_executable':str(p.resolve()),'is_symlink':p.is_symlink(),'executable_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'packages':sorted(distributions,key=lambda d:d['name'].lower()),'scientific_imports_executed':False}))
"""
r=subprocess.run([str(environment/'bin/python'),'-B','-c',query],capture_output=True,text=True,timeout=20)
print(json.dumps({'transport':{'path':str(p),'sha256':sha(p),'base64':base64.b64encode(p.read_bytes()).decode()},'environment_probe':{'returncode':r.returncode,'stderr':r.stderr,'metadata':json.loads(r.stdout) if r.returncode==0 else None},'environment_accepted':False,'scientific_execution':False}))
