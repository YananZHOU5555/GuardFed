"""Environment metadata closure only: no scientific imports or deserialization."""
from pathlib import Path
import hashlib,importlib.metadata as metadata,json,sys
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
expected={'torch':'2.8.0+cpu','torchvision':'0.23.0+cpu','numpy':'1.26.4','netcal':'1.3.6','scikit-learn':'1.7.1','scipy':'1.15.3','pandas':'2.3.1','pyro-ppl':'1.9.2','gpytorch':'1.15.2'}
actual={k:metadata.version(k) for k in expected};assert actual==expected and sys.version_info[:2]==(3,10)
exe=Path(sys.executable);assert str(exe)=='/workspace/guardfed_envs/logofair-frozen-post-cpu310-20261011_attempt002/bin/python' and not exe.is_symlink()
packages=[]
for d in metadata.distributions():
    pins={}
    for f in d.files or []:
        if f.name in ('METADATA','WHEEL','RECORD','direct_url.json') and '.dist-info/' in f.as_posix():
            p=Path(d.locate_file(f));pins[f.name]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    packages.append({'name':d.metadata['Name'],'version':d.version,'requires':d.requires or [],'metadata_files':pins})
assert not any(p['name'].lower().startswith(('nvidia-','cuda-')) for p in packages),'CUDA package slipped into CPU environment'
cfg=exe.parent.parent/'pyvenv.cfg'
print(json.dumps({'status':'EXACT_CPU_PACKAGE_METADATA_VERIFIED_NO_SCIENTIFIC_IMPORT','python':sys.version,'executable':{'path':str(exe),'resolved':str(exe.resolve()),'sha256':sha(exe),'bytes':exe.stat().st_size,'is_symlink':exe.is_symlink()},'base_prefix':sys.base_prefix,'pyvenv_cfg':{'path':str(cfg),'sha256':sha(cfg),'text':cfg.read_text()},'frozen_packages':actual,'distributions':sorted(packages,key=lambda p:p['name'].lower()),'scientific_imports_executed':False,'pickle_executed':False,'CNN_fit_executed':False,'scientific_result':None},allow_nan=False))
