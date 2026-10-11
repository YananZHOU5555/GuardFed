
from pathlib import Path
import base64,hashlib,json,subprocess,sys
H=lambda b:hashlib.sha256(b).hexdigest()
assert H(Path('/etc/vast-agents-guide.md').read_bytes())=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
repo=Path('/workspace/GuardFed-celeba-expanded')
w=repo/'tmp/gradient64_wiring_independent_review_20261011/repaired/wiring.py'
assert H(w.read_bytes())=='fdeccb0ee1838a1e1133014eadb0022d8745122556773b8d618ad7926fe58d08'
dest=Path('/workspace/guardfed_checks/celeba_gradient_fullcoverage_selected_20261011')
assert not dest.exists() and not dest.is_symlink()
i=repo/'tmp/gradient64_selected_inputs_20261011'
argv=['taskset','-c','105','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(w),'freeze-source','--summary',str(i/'COMPLETE64_SELECTION.json'),'--summary-sha256','a4dad47c67bf96f75cd100872ad5c596d46b853c2adcc2f6a4fe220b0cf8394c','--selected64-root',str(i/'SELECTED64_ROOT.json'),'--selected64-root-sha256','9aad5ca1eff130b2943013ad79145f50a8777debf5715997d715a0b202caad06','--out',str(dest)]
p=subprocess.run(argv,capture_output=True)
if dest.exists():
 (dest/'FREEZE.stdout').write_bytes(p.stdout);(dest/'FREEZE.stderr').write_bytes(p.stderr)
assert p.returncode==0,dict(returncode=p.returncode,stderr=p.stderr.decode(errors='replace'))
files={}
for f in sorted(dest.rglob('*')):
 if not f.is_file():continue
 assert not f.is_symlink() and f.stat().st_size<1024**2
 raw=f.read_bytes();files[f.relative_to(dest).as_posix()]=dict(sha256=H(raw),bytes=len(raw),base64=base64.b64encode(raw).decode())
print(json.dumps(dict(status='ACTUAL_FROZEN_CODE_SOURCE_ONLY_NOT_DISPATCHED',remote_out=str(dest),argv=argv,returncode=p.returncode,files=files,final_test=False,workers_started=0)))
