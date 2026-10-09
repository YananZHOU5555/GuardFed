"""Only rebind old verified archive/record verifiers to the new closed bytes."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
B=Path(__file__).resolve().parent;OLD=B.parent/'accepted_delta_after4_20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,text):
 with path.open('x',encoding='utf8',newline='\n') as f:f.write(text)
original=(OLD/'verify_offserver.py').read_text(encoding='utf8')
source=original.replace('hybrid_after4_delta.tar.gz','hybrid_after6_delta.tar.gz').replace('261596ba4a4a11d5877630a436fc0649f857e3f0746263060bd944075bd6afa9','efffebcaecacace9143e579f91d315ab13dd16fd20cce90481a3ba3072dd3c9d').replace('==35','==59').replace("'member_count':35","'member_count':59").replace("'old_accepted':4","'old_accepted':6")
compile(ast.parse(source),'verify_offserver.py','exec');write(B/'verify_offserver.py',source)
write(B/'VERIFIER_DIFF.patch',''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='accepted_after4/verify_offserver.py',tofile='accepted_after6/verify_offserver.py')))
bridge=B/'local_record_bridge_v2';bridge.mkdir(exist_ok=False)
for name in ('checked_record_body.py','SOURCE_REUSE.json','RUNTIME_DIFF.patch'):shutil.copyfile(OLD/'local_record_bridge_v2'/name,bridge/name)
original=(OLD/'local_record_bridge_v2/bridge.py').read_text(encoding='utf8')
source=original.replace('hybrid_after4_delta.tar.gz','hybrid_after6_delta.tar.gz').replace('143a489bcb28dc5234ba275137daae944dd95e20f26723240e8d583b27baff41','c796e51e497e6f8d03d337da019beeb4d24550095dece1bd9cf3d3c9a871d722').replace('261596ba4a4a11d5877630a436fc0649f857e3f0746263060bd944075bd6afa9','efffebcaecacace9143e579f91d315ab13dd16fd20cce90481a3ba3072dd3c9d').replace('20ea9d3a05368fd0aedd83172417d6b4102b9254f8c924ee7a3656467e64671e','4a76135321d4da21b5a08e70b615da90b345fc3a507b1fb06044ed71a39ddbaf')
compile(ast.parse(source),'bridge.py','exec');write(bridge/'bridge.py',source)
write(bridge/'PARENT_BINDING_DIFF.patch',''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='accepted_after4/local_record_bridge_v2/bridge.py',tofile='accepted_after6/local_record_bridge_v2/bridge.py')))
rows=[dict(path=p.name,sha256=sha(p),size=p.stat().st_size) for p in sorted(bridge.iterdir())]
write(bridge/'FILES_SHA256.json',json.dumps(dict(members=rows),indent=2)+'\n')
print(json.dumps(dict(verifier_sha256=sha(B/'verify_offserver.py'),bridge_seal_sha256=sha(bridge/'FILES_SHA256.json'),original_checked_record_body_bytes_exact=sha(bridge/'checked_record_body.py')==sha(OLD/'local_record_bridge_v2/checked_record_body.py'))))
