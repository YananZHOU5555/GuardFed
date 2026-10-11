from pathlib import Path
import hashlib,json,subprocess,datetime,sys,os
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
seal='eb08186667ef3ca700cd853aa4dfc919181c86f080b89e5b2b1484203051e3a3';assert sha(P/'FILES_SHA256.json')==seal
for n,pin in json.loads((P/'FILES_SHA256.json').read_bytes())['files'].items():assert sha(P/n)==pin['sha256']
proof=P/'LOGO_FIXED2_TRANSPORT_VERIFICATION.json';assert sha(proof)=='ebb9b246b205057ca7f756fae727093e375fa1b33a702d7c7379adf6b783fb6d'
t=json.loads(proof.read_bytes());assert t['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and t['all_members_verified'] and t['archive_members']==34 and t['archive_sha256']=='81b0e85cec0c0fa89377971643b9c7499232dfbddd8e860c4b1eb433e9c65c52'
assert sum(n.endswith('.npz') for n in t['members'])==2
sys.path.insert(0,str(R/'tmp'));from guardfed_local_storage import check_bulk_storage
save('AUDIT_F_VOLUME.json',check_bulk_storage())
x=json.loads((H/'ACTUAL_TRANSPORT_INPUTS.json').read_bytes());x.update(status='ROOT_AUTHORIZED_ACTUAL_LOGO_FIXED2_SAVED_AUDIT_NOT_ADOPTED',root_authorized_transport=False,root_authorized_audit=True,transport={'path':str(proof),'sha256':sha(proof)},archive_sha256=t['archive_sha256'])
assert x['source_review']['sha256']=='6ffb69c7d7335c0d720c7b243732a7f080d766e659d3f12915cc6fa834ad7502'
save('ACTUAL_AUDIT_INPUTS.json',x);inp=H/'ACTUAL_AUDIT_INPUTS.json'
cmd=[sys.executable,str(P/'audit_saved.py'),'--inputs',str(inp),'--inputs-sha256',sha(inp),'--source-seal-sha256',seal]
save('AUDIT_START.json',dict(command=cmd,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs_sha256=sha(inp)))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(cmd,cwd=R,env=env,capture_output=True,timeout=300)
for n,b in [('AUDIT.stdout',r.stdout),('AUDIT.stderr',r.stderr)]:
 with (H/n).open('xb') as f:f.write(b)
save('AUDIT_COMMAND.json',dict(command=cmd,exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(H/'AUDIT.stdout'),stderr_sha256=sha(H/'AUDIT.stderr'),inputs_sha256=sha(inp)))
print(r.stdout.decode(errors='replace'));print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
