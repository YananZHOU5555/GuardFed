from pathlib import Path
import hashlib,json,subprocess,datetime,sys,os
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
seal='40fa2578e9247d3bc746262915d4f54da470edcb980e364aa5a749babf94fd9e';assert sha(P/'FILES_SHA256.json')==seal
for n,pin in json.loads((P/'FILES_SHA256.json').read_bytes())['files'].items():assert sha(P/n)==pin['sha256']
proof=P/'LOGO100_DELTA98_TRANSPORT_VERIFICATION.json';assert sha(proof)=='3b8fc7546d3644fd82eee114310be6198d0cf4222dc55575caebd31f6a737df5'
t=json.loads(proof.read_bytes());assert t['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and t['all_members_verified'] and t['archive_members']==885 and t['archive_sha256']=='cabf8047e3d76b6ece725509265197273f2f821edd0b4bb082ce9714540213db'
assert sum(n.endswith('.npz') for n in t['members'])==98 and t['remote']['new_arrays']==98 and t['remote']['reused_arrays']==2 and len(t['remote']['reused_member_pins'])==33
sys.path.insert(0,str(R/'tmp'));from guardfed_local_storage import check_bulk_storage
save('AUDIT_F_VOLUME.json',check_bulk_storage())
x=json.loads((H/'ACTUAL_TRANSPORT_INPUTS.json').read_bytes());x.update(status='ROOT_AUTHORIZED_ACTUAL_LOGO100_ALL100_SAVED_AUDIT_NOT_ADOPTED',root_authorized_transport=False,root_authorized_audit=True,transport={'path':str(proof),'sha256':sha(proof)},archive_sha256=t['archive_sha256'])
save('ACTUAL_AUDIT_INPUTS.json',x);inp=H/'ACTUAL_AUDIT_INPUTS.json'
cmd=[sys.executable,str(P/'audit_saved.py'),'--inputs',str(inp),'--inputs-sha256',sha(inp),'--source-seal-sha256',seal]
save('AUDIT_START.json',dict(command=cmd,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs_sha256=sha(inp)))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(cmd,cwd=R,env=env,capture_output=True,timeout=300)
for n,b in [('AUDIT.stdout',r.stdout),('AUDIT.stderr',r.stderr)]:
 with (H/n).open('xb') as f:f.write(b)
save('AUDIT_COMMAND.json',dict(command=cmd,exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(H/'AUDIT.stdout'),stderr_sha256=sha(H/'AUDIT.stderr'),inputs_sha256=sha(inp)))
print(r.stdout.decode(errors='replace'));print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
