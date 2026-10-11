from pathlib import Path
import base64,hashlib,json,subprocess,datetime
H=Path(__file__).resolve().parent
P=H.parent/'FedAvg_cuda100_saved_cache_closure_prepare_20261011'
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
seal=json.loads((P/'FILES_SHA256.json').read_bytes())
files={}
for n in ['build_root_index.py','cache_verifier.py','EXACT100_REFS.json','original_prefix_functions.py.txt']:
    b=(P/n).read_bytes();assert sha(b)==seal['files'][n]['sha256'];files[n]={'sha256':sha(b),'base64':base64.b64encode(b).decode()}
proofs={
 'ROOT_SCIENTIFIC_ADOPTION.json':(P/'root_adoption/ROOT_SCIENTIFIC_ADOPTION.json','83a047fdd4d9c997c3ca31ff713de134575f1bcc6c75f6da579031f604efb083'),
 'SAVED_OUTPUTS_AUDIT.json':(P/'actual/SAVED_OUTPUTS_AUDIT.json','daf6118b869f8887cf90d5f3d38826432e471c701a60794edf1d75f63992b23c'),
 'TRANSPORT_VERIFICATION.json':(P/'OLD800_FIXED3_ATTEMPT1_TRANSPORT_VERIFICATION.json','e02c2efb1b1ed84952eeb87ae298690a64b130ad25be61c590d3bc709fb30c74')}
for n,(p,s) in proofs.items():
    b=p.read_bytes();assert sha(b)==s;files[n]={'sha256':s,'base64':base64.b64encode(b).decode()}
template=json.loads((P/'ROOT_INDEX_INPUTS.template.json').read_bytes())
base='/workspace/guardfed_checks/FedAvg100_cuda_adopted_cache_index_20261011_actual002'
template['metadata']['path']='/workspace/GuardFed-revision/data/celeba/derived/rgb64_v1/metadata.npz'
payload={'base':base,'files':files,'template':template,'selection_original':'/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt003/exports/FedAvg100_mixed_saved_audit_attempt001/COHORT_SELECTION.json'}
remote=r'''
from pathlib import Path
import base64,hashlib,json,subprocess,datetime,sys,os
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
x=PAYLOAD
guide=sha('/etc/vast-agents-guide.md');assert guide=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
mp=Path(x['template']['metadata']['path']);assert mp.is_absolute() and mp.is_file() and not mp.is_symlink() and mp.resolve()==mp and sha(mp)==x['template']['metadata']['sha256']=='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091'
B=Path(x['base']);assert B.is_absolute() and '..' not in B.parts and not B.exists();B.mkdir()
for name,entry in x['files'].items():
 p=B/name
 with p.open('xb') as f:f.write(base64.b64decode(entry['base64']))
 assert sha(p)==entry['sha256']
transport=json.loads((B/'TRANSPORT_VERIFICATION.json').read_bytes());sel=Path(x['selection_original']);expected=transport['members']['base/COHORT_SELECTION.json']['sha256'];assert sha(sel)==expected
with (B/'COHORT_SELECTION.json').open('xb') as f:f.write(sel.read_bytes())
assert sha(B/'COHORT_SELECTION.json')==expected
inputs=x['template'];inputs['status']='ROOT_AUTHORIZED_ADOPTED_CACHE_INDEX_METADATA_ONLY';inputs['root_authorized_index']=True
for role,name in [('root','ROOT_SCIENTIFIC_ADOPTION.json'),('audit','SAVED_OUTPUTS_AUDIT.json'),('transport','TRANSPORT_VERIFICATION.json'),('selection','COHORT_SELECTION.json')]:inputs[role]={'path':str(B/name),'sha256':sha(B/name)}
dump(B/'ROOT_INDEX_INPUTS.json',inputs)
cmd=[sys.executable,'-B',str(B/'build_root_index.py'),'--inputs',str(B/'ROOT_INDEX_INPUTS.json'),'--inputs-sha256',sha(B/'ROOT_INDEX_INPUTS.json'),'--output',str(B/'ROOT_CACHE_INDEX.json')]
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(cmd,capture_output=True,timeout=240)
for name,raw in [('BUILD.stdout',r.stdout),('BUILD.stderr',r.stderr)]:
 with (B/name).open('xb') as f:f.write(raw)
pins={p.name:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in B.iterdir() if p.is_file()}
for name,entry in x['files'].items():assert pins[name]['sha256']==entry['sha256']
proof={'status':'INDEX_METADATA_BUILD_PASS' if r.returncode==0 else 'INDEX_METADATA_BUILD_FAILED','command':cmd,'exit_code':r.returncode,'start_utc':start,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'guide_sha256':guide,'files':pins,'new_CNN':0,'new_fit':0,'arrays_downloaded':0,'source_byte_exact':True,'selection_original':str(sel),'selection_sha256':expected}
dump(B/'BUILD_COMMAND.json',proof)
print(json.dumps(proof));sys.exit(r.returncode)
'''.replace('PAYLOAD',repr(payload))
(H/'REMOTE_DEPLOY_BODY.py').write_bytes(remote.encode())
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 110 nice -n 10 ionice -c 3 python3 -B -']
dump(H/'DEPLOY_START.json',{'command':cmd,'payload_sha256':sha(remote.encode()),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
r=subprocess.run(cmd,input=remote.encode(),capture_output=True,timeout=300)
for n,b in [('DEPLOY.stdout',r.stdout),('DEPLOY.stderr',r.stderr)]:
    with (H/n).open('xb') as f:f.write(b)
dump(H/'DEPLOY_COMMAND.json',{'command':cmd,'exit_code':r.returncode,'stdout_sha256':sha(r.stdout),'stderr_sha256':sha(r.stderr),'payload_sha256':sha(remote.encode()),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print(r.stdout.decode(errors='replace'));print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
