from pathlib import Path
import json,hashlib,subprocess,datetime,sys,os
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def invoke(cmd,stem,**kw):
 r=subprocess.run(cmd,capture_output=True,**kw)
 for n,b in [('stdout',r.stdout),('stderr',r.stderr)]:
  with (H/(stem+'.'+n)).open('xb') as f:f.write(b)
 save(stem+'_COMMAND.json',dict(command=cmd,exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(H/(stem+'.stdout')),stderr_sha256=sha(H/(stem+'.stderr'))))
 return r
seal='eb08186667ef3ca700cd853aa4dfc919181c86f080b89e5b2b1484203051e3a3'
assert sha(P/'FILES_SHA256.json')==seal
for name,pin in json.loads((P/'FILES_SHA256.json').read_bytes())['files'].items():assert sha(P/name)==pin['sha256']
review=P/'ROOT_SOURCE_REVIEW.json';assert sha(review).startswith('6ffb69c7')
reviewdata=json.loads(review.read_bytes());assert reviewdata['source_adoptable'] and reviewdata['source_seal_sha256']==seal
original=R/'tmp/LoGo100_cuda_actual_root_20261011/observe001/REMOTE_EXECUTED.py'
code=original.read_bytes();save('OBSERVATION_SOURCE.json',dict(path=str(original),sha256=sha(original)))
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','python3 -B -']
r=invoke(cmd,'OBSERVATION',input=code,timeout=60);assert r.returncode==0,r.stderr
s=json.loads(r.stdout);save('OBSERVATION.json',s)
assert s['service']['returncode'] in (0,3) and s['service']['stdout'].split()[:2]==['guardfed_uniform_cuda_LoGo100_canary_valid','EXITED']
assert not s['processes'] and not s['unreadable_pids'] and s['failure'] is None
x=json.loads((P/'ROOT_INPUTS.template.json').read_bytes())
assert s['complete']['record']['ids']==x['exact_ids'] and s['complete']['record']['server_saved_accepted'] is True
assert s['progress']['record']['accepted_ids']==x['exact_ids'] and s['progress']['record']['failed']==0
assert all(s['tasks'][i]['worker_command']['record']['exit_code']==0 for i in x['exact_ids'])
base='/workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001'
x.update(status='ROOT_AUTHORIZED_ACTUAL_LOGO_FIXED2_TRANSPORT_NOT_ADOPTED',root_authorized_transport=True,server_base=base,service='guardfed_uniform_cuda_LoGo100_canary_valid',source_review={'path':str(review),'sha256':sha(review)},root_queue_auth={'path':base+'/ROOT_QUEUE_AUTH.json','sha256':'9713c11187351d93f9f5869963926bc0bc5f04159f315f946807c7fa040cfad9'},deployment_seal={'path':base+'/DEPLOYMENT_SEAL.json','sha256':'ca5eabc8d84559a934a0c822840af1a2dab3dee89e96c1bb0b80b5dd5e7154c1'})
x['task_accepted']={i:{k:s['tasks'][i]['accepted'][k] for k in ['path','sha256']} for i in x['exact_ids']}
prior=json.loads((R/'tmp/LoGo100_cuda_actual_root_20261011/observe001/stdout').read_bytes())
assert all(x['task_accepted'][i]['sha256']==prior['tasks'][i]['accepted']['sha256'] for i in x['exact_ids'])
save('ACTUAL_TRANSPORT_INPUTS.json',x);inp=H/'ACTUAL_TRANSPORT_INPUTS.json'
sys.path.insert(0,str(R/'tmp'));from guardfed_local_storage import check_bulk_storage
save('F_VOLUME.json',check_bulk_storage(250*1024**2))
cmd=[sys.executable,str(P/'transport.py'),'--inputs',str(inp),'--inputs-sha256',sha(inp),'--source-seal-sha256',seal]
save('TRANSPORT_START.json',dict(command=cmd,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs_sha256=sha(inp)))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
r=invoke(cmd,'TRANSPORT',cwd=R,env=env,timeout=450)
print(r.stdout.decode(errors='replace'));print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
