"""Read actual exact100 metadata; bind transport only after every saved gate exists."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess

R = Path(__file__).resolve().parents[1]
D = R / 'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011'
H = R / 'tmp/FedAvg100_cuda_actual_root_20261011'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
inp = read(D / 'ROOT_INPUTS.template.json')
review = R / 'tmp/FedAvg_cuda100_saved_cache_closure_root_review_20261011/SOURCE_REVIEW.json'
assert sha(review) == 'f8c9c4f0c3e268030039daad9f6378e8d08df7315e225f6555f1893414529135'
code = '''
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
x=__INPUTS__
rows={};missing=[];failures=[];workers=[]
for identity in x['exact_ids']:
 b=Path(x['producers'][x['record_producers'][identity]])
 p=b/'tasks'/identity/'ACCEPTED.json'
 if not p.is_file():missing.append(identity);continue
 a=read(p);assert a['id']==identity
 for role in ['receipt','arrays','binding','acceptance','approval']:
  assert sha(a[role]['path'])==a[role]['sha256'],(identity,role)
 z=read(a['acceptance']['path'])
 assert z['status']=='PER_RECORD_ORIGINAL800_CUDA_SAVED_WHOLE_ACCEPTED'
 assert z['native_comparison']['accepted'] and z['native_comparison']['max_abs_difference']<=1e-12
 rows[identity]=sha(p)
 f=b/'runs'/(identity+'.formal_failure.json')
 if f.exists():failures.append(str(f))
for p in Path('/proc').glob('[0-9]*'):
 try:
  argv=[z.decode(errors='replace') for z in (p/'cmdline').read_bytes().split(b'\\0') if z]
  if any(z==b+'/worker.py' for b in x['producers'].values() for z in argv):
   assert '--approval' in argv
   a=read(argv[argv.index('--approval')+1])
   if a['id'] in x['exact_ids']:workers.append({'pid':int(p.name),'id':a['id']})
 except (FileNotFoundError,ProcessLookupError):pass
dep={k:sha(Path(b)/'DEPLOYMENT_SEAL.json') for k,b in x['producers'].items()}
cohorts={sha(Path(b)/'EXACT800_INPUTS.json') for b in x['producers'].values()};assert len(cohorts)==1
print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'accepted_sha256':rows,'missing':missing,'selected_failures':failures,'selected_workers':workers,'deployment_sha256':dep,'cohort_sha256':cohorts.pop()}))
'''.replace('__INPUTS__', repr(inp))
compile(code, '<fedavg100-bind>', 'exec')
p = subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','python3 -'],input=code.encode(),capture_output=True,timeout=100)
assert p.returncode == 0, p.stderr.decode(errors='replace')
v = json.loads(p.stdout)
H.mkdir(exist_ok=True)
obs = H / ('OBSERVATION_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
with obs.open('x',encoding='utf8',newline='\n') as f: json.dump(v,f,indent=2);f.write('\n')
if v['missing'] or v['selected_workers'] or v['selected_failures']:
    print(json.dumps({'status':'NOT_READY_NO_AUTHORIZATION_WRITTEN','accepted':len(v['accepted_sha256']),'missing':len(v['missing']),'selected_workers':v['selected_workers'],'failures':v['selected_failures'],'observation':str(obs),'sha256':sha(obs)}))
    raise SystemExit(0)
assert len(v['accepted_sha256']) == 100
prior=read(R/'tmp/baseline_old800_runtime_root_20261011_attempt003/ROOT_FIXED3_AUDIT_INPUTS.json')
inp.update(status='ROOT_AUTHORIZED_EXACT_FEDAVG100_TRANSPORT_NOT_ADOPTED',root_authorized_transport=True,
           source_review={'path':str(review),'sha256':sha(review)},accepted_sha256=v['accepted_sha256'],
           deployment_sha256=v['deployment_sha256'],cohort_sha256=v['cohort_sha256'],
           attempt002_failure=prior['attempt002_failure'],preserved_prior_failure=prior['preserved_prior_failure'],
           actual_observation={'path':str(obs),'sha256':sha(obs)},new_scientific_adoptions=0)
target=H/'ROOT_TRANSPORT_INPUTS.json'
with target.open('x',encoding='utf8',newline='\n') as f:json.dump(inp,f,indent=2);f.write('\n')
print(json.dumps({'status':'EXACT100_ACTUAL_LINUX_GATES_BOUND_TRANSPORT_ONLY','path':str(target),'sha256':sha(target)}))
