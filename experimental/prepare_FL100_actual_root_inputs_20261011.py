"""Read-only actual resource observation, then bind already reviewed fixed3 inputs."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,subprocess,time
R=Path.cwd();D=R/'tmp/FL100_cuda_actual_root_20261011';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not D.exists();D.mkdir()
reviews={
 'formal_api':R/'tmp/FL100_uniform_cuda_formal_independent_review_20261011/SOURCE_REVIEW.json',
 'caller':R/'tmp/FL100_cuda_runtime_caller_independent_review_20261011/SOURCE_REVIEW.json',
 'runtime_mapping':R/'tmp/FL100_cuda_runtime_caller_independent_review_20261011/RUNTIME_MAPPING_REVIEW.json',
}
expected={'formal_api':'c5859287b94bc19f14f56ea36bccbde52e0734383841fdeab91486fc40e7a509','caller':'a2a165222fd62066a16d67a963f0b4c126d88c6c2e6b5b1619346e80be32278e','runtime_mapping':'575c782ea9eea8eca937d3f262639e88d85130b828e5154dac1d23c572e91448'}
for k,p in reviews.items():assert sha(p)==expected[k]
code=r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
b=Path('/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt003')
auth=json.loads((b/'ROOT_QUEUE_AUTH.json').read_bytes());protected=auth['protected_services'];observed={}
names=list(dict.fromkeys(list(protected)+['sglang','guardfed_celeba_mechanism_formal','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_fednga_fullcoverage_v1','guardfed_uniform_cuda500_resume2_valid','guardfed_uniform_cuda800_remaining_v3_valid']))
for n in names:
 p=subprocess.run(['supervisorctl','status',n],capture_output=True,text=True,timeout=20);words=p.stdout.split();assert len(words)>=2 and words[0]==n,(n,p.stdout,p.stderr)
 assert words[1] in ['RUNNING','EXITED','STOPPED'],(n,p.stdout)
 observed[n]={'status':words[1],'stdout':p.stdout.strip()}
assert observed['sglang']['status']=='STOPPED'
siblings=set()
for part in Path('/sys/devices/system/cpu/cpu111/topology/thread_siblings_list').read_text().strip().split(','):
 ends=[int(x) for x in part.split('-')];siblings.update(range(ends[0],ends[-1]+1))
restricted=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  for t in (p/'task').iterdir():
   mask=set(os.sched_getaffinity(int(t.name)))
   if len(mask)<=16 and mask&siblings:restricted.append({'pid':int(p.name),'tid':int(t.name),'affinity':sorted(mask)})
 except (FileNotFoundError,ProcessLookupError):pass
assert not restricted,restricted
p=subprocess.run(['nvidia-smi','--id=0','--query-gpu=uuid,memory.free,gpu_recovery_action','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stderr
uuid,free,recovery=[x.strip() for x in p.stdout.split(',')]
assert uuid=='GPU-da357477-30a7-fddc-344b-a20513b9a2d0' and int(free)>=2048 and recovery=='None'
author=b/'author.json';assert sha(author)=='a1f1be152a0aa5668b91d73d7297ea3e9275efaac8d82c3b22cfbe97b7b08fce'
print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'cpu111_siblings':sorted(siblings),'conflicting_restricted_tasks':restricted,'GPU_uuid':uuid,'GPU_free_MiB':int(free),'RecoveryAction':recovery,'services':observed,'author_decision':{'path':str(author),'sha256':sha(author)},'read_only':True,'not_formal_resource_callback':True}))
'''
p=subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','python3 -'],input=code.encode(),capture_output=True,timeout=90)
(D/'RESOURCE_STDOUT').write_bytes(p.stdout);(D/'RESOURCE_STDERR').write_bytes(p.stderr)
assert p.returncode==0,p.stderr.decode(errors='replace')
v=json.loads(p.stdout);(D/'ROOT_READONLY_RESOURCE_OBSERVATION.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
P=R/'tmp/FL100_cuda_runtime_caller_prepare_20261011'
inp=json.loads((P/'ROOT_INSTALL_INPUTS.template.json').read_bytes())
inp.update(status='ROOT_AUTHORIZED_FL100_FIXED3_CALLER_V1',execution_authorized=True,
 caller_source_seal_sha256=sha(P/'FILES_SHA256.json'),mapping_source_seal_sha256=sha(R/'tmp/FL100_cuda_runtime_mapping_prepare_20261011/FILES_SHA256.json'),
 gpu_uuid=v['GPU_uuid'],deadline_unix=time.time()+48*3600,author_decision=v['author_decision'],source_reviews_accepted_by_root=True,
 reviews={k:{'path':str(p),'sha256':sha(p)} for k,p in reviews.items()},
 protected_services={k:([z['status']] if z['status']!='RUNNING' else ['RUNNING','EXITED']) for k,z in v['services'].items()})
assert inp['caller_source_seal_sha256']=='a3778dbaf1b68d2540993d2ba47a1b93625a5e03b6657ce301834b5db6f831cb'
assert inp['mapping_source_seal_sha256']=='c08b726a846c09253632873358b2e5d761d3a1e292388780307938b9063419f0'
with (D/'ROOT_INSTALL_INPUTS.json').open('x',encoding='utf8',newline='\n') as f:json.dump(inp,f,indent=2);f.write('\n')
print(json.dumps({'path':str(D/'ROOT_INSTALL_INPUTS.json'),'sha256':sha(D/'ROOT_INSTALL_INPUTS.json'),'observation_utc':v['utc'],'CPU111_siblings':v['cpu111_siblings'],'GPU0_uuid':v['GPU_uuid'],'reviews':{k:sha(p) for k,p in reviews.items()},'actual_service_not_started_by_this_script':True}))
