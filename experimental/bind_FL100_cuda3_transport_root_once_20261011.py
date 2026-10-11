from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
R=Path.cwd();D=R/'tmp/FL100_cuda_actual_root_20261011';H=R/'tmp/FL100_cuda3_saved_closure_prepare_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=R/'tmp/FL100_cuda3_saved_closure_root_review_20261011/SOURCE_REVIEW.json'
assert sha(review)=='fdd72a90a76ab5adb2c13b9dbfbb05eee46b26d65ec7ab39e965ac54894aa11c'
assert sha(H/'FILES_SHA256.json')=='88d9b7d39df99fa1da79cc403e6009deef67f92c2f2691ad3f133fc7b7e35ac8'
obs_path=D/'OBSERVATION_20261011T080528Z.json';obs=json.loads(obs_path.read_bytes())
assert sha(obs_path)=='74130a83cf4ef9b4644c6ba7d240e455e2c9bb51e685aa8ba538b5bf68ff7dfe'
inp=json.loads((H/'ROOT_INPUTS.template.json').read_bytes());ids=inp['exact_ids']
assert obs['canary_complete']['ids']==ids and not obs['failures'] and len(obs['accepted'])==3
code=r'''
from pathlib import Path
import hashlib,json,subprocess
B=Path('/workspace/guardfed_checks/uniform_cuda_FL100_20261011_canary1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert not (B/'canary/QUEUE_FAILURE.json').exists()
p=subprocess.run(['supervisorctl','status','guardfed_uniform_cuda_FL100_canary_valid'],capture_output=True,text=True,timeout=20)
assert p.stdout.split()[:2]==['guardfed_uniform_cuda_FL100_canary_valid','EXITED']
ids=json.loads((B/'canary/COMPLETE.json').read_bytes())['ids']
print(json.dumps({'base':str(B),'accepted_sha256':{i:sha(B/'tasks'/i/'ACCEPTED.json') for i in ids},'deployment_sha256':sha(B/'DEPLOYMENT_SEAL.json'),'cohort_sha256':sha(B/'adapter/EXACT100.json'),'canary_complete_sha256':sha(B/'canary/COMPLETE.json')}))
'''
p=subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','python3 -'],input=code.encode(),capture_output=True,timeout=60)
assert p.returncode==0,p.stderr.decode(errors='replace');v=json.loads(p.stdout)
assert v['accepted_sha256']=={a['id']:a['sha256'] for a in obs['accepted']}
assert v['deployment_sha256']=='1e2384f2e87caafb56cf047a2dc6f8037621a246472d86d8d265195bde016a7d'
inp.update(v,status='ROOT_AUTHORIZED_FL_FIXED3_TRANSPORT_ONLY_NOT_ADOPTED',root_authorized_transport=True,
 source_review={'path':str(review),'sha256':sha(review)},actual_observation={'path':str(obs_path),'sha256':sha(obs_path)},new_scientific_adoptions=0)
target=D/'ROOT_FIXED3_TRANSPORT_INPUTS.json'
with target.open('x',encoding='utf8',newline='\n') as f:json.dump(inp,f,indent=2);f.write('\n')
print(json.dumps({'path':str(target),'sha256':sha(target),'ids':ids,'new_scientific_adoptions':0}))
