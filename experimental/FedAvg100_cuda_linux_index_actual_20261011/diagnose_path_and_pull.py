from pathlib import Path
import subprocess,json,hashlib
H=Path(__file__).resolve().parent
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','python3 -B -']
body=r'''
from pathlib import Path
import json,hashlib
p=Path('/workspace/GuardFed-celeba-expanded/data/celeba/derived/rgb64_v1/metadata.npz');r=p.resolve()
expected='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads(Path('/workspace/guardfed_checks/FedAvg100_cuda_adopted_cache_index_20261011_actual001/COHORT_SELECTION.json').read_bytes())
contracts=[x['data_contract'] for x in s['records']]
print(json.dumps({'original':str(p),'resolved':str(r),'original_exists':p.exists(),'resolved_is_file':r.is_file(),'resolved_is_symlink':r.is_symlink(),'resolved_stable':r.resolve()==r,'sha256':sha(r),'original_sha256':sha(p),'expected':expected,'first_data_contract':contracts[0],'metadata_contract_references':[x for x in contracts[0].items() if 'metadata' in x[0]]}))
assert p!=r and r.is_absolute() and r.is_file() and not r.is_symlink() and r.resolve()==r and sha(r)==expected==sha(p)
'''
r=subprocess.run(cmd,input=body.encode(),capture_output=True,timeout=60)
(H/'METADATA_PATH_DIAGNOSIS.stdout').write_bytes(r.stdout);(H/'METADATA_PATH_DIAGNOSIS.stderr').write_bytes(r.stderr)
(H/'METADATA_PATH_DIAGNOSIS_COMMAND.json').write_text(json.dumps({'command':cmd,'payload_sha256':hashlib.sha256(body.encode()).hexdigest(),'exit_code':r.returncode},indent=2)+'\n',encoding='utf8')
print(r.stdout.decode());assert r.returncode==0
base='/workspace/guardfed_checks/FedAvg100_cuda_adopted_cache_index_20261011_actual001/'
names=['ROOT_INDEX_INPUTS.json','ROOT_CACHE_INDEX.json','BUILD_COMMAND.json','BUILD.stdout']
pull=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=20']+['root@89.22.197.55:'+base+n for n in names]+[str(H)]
rr=subprocess.run(pull,capture_output=True,timeout=60)
(H/'PULL_COMMAND.json').write_text(json.dumps({'command':pull,'exit_code':rr.returncode},indent=2)+'\n',encoding='utf8');assert rr.returncode==0
proof=json.loads((H/'BUILD_COMMAND.json').read_bytes())
for n in names+['BUILD.stderr']:assert hashlib.sha256((H/n).read_bytes()).hexdigest()==proof['files'][n]['sha256'] if n!='BUILD_COMMAND.json' else True
