"""Derive the one-shot installer; this preparation executes no remote code."""
from pathlib import Path
import hashlib
R=Path(__file__).resolve().parents[1]
parent=R/'tmp/install_old800_cuda_canary_attempt002_root_once_20261011.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='b2ee2ddbeb9da435aa8164926b22b7ce13eb2d48abbde944cd996d73e6fe561e'
original=parent.read_text(encoding='utf-8')
new=original.replace('baseline_old800_runtime_root_20261011_attempt002','baseline_old800_runtime_root_20261011_attempt003')
new=new.replace('ROOT_CALLER_ATTEMPT002_REVIEW.json','ROOT_CALLER_ATTEMPT003_REVIEW.json')
new=new.replace('guardfed_uniform_cuda800_canary_v2_valid','guardfed_uniform_cuda800_canary_v3_valid')
old="assert not list((failed_base/'tasks').glob('*/ACCEPTED.json')) and not list((failed_base/'runs').glob('*/receipt.json'))"
extra=r'''
previous=Path('/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002')
assert sha(previous/'canary/QUEUE_FAILURE.json')=='6ecd9f3e90adc44713d976772ac6d05bb29d70ee4325a10b10356b1d03f94358'
assert sha(previous/'runs/FedAA-DDPG_policy0.001_keep16_local0.001_IID_Benign_seed91001.formal_failure.json')=='32500f3565b680041b1d0d8f7f479973b79c2a7e4446d2e4ad6680f46e47fbec'
previous_status=subprocess.run(['supervisorctl','status','guardfed_uniform_cuda800_canary_v2_valid'],capture_output=True,text=True,timeout=20)
assert previous_status.stdout.split()[1]=='EXITED','No duplicate producer from previous attempt'
assert len(list((previous/'tasks').glob('*/ACCEPTED.json')))==1
assert not (previous/'runs/FedAA-DDPG_policy0.001_keep16_local0.001_IID_Benign_seed91001/receipt.json').exists()
reused_cuda={'FedAvg_IID_Benign_seed91001':{
 'task_accepted':{'path':str(previous/'tasks/FedAvg_IID_Benign_seed91001/ACCEPTED.json'),'sha256':'a0507114f5c5de34fcfcff6527df6cf3032f5126755771ac8fe410d346d0f2b1'},
 'deployment_seal':{'path':str(previous/'DEPLOYMENT_SEAL.json'),'sha256':'c411816b1f73c47e400f3cbc1adbd6e52e89975f360f131429f5279cddf70a4f'},
 'queue_auth':{'path':str(previous/'ROOT_QUEUE_AUTH.json'),'sha256':'c5684f028a11f59d8d7eab3dd229c91e22a0aa329f70441e453d526720b46270'}}}
'''
assert new.count(old)==1;new=new.replace(old,old+extra)
old="'source_review_sha256':'__API_REVIEW_SHA__','caller_review_sha256':'__CALLER_REVIEW_SHA__'}"
replacement="'source_review_sha256':'__API_REVIEW_SHA__','caller_review_sha256':'__CALLER_REVIEW_SHA__',\n 'reused_cuda':reused_cuda,'recovery_from':str(previous),'recovery_new_ids':prepared['new_canary_ids']}"
assert new.count(old)==1;new=new.replace(old,replacement)
old="first=prepared['canary_ids'][0]"
replacement="sys.path.insert(0,str(base))\nfrom reuse_fedavg import verify_reused\nverified=verify_reused(auth,cohort)\nassert list(verified)==prepared['reused_canary_ids']\nfirst=prepared['new_canary_ids'][0]"
assert new.count(old)==1;new=new.replace(old,replacement)
new=new.replace("'status':'ROOT_INSTALLED_AND_STARTED_FIXED3_CANARIES_NOT800_COMPLETE'", "'status':'ROOT_INSTALLED_RECOVERY_NEW2_REUSED1_NOT800_COMPLETE'")
old="'canary_ids':prepared['canary_ids'],'cohort':pin(base/'EXACT800_INPUTS.json'),"
replacement="'canary_ids':prepared['canary_ids'],'new_ids':prepared['new_canary_ids'],'reused_ids':list(verified),\n 'cohort':pin(base/'EXACT800_INPUTS.json'),"
assert new.count(old)==1;new=new.replace(old,replacement)
new=new.replace("assert caller_review['worker_sha256']==H((D/'worker.py').read_bytes())", "assert caller_review['worker_sha256']==H((D/'worker.py').read_bytes())\nassert caller_review['reuse_validator_sha256']==H((D/'reuse_fedavg.py').read_bytes())")
compile(new,'install_old800_cuda_canary_attempt003_root_once_20261011.py','exec')
target=R/'tmp/install_old800_cuda_canary_attempt003_root_once_20261011.py'
with target.open('x',encoding='utf-8',newline='\n') as f:f.write(new)
print({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'status':'SOURCE_ONLY_NOT_EXECUTED'})
