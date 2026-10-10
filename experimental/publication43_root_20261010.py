"""Freeze the reviewed source/current-startup increment on the external checkout."""
from pathlib import Path
import hashlib,importlib.util,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'tmp/publication_increment43_20261010'
HERE=ROOT/'tmp/publication43_root_20261010';HERE.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(PACKAGE/'FILES_SHA256.json')=='80151e709caae9a97443631c58987507bcb6f4cdcb58791d0910caeec682eed4'
for n,pin in json.loads((PACKAGE/'FILES_SHA256.json').read_bytes())['files'].items():
 assert sha(PACKAGE/n)==pin['sha256']
spec=importlib.util.spec_from_file_location('publication43_root_publisher',PACKAGE/'publish_increment43.py')
pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
draft=json.loads((PACKAGE/'CURATED_INPUTS_DRAFT.json').read_bytes())
startup=ROOT/'tmp/celeba_mechanism_remaining620_root_operations_20261010/ROOT_STARTUP_REVIEW_V2.json'
assert sha(startup)=='ecd9196f830548a052ed4d480a05543ea421d84d618269c478a2d7e5d7e4fea4'
draft['bindings']['remaining620_startup']=dict(path=startup.relative_to(ROOT).as_posix(),sha256=sha(startup),expect={
 '/actual_service':'guardfed_celeba_mechanism_remaining620_valid_v2a',
 '/remote_strict_closed':1,'/accepted_offserver':0,'/root_adopted':0,'/new_training':0,'/test':False,
 '/all_closed_original_native_rows_exact':True})
state=ROOT/'docs/server_deployment_20260923/training_20260923/TRAINING_STATE.json'
draft['bindings']['current_state']['sha256']=sha(state)
draft['status']='ROOT_CURATED_ACTUAL_STARTUP_INPUTS_NOT_YET_STAGED'
draft['root_binding_note']='Actual cutoff: native188 / accepted mechanism three-view180 / FL22; first620 remote closure is explicitly not adopted offserver. Source/author decisions/real queues are included; no future model hashes or unseen-test claim.'
adds=[]
ops=ROOT/'tmp/celeba_mechanism_remaining620_root_operations_20261010'
adds.extend(p.relative_to(ROOT).as_posix() for p in ops.iterdir() if p.is_file() and p.suffix in {'.py','.sh','.conf','.json'})
for folder in ['tmp/celeba_mechanism_remaining_evaluation_transport_20261010','tmp/celeba_remaining620_transport_independent_review_20261010','tmp/celeba_logofair32_root_adoption_20261010']:
 base=ROOT/folder
 if folder not in draft['allowed_paths']:draft['allowed_paths'].append(folder)
 seal=base/'FILES_SHA256.json'
 if seal.exists():
  for n in json.loads(seal.read_bytes())['files']:adds.append(folder+'/'+n)
  adds.append(folder+'/FILES_SHA256.json')
 else:
  adds.extend(p.relative_to(ROOT).as_posix() for p in base.iterdir() if p.is_file() and p.suffix in {'.py','.json','.md'})
adds+=['tmp/celeba_gradient_screen64_v2_root_operations_20261010/ATTEMPT2_OBSERVATION_20261010T043421773777Z.json',
 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/MONITOR_HANDOFF.md',
 'tmp/publication43_root_20261010.py']
for p in adds:
 if not any(p==a or p.startswith(a.rstrip('/')+'/') for a in draft['allowed_paths']):draft['allowed_paths'].append(p)
draft['files']=sorted(set(draft['files']+adds))
path=HERE/'ROOT_REVIEWED_SPEC.json'
with path.open('x',encoding='utf8') as f:f.write(json.dumps(draft,ensure_ascii=False,indent=2)+'\n')
planned=pub.plan(draft)
assert planned['total_bytes']<10_000_000
storage=pub.storage(planned['total_bytes']*3)
assert pub.git('rev-parse','HEAD').decode().strip()==pub.PARENT
assert not pub.git('status','--porcelain','--untracked-files=all').strip()
# Materialize the prior tracked attribute file exactly. A sparse checkout need
# not have it in the working tree; retaining parent rules avoids dropping them.
old=pub.git('show','HEAD:.gitattributes')
attr=pub.CHECKOUT/'.gitattributes'
if attr.exists():assert attr.read_bytes()==old
else:attr.write_bytes(old)
assert not pub.git('status','--porcelain','--untracked-files=all').strip()
assert pub.git('ls-remote','--exit-code','origin','refs/heads/'+pub.BRANCH).decode().split()[0]==pub.PARENT
pub.freeze(path,sha(path),'freeze001')
frozen=pub.OUTPUT_ROOT/'freeze001/FROZEN_INPUTS.json'
with (HERE/'ROOT_FREEZE_REVIEW.json').open('x',encoding='utf8') as f:
 f.write(json.dumps(dict(status='ROOT_CURATED_SOURCE_AND_ACTUAL_STARTUP_FREEZE_PASS',spec_sha256=sha(path),
  frozen_path=str(frozen),frozen_sha256=sha(frozen),files=len(planned['files']),bytes=planned['total_bytes'],
  old_gitattributes_materialized_exact=True,old_gitattributes_sha256=hashlib.sha256(old).hexdigest(),
  current_parent_and_remote_exact=True,storage=storage,accepted=draft['accepted'],test=False,goal_complete=False),indent=2)+'\n')
pub.stage(frozen,sha(frozen),'stage001')
print(json.dumps(dict(status='ROOT_STAGED_NOT_COMMITTED',spec_sha256=sha(path),frozen_sha256=sha(frozen))))
