"""Rebind existing strict transfer/adoption helpers after actual V2 startup."""
from pathlib import Path
import ast, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009'
EX=BASE/'execution_candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
startup=read(EX/'ROOT_STARTUP_OBSERVATION.json')
assert startup['status']=='ROOT_AFTER82_V2_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
assert startup['deployment_receipt_sha256']==sha(EX/'deployment_receipt.json')
root_approval=read(EX/'deployment_receipt.json')['root_approval_sha256']
assert root_approval==sha(EX/'ROOT_APPROVED.json')
pins=[
 ('d05a0b81620d1791858a9f71405252443e593600f390359175ff961dde0e2bec',sha(BASE/'FILES_SHA256.json')),
 ('0fe232aab75a870b7840fd6ef0bba85b3d12c541ca49a0c0976f3f8a7095352f',sha(EX/'EXECUTION_SOURCE_SHA256.json')),
 ('72aca76f626580faf465c61f35b2274068e76782eb1bcef543e98da23c2e6d1e',sha(BASE/'inventory_actual92_Full100refs.json')),
 ('a95018b7ccc62472d7b1f796e7a2d1307a72bb4bd4905397c9f623ca6a4d2f60',sha(EX/'ROOT_STARTUP_OBSERVATION.json')),
 ('0438a11351de81103f52983dd86b5eee5f5ecbabdef4585dbb44d01602af838d',root_approval),
]
results=[]
for role in ('backup','adopt'):
 old=ROOT/f'tmp/{role}_mechanism_after71_root_20261009.py'
 text=old.read_text(encoding='utf8')
 text=text.replace('after71','after82_v2').replace('AFTER71','AFTER82_V2').replace('exact11','exact10')
 text=text.replace('actual82_Full100refs','actual92_Full100refs').replace('==11','==10')
 for before,after in pins:
  assert before in text,(role,before)
  text=text.replace(before,after)
 if role=='backup':
  before="'python -c '+shlex.quote(code)],capture_output=True,timeout=120)"
  assert before in text
  text=text.replace(before,"'python -B -'],input=code.encode(),capture_output=True,timeout=120)")
 else:
  before="tmp/celeba_mechanism_valid_incremental_next11_20261009/execution_candidate/backups/incremental_20261009T152532Z/ROOT_ADOPTION_REVIEW.json"
  assert before in text
  text=text.replace(before,"tmp/celeba_mechanism_valid_incremental_after71_20261009/execution_candidate/backups/incremental_20261009T163050Z/ROOT_ADOPTION_REVIEW.json")
  for before,after in [
   ('692ecd168ecab0b9c960965decb68424ce2ad80a5cf7ca452d2739da6b0a768a','fb2bc745e7a6b6aa3d7d4cb898aefa31c6acbf73bee7a8e1a30ab667e03c204c'),
   ("read(prior)['cumulative_three_view_models']==71","read(prior)['cumulative_three_view_models']==82"),
   ("len(set(scope['excluded_prior_ids']))==71","len(set(scope['excluded_prior_ids']))==82"),
   ('(99,264,33)','(90,240,30)'),
   ('prior_three_view_models=71,accepted_new=11,cumulative_three_view_models=82','prior_three_view_models=82,accepted_new=10,cumulative_three_view_models=92'),
   ('original71_unchanged','original82_unchanged'),('prior71_root_adoption_sha256','prior82_root_adoption_sha256')]:
   assert before in text,(role,before)
   text=text.replace(before,after)
 assert 'shlex.quote(' not in text
 ast.parse(text)
 target=ROOT/f'tmp/{role}_mechanism_after82_v2_root_20261009.py'
 with target.open('x',encoding='utf8',newline='\n') as stream:stream.write(text)
 results.append(dict(role=role,old_sha256=sha(old),new_path=target.relative_to(ROOT).as_posix(),new_sha256=sha(target)))
print(json.dumps(dict(status='ROOT_AFTER82_V2_CLEAN_BACKUP_AND_ADOPTION_HELPERS_PREPARED_NOT_EXECUTED',helpers=results)))
