"""Bind existing strict transfer operations to the measured C-after1 startup."""
from pathlib import Path
import ast,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'tmp/celeba_mechanism_valid_C1_gate_20261009'
BASE=ROOT/'tmp/celeba_mechanism_valid_C_after1_20261009';EX=BASE/'execution_candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(EX/'ROOT_STARTUP_OBSERVATION.json')=='f909a8551525930f345d77f930105ba91b476cea2b8f3fa0cee2415e2e33677c'
assert sha(EX/'deployment_receipt.json')=='22ac5ad68f377c02f9656a0d9296edc6b014b2f4ee9e2a3fb059d2749a4ac36f'
approval=read(EX/'deployment_receipt.json')['root_approval_sha256']
assert approval==sha(EX/'ROOT_APPROVED.json')
pins=[(sha(OLD/'FILES_SHA256.json'),sha(BASE/'FILES_SHA256.json')),
 (sha(OLD/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),sha(EX/'EXECUTION_SOURCE_SHA256.json')),
 (sha(OLD/'inventory_actual101_Full100refs.json'),sha(BASE/'inventory_actual112_Full100refs.json')),
 (sha(OLD/'execution_candidate/ROOT_STARTUP_OBSERVATION.json'),sha(EX/'ROOT_STARTUP_OBSERVATION.json')),
 (sha(OLD/'execution_candidate/ROOT_APPROVED.json'),approval)]
prepared=[]
for role in ('backup','adopt'):
 old=ROOT/f'tmp/{role}_mechanism_C1_gate_root_20261009.py';code=old.read_text(encoding='utf8')
 for before,after in [('celeba_mechanism_valid_C1_gate_20261009',BASE.name),('C1_gate','C_after1'),('exact1','exact11'),
  ('actual101_Full100refs','actual112_Full100refs')]+pins:
  assert before in code,(role,before)
  code=code.replace(before,after)
 code,n=re.subn(r'==1(?!\d)','==11',code)
 assert n>=3,(role,n)
 if role=='adopt':
  for before,after in [('C1_GATE','C_AFTER1'),
   ('tmp/celeba_mechanism_valid_incremental_after92_20261009/execution_candidate/backups/incremental_20261009T183102Z/ROOT_ADOPTION_REVIEW.json',
    'tmp/celeba_mechanism_valid_C1_gate_20261009/execution_candidate/backups/incremental_20261009T185829Z/ROOT_ADOPTION_REVIEW.json'),
   ('9050eb059a797c70f0ca977294989b5ae5757286dbc85b36d529012cb5ab72ee','d045665b066dafc25f9970adfdffef9c9a8a388575ec87b9b54d5dcabfa65cab'),
   ("read(prior)['cumulative_three_view_models']==100","read(prior)['cumulative_three_view_models']==101"),
   ("len(set(scope['excluded_prior_ids']))==100","len(set(scope['excluded_prior_ids']))==101"),
   ('(9,24,3)','(99,264,33)'),
   ('prior_three_view_models=100,accepted_new=1,cumulative_three_view_models=101','prior_three_view_models=101,accepted_new=11,cumulative_three_view_models=112'),
   ('original100_unchanged','original101_unchanged'),('prior100_root_adoption_sha256','prior101_root_adoption_sha256')]:
   assert before in code,(role,before)
   code=code.replace(before,after)
 ast.parse(code)
 target=ROOT/f'tmp/{role}_mechanism_C_after1_root_20261009.py'
 if target.exists():assert target.read_text(encoding='utf8')==code,'Preserve differing prepared helper'
 else:
  with target.open('x',encoding='utf8',newline='\n') as stream:stream.write(code)
 prepared.append(dict(role=role,path=target.relative_to(ROOT).as_posix(),sha256=sha(target),source_sha256=sha(old)))
print(json.dumps(dict(status='ROOT_C_AFTER1_STRICT_TRANSFER_HELPERS_PREPARED_NOT_EXECUTED',helpers=prepared)))
