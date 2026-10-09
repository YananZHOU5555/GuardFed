"""Bind existing strict transfer operations to the actual single C gate startup."""
from pathlib import Path
import ast, hashlib, json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_C1_gate_20261009'
EX=BASE/'execution_candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(EX/'ROOT_STARTUP_OBSERVATION.json')=='25f13ee8043ba002aea631daae3e616279f563c44587c4303111713a6a1a7e3c'
assert sha(EX/'deployment_receipt.json')=='85cb06255a69ea32bae732dc2ba39232b5cc2e10fbe1fbed8c17bb414c11b4b9'
approval=read(EX/'deployment_receipt.json')['root_approval_sha256']
assert approval==sha(EX/'ROOT_APPROVED.json')
pins=[
 ('832e02a7ab0bc58dd22373c1793c39fc7926a4e961d4fba0c750964eb7b7a94a',sha(BASE/'FILES_SHA256.json')),
 ('68b11d80698d1073250736ad73695a79b29ed26d0bbc133441876a56f6f5c5bf',sha(EX/'EXECUTION_SOURCE_SHA256.json')),
 ('156cfea40a65e47f122b96a2b08369f387bf5b7397a0c82e05fe616888ef61bd',sha(BASE/'inventory_actual101_Full100refs.json')),
 ('c773018542610620d1640f21bf8b8c443b7bff442898a922fc56ea93ee228030',sha(EX/'ROOT_STARTUP_OBSERVATION.json')),
 ('90f2b12d1c966ba9f3818f23ca5d8563ec9e3ea64e57c142c29c9f3aaf3a1560',approval),
]
prepared=[]
for role in ('backup','adopt'):
 old=ROOT/f'tmp/{role}_mechanism_after92_root_20261009.py'
 code=old.read_text(encoding='utf8')
 for before,after in [('celeba_mechanism_valid_incremental_after92_20261009',BASE.name),
       ('after92','C1_gate'),('exact8','exact1'),
       ('actual100_Full100refs','actual101_Full100refs'),('==8','==1')]+pins:
  assert before in code,(role,before)
  code=code.replace(before,after)
 if role=='adopt':
  assert 'AFTER92' in code
  code=code.replace('AFTER92','C1_GATE')
  for before,after in [
   ('tmp/celeba_mechanism_valid_incremental_after82_v2_20261009/execution_candidate/backups/incremental_20261009T174922Z/ROOT_ADOPTION_REVIEW.json',
    'tmp/celeba_mechanism_valid_incremental_after92_20261009/execution_candidate/backups/incremental_20261009T183102Z/ROOT_ADOPTION_REVIEW.json'),
   ('b9e40d1ca565c0bcf146058433ff3e037ab4e824aa6972d1a3f3f47a088e8683','9050eb059a797c70f0ca977294989b5ae5757286dbc85b36d529012cb5ab72ee'),
   ("read(prior)['cumulative_three_view_models']==92","read(prior)['cumulative_three_view_models']==100"),
   ("len(set(scope['excluded_prior_ids']))==92","len(set(scope['excluded_prior_ids']))==100"),
   ('(72,192,24)','(9,24,3)'),
   ('prior_three_view_models=92,accepted_new=8,cumulative_three_view_models=100','prior_three_view_models=100,accepted_new=1,cumulative_three_view_models=101'),
   ('original92_unchanged','original100_unchanged'),('prior92_root_adoption_sha256','prior100_root_adoption_sha256')]:
   assert before in code,(role,before)
   code=code.replace(before,after)
 ast.parse(code)
 target=ROOT/f'tmp/{role}_mechanism_C1_gate_root_20261009.py'
 with target.open('x',encoding='utf8',newline='\n') as stream:stream.write(code)
 prepared.append(dict(role=role,path=target.relative_to(ROOT).as_posix(),sha256=sha(target),source_sha256=sha(old)))
print(json.dumps(dict(status='ROOT_C1_STRICT_TRANSFER_HELPERS_PREPARED_NOT_EXECUTED',helpers=prepared)))
