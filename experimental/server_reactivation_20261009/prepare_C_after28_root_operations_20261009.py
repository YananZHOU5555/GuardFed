"""Reuse the accepted C_after25 transport for the next fixed eight checkpoints."""
from pathlib import Path
import ast, hashlib, importlib.util, json, re

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'tmp/celeba_mechanism_C_after25_root_operations_20261009'
NEW=ROOT/'tmp/celeba_mechanism_C_after28_root_operations_20261009'
BASE=ROOT/'tmp/celeba_mechanism_valid_C_after28_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(BASE/'HANDOFF.json')=='8d27ceb2f9afcf34e2619562434f37b20b81a401db2099bbb50212c5818237fe'
expected=[f'minus_C_IID_FedSA_seed{s}' for s in (91009,91010)]+[f'minus_C_IID_S-DFA_seed{s}' for s in range(91001,91007)]
assert json.loads((BASE/'SCOPE.json').read_bytes())['selected_ids']==expected
NEW.mkdir(exist_ok=True)
replacements={
 'C_after25':'C_after28','C_AFTER25':'C_AFTER28','EXACT3':'EXACT8','exact3':'exact8',
 '383d53753e907d4810048fbd6fc77ccafbcd4bd3067cfe1fc3de3ce4b6390a87':'1d06eca4b5eccf94fa5c510ca867d4281e0a8fe3bce5a2b17b18864e43bdcc67',
 'adbad62e9a7fa3936790255e40bd29262441798d17e104a8cda26c1d9b722560':'c823dce69ef511ff15faf1b964805184a6a8c10f271214de1cc6706446a49ba2',
 '59876871743440676378cbd586b3092b158d1c315043b72cf2432b4360e2033a':'c42c5c4234bdf19b1f50921f4991d585027b5cf7aeb1913a0850ffa7e42b9276',
 'bd2e925311c2b4c0519836ee8546bff1d9829b489801d2c5bd243c31b29fca46':'ab6f5ee0067fbfff1a09c1dabb210ed0e04c1c2b0976b8d345b6360caa7785ae',
 'fe2f65d5a0353949b2d15cb5c04828d8fdd7e7574d7788f00d7b096ece3489de':'7cf324b92be73420b7f4497673519c779666664d13a98ad93f4207434c1de635',
 'inventory_actual128_Full100refs.json':'inventory_actual136_Full100refs.json',
 "review['native_accepted_snapshot'] == 128":"review['native_accepted_snapshot'] == 136",
 'old125_records_exact':'old128_records_exact','original125':'original128','prior125':'prior128',
 "[f'minus_C_IID_FedSA_seed{s}' for s in (91002, 91004, 91007)]":repr(expected),
 '== 125':'== 128','==125':'==128','== 3':'== 8','==3':'==8',
 '(27,72,9)':'(72,192,24)',
 'prior_three_view_models=125,accepted_new=3,cumulative_three_view_models=128':'prior_three_view_models=128,accepted_new=8,cumulative_three_view_models=136',
 'C_after20_20261009/execution_candidate/backups/incremental_20261009T210932Z':'C_after25_20261009/execution_candidate/backups/incremental_20261009T213828Z',
 'ba3edd176219106c781e8b437017ff7444cee8b3e47073b25932928682661ef3':'0d1661bdc21025b957fa4e6ac39d4c8924cb50ab1b91920268119941312a615a'
}
pattern='|'.join(re.escape(k) for k in sorted(replacements,key=len,reverse=True))
old_seal={r['path']:r['sha256'] for r in json.loads((OLD/'TRANSPORT_REBIND.json').read_bytes())['members']}
rows=[]
for name in ('deploy.py','observe.py','backup.py','adopt.py'):
 assert sha(OLD/name)==old_seal[name]
 before=(OLD/name).read_text(encoding='utf-8-sig')
 after=re.sub(pattern,lambda m:replacements[m.group()],before)
 if name=='deploy.py':
  after=after.replace("('guardfed_celeba_mechanism_valid_C_after20','EXITED')","('guardfed_celeba_mechanism_valid_C_after25','EXITED')").replace('PRIOR_C_AFTER20_EXITED','PRIOR_C_AFTER25_EXITED')
 ast.parse(after)
 assert after!=before and 'EXACT3' not in after and 'exact3' not in after
 with (NEW/name).open('x',encoding='utf8',newline='\n') as f:f.write(after)
 rows.append(dict(path=name,sha256=sha(NEW/name),accepted_transport_source_sha256=sha(OLD/name)))
checks=0
for name in ('backup.py','adopt.py'):
 spec=importlib.util.spec_from_file_location('after28_'+name[:-3],NEW/name)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 good=dict(service='guardfed_celeba_mechanism_valid_C_after28 EXITED',processes=[],batch_failure=None,
  batch_complete={'completed':8},completed=[{'id':i} for i in expected])
 mod.check_terminal(good,expected);checks+=1
 for field,value in [('service','RUNNING'),('processes',[1]),('batch_failure',{'failure':True}),('batch_complete',None),
  ('completed',[{'id':expected[0]}]*8),('completed',good['completed'][:7]),
  ('completed',[{'id':i} for i in expected[:7]+['wrong_id']])]:
  case=dict(good);case[field]=value
  try:mod.check_terminal(case,expected)
  except AssertionError:checks+=1
  else:raise AssertionError('Terminal refusal failed')
assert checks==16
with (NEW/'TRANSPORT_REBIND.json').open('x',encoding='utf8') as f:
 json.dump(dict(source_only=True,SSH=False,original_science_changed=False,exact8=True,prior128_not_replayed=True,
  terminal_positive_and_refusal_checks=checks,members=rows),f,indent=2);f.write('\n')
print(json.dumps(dict(transport_files=rows,terminal_positive_and_refusal_checks=checks)))
