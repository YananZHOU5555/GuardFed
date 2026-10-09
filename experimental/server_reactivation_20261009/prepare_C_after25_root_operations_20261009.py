"""Reuse the accepted C_after20 root transport for the fixed next three checkpoints."""
from pathlib import Path
import ast, hashlib, importlib.util, json

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'tmp/celeba_mechanism_C_after20_root_operations_20261009'
NEW = ROOT/'tmp/celeba_mechanism_C_after25_root_operations_20261009'
NEW.mkdir(exist_ok=True)
replacements = {
    'C_after20':'C_after25', 'C_AFTER20':'C_AFTER25', 'EXACT5':'EXACT3', 'exact5':'exact3',
    'df9d23010d0d0acc3d81b8f53b061c69e8eca8228febc3a1a2fc35b0cc1c3739':'383d53753e907d4810048fbd6fc77ccafbcd4bd3067cfe1fc3de3ce4b6390a87',
    'd359805ca590958da37a02735fdd2daf1efe615332d91c37118835b97e24b260':'adbad62e9a7fa3936790255e40bd29262441798d17e104a8cda26c1d9b722560',
    '414816e646a0ba979f878946a83e3e0d1636b5b62ca4e50707f6eeb96771a5ae':'59876871743440676378cbd586b3092b158d1c315043b72cf2432b4360e2033a',
    '4dcffa2615b679f217ac953456b7cafc91f90f6ff3abdc670f95b74a2635016d':'bd2e925311c2b4c0519836ee8546bff1d9829b489801d2c5bd243c31b29fca46',
    '88c1272b423d91908dbd60b8095c5a4434c2b339807f85ead179afdda802c679':'fe2f65d5a0353949b2d15cb5c04828d8fdd7e7574d7788f00d7b096ece3489de',
    'inventory_actual125_Full100refs.json':'inventory_actual128_Full100refs.json',
    "review['native_accepted_snapshot'] == 125":"review['native_accepted_snapshot'] == 128",
    'old120_records_exact':'old125_records_exact',
    '[f\'minus_C_IID_FedSA_seed{s}\' for s in (91001, 91003, 91005, 91006, 91008)]':'[f\'minus_C_IID_FedSA_seed{s}\' for s in (91002, 91004, 91007)]',
    '== 120':'== 125', '==120':'==125', '== 5':'== 3', '==5':'==3',
    '(45,120,15)':'(27,72,9)',
    'prior_three_view_models=120,accepted_new=5,cumulative_three_view_models=125':'prior_three_view_models=125,accepted_new=3,cumulative_three_view_models=128',
    'original120_unchanged':'original125_unchanged', 'prior120_root_adoption_sha256':'prior125_root_adoption_sha256',
    'original120_not_rerun':'original125_not_rerun',
    'C_after12_20261009/execution_candidate/backups/incremental_20261009T202623Z':'C_after20_20261009/execution_candidate/backups/incremental_20261009T210932Z',
    '817d5f8ebebb566ee4b851fd600edcddaf07a410d29e618d1e5a821c5748b775':'ba3edd176219106c781e8b437017ff7444cee8b3e47073b25932928682661ef3',
}
rows = []
for name in ('deploy.py','observe.py','backup.py','adopt.py'):
    before = (OLD/name).read_text(encoding='utf-8-sig'); after = before
    for a,b in replacements.items(): after=after.replace(a,b)
    if name=='deploy.py':
        after=after.replace("('guardfed_celeba_mechanism_valid_C_after12','EXITED')", "('guardfed_celeba_mechanism_valid_C_after20','EXITED')")
        after=after.replace('PRIOR_C_AFTER12_EXITED','PRIOR_C_AFTER20_EXITED')
    after=after.replace('and prior112.', 'and prior125.')
    ast.parse(after)
    assert after!=before and 'EXACT5' not in after and 'exact5' not in after
    with (NEW/name).open('x',encoding='utf8',newline='\n') as f:f.write(after)
    rows.append(dict(path=name,sha256=hashlib.sha256((NEW/name).read_bytes()).hexdigest(),
        accepted_transport_source_sha256=hashlib.sha256((OLD/name).read_bytes()).hexdigest()))

checks=0
expected=[f'minus_C_IID_FedSA_seed{s}' for s in (91002,91004,91007)]
for name in ('backup.py','adopt.py'):
    spec=importlib.util.spec_from_file_location('after25_'+name[:-3],NEW/name)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    good=dict(service='guardfed_celeba_mechanism_valid_C_after25 EXITED',processes=[],batch_failure=None,
        batch_complete={'completed':3},completed=[{'id':x} for x in expected])
    mod.check_terminal(good,expected);checks+=1
    cases=[]
    for field,value in [('service','RUNNING'),('processes',[1]),('batch_failure',{'failure':True}),('batch_complete',None),
                        ('completed',[{'id':expected[0]}]*3),('completed',good['completed'][:2]),
                        ('completed',[{'id':x} for x in expected[:2]+['wrong_id']])]:
        case=dict(good);case[field]=value;cases.append(case)
    for case in cases:
        try:mod.check_terminal(case,expected)
        except AssertionError:checks+=1
        else:raise AssertionError('Terminal refusal failed')
with (NEW/'TRANSPORT_REBIND.json').open('x',encoding='utf8') as f:
    json.dump(dict(source_only=True,SSH=False,original_science_changed=False,exact3=True,prior125_not_replayed=True,
        terminal_positive_and_refusal_checks=checks,members=rows),f,indent=2);f.write('\n')
assert checks==16
print(json.dumps({'transport_files':rows,'terminal_positive_and_refusal_checks':checks}))
