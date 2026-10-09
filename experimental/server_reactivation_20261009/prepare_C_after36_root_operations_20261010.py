"""Rebind the accepted transport to four newly accepted C checkpoints; never deploy here."""
from pathlib import Path
import ast, hashlib, importlib.util, json, re

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'tmp/celeba_mechanism_C_after28_root_operations_20261009'
NEW = ROOT/'tmp/celeba_mechanism_C_after36_root_operations_20261010'
BASE = ROOT/'tmp/celeba_mechanism_valid_C_after36_20261010'
PRIOR = ROOT/'tmp/celeba_mechanism_valid_C_after28_20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
expected = [f'minus_C_IID_S-DFA_seed{s}' for s in range(91007, 91011)]
assert read(BASE/'SCOPE.json')['selected_ids'] == expected
assert len(read(BASE/'SCOPE.json')['excluded_prior_ids']) == 136
for folder, seal in [(BASE,'FILES_SHA256.json'),(BASE/'execution_candidate','EXECUTION_SOURCE_SHA256.json')]:
    for row in read(folder/seal)['members']:
        assert sha(folder/row['path'])==row['sha256'] and (folder/row['path']).stat().st_size==row['size']
changes = {
    'celeba_mechanism_valid_C_after28_20261009': 'celeba_mechanism_valid_C_after36_20261010',
    'C_AFTER28':'C_AFTER36', 'C_after28':'C_after36', 'EXACT8':'EXACT4', 'exact8':'exact4',
    sha(PRIOR/'FILES_SHA256.json'):sha(BASE/'FILES_SHA256.json'),
    sha(PRIOR/'execution_candidate/EXECUTION_SOURCE_SHA256.json'):sha(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),
    sha(PRIOR/'PACKAGE_RECEIPT.json'):sha(BASE/'PACKAGE_RECEIPT.json'),
    sha(PRIOR/'PACKAGE_SHA256.json'):sha(BASE/'PACKAGE_SHA256.json'),
    sha(PRIOR/'inventory_actual136_Full100refs.json'):sha(BASE/'inventory_actual140_Full100refs.json'),
    'inventory_actual136_Full100refs.json':'inventory_actual140_Full100refs.json',
    "review['native_accepted_snapshot'] == 136":"review['native_accepted_snapshot'] == 140",
    'old128_records_exact':'old136_records_exact','original128':'original136','prior128':'prior136',
    repr(read(PRIOR/'SCOPE.json')['selected_ids']):repr(expected),
    '== 128':'== 136','==128':'==136','== 8':'== 4','==8':'==4',
    '(72,192,24)':'(36,96,12)',
    'prior_three_view_models=128,accepted_new=8,cumulative_three_view_models=136':'prior_three_view_models=136,accepted_new=4,cumulative_three_view_models=140',
    'C_after25_20261009/execution_candidate/backups/incremental_20261009T213828Z':'C_after28_20261009/execution_candidate/backups/incremental_20261009T221150Z',
    '0d1661bdc21025b957fa4e6ac39d4c8924cb50ab1b91920268119941312a615a':'cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0',
}
old_seal = {row['path']:row['sha256'] for row in read(OLD/'TRANSPORT_REBIND.json')['members']}
pattern = '|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True))
NEW.mkdir(exist_ok=False)
rows=[]
for name in ('deploy.py','observe.py','backup.py','adopt.py'):
    assert sha(OLD/name)==old_seal[name]
    before=(OLD/name).read_text(encoding='utf8')
    after=re.sub(pattern,lambda m:changes[m.group()],before)
    if name=='deploy.py':
        after=after.replace("('guardfed_celeba_mechanism_valid_C_after25','EXITED')","('guardfed_celeba_mechanism_valid_C_after28','EXITED')").replace('PRIOR_C_AFTER25_EXITED','PRIOR_C_AFTER28_EXITED')
    ast.parse(after)
    assert after!=before and 'utf-4' not in after and 'EXACT8' not in after and 'exact8' not in after
    with (NEW/name).open('x',encoding='utf8',newline='\n') as stream:stream.write(after)
    rows.append({'path':name,'sha256':sha(NEW/name),'accepted_transport_source_sha256':sha(OLD/name)})
checks=0
for name in ('backup.py','adopt.py'):
    spec=importlib.util.spec_from_file_location('after36_'+name[:-3],NEW/name)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    good=dict(service='guardfed_celeba_mechanism_valid_C_after36 EXITED',processes=[],batch_failure=None,batch_complete={'completed':4},completed=[{'id':i} for i in expected])
    mod.check_terminal(good,expected);checks+=1
    for field,value in [('service','RUNNING'),('processes',[1]),('batch_failure',{'failure':True}),('batch_complete',None),('completed',[{'id':expected[0]}]*4),('completed',good['completed'][:3]),('completed',[{'id':i} for i in expected[:3]+['wrong_id']])]:
        case=dict(good);case[field]=value
        try:mod.check_terminal(case,expected)
        except AssertionError:checks+=1
        else:raise AssertionError('Terminal refusal failed')
assert checks==16
with (NEW/'TRANSPORT_REBIND.json').open('x',encoding='utf8') as stream:
    json.dump(dict(source_only=True,SSH=False,original_science_changed=False,exact4=True,prior136_not_replayed=True,terminal_positive_and_refusal_checks=checks,members=rows),stream,indent=2);stream.write('\n')
print(json.dumps({'transport_files':rows,'terminal_positive_and_refusal_checks':checks}))
