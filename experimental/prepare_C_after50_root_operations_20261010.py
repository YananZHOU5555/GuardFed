"""Rebind accepted transport to the sealed six-model C/non-IID/Benign package."""
from pathlib import Path
import ast,hashlib,importlib.util,json,re
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'tmp/celeba_mechanism_C_after47_root_operations_20261010'
NEW=ROOT/'tmp/celeba_mechanism_C_after50_root_operations_20261010'
PRIOR=ROOT/'tmp/celeba_mechanism_valid_C_after47_20261010'
BASE=ROOT/'tmp/celeba_mechanism_valid_C_after50_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
expected=[f'minus_C_non-IID_Benign_seed{s}' for s in range(91001,91007)]
handoff=read(BASE/'HANDOFF.json');scope=read(BASE/'SCOPE.json')
assert scope['selected_ids']==expected and len(scope['excluded_prior_ids'])==150
assert handoff['native_accepted_snapshot']==156 and handoff['new_three_view_accepted']==0
for folder,seal in ((BASE,'FILES_SHA256.json'),(BASE/'execution_candidate','EXECUTION_SOURCE_SHA256.json')):
    for row in read(folder/seal)['members']:
        assert sha(folder/row['path'])==row['sha256'] and (folder/row['path']).stat().st_size==row['size']
changes={
 'celeba_mechanism_valid_C_after47_20261010':'celeba_mechanism_valid_C_after50_20261010',
 'celeba_mechanism_valid_C_after40_20261010':'celeba_mechanism_valid_C_after47_20261010',
 'C_AFTER47':'C_AFTER50','C_after47':'C_after50','C_AFTER40':'C_AFTER47','C_after40':'C_after47',
 'EXACT3':'EXACT6','exact3':'exact6',
 sha(PRIOR/'FILES_SHA256.json'):sha(BASE/'FILES_SHA256.json'),
 sha(PRIOR/'execution_candidate/EXECUTION_SOURCE_SHA256.json'):sha(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),
 sha(PRIOR/'PACKAGE_RECEIPT.json'):sha(BASE/'PACKAGE_RECEIPT.json'),
 sha(PRIOR/'PACKAGE_SHA256.json'):sha(BASE/'PACKAGE_SHA256.json'),
 sha(PRIOR/'inventory_actual150_Full100refs.json'):sha(BASE/'inventory_actual156_Full100refs.json'),
 'inventory_actual150_Full100refs.json':'inventory_actual156_Full100refs.json',
 "review['native_accepted_snapshot'] == 150":"review['native_accepted_snapshot'] == 156",
 'old147_records_exact':'old150_records_exact','original147':'original150','prior147':'prior150',
 repr(read(PRIOR/'SCOPE.json')['selected_ids']):repr(expected),
 '== 147':'== 150','==147':'==150','== 3':'== 6','==3':'==6',
 '(27,72,9)':'(54,144,18)',
 'prior_three_view_models=147,accepted_new=3,cumulative_three_view_models=150':'prior_three_view_models=150,accepted_new=6,cumulative_three_view_models=156',
 'incremental_20261009T233028Z':'incremental_20261009T235311Z',
 '64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea':'3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8',
}
old_seal={row['path']:row['sha256'] for row in read(OLD/'TRANSPORT_REBIND.json')['members']}
pattern='|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True))
NEW.mkdir(exist_ok=False);members=[]
for name in ('deploy.py','observe.py','backup.py','adopt.py'):
    assert sha(OLD/name)==old_seal[name]
    before=(OLD/name).read_text(encoding='utf8')
    after=re.sub(pattern,lambda m:changes[m.group()],before)
    ast.parse(after);assert after!=before and 'utf-6' not in after
    if name=='adopt.py':assert 'len(names)==7*len(expected)' in after
    with (NEW/name).open('x',encoding='utf8',newline='\n') as f:f.write(after)
    members.append(dict(path=name,sha256=sha(NEW/name),accepted_transport_source_sha256=sha(OLD/name)))
checks=0
for name in ('backup.py','adopt.py'):
    spec=importlib.util.spec_from_file_location('after50_'+name[:-3],NEW/name);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    good=dict(service='guardfed_celeba_mechanism_valid_C_after50 EXITED',processes=[],batch_failure=None,batch_complete={'completed':6},completed=[{'id':i} for i in expected])
    mod.check_terminal(good,expected);checks+=1
    for field,value in [('service','RUNNING'),('processes',[1]),('batch_failure',{'failure':True}),('batch_complete',None),('completed',[{'id':expected[0]}]*6),('completed',good['completed'][:5]),('completed',[{'id':i} for i in expected[:5]+['wrong_id']])]:
        case=dict(good);case[field]=value
        try:mod.check_terminal(case,expected)
        except AssertionError:checks+=1
        else:raise AssertionError('Terminal refusal failed')
assert checks==16
with (NEW/'TRANSPORT_REBIND.json').open('x',encoding='utf8') as f:
    json.dump(dict(source_only=True,SSH=False,original_science_changed=False,exact6=True,prior150_not_replayed=True,terminal_positive_and_refusal_checks=checks,members=members),f,indent=2);f.write('\n')
print(json.dumps(dict(transport_files=members,terminal_positive_and_refusal_checks=checks)))
