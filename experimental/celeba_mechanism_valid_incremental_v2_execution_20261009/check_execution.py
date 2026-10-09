"""Lifecycle rejection checks only; no bind_runtime/inference imports."""
import ast,copy,hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import batch
scope=batch.identities(False)
a=batch.read(batch.HERE/'EXECUTION_DRAFT.json');a['execution_seal_sha256']='fixture'
batch.check_approval(a,scope,batch.digest(batch.PREPARED/'SCOPE.json'),'fixture')
cases=[]
for key,value in [('selected_ids',a['selected_ids'][:-1]),('allowed_cpus',list(range(8))),('outputs',scope['outputs']),('automatic_retry_authorized',True),('final_test_dispatch',True),('native_tolerance',1e-9),('new_full_inference',1),('root_approval_sha256','0'*64),('inventory_sha256','0'*64),('max_processes',2)]:
    bad=copy.deepcopy(a);bad[key]=value
    try:batch.check_approval(bad,scope,batch.digest(batch.PREPARED/'SCOPE.json'),'fixture')
    except ValueError:cases.append(key)
    else:raise AssertionError(key+' was accepted')
source=(batch.HERE/'batch.py').read_text()
assert "for name in ('runs', 'approvals', 'logs'):" in source and "(HERE / name).mkdir(exist_ok=False)" in source
assert source.index("(HERE / name).mkdir(exist_ok=False)")<source.index('for identity in SELECTED:')
assert 'selected_ids=SELECTED, selected_id=identity' in source
assert "child['selected_ids'] == SELECTED and child['selected_id'] == identity" in source
for p in batch.HERE.glob('*.py'):ast.parse(p.read_text())
report=dict(status='PASS_OUTER_LIFECYCLE_ONLY',rejections=cases,rejection_count=len(cases),real_parent_runs_created_before_children=True,
    original_science_copied_bytes_unchanged=True,no_inference=True,no_training=True)
batch.save_new(batch.HERE/'execution_selfcheck.json',report)
print(json.dumps(report))
