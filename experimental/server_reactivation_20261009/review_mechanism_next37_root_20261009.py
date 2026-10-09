"""Review the exact accepted60-minus-closed23 source extension, without inference."""
from pathlib import Path
import ast, collections, datetime, hashlib, json, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_next37_20261009'
OLD=ROOT/'tmp/celeba_mechanism_valid_incremental_v2_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(BASE/'FILES_SHA256.json')=='95978fa42c28e9b4ff5b855b33c2dda56edc2b14fcfd56c3a29b0a9ba98135fd'
members=read(BASE/'FILES_SHA256.json')['members'];assert len(members)==16
for row in members:
    assert sha(BASE/row['path'])==row['sha256'] and (BASE/row['path']).stat().st_size==row['size']
old_text=(OLD/'bridge.py').read_text();new_text=(BASE/'bridge.py').read_text()
functions=lambda s:{n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
old_functions=functions(old_text);new_functions=functions(new_text)
assert set(old_functions)==set(new_functions)
changed=[name for name in old_functions if ast.dump(old_functions[name])!=ast.dump(new_functions[name])]
assert changed==['validate_inventory','require_approval']
for name in set(old_functions)-set(changed):
    assert ast.get_source_segment(old_text,old_functions[name])==ast.get_source_segment(new_text,new_functions[name])
scope=read(BASE/'SCOPE.json');inventory=read(BASE/'inventory_actual60_Full100refs.json')
ids=[r['id'] for r in inventory['records']]
assert scope['status']=='PREPARED_NOT_APPROVED' and len(ids)==len(set(ids))==60
assert len(scope['already_closed_replay_ids'])==23 and len(scope['selected_ids'])==37
assert not set(scope['already_closed_replay_ids']).intersection(scope['selected_ids'])
assert set(ids)==set(scope['already_closed_replay_ids'])|set(scope['selected_ids'])
selected={r['id']:r for r in inventory['records'] if r['id'] in scope['selected_ids']}
assert collections.Counter((r['distribution'],r['attack']) for r in selected.values())=={
    ('IID','FedSA'):7,('IID','S-DFA'):10,('IID','Sp-DFA'):10,('non-IID','Benign'):10}
assert all(r['terminal_round']==70 and r['original_split']=='valid' and r['original_n_eval']==19867 for r in selected.values())
assert scope['native_tolerance']==1e-12 and scope['compute_threads']==8 and scope['cpu_ids']==list(range(112,120))
assert scope['max_processes']==1 and scope['new_full_inference']==scope['full_weights_repacked']==0
assert not scope['final_test_dispatch'] and not scope['new_training'] and not scope['automatic_retry_authorized']
target=BASE/'root_source_review';target.mkdir(exist_ok=False)
subprocess.run([sys.executable,str(BASE/'selfcheck.py'),'--output-dir',str(target)],cwd=ROOT,check=True,capture_output=True)
checks=read(target/'selfcheck.json')
assert checks['status']=='PASS_LOCAL_IDENTITIES_AND_REJECTION_ONLY' and checks['rejection_count']==36
assert not checks['image_inference_performed'] and not checks['torch_or_numpy_imported']
proof=dict(status='ROOT_NEXT37_SOURCE_SCOPE_AND_NO_CNN_REJECTIONS_PASS_NOT_RUNTIME_APPROVAL',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_seal_sha256=sha(BASE/'FILES_SHA256.json'),source_members_verified=16,
    bridge_sha256=sha(BASE/'bridge.py'),inventory_sha256=sha(BASE/'inventory_actual60_Full100refs.json'),
    actual_native_accepted=60,closed_three_view=23,selected_new_three_view=37,
    changed_bridge_functions=changed,bind_runtime_and_other_scientific_functions_source_exact=True,
    no_CNN_check_sha256=sha(target/'selfcheck.json'),rejections=36,
    source_data_metrics_or_recipe_changed=False,execution_started=False,runtime_approval=False)
with (target/'ROOT_REVIEW.json').open('x',encoding='utf8') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(proof|{'root_review_sha256':sha(target/'ROOT_REVIEW.json')}))
