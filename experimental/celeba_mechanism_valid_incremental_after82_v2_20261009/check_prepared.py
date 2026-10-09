"""Pure approval/identity checks only. Never bind_runtime, Torch, CNN, SSH or approval files."""
import ast
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.with_name('celeba_mechanism_valid_incremental_after82_20261009')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_bytes())
def functions(text): return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    b = load('after82_v2_static_bridge', HERE / 'bridge.py')
    old = load('after82_preserved_static_bridge', OLD / 'bridge.py')
    inv = read(HERE / 'inventory_actual92_Full100refs.json')
    original = read(OLD / 'inventory_actual92_Full100refs.json')
    prior = read(ROOT / 'tmp/celeba_mechanism_valid_incremental_after71_20261009/inventory_actual82_Full100refs.json')
    baseline_path = ROOT / 'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json'
    assert sha(baseline_path) == b.BASELINE_INVENTORY_SHA
    assert len(b.validate_inventory(inv, read(baseline_path))) == 92
    assert inv['records'] == original['records'] and inv['full_references'] == original['full_references']
    assert {r['id']: r for r in prior['records']} == {r['id']: r for r in inv['records'] if r['id'] in b.EXCLUDED_PRIOR_IDS}
    assert prior['full_references'] == inv['full_references']
    assert (HERE/'FULL100_ACTUAL_SOURCE_BINDINGS.json').read_bytes() == (OLD/'FULL100_ACTUAL_SOURCE_BINDINGS.json').read_bytes()
    assert set(b.ACCEPTED_IDS) - set(b.EXCLUDED_PRIOR_IDS) == set(b.REPLAY_IDS) and len(b.REPLAY_IDS) == 10

    before = functions((OLD / 'bridge.py').read_text(encoding='utf-8-sig'))
    after = functions((HERE / 'bridge.py').read_text(encoding='utf-8-sig'))
    exact = [name for name in before if name not in ('require_approval', 'validate_inventory')]
    assert len(exact) == 10 and all(before[name] == after[name] for name in exact)
    assert before['require_approval'].replace('len(ids) == len(set(ids)) == 11', 'len(ids) == len(set(ids)) == 10') == after['require_approval']
    assert before['validate_inventory'].replace('Excluded-prior82/selected11 boundary changed', 'Excluded-prior82/selected10 boundary changed') == after['validate_inventory']
    assert sha(HERE/'execution_candidate/resource_extra.py') == sha(OLD/'execution_candidate/resource_extra.py')
    for name in ('verify_backup.py','verify_saved_increment.py','backup_completed.py'):
        expected=(OLD/'execution_candidate'/name).read_text(encoding='utf-8-sig').replace(OLD.name,HERE.name).replace(old.SCOPE,b.SCOPE)
        for rel in ('bridge.py','inventory_actual92_Full100refs.json','SCOPE.json','FILES_SHA256.json','execution_candidate/RUNTIME_BINDINGS.json'):
            expected=expected.replace(sha(OLD/rel),sha(HERE/rel))
        assert expected == (HERE/'execution_candidate'/name).read_text(encoding='utf-8-sig'), name
    for folder, seal_name in [(HERE,'FILES_SHA256.json'),(HERE/'execution_candidate','EXECUTION_SOURCE_SHA256.json')]:
        for row in read(folder/seal_name)['members']:
            assert sha(folder/row['path']) == row['sha256'] and (folder/row['path']).stat().st_size == row['size']

    approval = dict(status='APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY', scope=b.SCOPE,
        inventory_sha256=sha(HERE/'inventory_actual92_Full100refs.json'), bridge_sha256=sha(HERE/'bridge.py'),
        selected_ids=b.REPLAY_IDS, device='cpu', compute_threads=8, max_processes=1,
        allowed_cpus=list(range(112,120)), target_split='valid', final_test_dispatch=False, native_tolerance=1e-12)
    def check(a, selected=None):
        b.require_approval(a, approval['inventory_sha256'], selected or b.REPLAY_IDS[0], approval['bridge_sha256'])
    for identity in b.REPLAY_IDS:
        check(approval, identity)
    rejected=[]
    def reject(name, call):
        try: call()
        except (ValueError, KeyError) as error: rejected.append(dict(case=name, error=str(error)))
        else: raise AssertionError('Must reject: '+name)
    def mutate(name, fn):
        a=copy.deepcopy(approval); fn(a); reject(name,lambda:check(a))
    old_approval=copy.deepcopy(approval);old_approval['scope']=old.SCOPE
    reject('preserved_v1_valid_ten_reproduces_actual_cardinality_failure',lambda:old.require_approval(old_approval,approval['inventory_sha256'],b.REPLAY_IDS[0],approval['bridge_sha256']))
    mutate('nine_IDs',lambda a:a.__setitem__('selected_ids',a['selected_ids'][:-1]))
    mutate('eleven_IDs',lambda a:a['selected_ids'].append('minus_U_non-IID_Sp-DFA_seed91003'))
    mutate('duplicate_ID',lambda a:a['selected_ids'].__setitem__(-1,a['selected_ids'][0]))
    mutate('foreign_ID',lambda a:a['selected_ids'].__setitem__(0,'foreign'))
    mutate('already_closed82_ID',lambda a:a['selected_ids'].__setitem__(0,b.EXCLUDED_PRIOR_IDS[0]))
    mutate('Full_ID',lambda a:a['selected_ids'].__setitem__(0,inv['full_references'][0]['id']))
    mutate('wrong_bridge_source',lambda a:a.__setitem__('bridge_sha256','0'*64))
    mutate('wrong_inventory_source',lambda a:a.__setitem__('inventory_sha256','0'*64))
    mutate('old_scope',lambda a:a.__setitem__('scope',old.SCOPE))
    mutate('wrong_split',lambda a:a.__setitem__('target_split','test'))
    mutate('changed_tolerance',lambda a:a.__setitem__('native_tolerance',1e-6))
    mutate('test_dispatch',lambda a:a.__setitem__('final_test_dispatch',True))
    mutate('wrong_CPU_budget',lambda a:a.__setitem__('allowed_cpus',list(range(104,112))))
    mutate('wrong_thread_count',lambda a:a.__setitem__('compute_threads',1))
    mutate('multiple_processes',lambda a:a.__setitem__('max_processes',2))
    reject('unselected_worker_ID',lambda:check(approval,b.EXCLUDED_PRIOR_IDS[0]))
    reject('prepared_template_is_not_approval',lambda:check(read(HERE/'APPROVAL_TEMPLATE.json')))

    # Exercise the new service guard with fake /proc and supervisor outputs only.
    installer=(HERE/'execution_candidate/install_once.py').read_text(encoding='utf-8-sig')
    node=next(n for n in ast.parse(installer).body if isinstance(n,ast.FunctionDef) and n.name=='prior_failed_service_stopped')
    class Proc:
        name='123'
        def __truediv__(self, _): return self
        def read_bytes(self): return process_argv[0]
    class ProcRoot:
        def iterdir(self): return [Proc()] if process_argv[0] else []
    process_argv=[b'']; status=[SimpleNamespace(stdout='guardfed_celeba_mechanism_valid_after82 EXITED Oct09\n',stderr='',returncode=3)]
    ns=dict(batch=SimpleNamespace(require=b.require),subprocess=SimpleNamespace(run=lambda *args,**kwargs:status[0]),Path=lambda _:ProcRoot())
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pure_mock_service_guard>','exec'),ns)
    assert ns['prior_failed_service_stopped']()['original_batch_processes']==0
    status[0].stdout='guardfed_celeba_mechanism_valid_after82 RUNNING pid1\n'
    reject('prior_failed_service_still_RUNNING',lambda:ns['prior_failed_service_stopped']())
    status[0].stdout='guardfed_celeba_mechanism_valid_after82 EXITED Oct09\n'
    process_argv[0]=b'python\0/workspace/guardfed_checks/celeba_mechanism_valid_incremental_after82_20261009/execution_candidate/batch.py\0worker\0'
    reject('prior_after82_worker_exists',lambda:ns['prior_failed_service_stopped']())
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    result=dict(status='AFTER82_V2_PURE_APPROVAL_IDENTITY_AND_REFUSAL_CHECK_PASS', valid10_positive_worker_guards=10,
        native_inventory_count=92, closed82_records_exact=True, all92_records_exact=True, Full100_refs_and_actual900_binding_bytes_exact=True,
        exact_bridge_functions=exact, bind_runtime_including_replay_accept_source_exact=True,
        only_approval_guard_cardinality_and_diagnostic_changed=True, original_resource_extra_byte_exact=True,
        original_backup_and_saved_array_verifiers_exact_after_namespace_rebind=True,
        preserved_original_failure_reproduced=True, no_CNN=True, no_SSH=True, no_dispatch=True,
        actual_Linux_preflight='PENDING_ROOT', rejections=rejected, refusal_count=len(rejected))
    output=Path(sys.argv[1]) if len(sys.argv)==2 else HERE/'LOCAL_GUARD_CHECK.json'
    with output.open('x',encoding='utf-8') as out:
        json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:result[k] for k in ['status','valid10_positive_worker_guards','refusal_count','no_CNN','no_SSH','no_dispatch']}))


if __name__=='__main__': main()
