"""Read-only source/AST checks. All approval and host state fixtures stay in memory."""
from pathlib import Path
import ast
import copy
import hashlib
import json
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'tmp/celeba_hybrid_fullcoverage_canary_operations_20261010/v3'
HERE = Path(__file__).resolve().parent
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
checks = []

def expect(name, fn, passes):
    try:
        fn()
        result = True
    except (AssertionError, ValueError, KeyError):
        result = False
    assert result == passes, name
    checks.append(dict(name=name, expected='PASS' if passes else 'REFUSE', actual='PASS' if result else 'REFUSE'))

seal = read(SRC / 'FILES_SHA256.json')
assert digest(SRC / 'FILES_SHA256.json') == '04dbe344603a5243dbfa6a00bc89c31ea5208792d88e9371ad7f9873775d17ab'
assert len(seal['files']) == 7
for name, pin in seal['files'].items():
    assert digest(SRC / name) == pin['sha256'] and (SRC / name).stat().st_size == pin['bytes']
texts = {name: (SRC / name).read_text(encoding='utf8') for name in ('launch.py', 'remote_launch.py', 'main_health.py')}
trees = {name: ast.parse(value) for name, value in texts.items()}
ns = {'__file__': str(SRC / 'launch.py'), '__name__': 'not_main'}
exec(compile(trees['launch.py'], str(SRC / 'launch.py'), 'exec'), ns)
main = next(n for n in trees['launch.py'].body if isinstance(n, ast.FunctionDef) and n.name == 'main')
cut = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'source' for t in n.targets))
validation = copy.deepcopy(main); validation.name = 'validate_fixture'; validation.body = validation.body[:cut]
exec(compile(ast.fix_missing_locations(ast.Module(body=[validation], type_ignores=[])), '<original-local-validation-prefix>', 'exec'), ns)

def approval_fixture(change=None, stale_external_hash=False):
    off = dict(status='BOUND_METADATA_MEMBERS_RECEIVED_NOT_TRAINING', package_sha256='c' * 64)
    root = dict(status='ROOT_HYBRID100_BOUND_METADATA_ADOPTED', package_sha256='c' * 64, new=96, reused=4, canaries=7,
                execution_authorized=False, final_test=False, implementation_source_seal_sha256='d' * 64)
    approval = dict(status='ROOT_AUTHORIZED_SEVEN_HYBRID_CANARIES', scope='seven_same_horizon_3round_canaries', package_sha256='c' * 64,
                    helper_seal_sha256=digest(SRC / 'FILES_SHA256.json'), service='guardfed_celeba_hybrid_fullcoverage_canary', cpus=[104], cpu_threads=1,
                    formal100_started=False, final_test=False, implementation_source_seal_sha256='d' * 64)
    if change: change(approval, off, root)
    encoded = lambda value: json.dumps(value, sort_keys=True).encode()
    h = lambda value: hashlib.sha256(encoded(value)).hexdigest()
    baseline = dict(checked_utc='MOCK_ONLY_NOT_HOST_EVIDENCE', queue_completed=0, active=[])
    root['bound_offserver_sha256'] = h(off)
    approval.update(bound_offserver_sha256=h(off), bound_root_review_sha256=h(root), baseline_snapshot_sha256=h(baseline))
    values = {key: value for key, value in [('approval', approval), ('bound_offserver', off), ('bound_root_review', root), ('baseline', baseline)]}
    paths = {key: HERE / ('MOCK_' + key + '.json') for key in values}
    a = SimpleNamespace(**paths, **{key + '_sha256': h(value) for key, value in values.items()},
                        package_sha256='c' * 64, helper_seal_sha256=digest(SRC / 'FILES_SHA256.json'))
    if stale_external_hash: a.bound_root_review_sha256 = '0' * 64
    class Parser:
        def add_argument(self, *args, **kwargs): pass
        def parse_args(self): return a
    memory = {str(paths[key]): value for key, value in values.items()}
    ns['argparse'] = SimpleNamespace(ArgumentParser=Parser)
    ns['read'] = lambda p: copy.deepcopy(memory[str(p)]) if str(p) in memory else read(p)
    ns['sha'] = lambda p: h(memory[str(p)]) if str(p) in memory else digest(p)
    ns['validate_fixture']()

expect('actual validation prefix accepts exact7 metadata MOCK', approval_fixture, True)
for name, change in [
    ('root implementation seal drift', lambda a, o, r: r.update(implementation_source_seal_sha256='e' * 64)),
    ('offserver package drift', lambda a, o, r: o.update(package_sha256='e' * 64)),
    ('root canary count8', lambda a, o, r: r.update(canaries=8)),
    ('root already execution-authorized', lambda a, o, r: r.update(execution_authorized=True)),
    ('96 scope substituted', lambda a, o, r: a.update(scope='96_new_70round_valid_only')),
    ('wrong worker CPU105', lambda a, o, r: a.update(cpus=[105])),
]: expect(name, lambda change=change: approval_fixture(change), False)
expect('actual root external SHA mismatch', lambda: approval_fixture(stale_external_hash=True), False)

health_ns = {}; exec(compile(trees['main_health.py'], '<original-main-health>', 'exec'), health_ns)
for active in (0, 1, 8, 9):
    expect('main active=' + str(active), lambda active=active: health_ns['main_health'](dict(returncode=0, stdout='formal RUNNING pid1'), dict(active=[{}] * active, failed=[]), 'MOCK'), active in (1, 8))
expect('main failed item', lambda: health_ns['main_health'](dict(returncode=0, stdout='formal RUNNING pid1'), dict(active=[{}], failed=['bad']), 'MOCK'), False)
expect('main EXITED rc3', lambda: health_ns['main_health'](dict(returncode=3, stdout='formal EXITED Oct10'), dict(active=[{}], failed=[]), 'MOCK'), False)

def expression(tree, text):
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Assert) and text in ast.unparse(n.test)]
    assert len(matches) == 1, text
    return compile(ast.Expression(matches[0].test), '<original-assertion>', 'eval')

remote = trees['remote_launch.py']
worker_conflict = expression(remote, 'len(cpus) <= 16')
for cpus, passes, label in [([104], False, 'CPU104 owner'), (list(range(100,108)), False, 'auxiliary thread crossing CPU104'),
                            ([102,103], True, 'FL102103'), ([105], True, 'gradient105'), (list(range(112,120)), True, 'remaining112119'),
                            (list(range(128)), True, 'broad scheduler mask')]:
    expect('original all-thread worker predicate ' + label, lambda cpus=cpus: (_ for _ in ()).throw(AssertionError()) if not eval(worker_conflict, dict(cpus=cpus)) else None, passes)
transport_template = next(n.value.left.value for n in ast.walk(trees['launch.py']) if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
transport_tree = ast.parse(transport_template % ('/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/canary_operations', '{}'))
outer_conflict = expression(transport_tree, '107 in cpus')
expect('CPU107 restricted owner refuses', lambda: (_ for _ in ()).throw(AssertionError()) if not eval(outer_conflict, dict(cpus={107})) else None, False)
expect('CPU106 owner remains untouched', lambda: (_ for _ in ()).throw(AssertionError()) if not eval(outer_conflict, dict(cpus={106})) else None, True)
for fragment, values, good, bad in [
    ('planned <= cores', dict(planned=120,cores=122.88), None, dict(planned=123,cores=122.88)),
    ('free >=', dict(free=8*1024**3), None, dict(free=8*1024**3-1)),
    ('gpu_free[0] >=', dict(gpu_free=[4096]), None, dict(gpu_free=[4095])),
    ("recovery ==", dict(recovery=['None','None']), None, dict(recovery=['Reset','None'])),
]:
    code = expression(remote, fragment)
    expect(fragment + ' boundary accepts', lambda code=code,values=values: (_ for _ in ()).throw(AssertionError()) if not eval(code,values) else None, True)
    expect(fragment + ' boundary refuses', lambda code=code,bad=bad: (_ for _ in ()).throw(AssertionError()) if not eval(code,bad) else None, False)
grown_node = next(n.value for n in ast.walk(remote) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='grown' for t in n.targets))
grown_code = compile(ast.Expression(grown_node), '<original-growth-expression>', 'eval')
for label, completed, changes, passes in [('no growth',10,[dict(before=3,after=3)],False),('round grows',10,[dict(before=3,after=4)],True),('completion grows',11,[],True)]:
    expect(label,lambda completed=completed,changes=changes: (_ for _ in ()).throw(AssertionError()) if not eval(grown_code,dict(queue={'completed':[None]*completed},baseline={'queue_completed':10},changes=changes)) else None,passes)

# Inspect the original supervisor string and called entry only; do not evaluate the remote main.
config = next(n.value for n in ast.walk(remote) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='config' for t in n.targets))
config_text = ''.join(n.value for n in config.values if isinstance(n,ast.Constant))
for wanted in ('taskset -c 104','nice -n 10','run_canaries.py','autostart=false','autorestart=false','startretries=0','CUDA_VISIBLE_DEVICES="0"','OMP_NUM_THREADS="1"'):
    assert wanted in config_text, wanted
assert 'run_fullcoverage.py' not in texts['remote_launch.py'] and 'run_fullcoverage.py' not in texts['launch.py']
canary_path=ROOT/'tmp/celeba_hybrid_fullcoverage_implementation_v3_20261010/run_canaries.py'
canary=ast.parse(canary_path.read_text(encoding='utf8'))
child_calls=[n for n in ast.walk(canary) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess' and n.func.attr=='run']
assert len(child_calls)==1 and any(k.arg=='check' and isinstance(k.value,ast.Constant) and k.value.value is True for k in child_calls[0].keywords)
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen' for n in ast.walk(canary))
result=dict(status='LOCAL_SOURCE_AST_AND_MEMORY_FIXTURES_PASS_NOT_RUNTIME_EVIDENCE',source_seal_sha256=digest(SRC/'FILES_SHA256.json'),source_members_verified=7,
            source_files={n:digest(SRC/n) for n in seal['files']},checks=checks,check_n=len(checks),mock_proofs_created_on_disk=False,
            actual_bound_proof_read_or_generated=False,SSH=False,subprocess_commands_executed=False,Torch=False,CNN=False,
            no_96_automatic_entry=True,sequential_synchronous_child_check=True,
            called_canary_source_sha256=digest(canary_path))
with (HERE/'CHECKS.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],checks=len(checks),checks_sha256=digest(HERE/'CHECKS.json'))))
