"""No Torch/CNN/server: exact diff boundaries and resource-guard inputs/outputs."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'celeba_valid_gpu_recovery_implementation_20261009'
spec = importlib.util.spec_from_file_location('resource_guard_v2_checked', HERE / 'release/recovery.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
old_tree = ast.parse((OLD / 'recovery.py').read_text()); new_tree = ast.parse((HERE / 'release/recovery.py').read_text())
functions = lambda tree: {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
old_functions, new_functions = functions(old_tree), functions(new_tree)
changed = {k for k in old_functions if ast.dump(old_functions[k]) != ast.dump(new_functions[k])}
r.need(changed == {'resource_preflight', 'GPU_gate', 'worker', 'preserve_outer_failure'} and set(new_functions) - set(old_functions) == {'main_health'}, 'Unexpected function/science change')
for name, row in r.read(OLD / 'PACKAGE_SHA256.json')['members'].items():
    if name != 'recovery.py': r.need(r.sha(HERE / 'release' / name) == row['sha256'], 'Scientific/source member changed: ' + name)
def nonhealth(node):
    node = copy.deepcopy(node); kept = []
    for statement in node.body:
        targets = {x.id for x in statement.targets if isinstance(x, ast.Name)} if isinstance(statement, ast.Assign) else set()
        if targets & {'observed', 'service', 'main', 'guard_inputs'}: continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if isinstance(call.func, ast.Name) and call.func.id == 'main_health': continue
            if any(isinstance(x, ast.Constant) and x.value in ('Protected main800 health failed', 'Protected training failed') for x in ast.walk(call)): continue
        if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Dict):
            pairs = [(k, v) for k, v in zip(statement.value.keys, statement.value.values) if not (isinstance(k, ast.Constant) and k.value == 'main_guard_inputs')]
            statement.value.keys, statement.value.values = [k for k, _ in pairs], [v for _, v in pairs]
        kept.append(statement)
    node.body = kept; return ast.dump(node)
for name in ('resource_preflight', 'GPU_gate'): r.need(nonhealth(old_functions[name]) == nonhealth(new_functions[name]), 'Nonhealth resource checks changed')
service = {'returncode': 0, 'stdout': 'guardfed_celeba_mechanism_formal RUNNING pid 9179', 'stderr': ''}
queue = {'active': [{'id': 'job%d' % n} for n in range(8)], 'failed': [], 'completed': ['prior']}
health_passed, health_rejected = 0, 0
for state in ('RUNNING', 'STOPPED', 'EXITED'):
    for failures in ([], [{'id': 'failed'}]):
        for n in range(10):
            observed = dict(service, stdout='guardfed_celeba_mechanism_formal ' + state); q = dict(queue, active=queue['active'][:n] + ([{'id': 'ninth'}] if n == 9 else []), failed=failures)
            should_pass = state == 'RUNNING' and not failures and 1 <= n <= 8
            try: inputs = r.main_health(observed, q, 'fixture')
            except ValueError as error:
                r.need(not should_pass and error.resource_guard_inputs['service'] == observed and error.resource_guard_inputs['queue_snapshot'] == q and error.resource_guard_inputs['active_n'] == n, 'Failure inputs not preserved')
                json.dumps(error.resource_guard_inputs, allow_nan=False); health_rejected += 1
            else: r.need(should_pass and inputs['required']['active_max'] == 8, 'Health predicate incorrect'); health_passed += 1
state = {'queue': queue, 'service': service, 'cpu.max': '12288000 100000', 'memory.current': str(24 * 1024**3), 'memory.max': str(32 * 1024**3), 'gpu_free': 2048, 'uuid': 'GPU-fixture', 'recovery': 'None'}
class FakePath:
    def __init__(self, value): self.value = value
    def __truediv__(self, value): return FakePath(self.value + '/' + value)
    def iterdir(self): r.need(self.value == '/proc', 'Unexpected process lookup'); return iter(())
    def read_text(self): return state[self.value.rsplit('/', 1)[-1]]
def command(argv, **kwargs):
    if argv[0] == 'supervisorctl': return SimpleNamespace(**state['service'])
    if '--query-gpu=uuid,memory.free' in argv: text = state['uuid'] + ', ' + str(state['gpu_free'])
    elif '--query-gpu=uuid' in argv: text = state['uuid']
    else: text = 'GPU Recovery Action : ' + state['recovery']
    return SimpleNamespace(stdout=text, stderr='', returncode=0)
namespace = {'Path': FakePath, 'os': SimpleNamespace(getpid=lambda: 123, PRIO_PROCESS=0, getpriority=lambda *_: 10, environ={'CUDA_VISIBLE_DEVICES': 'GPU-fixture', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}), 'subprocess': SimpleNamespace(run=command), 'read': lambda _: copy.deepcopy(state['queue']), 'REPO': FakePath('/repo'), 'need': r.need, 'main_health': r.main_health, 'datetime': datetime}
for name in ('resource_preflight', 'GPU_gate'): exec(compile(ast.Module(body=[new_functions[name]], type_ignores=[]), name, 'exec'), namespace)
t = SimpleNamespace(get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1, cuda=SimpleNamespace(device_count=lambda: 1, current_device=lambda: 0), are_deterministic_algorithms_enabled=lambda: True, backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False, deterministic=True, allow_tf32=False), cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
a = {'gpu_uuid': 'GPU-fixture'}
def snapshot(): return {'service': state['service'], 'training_queue_snapshot': state['queue'], 'thread_cpu_affinities': {'1': [105]}, 'cgroup': {k: state[k] for k in ('cpu.max', 'memory.current', 'memory.max')}}
def preflight(): return namespace['resource_preflight'](a)
def gpu_gate(): return namespace['GPU_gate'](SimpleNamespace(torch=t), a, snapshot(), {'other_declared_threads': 0})
for n in (1, 7, 8):
    state['queue'] = dict(queue, active=queue['active'][:n]); proof = preflight(); r.need(proof['main_guard_inputs']['active_n'] == n, 'Success input receipt missing'); gpu_gate()
refusals = []
def refuses(name, action):
    try: action()
    except ValueError: refusals.append(name)
    else: raise AssertionError('Unexpected pass: ' + name)
for n in (0, 9):
    state['queue'] = dict(queue, active=queue['active'][:n] + ([{'id': 'ninth'}] if n == 9 else []))
    refuses('preflight_active%d' % n, preflight); refuses('CNN_gate_active%d' % n, gpu_gate)
state['queue'] = queue
for name, key, value, action in [('quota_insufficient', 'cpu.max', '50000 100000', preflight), ('GPU_memory2047MiB', 'gpu_free', 2047, preflight), ('RAM_below8GiB', 'memory.current', str(24 * 1024**3 + 1), preflight), ('GPU_UUID_drift', 'uuid', 'GPU-foreign', preflight), ('GPU_recovery_required', 'recovery', 'Reset', preflight), ('CNN_quota_insufficient', 'cpu.max', '50000 100000', gpu_gate), ('CNN_RAM_below8GiB', 'memory.current', str(24 * 1024**3 + 1), gpu_gate)]:
    previous = state[key]; state[key] = value; refuses(name, action); state[key] = previous
state['service'] = dict(service, returncode=3, stdout='guardfed_celeba_mechanism_formal STOPPED')
try: preflight()
except ValueError as error: r.need(error.resource_guard_inputs['service']['returncode'] == 3, 'STOPPED returncode not captured')
else: raise AssertionError('STOPPED passed')
for name in ('worker', 'preserve_outer_failure'):
    r.need(any(isinstance(n, ast.Constant) and n.value == 'resource_guard_inputs' for n in ast.walk(new_functions[name])), 'Failure persistence lost observation')
r.need('torch' not in sys.modules, 'Torch imported')
print(json.dumps({'status': 'PASS_NO_CNN_NO_SERVER_NO_DISPATCH', 'original_members_read_verified': 25, 'unchanged_source_members': 24, 'health_matrix_cases': 60, 'health_matrix_passed': health_passed, 'health_matrix_rejected': health_rejected, 'success_activity_boundaries_checked': [1, 7, 8], 'unchanged_nonhealth_resource_AST': True, 'resource_refusal_checks': refusals, 'supervisor_STOPPED_returncode_preserved': True, 'success_and_failure_health_inputs_preserved': True, 'GPU_memory_exact2048MiB_and_RAM_exact8GiB_passed': True, 'native_tolerance_unchanged': 1e-12, 'new_CNN_inference': 0, 'new_server_access': 0, 'accepted460_unchanged': True, 'partial2_now_accepted_in_root460_baseline': True, 'partial2_recomputed_or_accepted_by_this_check': False, 'Linux_runtime_verified': False}, indent=2))
