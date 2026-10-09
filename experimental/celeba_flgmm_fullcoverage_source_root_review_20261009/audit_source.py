"""Bounded AST/metadata review. No candidate main(), scientific import or child."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
P = R / 'tmp/celeba_flgmm_fullcoverage_source_20261009'
O = R / 'tmp/celeba_flgmm_screen_20261009_v2_frozen_release'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def functions(path, names, namespace):
    text = path.read_text(encoding='utf-8'); nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path) + '[AST_REVIEW_ONLY]', 'exec'), namespace)
    return {n.name: ast.get_source_segment(text, n) for n in nodes}


def queue_fixture(early_identity_error=False):
    """Execute the actual run() control flow with inert processes/paths only."""
    events = []; progress = []; processes = []
    class InertPath:
        def __init__(self, name='fixture'): self.name = name
        def __truediv__(self, name): return InertPath(self.name + '/' + str(name))
        def open(self, *args): return SimpleNamespace(close=lambda: None, closed=True)
        def mkdir(self, **kwargs): pass
        def glob(self, pattern): return []
        def __str__(self): return self.name
    class Process:
        def __init__(self, index): self.index = index; self.pid = index + 100; self.polls = 0; self.waited = False
        def poll(self):
            self.polls += 1
            return 1 if self.index == 0 else (None if self.polls == 1 else 0)
        def wait(self): self.waited = True; events.append('drain_wait_' + str(self.index)); return 0
    def popen(*args, **kwargs):
        process = Process(len(processes)); processes.append(process); events.append('launch_' + str(process.index)); return process
    calls = 0
    manifest = {'jobs': [{'id': f'fixture_{n}'} for n in range(96)]}
    def local_identity():
        nonlocal calls
        calls += 1
        if early_identity_error and calls == 3: raise ValueError('Fixture source drift while two peers active')
        return {}, manifest
    def write(path, value):
        if str(path).endswith('queue_progress.json'): progress.append(copy.deepcopy(value))
        else: events.append('failure_written')
    namespace = dict(HERE=InertPath(), Path=InertPath, local_identity=local_identity,
        authorized=lambda *a, **k: None, repo_identity=lambda *a: None, reused_records=lambda *a: [],
        inspect=lambda item: None if not processes else {'fixture_terminal': True},
        write_json=write, subprocess=SimpleNamespace(Popen=popen, STDOUT=-2), sys=SimpleNamespace(executable='NO_EXECUTABLE'),
        os=SimpleNamespace(environ={}), time=SimpleNamespace(time=lambda: 0, time_ns=lambda: 1, sleep=lambda _: None),
        traceback=SimpleNamespace(format_exc=lambda: 'INERT_FIXTURE_ONLY'), summarize=lambda _: events.append('SUMMARY_MUST_NOT_RUN'))
    # Initial inspect must return missing for all96 jobs; later inspect denotes a
    # terminal accepted peer. No model/metrics are represented by this fixture.
    namespace['inspect'] = lambda item: None if len(processes) == 0 else {'fixture_terminal': True}
    functions(P / 'run_fullcoverage.py', ['run'], namespace)
    previous = sys.modules.get('fcntl')
    sys.modules['fcntl'] = SimpleNamespace(flock=lambda *a: None, LOCK_EX=1, LOCK_NB=2)
    try:
        try: namespace['run'](InertPath('NO_REPO'))
        except (RuntimeError, ValueError): pass
        else: raise AssertionError('Fixture failure must fail stop')
    finally:
        if previous is None: del sys.modules['fcntl']
        else: sys.modules['fcntl'] = previous
    assert len(processes) == 2 and 'SUMMARY_MUST_NOT_RUN' not in events
    if early_identity_error:
        assert all(p.waited for p in processes)
        return dict(scenario='identity_error_with_two_active_peers', launched=2, both_peers_waited=True, summary_called=False)
    assert progress[-1]['failed'] and progress[-1]['completed_new'] == 2 and progress[-1]['pending'] == 94
    return dict(scenario='one_failed_child_one_terminal_peer', launched=2, actual_strict_success_fixture=1,
        failed_exits=1, reported_completed_new=2, dispatch_stopped=True, peer_drained=True, summary_called=False)


def main():
    assert sha(P / 'FILES_SHA256.json') == '8ccb07808954aae391c127e1a2c1bc3c41e5fe6612fb72c8481fc209ff4c80f9'
    seal = read(P / 'FILES_SHA256.json')
    for name, pin in seal['files'].items(): assert sha(P / name) == pin['sha256'] and (P / name).stat().st_size == pin['bytes']
    oldworker = (O / 'source/worker.py').read_text(); newworker = (P / 'source/worker.py').read_text()
    original = functions(O / 'source/worker.py', ['digest', 'write_json', 'aggregation_wrapper', 'load_core'], {})
    new = functions(P / 'source/worker.py', ['digest', 'write_json', 'aggregation_wrapper', 'load_core'], {})
    assert original == new
    start = '        os.environ["CUBLAS_WORKSPACE_CONFIG"]'; end = '        for name in ("trajectory_metrics", "round_summaries"):'
    assert oldworker[oldworker.index(start):oldworker.index(end)] == newworker[newworker.index(start):newworker.index(end)]
    unchanged = ['flgmm_adapter.py', 'sources/flgmm_pinned.py', 'sources/flgmm_fedavg.py', 'sources/flgmm_license.txt']
    assert all(sha(P / 'source' / name) == sha(O / 'source' / name) for name in unchanged)
    assert sha(P / 'rng_capture.py') == sha(R / 'tmp/celeba_flgmm_gpu_gate_v3_20261009/rng_capture.py')
    oldprotocol = read(O / 'source/protocol.json'); manifest = read(O / 'jobs/manifest.json')
    namespace = dict(copy=copy, itertools=__import__('itertools'), METHOD='FLGMM-author-code')
    functions(P / 'source/worker.py', ['validate_job'], namespace)
    functions(P / 'source/prepare_jobs.py', ['definitions', 'make_job'], namespace)
    # Metadata fixture uses the first declared recipe; it does not select or bind
    # it. Every original actual job remains available for root recipe adoption.
    candidate = oldprotocol['candidates'][0]
    protocol = dict(copy.deepcopy(oldprotocol), scope='flgmm_selected_100_valid_only', version='INERT_METADATA_FIXTURE', selected_recipe=candidate, candidates=[candidate], attacks=['Benign','F Flip','FedSA','S-DFA','Sp-DFA'], seeds=list(range(91001,91011)))
    keys = ['flgmm_adapter.py','worker.py','prepare_jobs.py','protocol.json','sources/flgmm_pinned.py','sources/flgmm_fedavg.py','sources/flgmm_license.txt']
    hashes = {name: sha(P / 'source' / name) for name in keys}
    jobs = list(namespace['definitions'](protocol, hashes)); gates = [namespace['make_job'](protocol, hashes, 'non-IID', a, 91001, 'preflight') for a in protocol['attacks']]
    for job in jobs + gates: namespace['validate_job'](job, protocol)
    newcells = {(j['distribution'],j['attack'],j['config']['seed']) for j in jobs}
    reused = {(d,a,91001) for d in protocol['distributions'] for a in ['Benign','S-DFA']}
    assert len(jobs) == 96 and len(gates) == 5 and newcells.isdisjoint(reused) and len(newcells | reused) == 100
    actualold = [item for item in manifest['jobs'] if item['tuning_candidate'] == candidate['id']]
    assert len(actualold) == 4
    for item in actualold:
        assert sha(O / 'jobs' / item['job']) == item['job_sha256']
        job = read(O / 'jobs' / item['job']); assert job['source_hashes'] == oldprotocol['source_hashes'] and job['adapter'] == candidate['adapter'] and job['config']['rounds'] == 70
    horizon = dict(ast=ast, copy=copy); functions(P / 'canary_reference.py', ['horizon_functions'], horizon)
    patched = horizon['horizon_functions'](oldworker); validator = next(n for n in patched.body if n.name == 'validate_job')
    ns = dict(METHOD='FLGMM-author-code'); exec(compile(ast.Module(body=[validator], type_ignores=[]), 'EXACT_OLD_VALIDATOR3', 'exec'), ns)
    oldjob = read(O / 'jobs' / next(x['job'] for x in actualold if x['distribution']=='non-IID' and x['attack']=='Benign'))
    oldjob['config']['rounds'] = 3; ns['validate_job'](oldjob, oldprotocol)
    newjob = next(j for j in gates if j['attack'] == 'Benign')
    differences = {k for k in oldjob['config'] if oldjob['config'][k] != newjob['config'][k]}
    assert differences == {'experiment_suite','experiment_tag'}
    for path in P.rglob('*.py'): ast.parse(path.read_text())
    fixtures = [queue_fixture(), queue_fixture(True)]
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    for name, pin in seal['files'].items(): assert sha(P / name) == pin['sha256']
    output = dict(status='READ_ONLY_SOURCE_METADATA_AST_REVIEW_COMPLETE', source_seal_sha256=sha(P/'FILES_SHA256.json'), source_members_unchanged=23,
        exact_worker_functions=list(original), runtime_rng_core_call_block_byte_exact=True, exact_adapter_author_files=unchanged, rng_recorder_exact_original_v3=True,
        fixture_new_jobs=96, fixture_reused_cells=4, fixture_canaries=5, fixture_union_unique100=True, fixture_recipe_not_selected_or_bound=True,
        actual_original_job_identities_checked=4, original_reference_horizon_only_constants=4, same3_horizon_config_differences=sorted(differences),
        queue_control_flow_fixtures=fixtures, scientific_imports=0, real_subprocesses=0, CNN=0, SSH=False,
        limitations=['No Linux/GPU/image execution or actual100 binding occurred.', 'Only metadata source compatibility is established; actual same-horizon canaries remain required.'])
    with (H / 'CHECKS.json').open('x', encoding='utf-8') as stream: json.dump(output, stream, indent=2); stream.write('\n')
    print(json.dumps(output))


if __name__ == '__main__': main()
