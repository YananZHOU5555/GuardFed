"""Actual queue control-flow fixtures and non-optimized entry rejection. No scientific work."""
import ast, copy, hashlib, json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
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
    assert progress[-1]['failed'] and progress[-1]['completed_new'] == 1 and progress[-1]['pending'] == 94
    return dict(scenario='one_failed_child_one_terminal_peer', launched=2, actual_strict_success_fixture=1,
        failed_exits=1, reported_completed_new=1, dispatch_stopped=True, peer_drained=True, summary_called=False)

def main():
    fixtures=[queue_fixture(),queue_fixture(True)]
    # Check the actual initializer on seven strict accepted skips, without paths/processes.
    node=next(n for n in ast.parse((P/'run_fullcoverage.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assignments=[n for n in ast.walk(node) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('pending','completed_new') for x in n.targets)]
    ns={'manifest':{'jobs':[{'id':i} for i in range(96)]},'inspect':lambda item: {'accepted':True} if item['id']<7 else None}
    exec(compile(ast.Module(body=assignments,type_ignores=[]),'ACCEPTED_SKIP_INITIALIZER','exec'),ns)
    assert len(ns['pending'])==89 and ns['completed_new']==7
    entries=['bind_stage.py','run_canaries.py','run_one.py','run_fullcoverage.py','canary_reference.py']
    checks=[]
    for entry in entries:
        for mode in ['normal','-O','-OO','env1','env2']:
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
            options=[]
            if mode.startswith('-'):options=[mode]
            if mode.startswith('env'):env['PYTHONOPTIMIZE']=mode[-1]
            process=subprocess.run([sys.executable,'-B',*options,str(P/entry),'--help'],capture_output=True,text=True,env=env,timeout=15)
            if mode=='normal':assert process.returncode==0,(entry,process.stderr)
            else:
                assert process.returncode!=0 and 'Optimized Python is forbidden' in process.stderr,(entry,mode,process.stderr)
            checks.append(dict(entry=entry,mode=mode,returncode=process.returncode,explicit_guard_rejected=mode!='normal'))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(json.dumps(dict(status='PASS_V2_ENGINEERING_FIXTURES_ONLY',queue_fixtures=fixtures,accepted_skip_initializer=7,
        entry_checks=checks,optimization_refusals=20,normal_help_positive=5,scientific_execution=0,real_training_children=0),indent=2))

if __name__=='__main__':main()
