"""Limited independent v2 review: actual run AST + entry --help, no science."""
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
P = R / 'tmp/celeba_flgmm_fullcoverage_source_v2_20261009'
O = R / 'tmp/celeba_flgmm_fullcoverage_source_20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())


def verify(path, expected):
    if sha(path / 'FILES_SHA256.json') != expected: raise ValueError('Seal drift')
    files = read(path / 'FILES_SHA256.json')['files']
    for name, pin in files.items():
        if sha(path / name) != pin['sha256'] or (path / name).stat().st_size != pin['bytes']:
            raise ValueError('Member drift: ' + name)
    return files


def fixture(skips, failure, partial=False):
    processes, progress, writes = [], [], []
    class InertPath:
        def __init__(self, name='fixture'): self.name = name
        def __truediv__(self, name): return InertPath(self.name + '/' + str(name))
        def open(self, *args): return SimpleNamespace(close=lambda: None, closed=True)
        def mkdir(self, **kwargs): pass
        def glob(self, pattern): return []
        def __str__(self): return self.name
    class InertProcess:
        def __init__(self, item): self.item=item; self.pid=100+len(processes); self.polls=0
        def poll(self):
            self.polls += 1
            if self.item == skips: return 1 if failure else 0
            return None if self.polls == 1 else 0
        def wait(self): return 0
    def popen(argv, **kwargs):
        child=InertProcess(int(argv[-1])); processes.append(child); return child
    def inspect(item):
        identity=int(item['id'])
        if not processes: return {'accepted':True} if identity < skips else None
        return None if partial and identity == skips else {'accepted':True}
    def write(path, value):
        if str(path).endswith('queue_progress.json'): progress.append(copy.deepcopy(value))
        else: writes.append(str(path))
    summarized=[]
    ns=dict(HERE=InertPath(), Path=InertPath,
        local_identity=lambda: ({},{'jobs':[{'id':str(n)} for n in range(96)]}),
        authorized=lambda *a,**k:None, repo_identity=lambda *a:None,
        reused_records=lambda *a:[], inspect=inspect, write_json=write,
        subprocess=SimpleNamespace(Popen=popen,STDOUT=-2),sys=SimpleNamespace(executable='NO_EXECUTABLE'),
        os=SimpleNamespace(environ={}),time=SimpleNamespace(time=lambda:0,time_ns=lambda:1,sleep=lambda _:None),
        traceback=SimpleNamespace(format_exc=lambda:'INERT_FIXTURE'),summarize=lambda _: summarized.append(True))
    node=next(n for n in ast.parse((P/'run_fullcoverage.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'V2_ACTUAL_RUN_AST','exec'),ns)
    prior=sys.modules.get('fcntl');sys.modules['fcntl']=SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)
    error=None
    try:
        try:ns['run'](InertPath('NO_REPO'))
        except RuntimeError as exc:error=str(exc)
    finally:
        if prior is None:del sys.modules['fcntl']
        else:sys.modules['fcntl']=prior
    expected=skips+1 if failure or partial else 96
    if progress[-1]['completed_new'] != expected: raise ValueError('Incorrect completed count')
    if len(processes)!=2:raise ValueError('Dispatch did not stop at expected two children')
    if failure or partial:
        if not error or summarized or not progress[-1]['failed']:raise ValueError('Failstop/summary regression')
    elif error or len(summarized)!=1 or progress[-1]['failed']:raise ValueError('Successful completion regression')
    return dict(initial_strict_skips=skips,first_exit=1 if failure else 0,
        first_exit_zero_partial=partial,launched=len(processes),
        reported_completed_new=progress[-1]['completed_new'],expected_strict_success=expected,
        pending=progress[-1]['pending'],failed=progress[-1]['failed'],summary_calls=len(summarized),
        failstop=bool(error),real_processes=0)


def main():
    old=verify(O,'8ccb07808954aae391c127e1a2c1bc3c41e5fe6612fb72c8481fc209ff4c80f9')
    new=verify(P,'fc9cd4133345d74949d22cddca12868393b8ac6ac6920d2f1deb293611584c30')
    changed=[n for n in old if sha(O/n)!=sha(P/n)]
    if set(changed)!={'bind_stage.py','run_fullcoverage.py','screen_common.py'}:raise ValueError('Unexpected inherited source change')
    for name in ['bind_stage.py','screen_common.py']:
        text=(P/name).read_text(); original=(O/name).read_text()
        guard="\nif sys.flags.optimize:\n    raise RuntimeError('Optimized Python is forbidden: scientific identity/comparison assertions must execute')\n"
        if text.replace(guard,'',1)!=original:raise ValueError('Additional guard file changes')
    difference=''.join(''.join(difflib.unified_diff((O/n).read_text().splitlines(True),(P/n).read_text().splitlines(True),fromfile='v1/'+n,tofile='v2/'+n)) for n in changed)
    with (H/'ACTUAL_SOURCE_DIFF.patch').open('x',encoding='utf-8',newline='\n') as f:f.write(difference)
    fixtures=[fixture(7,True),fixture(0,False,True),fixture(94,False)]
    checks=[]
    entries=['bind_stage.py','run_canaries.py','run_one.py','run_fullcoverage.py','canary_reference.py']
    for entry in entries:
        for mode in ['normal','-O','-OO','env1','env2']:
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
            options=[mode] if mode.startswith('-') else []
            if mode.startswith('env'):env['PYTHONOPTIMIZE']=mode[-1]
            proc=subprocess.run([sys.executable,'-B',*options,str(P/entry),'--help'],capture_output=True,text=True,env=env,timeout=15)
            if mode=='normal':
                if proc.returncode!=0 or 'usage:' not in proc.stdout:raise ValueError('Normal help failed '+entry)
            elif proc.returncode==0 or 'Optimized Python is forbidden' not in proc.stderr:
                raise ValueError('Optimization not explicitly refused '+entry+' '+mode)
            checks.append(dict(entry=entry,mode=mode,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    verify(O,'8ccb07808954aae391c127e1a2c1bc3c41e5fe6612fb72c8481fc209ff4c80f9')
    verify(P,'fc9cd4133345d74949d22cddca12868393b8ac6ac6920d2f1deb293611584c30')
    if 'torch' in sys.modules or 'numpy' in sys.modules:raise ValueError('Unexpected scientific import')
    output=dict(status='PASS_LIMITED_V2_REVIEW',changed_inherited_files=changed,
        inherited_byte_identical=len(old)-len(changed),v2_sealed_members=len(new),
        added_files=sorted(set(new)-set(old)),queue_actual_run_AST_fixtures=fixtures,
        optimization_refusals=20,normal_help_positive=5,entry_subprocesses=checks,
        science_worker_rng_comparator_acceptor_byte_identical=True,
        scientific_runs=0,SSH=False,stage_bound=False,canaries_executed=False,
        source_before_after_unchanged=True)
    with (H/'CHECKS.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(output,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k!='entry_subprocesses'}))


if __name__=='__main__':main()
