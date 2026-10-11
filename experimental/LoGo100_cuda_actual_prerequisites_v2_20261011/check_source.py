"""Stdlib synthetic process fixtures; no subprocess, F, SSH or installation."""
from pathlib import Path
import ast,hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'LoGo100_cuda_actual_prerequisites_20261011'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p in HERE.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
source=(HERE/'transfer_originals_proposal.py').read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='drain_failed')
ns={'subprocess':subprocess};exec(compile(ast.Module(body=[node],type_ignores=[]),'<fixture>','exec'),ns)
class Pipe:
    def close(self):raise BrokenPipeError('synthetic broken input')
class Process:
    def __init__(self,waits=0):self.stdin=Pipe();self.killed=0;self.calls=0;self.returncode=None;self.waits=waits
    def poll(self):return self.returncode
    def kill(self):self.killed+=1;self.returncode=-9
    def communicate(self,timeout):
        self.calls+=1
        if self.calls<=self.waits:raise subprocess.TimeoutExpired('synthetic',timeout,output=b'partial-out',stderr=b'partial-err')
        self.returncode=self.returncode or 1;return b'actual-receiver-out',b'actual-receiver-error'
cases=[]
p=Process();assert ns['drain_failed'](p,BrokenPipeError())==(b'actual-receiver-out',b'actual-receiver-error',None) and p.stdin is None;cases.append('BrokenPipe_closes_and_drains_exact_streams')
p=Process();assert ns['drain_failed'](p,subprocess.TimeoutExpired('x',600))[2] is None and p.killed==1;cases.append('timeout_kills_and_drains')
p=Process(1);assert ns['drain_failed'](p,ValueError())[2] is None and p.killed==1;cases.append('delayed_exit_bounded_kill_and_drain')
p=Process(2);out=ns['drain_failed'](p,ValueError());assert out[:2]==(b'partial-out',b'partial-err') and out[2];cases.append('unresolved_drain_preserves_available_bytes_and_error')
assert ns['drain_failed'](None,FileNotFoundError())==(b'',b'',None);cases.append('Popen_failure_has_recordable_null_exit')
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
tr=next(n for n in main.body if isinstance(n,ast.Try))
assert tr.finalbody and any(isinstance(n,ast.Raise) for h in tr.handlers for n in h.body)
assert "'exception':error" in source and "'drain_error':drain_error" in source and 'communicated=True' in source
assert "inventory=ORIGINAL/'ACTUAL_ORIGINAL_INVENTORY.json'" in source
shell=(HERE/'create_cpu_environment_proposal.sh').read_text()
for phrase in ['test ! -L "$LoGo_ENV"','test ! -e "$UV_PYTHON_INSTALL_DIR"','test ! -L "$UV_PYTHON_INSTALL_DIR"','LoGo_PYTHON=$(readlink -f "$LoGo_PYTHON")','"$UV_PYTHON_INSTALL_DIR"/*) test -f "$LoGo_PYTHON"']:
    assert phrase in shell
cases.append('fresh_both_paths_no_dangling_links_and_resolved_managed_prefix')
assert (HERE/'receive_originals_proposal.py').read_bytes()==(PARENT/'receive_originals_proposal.py').read_bytes()
assert sha(PARENT/'FILES_SHA256.json')=='eca43efd64fef253f4227a75945fbe216215452fff7b3d153140e3bbae2e61f3'
assert sha(PARENT/'ACTUAL_ORIGINAL_INVENTORY.json')=='926b478ea38d977f21151ddf3d2f7ca1cd14e6430caa2abcd8e1a699356b2a2b'
report={'status':'PASS_SOURCE_ONLY_SYNTHETIC_NO_EXECUTION','checks':cases,'receiver_byte_exact_parent':True,'old_seal_and_inventory_unchanged':True,'actual_transfer':None,'actual_install':None,'actual_science':None}
(HERE/'SOURCE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))
