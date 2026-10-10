"""One-time source derivation only. Never SSH, transport, refit or verify arrays."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
OLD_BASE = '/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010'
BASE = '/workspace/guardfed_checks/celeba_flgmm_three_view_closed_batch_20261011'
OLD_PACKAGE = '49c42b90214eae57d456f8ded1d2a4ff2de2762a78d6bcf5c4ae1a5766de9434'
PACKAGE = '43b53caf0d6a0b4ac268fdad6597065cd76eee26d52f6d4929d45478e47cf023'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old))
    return text.replace(old, new)


def save(name, text):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)


def main():
    candidate = ROOT / 'tmp/celeba_flgmm_three_view_closed_batch_preparation_20261011'
    assert sha(candidate / 'FILES_SHA256.json') == PACKAGE
    m = json.loads((candidate / 'MANIFEST.json').read_text(encoding='utf-8'))
    ids = m['exact_ids']
    assert len(ids) == 47 and len(set(ids)) == 47
    path = ROOT / 'tmp/celeba_added_cnn_exact3_root_execution_20261010/linux_saved_root_remote.py'
    assert sha(path) == '4a4e3d9a533bb87440e92107fde5b08d78369eee07e71bb10799d5d0b6d12f5e'
    before = path.read_text(encoding='utf-8')
    code = replace(before, OLD_BASE, BASE)
    code = replace(code, OLD_PACKAGE, PACKAGE)
    code = replace(code, 'payload=json.load(sys.stdin)', """import argparse,subprocess
ap=argparse.ArgumentParser(description='Root-only unchanged original whole saved checker; no CNN')
ap.add_argument('--gate-result-sha256',required=True)
ap.add_argument('--allow-original-cached-root-refit',action='store_true',required=True)
a=ap.parse_args()
saved=pathlib.Path('/workspace/guardfed_checks/celeba_flgmm_three_view_closed_batch_20261011/source/originals/saved_science.py')
payload={'saved_check_source':saved.read_text(encoding='utf-8'),'source_sha256':hashlib.sha256(saved.read_bytes()).hexdigest()}""")
    code = replace(code, 'LINUX_EXACT_ROOT_CHECK.json', 'LINUX_SAVED_CHECK.json')
    code = replace(code, 'LINUX_EXACT_ROOT_FAILURE.json', 'LINUX_SAVED_FAILURE.json', 2)
    code = replace(code, "import numpy as np,pandas as pd,torch", """out=base/'outputs/attempt001'
assert hashlib.sha256((out/'GATE_RESULT.json').read_bytes()).hexdigest()==a.gate_result_sha256
gate=c.read(out/'GATE_RESULT.json')
assert not (out/'FAILURE.json').exists() and not list(out.rglob('FAILURE.json'))
assert gate['status']=='FLGMM_EXACT47_THREE_VIEW_PASS_NOT_ROOT_ADOPTED'
assert gate['package_sha256']=='43b53caf0d6a0b4ac268fdad6597065cd76eee26d52f6d4929d45478e47cf023'
assert len(gate['receipts'])==47 and [r['id'] for r in gate['receipts']]==m['exact_ids']
status=subprocess.run(['supervisorctl','status','guardfed_flgmm_closed47_valid'],capture_output=True,text=True)
assert status.returncode==3 and status.stdout.split()[:2]==['guardfed_flgmm_closed47_valid','EXITED'],status.stdout
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 try:
  cmd=(proc/'cmdline').read_bytes().decode(errors='replace').split('\\0')
  assert not any(pathlib.Path(token).name=='candidate.py' and 'celeba_flgmm_three_view_closed_batch_20261011' in token for token in cmd),'Live replay worker'
 except (FileNotFoundError,ProcessLookupError):pass
 for task in (proc/'task').glob('*'):
  try:
   affinity=set(os.sched_getaffinity(int(task.name)))
   assert not (len(affinity)<=8 and affinity.intersection(os.sched_getaffinity(0))),'Restricted-thread CPU overlap'
  except ProcessLookupError:pass
import numpy as np,pandas as pd,torch""")
    code = replace(code, "gate=c.read(base/'outputs/attempt001/GATE_RESULT.json');assert gate['status']=='EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED'", "assert gate['status']=='FLGMM_EXACT47_THREE_VIEW_PASS_NOT_ROOT_ADOPTED'")
    code = replace(code, 'LINUX_ORIGINAL_EXACT3_SAVED_ROOT_AND_ARRAY_CHECK_PASS_NOT_ROOT_ADOPTED', 'LINUX_ORIGINAL_FLGMM47_WHOLE_SAVED_CHECK_PASS_NOT_ROOT_ADOPTED')
    code = replace(code, "'cached_root_refits':3", "'cached_root_refits':47")
    code = replace(code, "'original_check_saved_sha256':payload['source_sha256'],", "'original_check_saved_sha256':payload['source_sha256'],'gate_result_sha256':a.gate_result_sha256,'package_sha256':'"+PACKAGE+"',")
    code = replace(code, 'LINUX_EXACT_ROOT_FAILED_PRESERVED_NO_RETRY', 'LINUX_FLGMM47_WHOLE_SAVED_FAILED_PRESERVED_NO_RETRY')
    compile(code, 'linux_saved_remote.py', 'exec')
    save('linux_saved_remote.py', code)
    save('LINUX_SOURCE_DIFF.patch', ''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile=str(path),tofile='linux_saved_remote.py')))
    transport_path = ROOT / 'tmp/transport_added_cnn_exact3_root_20261010.py'
    assert sha(transport_path) == 'e17d6d9b79cf036fbf645876f9d7d77c6ee44145056f1ab737f4a71e047cbd09'
    before_transport = transport_path.read_text(encoding='utf-8')
    tree = ast.parse(before_transport)
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    remote_assign = next(n for n in main_node.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets))
    old_remote = ast.literal_eval(remote_assign.value)
    remote = replace(old_remote, OLD_BASE, BASE)
    old_ids = "['FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage','CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage','FedNGA_eta0.01_non-IID_Benign_seed91001_screen']"
    remote = replace(remote, old_ids, repr(ids))
    remote = replace(remote, OLD_PACKAGE, PACKAGE)
    remote = replace(remote, 'EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED', 'FLGMM_EXACT47_THREE_VIEW_PASS_NOT_ROOT_ADOPTED')
    remote = replace(remote, "assert 'EXITED' in status.stdout and 'RUNNING' not in status.stdout,status.stdout", """assert status.returncode==3 and status.stdout.split()[:2]==['guardfed_flgmm_closed47_valid','EXITED'],status.stdout
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 try:
  command=(proc/'cmdline').read_bytes().decode(errors='replace').split('\\0')
  assert not any(pathlib.Path(token).name=='candidate.py' and 'celeba_flgmm_three_view_closed_batch_20261011' in token for token in command),'Live replay worker'
 except (FileNotFoundError,ProcessLookupError):pass
 for task in (proc/'task').glob('*'):
  try:
   affinity=set(os.sched_getaffinity(int(task.name)))
   assert not (len(affinity)<=8 and affinity.intersection(os.sched_getaffinity(0))),'Restricted-thread CPU overlap'
  except ProcessLookupError:pass
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.getpriority(os.PRIO_PROCESS,0)>=10 and len(os.sched_getaffinity(0))==1
proof=base/'LINUX_SAVED_CHECK.json'
assert hashlib.sha256(proof.read_bytes()).hexdigest()=='__LINUX_PROOF_SHA256__'
linux=json.loads(proof.read_text())
assert linux['status']=='LINUX_ORIGINAL_FLGMM47_WHOLE_SAVED_CHECK_PASS_NOT_ROOT_ADOPTED'
assert len(linux['records'])==47 and [r['id'] for r in linux['records']]==ids
assert linux['original_check_saved_sha256']=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
assert all(r['root_receipt_exact'] and r['cached_root_fit_exact'] and r['saved_predictions_metrics_counts_exact'] for r in linux['records'])
assert linux['new_CNN']==0 and linux['new_training']==0 and linux['test'] is False""")
    remote = replace(remote, "assert linux['new_CNN']==0 and linux['new_training']==0 and linux['test'] is False", "assert linux['new_CNN']==0 and linux['new_training']==0 and linux['test'] is False\nassert linux['gate_result_sha256']=='__GATE_RESULT_SHA256__' and linux['package_sha256']=='"+PACKAGE+"'")
    remote = replace(remote, 'guardfed_added_cnn_exact3_gate', 'guardfed_flgmm_closed47_valid')
    remote = replace(remote, 'import datetime,hashlib,io,json,pathlib,subprocess,sys,zipfile', 'import datetime,hashlib,io,json,os,pathlib,subprocess,sys,zipfile')
    remote = replace(remote, "'metadata.npz':pathlib.Path('/workspace/GuardFed-celeba-expanded/data/celeba/derived/rgb64_v1/metadata.npz')", "'LINUX_SAVED_CHECK.json':proof")
    remote = replace(remote, "for name in ['stdout.log','stderr.log']:files['execution/'+name]=base/'execution'/name\n", '')
    remote = replace(remote, '20_000_000', '128_000_000')
    remote = replace(remote, "assert observed['metadata.npz']['sha256']=='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091'\n", '')
    remote = replace(remote, 'EXACT3_SAVED_ARRAY_TRANSPORT_SOURCE_MEMBERS_UNCHANGED_NOT_SCIENTIFIC_ACCEPTANCE', 'FLGMM47_SAVED_ARRAY_TRANSPORT_SOURCE_MEMBERS_UNCHANGED_NOT_ROOT_ADOPTED')
    remote = replace(remote, "assert not (out/'FAILURE.json').exists()", "assert not (out/'FAILURE.json').exists() and not list(out.rglob('FAILURE.json'))\nassert hashlib.sha256((out/'GATE_RESULT.json').read_bytes()).hexdigest()=='__GATE_RESULT_SHA256__'")
    remote = replace(remote, "for rel,path in files.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==observed[rel]['sha256']", "for row in linux['records']:\n assert row['receipt_sha256']==observed['bundle/'+row['id']+'/receipt.json']['sha256'] and row['array_sha256']==observed['bundle/'+row['id']+'/validation_predictions.npz']['sha256']\nfor rel,path in files.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==observed[rel]['sha256']")
    compile(remote.replace('__LINUX_PROOF_SHA256__','0'*64).replace('__GATE_RESULT_SHA256__','0'*64), 'transport_remote.py', 'exec')
    save('transport_remote.py', remote)
    save('TRANSPORT_REMOTE_DIFF.patch', ''.join(difflib.unified_diff(old_remote.splitlines(True),remote.splitlines(True),fromfile='original exact3 remote transport',tofile='transport_remote.py')))
    save('FIXED_IDS.json', json.dumps({'exact_ids':ids,'candidate_package_sha256':PACKAGE,'candidate_manifest_sha256':sha(candidate/'MANIFEST.json')},indent=2)+'\n')


if __name__ == '__main__':
    main()
