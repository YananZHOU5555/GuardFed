"""Single ordered original Linux check and minimum F transport; failure stops."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
H=Path(__file__).resolve().parent;S=H.parent/'saved';A=H/'saved001'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
gate=read(A/'GATE_PIN.json')
age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(gate['utc'])).total_seconds()
assert gate['completed']==13 and not gate['cpu110_narrow_conflicts'] and 0<=age<=300
assert gate['guide_sha256']=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert gate['saved_check_source_sha256']=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
for rel,pin in read(S/'STATIC_SOURCE_SHA256.json')['files'].items():assert sha(S/rel)==pin['sha256']
def run(name,cmd):
 assert not (A/(name+'_LOCAL_COMMAND.json')).exists()
 (A/(name+'_LOCAL_COMMAND.json')).write_text(json.dumps({'argv':cmd,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
 p=subprocess.run(cmd,capture_output=True)
 (A/(name+'_LOCAL.stdout')).write_bytes(p.stdout);(A/(name+'_LOCAL.stderr')).write_bytes(p.stderr)
 (A/(name+'_LOCAL_EXIT.json')).write_text(json.dumps({'exit':p.returncode})+'\n')
 if p.returncode:sys.stderr.write(p.stderr.decode(errors='replace'));raise RuntimeError(name+' failed; preserved, no retry')
 print(p.stdout.decode(errors='replace').strip(),flush=True)
run('LINUX',[sys.executable,'-B',str(S/'check_linux.py'),'--gate-result-sha256',gate['gate_sha256'],'--report-dir',str(A),'--allow-original-cached-root-refit'])
run('TRANSPORT',[sys.executable,'-B',str(S/'transport.py'),'--gate-result-sha256',gate['gate_sha256'],'--linux-proof-sha256',sha(A/'LINUX_SAVED_CHECK.json'),'--destination','F:/YananResearchStorage/GuardFed/fl_three_view_after48_20261011/attempt001','--report-dir',str(A)])
run('AUDIT_BIND',[sys.executable,'-B',str(H/'bind_audit.py')])
print('ACTUAL_LINUX_AND_F_COMPLETE_ZERO_FIT_AUDIT_NOT_YET_EXECUTED',flush=True)
