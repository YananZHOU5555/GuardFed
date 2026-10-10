"""One compact observation on explicit invocation; no busy polling or science."""
from pathlib import Path
import base64,datetime,hashlib,json,subprocess,sys
H=Path(__file__).resolve().parent
name='GATE_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
source='''import pathlib,subprocess,json,hashlib,datetime,os
P=pathlib.Path;base=P('/workspace/guardfed_checks/fl_three_view_after48_20261011');out=base/'outputs/attempt001'
status=subprocess.run(['supervisorctl','status','guardfed_flgmm_after48_exact13_valid'],capture_output=True,text=True)
members={}
for p in sorted(out.glob('*/receipt.json')):
 b=p.read_bytes();members[p.parent.name]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
failures={str(p):p.read_text() for p in out.rglob('FAILURE.json')}
gate=out/'GATE_RESULT.json'; saved_check=base/'LINUX_SAVED_CHECK.json'
conflict=[]
for proc in P('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 for t in (proc/'task').glob('*'):
  try:
   af=set(os.sched_getaffinity(int(t.name)))
   if len(af)<=8 and 110 in af:conflict.append([int(proc.name),int(t.name),sorted(af)])
  except ProcessLookupError:pass
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'service':{'exit':status.returncode,'stdout':status.stdout,'stderr':status.stderr},'completed':members,'failures':failures,'stderr_tail':(base/'execution/stderr.log').read_text()[-8000:],'gate':json.loads(gate.read_bytes()) if gate.exists() else None,'gate_sha256':hashlib.sha256(gate.read_bytes()).hexdigest() if gate.exists() else None,'cpu110_narrow_conflicts':conflict,'linux_saved_exists':saved_check.exists(),'cpu_quota':P('/sys/fs/cgroup/cpu.max').read_text(),'memory_current':P('/sys/fs/cgroup/memory.current').read_text(),'guide_sha256':hashlib.sha256(P('/etc/vast-agents-guide.md').read_bytes()).hexdigest(),'saved_check_source_sha256':hashlib.sha256((base/'source/originals/saved_science.py').read_bytes()).hexdigest()}))
'''
compile(source,'observe','exec')
code="import base64;exec(compile(base64.b64decode('"+base64.b64encode(source.encode()).decode()+"'),'<gate-observe>','exec'))"
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','python3 -c "'+code+'"']
p=subprocess.run(cmd,capture_output=True,timeout=45)
(H/(name+'.stdout')).write_bytes(p.stdout);(H/(name+'.stderr')).write_bytes(p.stderr)
(H/(name+'_COMMAND.json')).write_text(json.dumps({'argv':cmd,'exit':p.returncode},indent=2)+'\n')
assert p.returncode==0,p.stderr.decode(errors='replace')
j=json.loads(p.stdout);(H/(name+'.json')).write_text(json.dumps(j,indent=2)+'\n')
print(json.dumps({'utc':j['utc'],'service':j['service']['stdout'].strip(),'completed':len(j['completed']),'failure':len(j['failures']),'gate_sha256':j['gate_sha256'],'cpu110_conflicts':j['cpu110_narrow_conflicts']}))
if j['gate'] is not None:
 (H/'saved001').mkdir(exist_ok=True)
 with (H/'saved001/GATE_PIN.json').open('x') as f:json.dump({'gate_sha256':j['gate_sha256'],'observation':name+'.json','guide_sha256':j['guide_sha256'],'saved_check_source_sha256':j['saved_check_source_sha256'],'completed':len(j['completed']),'cpu110_narrow_conflicts':j['cpu110_narrow_conflicts'],'utc':j['utc']},f,indent=2)
