from pathlib import Path
import datetime,hashlib,json,subprocess
H=Path(__file__).resolve().parent
code='''from pathlib import Path
import datetime,hashlib,json
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
B=Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
q=json.loads((B/'formal_queue_progress.json').read_bytes())
ids=[f'minus_F_IID_F Flip_seed{s}' for s in range(91001,91011)]
done=[i for i in ids if i in q['completed']]
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='minus_F_IID_F Flip',expected_ids=ids,completed_ids=done,ten_complete=len(done)==10,main_terminal=len(q['completed']),active=len(q['active']),failed=q['failed'],scientific_acceptance=False)))
'''
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55','python -B -']
with (H/'SCENE_PROBE_COMMAND.json').open('x',encoding='utf8') as f:json.dump(dict(argv=cmd,source_sha256=hashlib.sha256(code.encode()).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2)
with (H/'SCENE_PROBE_REMOTE.py').open('x',encoding='utf8',newline='\n') as f:f.write(code)
p=subprocess.run(cmd,input=code.encode(),capture_output=True,timeout=30)
for n,v in [('SCENE_PROBE.stdout',p.stdout),('SCENE_PROBE.stderr',p.stderr)]:
 with (H/n).open('xb') as f:f.write(v)
with (H/'SCENE_PROBE_EXIT.json').open('x',encoding='utf8') as f:json.dump(dict(exit_code=p.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
p.check_returncode();j=json.loads(p.stdout)
with (H/'SCENE_PROBE.json').open('x',encoding='utf8') as f:json.dump(j,f,indent=2)
print(json.dumps(j))
