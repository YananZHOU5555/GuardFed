from pathlib import Path
import ast,datetime,hashlib,json,subprocess
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
def save(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
guide=subprocess.run(ssh+['cat /etc/vast-agents-guide.md'],capture_output=True,timeout=30)
for n,b in [('SERVER_GUIDE.md',guide.stdout),('GUIDE_STDERR.txt',guide.stderr)]:
 with (H/n).open('xb') as f:f.write(b)
save('GUIDE_RECEIPT.json',dict(command=ssh+['cat /etc/vast-agents-guide.md'],exit_code=guide.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),guide_sha256=sha(H/'SERVER_GUIDE.md')))
guide.check_returncode();assert sha(H/'SERVER_GUIDE.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
original=R/'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/observe.py'
tree=ast.parse(original.read_text(encoding='utf8'));node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets));body=ast.literal_eval(node.value)
cmd=ssh+['env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 107 ionice -c 3 nice -n 10 python -B -']
start=datetime.datetime.now(datetime.timezone.utc).isoformat();result=subprocess.run(cmd,input=body.encode(),capture_output=True,timeout=75)
for n,b in [('OBSERVE_STDOUT.txt',result.stdout),('OBSERVE_STDERR.txt',result.stderr)]:
 with (H/n).open('xb') as f:f.write(b)
save('OBSERVE_COMMAND.json',dict(command=cmd,exit_code=result.returncode,start=start,finish=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_observer_path=original.relative_to(R).as_posix(),original_observer_sha256=sha(original),remote_body_sha256=hashlib.sha256(body.encode()).hexdigest(),remote_body_unchanged=True))
result.check_returncode();value=json.loads(result.stdout);save('SNAPSHOT.json',value)
print(json.dumps(dict(utc=value['utc'],source_members_match=value['source_members_match'],queue=value['queue'],failure_paths=value['failure_paths'],service=value['coverage_service']['stdout'].strip(),snapshot_sha256=sha(H/'SNAPSHOT.json'))))
