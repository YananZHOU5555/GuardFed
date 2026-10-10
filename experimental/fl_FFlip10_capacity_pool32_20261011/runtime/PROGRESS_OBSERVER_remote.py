from pathlib import Path
import os,json,time,subprocess,hashlib,datetime
P=Path
assert hashlib.sha256(P('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
base=P('/workspace/guardfed_checks/fl_FFlip10_capacity_pool32_20261011')
pid=99622
proc=P('/proc')/str(pid)
def snap():
 if not proc.exists():return None
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,ticks=int(stat[11])+int(stat[12]),argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0'),affinity=sorted(os.sched_getaffinity(pid)))
a=snap();time.sleep(2);b=snap()
q=subprocess.run(['supervisorctl','status','guardfed_flgmm_FFlip10_capacity_pool32_valid'],capture_output=True,text=True)
out=base/'outputs/attempt001'
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),service=dict(exit=q.returncode,stdout=q.stdout,stderr=q.stderr),before=a,after=b,output_names=sorted(str(x.relative_to(out)) for x in out.rglob('*.json')) if out.exists() else [],new_scientific_acceptances=0)
for name in ['execution/stdout.log','execution/stderr.log']:
 f=base/name;r[name]=f.read_bytes()[-4000:].decode(errors='replace') if f.exists() else None
r['receipt_summary']=[]
for f in sorted(out.glob('*/receipt.json')):
 j=json.loads(f.read_bytes());r['receipt_summary'].append(dict(id=j['id'],native=j['native_comparison']))
for name in ['GATE_RESULT.json','FAILURE.json']:
 f=out/name
 if f.exists():r[name]=dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),value=json.loads(f.read_bytes()))
print(json.dumps(r))
