from pathlib import Path
import json,hashlib,datetime,subprocess
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
B=Path('/workspace/guardfed_checks/uniform_cuda_formal_20261011_resume2')
raw=(B/'progress.json').read_bytes();p=json.loads(raw)
c=json.loads((B/'cohort.json').read_bytes());expected=[r['id'] for r in c['records'] if r['id'].startswith('GuardFed-AD2+_')]
assert len(expected)==len(set(expected))==100
ids=p['remote_saved_accepted_ids'];assert len(ids)==len(set(ids)) and set(ids)<=set(c['ordered_ids'])
full=[i for i in ids if i in expected];controls=[i for i in ids if i not in expected]
r=subprocess.run(['supervisorctl','status','guardfed_uniform_cuda500_resume2_valid'],capture_output=True,text=True,timeout=20)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),service=dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr),cohort_sha256=sha(B/'cohort.json'),progress_sha256=hashlib.sha256(raw).hexdigest(),progress_utc=p['utc'],accepted_total=len(ids),accepted_Full=len(full),accepted_controls=len(controls),missing_Full=[i for i in expected if i not in full],controls=controls,active=p['active'],failed=p['failed'],queue_failure_exists=(B/'QUEUE_FAILURE.json').exists(),Full100_complete_in_progress=(len(full)==100),scientific_root_adoption_created=0,observation_only=True)))
