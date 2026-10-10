"""Verify actual process and physical-GPU identity; no scientific-result claim."""
from pathlib import Path
import datetime,hashlib,json
HERE=Path(__file__).resolve().parent
path=HERE/'LATEST_ATTEMPT2_OBSERVATION.json';d=json.loads(path.read_bytes())
assert d['service']['returncode']==0 and len(d['processes'])==2 and len(d['rows'])==1
assert d['rows'][0]['progress']['round']>=4 and not d['failure_paths']
assert all(x['nice']==10 and 'idle' in x['io'].lower() and all(t['cpus']==[105] for t in x['threads']) for x in d['processes'])
r=d['rows'][0]['provenance']
assert r['environment']['torch']=='2.11.0+cu128' and r['environment']['device']=='cuda'
gpu_rows=[[x.strip() for x in line.split(',')] for line in d['gpu_compute_apps']['stdout'].splitlines() if line.strip()]
worker=d['rows'][0]['progress']['pid'];matches=[row for row in gpu_rows if int(row[0])==worker]
assert d['gpu_compute_apps']['returncode']==0 and len(matches)==1 and matches[0][1]==d['resource_proof']['gpu_uuid']
proof=dict(status='ROOT_ACTUAL_GRADIENT64_SINGLE_GPU_WORKER_AND_ROUND_PROGRESS_VERIFIED',
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_unix=d['at_unix'],
 observation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physical_GPU=1,actual_GPU_UUID=matches[0][1],
 logical_device=r['environment']['device'],actual_CPU=[105],actual_workers=1,coordinators=1,nice=10,idle_io=True,
 observed_round=d['rows'][0]['progress']['round'],resource_proof=d['resource_proof_path'],
 scientific_completed=0,offserver_accepted=0,old_preflight_failures_preserved=True,
 earlier_root_review_failure='Expected cuda:0 but actual provenance serializes cuda; only the root reviewer assumption was wrong, original outputs unchanged')
(HERE/'ROOT_STARTUP_REVIEW.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
print(json.dumps(proof))
