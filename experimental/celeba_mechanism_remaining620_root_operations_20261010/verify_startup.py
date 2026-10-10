"""Join actual first CPU evaluator closure to already accepted native evidence."""
from pathlib import Path
import datetime,hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
d=read(HERE/'LATEST_OBSERVATION.json')
observations=[p for p in HERE.glob('OBSERVATION_*.json') if sha(p)==sha(HERE/'LATEST_OBSERVATION.json')]
assert len(observations)==1
observation=observations[0]
assert d['service']['returncode']==0 and d['service']['stdout'].split()[:2]==['guardfed_celeba_mechanism_remaining620_valid_v2a','RUNNING']
assert d['source_seal_sha256']=='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
assert d['root_approval_sha256']==sha(HERE/'ROOT_APPROVED.json')=='55e0c1a5c08fd00a33ff1caaa862b5b1e67c4328559b750a95b6d0a5d1aebb6c'
assert not d['failures'] and d['remote_closed_n']>=1 and d['accepted_offserver']==0
assert len([p for p in d['processes'] if 'manage' in p['argv']])==1
assert len([p for p in d['processes'] if 'worker' in p['argv']])<=1
for p in d['processes']:
 assert p['nice']==10 and p['ionice']=='idle' and p['safe_env']['CUDA_VISIBLE_DEVICES']==''
 assert all(t['cpus']==list(range(112,120)) for t in p['threads'])
 if 'worker' in p['argv']:
  assert p['safe_env']['OMP_NUM_THREADS']==p['safe_env']['MKL_NUM_THREADS']=='8'
native=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T034047Z/inspection/inspection.json'
assert sha(native)=='327c3b208bb074ed8b6fad484f5e2b10168cce89fd56bb6e87b0b47b7b8acfb8'
records={r['id']:r for r in read(native)['records']}
closed=[]
for task in d['tasks']:
 c=task['remote_complete']
 if not c:continue
 b=task['binding'];r=b['record']['accepted_v4_row']
 assert records[task['id']]==r and b['checkpoint_sha256']==r['checkpoint_sha256']==c['checkpoint_sha256']
 assert c['status']=='REMOTE_STRICT_CLOSED_PENDING_OFFSERVER' and c['native_difference']<=1e-12 and c['accepted_offserver']==0
 assert c['parent_review_sha256']==d['root_approval_sha256']
 closed.append(dict(id=task['id'],checkpoint_sha256=c['checkpoint_sha256'],native_difference=c['native_difference'],binding_sha256=task['hashes']['binding.json'],remote_complete_sha256=task['hashes']['REMOTE_COMPLETE.json']))
proof=dict(status='ROOT_ACTUAL_REMAINING620_CPU_QUEUE_AND_FIRST_REMOTE_STRICT_CLOSURE_VERIFIED',
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_utc=d['utc'],source_seal_sha256=d['source_seal_sha256'],
 actual_service='guardfed_celeba_mechanism_remaining620_valid_v2a',observation_path=observation.relative_to(ROOT).as_posix(),
 observation_sha256=sha(observation),root_approval_sha256=d['root_approval_sha256'],
 native188_inspection_sha256=sha(native),all_closed_original_native_rows_exact=True,remote_strict_closed=len(closed),closed=closed,
 accepted_offserver=0,root_adopted=0,allowed_cpus=list(range(112,120)),compute_threads=8,max_CNN_workers=1,
 first_wrapper_syntax_failure_preserved=True,fixed_wrapper_sha256=sha(HERE/'service_v2a.sh'),
 dispatch_receipt_sha256=sha(HERE/'V2A_DISPATCH_RECEIPT.json'),scientific_source_unchanged=True,
 new_training=0,new_Full_inference=0,test=False,automatic_retry=False)
p=HERE/'ROOT_STARTUP_REVIEW_V2.json'
if (HERE/'ROOT_STARTUP_REVIEW.json').exists():
 proof['prior_startup_review_sha256']=sha(HERE/'ROOT_STARTUP_REVIEW.json')
 proof['metadata_amendment']='Bind immutable actual observation filename; same measured bytes and scientific scope.'
with p.open('x',encoding='utf8') as f:f.write(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(path=str(p),sha256=sha(p),remote_strict_closed=len(closed),accepted_offserver=0)))
