"""Check readonly snapshot growth, with zero scientific acceptance."""
from pathlib import Path
import datetime,hashlib,json
R=Path(__file__).resolve().parents[1];D=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=D/'root_five_queue_20261010T113833Z.raw.json';now=D/'root_five_queue_20261010T123338Z.raw.json'
assert sha(before)=='480ac705d3fdbcefc6feb9be3f5ba0906b159a9b1625953b3b9631a9116e9bbe'
assert sha(now)=='a62f9cd234d93c8d244e047c57d0f7f034d89e969a4746152e9fb84df7404f2e'
a,b=(json.loads(p.read_bytes()) for p in (before,now))
assert not b['source_data_binding']['changed_paths'] and b['new_accepted']==0
for x in ('FLGMM','gradient64','remaining620','Hybrid96'):
    assert b[x]['service']['returncode']==0 and 'RUNNING' in b[x]['service']['stdout']
    assert not b[x].get('failure_paths',b[x].get('failures',b[x].get('failure_files',[])))
    assert not b[x].get('queue_failed',False)
    if 'source' in b[x]:assert not b[x]['source']['changed_members']
assert not b['main_mechanism']['queue']['failed'] and b['main_mechanism']['worker_count']==8
assert not b['Hybrid96']['queue_progress']['failed']
growth={}
for x in ('FLGMM','gradient64'):
    old,new=set(a[x]['terminal_ids']),set(b[x]['terminal_ids']);assert old<=new
    growth[x]={'terminal_before':len(old),'terminal_now':len(new),'new_terminal':sorted(new-old),'active_now':b[x]['active']}
old,new=({r['id'] for r in d['Hybrid96']['terminal_records']} for d in (a,b));assert old<=new
growth['Hybrid96']={'terminal_before':len(old),'terminal_now':len(new),'new_terminal':sorted(new-old),'active_now':b['Hybrid96']['queue_progress']['active']}
old,new=(set(d['main_mechanism']['queue']['completed']) for d in (a,b));assert old<new
growth['main']={'terminal_before':len(old),'terminal_now':len(new),'new_terminal':sorted(new-old),'workers':b['main_mechanism']['workers']}
assert set(a['remaining620']['remote_closed_ids'])<=set(b['remaining620']['remote_closed_ids'])
for v in b['logs'].values():
    assert all(not z['possible_error_lines'] for z in v['logs'])
out=dict(status='ROOT_FIVE_LIVE_QUEUES_IDENTITY_AND_PROGRESS_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_utc=b['utc'],snapshots=[dict(path=p.relative_to(R).as_posix(),sha256=sha(p)) for p in (before,now)],growth=growth,main_workers=8,remaining620_closed=b['remaining620']['remote_closed_n'],source_changes=[],failures=[],source_data_scope=b['source_data_binding']['bulk_identity_scope'],new_scientific_acceptance=0,claim='Observed terminal-set growth with correct source/worker identities, not scientific result acceptance; gradient active metadata was sampled during handover.')
p=D/'ROOT_FIVE_QUEUE_GROWTH_20261010T1233.json'
with p.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=out['status'],sha256=sha(p),main=len(new),FL=growth['FLGMM']['terminal_now'],gradient=growth['gradient64']['terminal_now'],Hybrid=growth['Hybrid96']['terminal_now'],remaining=out['remaining620_closed'])))
