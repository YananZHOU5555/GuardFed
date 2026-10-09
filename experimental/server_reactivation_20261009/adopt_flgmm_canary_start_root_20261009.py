"""Verify measured startup and protected-resource evidence, without accepting a canary."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
attempt=ROOT/'tmp/celeba_flgmm_fullcoverage_canary_operations_20261009/attempt_20261009T201641646912Z'
snapshot=ROOT/'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/observation_20261009T201720583136Z/SNAPSHOT.json'
start=read(attempt/'START.json');resource=read(attempt/'RESOURCE.json');obs=read(snapshot)
assert start['status']=='START_COMMAND_SUCCEEDED_NOT_GATE_ACCEPTANCE' and start['resource_sha256']==sha(attempt/'RESOURCE.json')
assert start['canary_scope']==7 and not start['fullcoverage_started']
assert resource['main_growth_observed'] and resource['protected_main_healthy'] and len(resource['main_actual_workers'])==8
assert resource['planned_total_cpu_threads']<=resource['cpu_quota_cores'] and resource['free_memory_bytes']>=8*1024**3
assert resource['gpu_recovery_actions']==['None','None'] and min(resource['gpu_free_memory_mib'])>=2048
assert resource['old_flgmm']['stdout'].split()[1]=='EXITED' and resource['sglang']['stdout'].split()[1]=='STOPPED'
assert obs['source_members_match'] and not obs['failure_paths'] and not obs['recent_log_error_matches']
assert obs['canary_service']['stdout'].split()[1]=='RUNNING'
assert 1<=len(obs['processes'])<=2 and all(p['affinity']==[102,103] for p in obs['processes'])
assert not any(r.get('result_present') or r.get('progress') for r in obs['rows'] if r['kind']=='new')
proof=dict(status='ROOT_ACTUAL_SEVEN_CANARY_STARTUP_AND_RESOURCE_VERIFIED_NOT_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    start_receipt_path=(attempt/'START.json').relative_to(ROOT).as_posix(),start_receipt_sha256=sha(attempt/'START.json'),
    resource_path=(attempt/'RESOURCE.json').relative_to(ROOT).as_posix(),resource_sha256=sha(attempt/'RESOURCE.json'),
    snapshot_path=snapshot.relative_to(ROOT).as_posix(),snapshot_sha256=sha(snapshot),observed_utc=obs['utc'],service=obs['canary_service']['stdout'].strip(),
    actual_processes=obs['processes'],main_growth_verified=True,source_members_match=True,canary_scope=7,canaries_accepted=0,
    root_approval_sha256=sha(attempt/'APPROVAL.json'),planned_new70round=96,reused70round=4,formal100_started=False,final_test=False,
    limit='Sequential seven3-round checks are running. Tg20 transition is beyond horizon; no70-round equivalence or scientific acceptance inferred from startup.')
out=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_CANARY_STARTUP.json'
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out))))
