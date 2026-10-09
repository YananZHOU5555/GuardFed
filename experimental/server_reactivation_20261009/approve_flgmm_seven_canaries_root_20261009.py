"""Root-reviewed exact-seven launch authorization; no SSH or scientific execution."""
from pathlib import Path
import datetime, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
OPS=ROOT/'tmp/celeba_flgmm_fullcoverage_canary_operations_20261009'
OUT=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
seal='47ad62f6625588b69fc5a4ecf3a3ded5abb586fb5f469612538035be8c1f5665'
assert sha(OPS/'FILES_SHA256.json')==seal
files=read(OPS/'FILES_SHA256.json')['files'];assert len(files)==8
for n,pin in files.items():
    assert sha(OPS/n)==pin['sha256'] and (OPS/n).stat().st_size==pin['bytes']
bound=OUT/'ROOT_BOUND_ADOPTION.json'
assert sha(bound)=='fcecfc0a3582695edfd54c70db38e7dafdd5bf46dcdff8212b9dfc03fc7506fc'
assert read(bound)['archive_members_verified']==144
baseline=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_live_20261009T201240Z.json'
assert sha(baseline)=='7f2e9f8467acdc657b2fe026fb4ee87ba60aabe4138dbea0be72bcc6ff7ad972'
assert read(baseline)['queue_completed']==120 and len(read(baseline)['active'])==8 and read(baseline)['failed']==[]
assert read(OPS/'SELF_CHECK.json')['check_count']==25
review=dict(status='ROOT_SOURCE_REVIEW_PASS_EXACT_SEVEN_CANARY_LAUNCH',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    helper_seal_sha256=seal,verified_helper_members=8,bound_adoption_sha256=sha(bound),baseline_snapshot_sha256=sha(baseline),
    source_review='Read launch.py, remote_launch.py and original main_health.py. Existing bound run_canaries.py only; no numerical source modification.',
    runtime_guards='Guide/source/data and empty namespace, no duplicate stage producer, old FL EXITED, sglang STOPPED, CPU102/103 thread affinity, actual cgroup quota/RAM/OOM, two GPU free memory/recovery, fresh resource and measured main progress required remotely before start.',
    service_scope='Only guardfed_celeba_flgmm_fullcoverage_canary; autostart/autorestart false; no blind retry.',
    formal100_started=False,final_test=False,scientific_canaries_accepted=0)
save(OUT/'ROOT_CANARY_LAUNCH_SOURCE_REVIEW.json',review)
approval=read(OPS/'APPROVAL_TEMPLATE.json');approval.update(status='ROOT_AUTHORIZED_SEVEN_FLGMM_CANARIES',helper_seal_sha256=seal,baseline_snapshot_sha256=sha(baseline))
save(OUT/'ROOT_SEVEN_CANARY_APPROVAL.json',approval)
print(json.dumps(dict(approval=str(OUT/'ROOT_SEVEN_CANARY_APPROVAL.json'),approval_sha256=sha(OUT/'ROOT_SEVEN_CANARY_APPROVAL.json'),review_sha256=sha(OUT/'ROOT_CANARY_LAUNCH_SOURCE_REVIEW.json'))))
