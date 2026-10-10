"""Bind actual metadata and independent launcher review to seven fixed canaries."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
OPS=ROOT/'tmp/celeba_hybrid_fullcoverage_canary_operations_20261010/v3'
REVIEW=ROOT/'tmp/celeba_hybrid_fullcoverage_canary_v3_independent_review_20261010/REVIEW.json'
BASELINE=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_live_20261010T085659Z.json'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')

def main():
    assert sha(OPS/'FILES_SHA256.json')=='04dbe344603a5243dbfa6a00bc89c31ea5208792d88e9371ad7f9873775d17ab'
    for name,pin in read(OPS/'FILES_SHA256.json')['files'].items():
        assert sha(OPS/name)==pin['sha256'] and (OPS/name).stat().st_size==pin['bytes']
    for name,pin in read(REVIEW.parent/'FILES_SHA256.json')['files'].items():
        assert sha(REVIEW.parent/name)==pin['sha256']
    review=read(REVIEW)
    assert 'PASS' in review['status'] and not review.get('confirmed_blockers',[])
    assert '04dbe344603a5243dbfa6a00bc89c31ea5208792d88e9371ad7f9873775d17ab' in json.dumps(review)
    bound=read(HERE/'ROOT_BOUND_ADOPTION.json');offserver=HERE/'BOUND_TRANSFER_VERIFICATION.json'
    assert bound['status']=='ROOT_HYBRID100_BOUND_METADATA_ADOPTED' and (bound['new'],bound['reused'],bound['canaries'])==(96,4,7)
    assert bound['bound_offserver_sha256']==sha(offserver)
    assert bound['execution_authorized'] is False and bound['final_test'] is False
    assert sha(BASELINE)=='e5bed5789a8336249c67230f3464c3f7416193bbe37b8a09c80ba14ef8de216e'
    baseline=read(BASELINE)
    assert baseline['queue_completed']==230 and len(baseline['active'])==8 and not baseline['failed'] and not baseline['recent_active_log_errors']
    assert 0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(baseline['checked_utc'])).total_seconds()<3600
    source_review=dict(status='ROOT_LAUNCH_SOURCE_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        helper_source_seal_sha256=sha(OPS/'FILES_SHA256.json'),independent_review_sha256=sha(REVIEW),
        root_bound_sha256=sha(HERE/'ROOT_BOUND_ADOPTION.json'),scientific_source_changes=0,
        scope='Exact seven sequential3round image gates, CPU104/GPU0; no96 runner; no automatic retry.',
        diagnostic_limit='Inherited fail-stop main-health repr/traceback may omit exception resource_guard_inputs; launch RESOURCE includes actual observations only if resource check returns. Not a scientific guard relaxation.',
        source_only_review=True,canaries_accepted=0,final_test=False)
    save(HERE/'ROOT_CANARY_LAUNCH_SOURCE_REVIEW.json',source_review)
    approval=dict(status='ROOT_AUTHORIZED_SEVEN_HYBRID_CANARIES',scope='seven_same_horizon_3round_canaries',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),service='guardfed_celeba_hybrid_fullcoverage_canary',
        package_sha256=bound['package_sha256'],helper_seal_sha256=sha(OPS/'FILES_SHA256.json'),
        implementation_source_seal_sha256=bound['implementation_source_seal_sha256'],
        bound_offserver_sha256=sha(offserver),bound_root_review_sha256=sha(HERE/'ROOT_BOUND_ADOPTION.json'),
        baseline_snapshot_sha256=sha(BASELINE),cpus=[104],cpu_threads=1,formal100_started=False,final_test=False,
        independent_launcher_review_sha256=sha(REVIEW),
        authorization='User complete-rebuttal experimental authorization; root-selected fixed recipe from original32, seven pipeline gates only.')
    save(HERE/'ROOT_SEVEN_CANARY_APPROVAL.json',approval)
    print(json.dumps(dict(approval_sha256=sha(HERE/'ROOT_SEVEN_CANARY_APPROVAL.json'),review_sha256=sha(REVIEW),bound_root_sha256=sha(HERE/'ROOT_BOUND_ADOPTION.json'),baseline_sha256=sha(BASELINE))))
if __name__=='__main__':main()
