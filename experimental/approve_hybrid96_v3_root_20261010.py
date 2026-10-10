"""One frozen validation approval after actual seven-canary root adoption."""
from pathlib import Path
import argparse,datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
OPS=ROOT/'tmp/celeba_hybrid_fullcoverage_launch_operations_20261010/v3'
REVIEW=ROOT/'tmp/celeba_hybrid96_launch_source_review_20261010/REVIEW.json'
BASELINE=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_live_20261010T093819Z.json'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--independent-review-sha256',required=True);a=parser.parse_args()
    assert sha(REVIEW)==a.independent_review_sha256
    for name,pin in read(REVIEW.parent/'FILES_SHA256.json')['files'].items():assert sha(REVIEW.parent/name)==pin['sha256']
    review=read(REVIEW);assert 'PASS' in review['status'] and not review.get('blockers',[])
    assert sha(OPS/'FILES_SHA256.json')=='136a44d634d049418ff81d867c706298797844ce9b1bd5936fb36c6d11174509'
    assert sha(OPS/'FILES_SHA256.json') in json.dumps(review)
    for name,pin in read(OPS/'FILES_SHA256.json')['files'].items():assert sha(OPS/name)==pin['sha256'] and (OPS/name).stat().st_size==pin['bytes']
    closure_path=HERE/'ROOT_SEVEN_CANARY_CLOSURE.json';assert sha(closure_path)=='eda03511ec037be8d06401886355b3c49929de3523bed272ca8df94e5799b05a'
    closure=read(closure_path);offserver=Path(closure['offserver_path']);gate=Path(closure['gate_path'])
    assert closure['status']=='ROOT_SEVEN_HYBRID_CANARIES_OFFSERVER_ADOPTED'
    assert sha(offserver)==closure['offserver_sha256'] and sha(gate)==closure['gate_sha256']
    assert closure['canary_authorization_sha256']=='7de75426971da9509ae91b81a081977c5aa4160b6d12e706b0b3cca55eb91882'
    assert (closure['accepted_new_canaries'],closure['same_horizon_pairs'],closure['total_canary_runs'],closure['rounds'],closure['formal_table_samples'])==(7,2,7,3,0)
    assert sha(BASELINE)=='17b0b754308bcf4089e5a462e4973418e9c56c970c2e12ce4622a33c4c28bda8'
    baseline=read(BASELINE)
    assert baseline['queue_completed']==236 and len(baseline['active'])==8 and not baseline['failed'] and not baseline['recent_active_log_errors']
    assert 0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(baseline['checked_utc'])).total_seconds()<3600
    bound=read(HERE/'ROOT_BOUND_ADOPTION.json')
    assert bound['package_sha256']==closure['package_sha256'] and (bound['new'],bound['reused'],bound['canaries'])==(96,4,7)
    source_review=dict(status='ROOT_HYBRID96_LAUNCH_SOURCE_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        helper_source_seal_sha256=sha(OPS/'FILES_SHA256.json'),independent_review_sha256=sha(REVIEW),
        root_closure_sha256=sha(closure_path),scope='96 frozen new70-round validation-only jobs,4 exact reuses; CPU104/GPU0/one worker/one compute thread; no automatic retry.',
        findings='Root read full launch/remote sources: real7 gate inventory and original4 records are rechecked; released original lock retained; source/data identities, resource freshness and protected-main growth required; old canary authorization archived before atomic replacement. Partial/failure evidence refuses restart.',
        scientific_source_changes=0,scientific70_accepted=0,source_review_is_execution=False,final_test=False)
    save(HERE/'ROOT_HYBRID96_LAUNCH_SOURCE_REVIEW.json',source_review)
    approval=read(OPS/'APPROVAL_TEMPLATE.json')['approval_fields']
    approval.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),package_sha256=bound['package_sha256'],
        helper_seal_sha256=sha(OPS/'FILES_SHA256.json'),implementation_source_seal_sha256=bound['implementation_source_seal_sha256'],
        gate_acceptance_sha256=sha(gate),gate_offserver_sha256=sha(offserver),root_closure_sha256=sha(closure_path),
        baseline_snapshot_sha256=sha(BASELINE),canary_authorization_sha256=closure['canary_authorization_sha256'],
        independent_launcher_review_sha256=sha(REVIEW),root_launcher_review_sha256=sha(HERE/'ROOT_HYBRID96_LAUNCH_SOURCE_REVIEW.json'),
        authorization='Existing user full-rebuttal authorization; frozen original32 validation winner and complete17-method coverage. No new tuning/seed selection/test or scientific claim authorized by this receipt.')
    assert not any(v is None for v in approval.values())
    save(HERE/'ROOT_HYBRID96_APPROVAL.json',approval)
    print(json.dumps(dict(approval_sha256=sha(HERE/'ROOT_HYBRID96_APPROVAL.json'),root_source_review_sha256=sha(HERE/'ROOT_HYBRID96_LAUNCH_SOURCE_REVIEW.json'),closure_sha256=sha(closure_path),baseline_sha256=sha(BASELINE))))
if __name__=='__main__':main()
