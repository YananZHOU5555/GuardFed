"""Root-reviewed authorization for frozen96-new coverage after actual seven-run closure."""
from pathlib import Path
import argparse,datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OPS=ROOT/'tmp/celeba_flgmm_fullcoverage_launch_operations_20261009'
BOUND=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
parser=argparse.ArgumentParser();parser.add_argument('--closure-sha256',required=True);parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--baseline-sha256',required=True);a=parser.parse_args()
seal='215531d2dcddaca80b82fd8f8683c39d869d077d702a28ecd47b3f94be883b08'
assert sha(OPS/'FILES_SHA256.json')==seal
files=read(OPS/'FILES_SHA256.json')['files'];assert len(files)==10
for n,pin in files.items():assert sha(OPS/n)==pin['sha256'] and (OPS/n).stat().st_size==pin['bytes']
assert (OPS/'main_health.py').read_bytes()==(ROOT/'tmp/celeba_flgmm_fullcoverage_canary_operations_20261009/main_health.py').read_bytes()
closure=BOUND/'ROOT_SEVEN_CANARY_CLOSURE.json';assert sha(closure)==a.closure_sha256
c=read(closure);proof=ROOT/c['offserver_path'];p=read(proof)
assert c['status']=='ROOT_SEVEN_CANARY_CLOSURE_ADOPTED' and sha(proof)==c['offserver_sha256']
assert p['status']=='PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON' and p['receipt_sha256']==c['receipt_sha256'] and p['archive_sha256']==c['archive_sha256']
assert (c['accepted_new_canaries'],c['same_horizon_pairs'],c['total_canary_runs'],c['rounds'],c['formal_table_samples'])==(5,2,7,3,0)
assert sha(a.baseline)==a.baseline_sha256 and not read(a.baseline)['failed'] and len(read(a.baseline)['active'])==8
approval=read(OPS/'APPROVAL_TEMPLATE.json')
approval.update(status='ROOT_AUTHORIZED_FLGMM96_VALID_ONLY',helper_seal_sha256=seal,gate_offserver_sha256=sha(proof),
    root_closure_sha256=sha(closure),gate_acceptance_sha256=c['gate_sha256'],canary_authorization_sha256=c['canary_authorization_sha256'],baseline_snapshot_sha256=a.baseline_sha256,
    reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_scientific_source_review='Existing immutable worker/recipe/70-round valid-only manifest;10 launch helper members reviewed. Original gate and four reuse references rechecked before resource collection; preserved canary authorization atomically replaced, no other scientific source/config modified.')
out=BOUND/'ROOT_FLGMM96_COVERAGE_APPROVAL.json'
with out.open('x',encoding='utf8') as f:json.dump(approval,f,indent=2);f.write('\n')
print(json.dumps(dict(approval=str(out),approval_sha256=sha(out),closure_sha256=sha(closure),offserver_path=str(proof),offserver_sha256=sha(proof),baseline_path=str(a.baseline),baseline_sha256=a.baseline_sha256)))
