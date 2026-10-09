"""Adopt reviewed exact C8 source and root operations prior to one-shot invocation."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
base=ROOT/'tmp/celeba_mechanism_valid_C_after12_20261009'
review=ROOT/'tmp/celeba_mechanism_C_after12_root_independent_20261009/ROOT_INDEPENDENT_REVIEW.json'
assert sha(review)=='3b2c936c91a78e6caa39fb007c8d485c6e7870ebfab3bae49831adc1c60c64db'
r=read(review);assert r['source_adoptable'] and r['source_tar_members_verified']==30
package=base/'PACKAGE_SHA256.json';assert sha(package)==r['package_sha256']
pins=read(package)
members=pins.get('members',pins.get('files'))
if isinstance(members,dict):
    for n,v in members.items():assert sha(base/n)==(v['sha256'] if isinstance(v,dict) else v)
else:
    for v in members:assert sha(base/v['path'])==v['sha256']
assert len(members)==39 and len(r['exact_selected_ids'])==8
assert r['old112_records_exact'] and r['Full100_actual900_source_records_exact'] and r['unchanged_bridge_scientific_function_count']==11
ops=ROOT/'tmp/celeba_mechanism_C_after12_root_operations_20261009'
expected={'deploy.py':'a32823cc19d56097339b2e436c5ca705ac1fe507f404c9dcd19829a73d06e919','observe.py':'711be21423a5038f642aae886ddff963d38d5c3fbcea17b177de9be6dd2d39b9'}
for n,h in expected.items():assert sha(ops/n)==h
proof=dict(status='ROOT_EXACT8_SOURCE_AND_ONE_SHOT_OPERATIONS_ADOPTED_FOR_LINUX_PREFLIGHT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_review_sha256=sha(review),package_sha256=sha(package),package_members_verified=39,operations=expected,
    selected_ids=r['exact_selected_ids'],prior112_excluded=True,Full100_reused=True,
    source_review='Reviewed deploy stdin/transport/authority and unchanged installer contracts; bounded original CPU112..119 eight-thread valid-only evaluator.',
    Linux_runtime_preflight_required=True,new_scientific_acceptance=0,new_training=0,test_dispatch=False)
out=ops/'ROOT_SOURCE_ADOPTION.json'
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out))))
