"""Adopt the independently checked four non-IID Benign native terminals."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
TAG='root_delta_20261010T003657Z'
source=ROOT/'tmp/celeba_mechanism_valid_C_after56_20261010/ROOT_NATIVE_INCREMENT_REVIEW.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(source)=='369c5d80169f5188604f46565116a184de3b1c84eaf53ac4bc3794279a0e6ee4'
review,actual=read(source),read(BASE/TAG/'ROOT_DELTA_VERIFICATION.json')
expected=[f'minus_C_non-IID_Benign_seed{s}' for s in range(91007,91011)]
assert review['status']=='ROOT_NATIVE_INCREMENT_MEMBERS_OLD256_RECORDS_AND_EXACT_C_DELTA_PASS'
assert review['native_accepted']==review['ledger_accepted_unique_ids']==actual['total_new_strict_and_offserver']==160
assert review['added_n']==4 and review['added_ids']==actual['new_ids']==expected
for name,key in [(TAG+'.tar.gz','archive_sha256'),(TAG+'.tar.gz.receipt.json','receipt_sha256'),
 (TAG+'_offserver_verification.json','offserver_proof_sha256'),('mechanism_inspection_v4_'+TAG+'/inspection.json','inspection_sha256'),
 (TAG+'/verified_ledger.json','ledger_sha256')]:assert sha(BASE/name)==review[key]==actual[key]
assert review['source_root_proof_sha256']==sha(BASE/TAG/'ROOT_DELTA_VERIFICATION.json')
assert review['archive_members_verified']==62 and review['ledger_entries_verified']==25
for key in ('ledger_previous24_entries_exact','receipt_chain_verified','original256_records_exact',
 'original256_raw_json_record_bytes_exact','original256_record_order_preserved','minus_U_complete_100'):assert review[key] is True
assert (review['full_reused'],review['inspection_records'],review['minus_C_accepted'])==(100,260,60)
assert [r['id'] for r in review['new_identity_checks']]==expected
assert all(r['complete_rounds']==70 and r['valid_n']==19867 and r['native_metrics_match_inspection'] for r in review['new_identity_checks'])
target=BASE/TAG/'ROOT_INDEPENDENT_REVIEW.json'
with target.open('xb') as f:f.write(source.read_bytes())
assert sha(target)==sha(source)
print(json.dumps(dict(status='ROOT_NATIVE160_INDEPENDENT_REVIEW_ADOPTED',sha256=sha(target),new_ids=expected,scene_n=10)))
