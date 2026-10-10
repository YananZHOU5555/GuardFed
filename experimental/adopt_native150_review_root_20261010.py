"""Adopt the independently executed exact-three native increment review."""
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
TAG = 'root_delta_20261009T233607Z'
source = ROOT/'tmp/celeba_mechanism_valid_C_after47_20261010/ROOT_NATIVE_INCREMENT_REVIEW.json'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_bytes())
assert sha(source) == 'c0e78899ce232b81c7c2849ed9d05283c3c740bc962176074ad7281c72c24f4a'
review, actual = read(source), read(BASE/TAG/'ROOT_DELTA_VERIFICATION.json')
expected = [f'minus_C_IID_Sp-DFA_seed{seed}' for seed in (91008, 91009, 91010)]
assert review['status'] == 'ROOT_NATIVE_INCREMENT_MEMBERS_OLD247_RECORDS_AND_EXACT_C_DELTA_PASS'
assert review['native_accepted'] == review['ledger_accepted_unique_ids'] == actual['total_new_strict_and_offserver'] == 150
assert review['added_n'] == 3 and review['added_ids'] == actual['new_ids'] == expected
for name, key in [(TAG+'.tar.gz', 'archive_sha256'), (TAG+'.tar.gz.receipt.json', 'receipt_sha256'),
                  (TAG+'_offserver_verification.json', 'offserver_proof_sha256'),
                  ('mechanism_inspection_v4_'+TAG+'/inspection.json', 'inspection_sha256'),
                  (TAG+'/verified_ledger.json', 'ledger_sha256')]:
    assert sha(BASE/name) == review[key] == actual[key]
assert review['source_root_proof_sha256'] == sha(BASE/TAG/'ROOT_DELTA_VERIFICATION.json')
assert review['archive_members_verified'] == 54 and review['ledger_entries_verified'] == 23
for key in ('ledger_previous22_entries_exact', 'receipt_chain_verified', 'original247_records_exact',
            'original247_raw_json_record_bytes_exact', 'original247_record_order_preserved', 'minus_U_complete_100'):
    assert review[key] is True
assert review['full_reused'] == 100 and review['inspection_records'] == 250 and review['minus_C_accepted'] == 50
assert [row['id'] for row in review['new_identity_checks']] == expected
assert all(row['complete_rounds'] == 70 and row['valid_n'] == 19867 and row['native_metrics_match_inspection'] for row in review['new_identity_checks'])
target = BASE/TAG/'ROOT_INDEPENDENT_REVIEW.json'
with target.open('xb') as stream:
    stream.write(source.read_bytes())
assert sha(target) == sha(source)
print(json.dumps({'status': 'ROOT_NATIVE150_INDEPENDENT_REVIEW_ADOPTED', 'sha256': sha(target), 'new_ids': expected}))
