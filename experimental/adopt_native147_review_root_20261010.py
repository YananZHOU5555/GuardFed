"""Adopt the independently executed review of the newly collected seven terminals."""
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
TAG = 'root_delta_20261009T230901Z'
source = ROOT/'tmp/celeba_mechanism_valid_C_after40_20261010/ROOT_NATIVE_INCREMENT_REVIEW.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
assert sha(source) == '5c01316cac5008fa3ec5fa21203fee8f0b7f0f94bd5bb0793bd9f4437a7627d5'
review = read(source)
actual = read(BASE/TAG/'ROOT_DELTA_VERIFICATION.json')
expected = [f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91001, 91008)]
assert review['status'] == 'ROOT_NATIVE147_DELTA_MEMBERS_OLD240_RECORDS_AND_EXACT7_C_SPDFA_PASS'
assert review['native_accepted'] == review['ledger_accepted_unique_ids'] == actual['total_new_strict_and_offserver'] == 147
assert review['added_n'] == 7 and review['added_ids'] == actual['new_ids'] == expected
for name, key in [(TAG+'.tar.gz', 'archive_sha256'), (TAG+'.tar.gz.receipt.json', 'receipt_sha256'),
                  (TAG+'_offserver_verification.json', 'offserver_proof_sha256'),
                  ('mechanism_inspection_v4_'+TAG+'/inspection.json', 'inspection_sha256'),
                  (TAG+'/verified_ledger.json', 'ledger_sha256')]:
    assert sha(BASE/name) == review[key] == actual[key]
assert review['source_root_proof_sha256'] == sha(BASE/TAG/'ROOT_DELTA_VERIFICATION.json')
assert review['archive_members_verified'] == 86 and review['ledger_entries_verified'] == 22
for key in ('ledger_previous21_entries_exact', 'receipt_chain_verified', 'original240_records_exact',
            'original240_raw_json_record_bytes_exact', 'original240_record_order_preserved',
            'minus_U_complete_100', 'minus_C_partial_47'):
    assert review[key] is True
assert review['full_reused'] == 100 and review['inspection_records'] == 247
assert review['minus_C_IID_SpDFA_accepted_seeds'] == list(range(91001, 91008))
assert review['minus_C_IID_SpDFA_complete'] is False
assert [row['id'] for row in review['new_identity_checks']] == expected
assert all(row['complete_rounds'] == 70 and row['valid_n'] == 19867 and row['native_metrics_match_inspection'] for row in review['new_identity_checks'])
target = BASE/TAG/'ROOT_INDEPENDENT_REVIEW.json'
with target.open('xb') as stream:
    stream.write(source.read_bytes())
assert sha(target) == sha(source)
print(json.dumps({'status': 'ROOT_NATIVE147_INDEPENDENT_REVIEW_ADOPTED', 'sha256': sha(target), 'new_ids': expected}))
