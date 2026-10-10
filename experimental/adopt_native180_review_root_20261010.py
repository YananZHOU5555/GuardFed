"""Reuse the native increment adopter with actual C80 pins and original nested paths."""
from pathlib import Path
import ast, hashlib, re

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT/'tmp/adopt_native170_review_root_20261010.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == 'cd44d6f8993a0318f2158107e5b7d37b5f174e99f84a3194b330c01f6c5f2396'
old = parent.read_text(encoding='utf8')
bindings = {
    'root_delta_20261010T013518Z': 'root_delta_20261010T023113Z',
    'celeba_mechanism_valid_C_after60_20261010': 'celeba_mechanism_valid_C_after70_20261010',
    'e45971c7f3db346f54594a7eccfc2f41c4bf2a4a7d015fab020a06dc7a52343e': 'fc650143107f70e2017bb58cc864dd4f2cda2c7ac8e3894558cca81c86fae703',
    "f'minus_C_non-IID_F Flip_seed{s}'": "f'minus_C_non-IID_FedSA_seed{s}'",
    'OLD260_RECORDS': 'OLD270_RECORDS',
    "['total_new_strict_and_offserver']==170": "['total_new_strict_and_offserver']==180",
    "(TAG+'.tar.gz','archive_sha256')": "(TAG+'/'+TAG+'.tar.gz','archive_sha256')",
    "(TAG+'.tar.gz.receipt.json','receipt_sha256')": "(TAG+'/'+TAG+'.tar.gz.receipt.json','receipt_sha256')",
    "(TAG+'_offserver_verification.json','offserver_proof_sha256')": "(TAG+'/OFFSERVER_VERIFICATION.json','offserver_proof_sha256')",
    "('mechanism_inspection_v4_'+TAG+'/inspection.json','inspection_sha256')": "(TAG+'/inspection/inspection.json','inspection_sha256')",
    "review['ledger_entries_verified']==26": "review['ledger_entries_verified']==27",
    'ledger_previous25_entries_exact': 'ledger_previous26_entries_exact',
    'original260_records_exact': 'original270_records_exact',
    'original260_raw_json_record_bytes_exact': 'original270_raw_json_record_bytes_exact',
    'original260_record_order_preserved': 'original270_record_order_preserved',
    '==(100,270,70)': '==(100,280,80)',
    'ROOT_NATIVE170_INDEPENDENT_REVIEW_ADOPTED': 'ROOT_NATIVE180_INDEPENDENT_REVIEW_ADOPTED',
}
for before in bindings:
    assert old.count(before) == 1, before
bound = re.sub('|'.join(re.escape(k) for k in sorted(bindings, key=len, reverse=True)),
               lambda m: bindings[m.group()], old)
ast.parse(bound)
exec(compile(bound, str(parent), 'exec'), {'__file__': str(parent), '__name__': '__main__'})
