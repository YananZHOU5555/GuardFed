"""Rebind the bounded existing native checker; preserve its scientific checks."""
from pathlib import Path
import ast, difflib, hashlib, json, re
H = Path(__file__).resolve().parent
O = H.with_name('celeba_mechanism_valid_C_after50_20261010')
s = (O/'verify_native_snapshot.py').read_text(encoding='utf-8-sig'); before = s
changes = {
    'native150 successor': 'native156 successor',
    "OLD='root_delta_20261009T233607Z'": "OLD='root_delta_20261010T000754Z'",
    'TOTAL==150+len(IDS)': 'TOTAL==156+len(IDS)',
    '7e04be828e3b6824dbb13e40b0cc8d3a9435f48541a0fafebc2748f259691353': '4dcaa43e2278714684a1cfd4800c165f71d72ebca48c8a1519b3c879f55fbf6c',
    '5ad6f7753271c882b20a8ecc975a3fbdc9228e8aaa894222e63f80f118adb439': '9a56b5533c05727dc726fa013e3fdd9df035ec3be633747fff13477f55d83d44',
    'len(oldids)==250': 'len(oldids)==256',
    'native150_successor_existing_evidence': 'native156_successor_existing_evidence',
    "len(ledger['entries'])==24": "len(ledger['entries'])==25",
    "len(priorledger['entries'])==23": "len(priorledger['entries'])==24",
    'OLD250_RECORDS': 'OLD256_RECORDS',
    'ledger_entries_verified=24': 'ledger_entries_verified=25',
    'ledger_previous23_entries_exact': 'ledger_previous24_entries_exact',
    'original250_': 'original256_',
}
for k in changes: assert k in s, k
s = re.sub('|'.join(re.escape(k) for k in sorted(changes, key=len, reverse=True)), lambda m: changes[m[0]], s)
needle = " IDS=root['new_ids'];TOTAL=root['total_new_strict_and_offserver']\n"
assert s.count(needle) == 1
s = s.replace(needle, needle + " assert IDS==[f'minus_C_non-IID_Benign_seed{i}' for i in range(91007,91011)] and TOTAL==160\n" +
    " assert root['archive_sha256']=='25c0dd3f7f006ab4bb6cb3418e07f96a63add7f36c8eb2a3c2e5b5b12ea4e2d8'\n" +
    " assert root['receipt_sha256']=='8ddfbbd54ce7e0fec6dc62ad08a1256a4644af96cea0dd55d3546e68874fc5de'\n" +
    " assert root['offserver_proof_sha256']=='f7d65a7a17369d0542b5ca85245ad41efe546c6693dc64039f1571834d85542d'\n" +
    " assert root['ledger_sha256']=='ffa58131d986fcdb6a3e580ce3cad5ef221016ce96109ce726ce32a7dd80c093'\n" +
    " assert root['inspection_sha256']=='be2f7e00ef45349bd57050938195c5972c2688dd1b8b2e62706fdce4c6d6c6b3'\n")
ast.parse(s)
with (H/'verify_native_snapshot.py').open('x', encoding='utf-8', newline='\n') as f: f.write(s)
with (H/'NATIVE_CHECKER_SOURCE_DIFF.patch').open('x', encoding='utf-8', newline='\n') as f:
    f.writelines(difflib.unified_diff(before.splitlines(True), s.splitlines(True), fromfile='sealed_C_after50/verify_native_snapshot.py', tofile='C_after56/verify_native_snapshot.py'))
print(json.dumps(dict(status='NATIVE_CHECKER_REBOUND_NOT_EXECUTED', checker_sha256=hashlib.sha256(s.encode()).hexdigest(), predecessor_native=156, expected_new=4)))
