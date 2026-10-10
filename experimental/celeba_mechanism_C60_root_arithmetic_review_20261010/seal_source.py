"""Freeze this independent source; no actual C60 evidence is consumed."""
from pathlib import Path
import ast, hashlib, json
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
check=json.loads((H/'SOURCE_CHECK.json').read_bytes())
assert check['actual_handoff_read'] is False and check['actual_snapshot_read'] is False
assert not (H/'ROOT_ARITHMETIC_REVIEW.json').exists()
for p in H.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
handoff=dict(status='PREPARED_INDEPENDENT_C60_READER_WAITING_ACTUAL_HANDOFF_AND_DELIVERY_SHA',
    review_entry='review.py',review_sha256=sha(H/'review.py'),source_connections_sha256=sha(H/'source_connections.py'),
    actual_handoff_sha256=None,actual_delivery_seal_sha256=None,actual_C4_root_adoption_sha256=None,
    contract='tmp/adopt_C60_table_root_20261010.py:review_guard',source_check_sha256=sha(H/'SOURCE_CHECK.json'),
    actual_arithmetic_review=False,adoption_performed=False,canonical_modified=False,new_CNN=0,new_training=0,new_Full_inference=0,test=False)
with (H/'PREPARED_HANDOFF.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(handoff,f,indent=2);f.write('\n')
files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(H.iterdir()) if p.is_file() and p.name!='SOURCE_FILES_SHA256.json'}
with (H/'SOURCE_FILES_SHA256.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(dict(status='SOURCE_ONLY_NO_ACTUAL_ARITHMETIC_PASS',files=files),f,indent=2);f.write('\n')
for name,pin in files.items():assert sha(H/name)==pin['sha256']
print(json.dumps(dict(status=handoff['status'],source_seal_sha256=sha(H/'SOURCE_FILES_SHA256.json'),members=len(files),review_sha256=sha(H/'review.py'))))
