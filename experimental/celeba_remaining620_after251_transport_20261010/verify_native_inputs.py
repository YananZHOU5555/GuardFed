"""Local pin/ID gate; import performs no execution or server operation."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def verify_native_inputs():
 e=read(H/'EXECUTION_INPUTS.json');p=read(H/'PREPARED.json')
 assert sha(H/'SOURCE_FILES_SHA256.json')==e['source_seal_sha256']
 for name,pin in read(H/'SOURCE_FILES_SHA256.json')['files'].items():
  assert sha(H/name)==pin['sha256'] and (H/name).stat().st_size==pin['bytes']
 assert e['native_root_verified'] is True and e['exact_candidate_ids']==p['candidate_ids']
 for field in ['native_root','native_inspection','native_ledger']:
  assert sha(e[field+'_path'])==e[field+'_sha256']
 for field in ['prior_root','prior_index']:assert sha(p[field+'_path'])==p[field+'_sha256']
 proof=read(e['native_root_path']);inspection=read(e['native_inspection_path']);ledger=read(e['native_ledger_path'])
 assert proof['root_adopted'] is True and proof['inspection_sha256']==e['native_inspection_sha256'] and proof['ledger_sha256']==e['native_ledger_sha256']
 assert proof['total_new_strict_and_offserver']>=260
 assert set(p['candidate_ids'])<=set(proof['new_ids'])<=set(inspection['accepted_new_ids'])
 assert set(p['candidate_ids']).isdisjoint(read(p['prior_index_path'])['all_ids'])
 byid={r['id']:r for r in inspection['records']}
 assert len(byid)==len(inspection['records']) and all(i in byid for i in p['candidate_ids'])
 assert ledger['entries'] and p['prior_accepted']==251
 return e
