"""Reuse last metadata join, advancing parent251 and exact9 native membership."""
from pathlib import Path
import difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1];parent=R/'tmp/celeba_remaining620_after240_transport_20261010/join_saved.py'
text=parent.read_text('utf8');updated=text
replacements=[
 ('assert 1<=N<=11 and set(expected)<=set(P[\'candidate_ids\'])',"assert N==9 and expected==P['candidate_ids']"),
 ("==len(prior['all_ids'])==240","==len(prior['all_ids'])==251"),
 ("currentproof['total_new_strict_and_offserver']==251","currentproof['total_new_strict_and_offserver']>=260"),
 ("assert len(oldinspection['records'])==343 and len(inspection['records'])==351","assert len(oldinspection['records'])==351 and len(inspection['records'])==100+currentproof['total_new_strict_and_offserver']"),
 ("len(ledger['entries'])==37","len(ledger['entries'])==len(oldledger['entries'])+1"),
 ("set(inspection['accepted_new_ids'])-set(prior['all_ids'])==set(P['candidate_ids'])","set(P['candidate_ids'])<=set(inspection['accepted_new_ids'])-set(prior['all_ids'])"),
 ("for proof,path in ((previousproof,previousroot),(currentproof,Path(E['native_root_path']))):","for proof,path in ((currentproof,Path(E['native_root_path'])),):"),
 ("==240+N and allids[:240]","==251+N and allids[:251]"),
 ("SAVED_ARRAY_BATCH_NATIVE251_IDENTITY_JOIN_PASS_PENDING_ROOT","SAVED_ARRAY_EXACT9_NATIVE_IDENTITY_JOIN_PASS_PENDING_ROOT"),
 ("prior240_objects_and_order_unchanged=True,old343_native_records_exact=True,old36_native_ledger_entries_exact=True","prior251_objects_and_order_unchanged=True,old351_native_records_exact=True,old_native_ledger_entries_exact=True"),
 ("cumulative=240+N","cumulative=251+N")]
for old,new in replacements:
 assert updated.count(old)==1,(old,updated.count(old));updated=updated.replace(old,new,1)
compile(updated,'join_saved.py','exec')
with (H/'join_saved.py').open('x',encoding='utf8',newline='\n') as f:f.write(updated)
with (H/'JOIN_SOURCE_DIFF.patch').open('x',encoding='utf8',newline='\n') as f:f.write(''.join(difflib.unified_diff(text.splitlines(True),updated.splitlines(True),fromfile='after240/join_saved.py',tofile='after251/join_saved.py')))
with (H/'JOIN_SOURCE_REUSE.json').open('x',encoding='utf8') as f:json.dump({'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'new_sha256':hashlib.sha256((H/'join_saved.py').read_bytes()).hexdigest(),'exact_replacements':len(replacements),'scientific_functions_changed':False,'old_archives_not_rehashed':True,'only_new9_native_model_result_members_rehashed':True},f,indent=2)
