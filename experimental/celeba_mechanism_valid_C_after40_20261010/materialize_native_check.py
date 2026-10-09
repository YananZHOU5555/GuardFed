"""Exact source-only rebind of the previously accepted native increment checker."""
from pathlib import Path
import ast, hashlib, json
H=Path(__file__).resolve().parent
OLD=H.with_name('celeba_mechanism_valid_C_after36_20261010')
source=(OLD/'verify_native_snapshot.py').read_text(encoding='utf-8-sig')
selected=[f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91001,91008)]
def replace(a,b):
 global source
 assert source.count(a)==1,(a,source.count(a))
 source=source.replace(a,b,1)
replace('Bounded local native140 delta verification.', 'Bounded local native147 delta verification.')
replace("TAG='root_delta_20261009T223148Z';OLD='root_delta_20261009T214755Z'", "TAG='root_delta_20261009T230901Z';OLD='root_delta_20261009T223148Z'")
node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='IDS' for t in n.targets))
replace(ast.get_source_segment(source,node),'IDS='+repr(selected))
replace("root=read(B/TAG/'ROOT_DELTA_VERIFICATION.json')", "assert sha(B/TAG/'ROOT_DELTA_VERIFICATION.json')=='b8b5addc893455142ef1ebefd0e2dac8203bc7c18ad7ce09bff966aa8d20116d'\nroot=read(B/TAG/'ROOT_DELTA_VERIFICATION.json')\nassert root['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and root['tag']==TAG")
for a,b in [
 ("root['total_new_strict_and_offserver']==140", "root['total_new_strict_and_offserver']==147"),
 ('74426180abdf1fccb40af004df06ecae2fc1c6316166ffce63c44d160c077c2a','da22ea858d18c047eb2e9ea108a46bafab19328b7672aed51dc921e703d03d77'),
 ("root['ledger_sha256']=='1cc05e3c627653d9b37cc0b4eb6081454fd7aa2d0bb5d5ce587e4a140444336e'", "root['ledger_sha256']=='dc92864bf8d536c0694580cce4676472a0e2388ec99e6eb6fc5957ac501229fd'"),
 ('42c3b021b27ba53eedb2e2f0d684eab9a0c8aa5af8b3b916591cd297399a3794','1cc05e3c627653d9b37cc0b4eb6081454fd7aa2d0bb5d5ce587e4a140444336e'),
 ('cc541f4e628f70888346684ae43b3ae949ff3bd03af967fa80c7c3f73a98d175','426397ba48fbc679bcb88b579d53a05487a94ee495565af59a7de1145e87e515'),
 ('len(rows)==240 and len(oldids)==236','len(rows)==247 and len(oldids)==240'),
 ("inspection['new_count']==140", "inspection['new_count']==147"),
 ("r['variant']=='minus_C'])==40", "r['variant']=='minus_C'])==47"),
 ("'native140_existing_evidence'", "'native147_existing_evidence'"),
 ("verification['members_verified']==proof['members_verified']==62", "verification['members_verified']==proof['members_verified']"),
 ("len(ledger['entries'])==21", "len(ledger['entries'])==22"),
 ("len(priorledger['entries'])==20", "len(priorledger['entries'])==21"),
 ('len(accepted)==140','len(accepted)==147'),
 ("'FedSA' if identity.startswith('minus_C_IID_FedSA_') else 'S-DFA'", "'Sp-DFA'")]:replace(a,b)
start=source.index("target=H/'ROOT_NATIVE140_INDEPENDENT_REVIEW.json'")
source=source[:start]+'''target=H/'ROOT_NATIVE_INCREMENT_REVIEW.json'
report=dict(status='ROOT_NATIVE147_DELTA_MEMBERS_OLD240_RECORDS_AND_EXACT7_C_SPDFA_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_sha256=sha(archive),inspection_sha256=sha(inspection_path),ledger_sha256=sha(ledger_path),receipt_sha256=sha(receipt_path),offserver_proof_sha256=sha(proof_path),source_root_proof_sha256=sha(B/TAG/'ROOT_DELTA_VERIFICATION.json'),previous_inspection_sha256=sha(old_path),previous_ledger_sha256=sha(old_ledger_path),native_accepted=147,added_n=7,added_ids=IDS,archive_members_verified=verification['members_verified'],ledger_entries_verified=22,ledger_accepted_unique_ids=147,ledger_previous21_entries_exact=True,receipt_chain_verified=True,original240_records_exact=True,original240_raw_json_record_bytes_exact=True,original240_record_order_preserved=True,inspection_records=247,full_reused=100,minus_U_complete_100=True,minus_C_partial_47=True,minus_C_IID_SpDFA_accepted_seeds=[91001,91002,91003,91004,91005,91006,91007],minus_C_IID_SpDFA_complete=False,new_identity_checks=identities,frozen_repo_source_members_verified=source_members,strict_source_script_sha256=sha(ep),manifest_sha256=sha(manifest_path),full_inventory_sha256=sha(base_path),old_archive_members_rescanned=False,live_data_files_rehashed=False,independent_mean_table_generated=False,three_view_acceptance_changed=False,oldFull_models_repacked=0,new_inference=0,test=False,whole_rebuttal_complete=False,verification_script_sha256=sha(Path(__file__)))
with target.open('x',encoding='utf-8',newline='\\n') as f:json.dump(report,f,indent=2);f.write('\\n')
print(json.dumps({'status':report['status'],'root_review_path':str(target),'root_review_sha256':sha(target),'native_accepted':147,'new_ids':IDS,'members':verification['members_verified']}))
'''
prefix,body=source.split("assert sha(B/TAG/'ROOT_DELTA_VERIFICATION.json')",1)
body="assert sha(B/TAG/'ROOT_DELTA_VERIFICATION.json')"+body
source=prefix+'def main():\n'+''.join(' '+line+'\n' for line in body.splitlines())+'''\nif __name__=='__main__':
 try:main()
 except BaseException as exc:
  import traceback
  p=H/'NATIVE_INCREMENT_REVIEW_FAILURE.json'
  if not p.exists():
   with p.open('x',encoding='utf-8') as f:json.dump({'status':'FAILED_NOT_ACCEPTED','error':repr(exc),'traceback':traceback.format_exc(),'checker_sha256':sha(Path(__file__)),'new_inference':0},f,indent=2)
  raise
'''
ast.parse(source)
with (H/'verify_native_snapshot.py').open('x',encoding='utf-8',newline='\n') as f:f.write(source)
print(json.dumps({'checker':str(H/'verify_native_snapshot.py'),'sha256':hashlib.sha256(source.encode()).hexdigest(),'selected':selected,'CNN':False}))
