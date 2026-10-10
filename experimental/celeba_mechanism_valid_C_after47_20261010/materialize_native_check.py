"""Prepare a bounded native147 successor check without reading a future snapshot."""
from pathlib import Path
import ast,hashlib,json
H=Path(__file__).resolve().parent;O=H.with_name('celeba_mechanism_valid_C_after40_20261010')
s=(O/'verify_native_snapshot.py').read_text(encoding='utf-8-sig')
def change(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b,1)
change('Bounded local native147 delta verification.', 'Bounded local verification of one externally pinned native147 successor.')
change("TAG='root_delta_20261009T230901Z';OLD='root_delta_20261009T223148Z'", "OLD='root_delta_20261009T230901Z'")
n=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='IDS' for t in n.targets))
change(ast.get_source_segment(s,n),'')
start=s.index(" assert sha(B/TAG/'ROOT_DELTA_VERIFICATION.json')")
end=s.index(' archive=B/',start)
s=s[:start]+''' import argparse,re
 p=argparse.ArgumentParser();p.add_argument('--root-delta',type=Path,required=True);p.add_argument('--root-delta-sha256',required=True);a=p.parse_args()
 root_path=a.root_delta.resolve()
 assert root_path.name=='ROOT_DELTA_VERIFICATION.json' and root_path.parent.parent==B.resolve()
 TAG=root_path.parent.name
 assert re.fullmatch(r'root_delta_\\d{8}T\\d{6}Z',TAG) and TAG!=OLD
 assert re.fullmatch('[0-9a-f]{64}',a.root_delta_sha256) and sha(root_path)==a.root_delta_sha256
 root=read(root_path)
 assert root['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and root['tag']==TAG
 IDS=root['new_ids'];TOTAL=root['total_new_strict_and_offserver']
 assert isinstance(IDS,list) and IDS and len(IDS)==len(set(IDS)) and all(isinstance(i,str) and i.startswith('minus_C_') for i in IDS)
 assert type(TOTAL) is int and TOTAL==147+len(IDS)
'''+s[end:]
for a,b in [
 ("root['previous_ledger_sha256']=='1cc05e3c627653d9b37cc0b4eb6081454fd7aa2d0bb5d5ce587e4a140444336e'", "root['previous_ledger_sha256']=='dc92864bf8d536c0694580cce4676472a0e2388ec99e6eb6fc5957ac501229fd'"),
 ("sha(old_path)=='426397ba48fbc679bcb88b579d53a05487a94ee495565af59a7de1145e87e515'", "sha(old_path)=='bcda76141f61f88c13c177f3cdee9853310001d8dedc74bad3bf9b3f7af3f285'"),
 ('len(rows)==247 and len(oldids)==240','len(rows)==TOTAL+100 and len(inspection[\'records\'])==TOTAL+100 and len(oldids)==247'),
 ("inspection['new_count']==147", "inspection['new_count']==TOTAL"),
 ("r['variant']=='minus_C'])==47", "r['variant']=='minus_C'])==TOTAL-100"),
 ("'native147_existing_evidence'", "'native147_successor_existing_evidence'"),
 ("len(ledger['entries'])==22", "len(ledger['entries'])==23"),
 ("len(priorledger['entries'])==21", "len(priorledger['entries'])==22"),
 ('len(accepted)==147','len(accepted)==TOTAL'),
 ("assert (job['distribution'],job['attack'],job['config']['seed'])==('IID','Sp-DFA',int(identity[-5:]))", "assert identity==f\"minus_C_{job['distribution']}_{job['attack']}_seed{job['config']['seed']}\" and job['config']['seed']==int(identity[-5:])")]:change(a,b)
start=s.index(" target=H/'ROOT_NATIVE_INCREMENT_REVIEW.json'");end=s.index("\nif __name__=='__main__':",start)
s=s[:start]+''' target=H/'ROOT_NATIVE_INCREMENT_REVIEW.json'
 report=dict(status='ROOT_NATIVE_INCREMENT_MEMBERS_OLD247_RECORDS_AND_EXACT_C_DELTA_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tag=TAG,archive_sha256=sha(archive),inspection_sha256=sha(inspection_path),ledger_sha256=sha(ledger_path),receipt_sha256=sha(receipt_path),offserver_proof_sha256=sha(proof_path),source_root_proof_sha256=sha(root_path),previous_inspection_sha256=sha(old_path),previous_ledger_sha256=sha(old_ledger_path),native_accepted=TOTAL,added_n=len(IDS),added_ids=IDS,archive_members_verified=verification['members_verified'],ledger_entries_verified=23,ledger_accepted_unique_ids=TOTAL,ledger_previous22_entries_exact=True,receipt_chain_verified=True,original247_records_exact=True,original247_raw_json_record_bytes_exact=True,original247_record_order_preserved=True,inspection_records=TOTAL+100,full_reused=100,minus_U_complete_100=True,minus_C_accepted=TOTAL-100,new_identity_checks=identities,frozen_repo_source_members_verified=source_members,strict_source_script_sha256=sha(ep),manifest_sha256=sha(manifest_path),full_inventory_sha256=sha(base_path),old_archive_members_rescanned=False,live_data_files_rehashed=False,independent_mean_table_generated=False,three_view_acceptance_changed=False,oldFull_models_repacked=0,new_inference=0,test=False,whole_rebuttal_complete=False,verification_script_sha256=sha(Path(__file__)))
 with target.open('x',encoding='utf-8',newline='\\n') as f:json.dump(report,f,indent=2);f.write('\\n')
 print(json.dumps({'status':report['status'],'root_review_path':str(target),'root_review_sha256':sha(target),'native_accepted':TOTAL,'new_ids':IDS,'members':verification['members_verified']}))
'''+s[end:]
ast.parse(s)
with (H/'verify_native_snapshot.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
with (H/'PREPARED_CHECK_ENTRY.json').open('x',encoding='utf-8') as f:json.dump({'status':'SOURCE_ONLY_WAITING_ACTUAL_NATIVE_DELTA_PINS','checker_sha256':hashlib.sha256(s.encode()).hexdigest(),'prior_native_accepted':147,'prior_records':247,'prior_ledger_entries':22,'new_count':None,'new_ids':None,'new_snapshot_read':False,'CNN':False,'SSH':False},f,indent=2);f.write('\n')
print(json.dumps({'status':'SOURCE_ONLY_WAITING_ACTUAL_NATIVE_DELTA_PINS','checker_sha256':hashlib.sha256(s.encode()).hexdigest()}))
