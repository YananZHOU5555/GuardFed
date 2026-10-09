"""Bounded local native128 delta verification. No images, science or remote calls."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,sys,tarfile
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
B=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
TAG='root_delta_20261009T212311Z';OLD='root_delta_20261009T204215Z'
IDS=[f'minus_C_IID_FedSA_seed{s}' for s in (91002,91004,91007)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
root=read(B/TAG/'ROOT_DELTA_VERIFICATION.json')
assert root['total_new_strict_and_offserver']==128 and root['new_ids']==IDS
assert root['archive_sha256']=='6635f82fe688f2cb88d419a03d915ec6a3a31e7d02f66396205e893a39a1ecc0'
assert root['ledger_sha256']=='f9e3aac3667fbf706c6e08d4d5c02818209820753b85473b75162435024d4212'
archive=B/(TAG+'.tar.gz');receipt_path=B/(TAG+'.tar.gz.receipt.json');proof_path=B/(TAG+'_offserver_verification.json')
inspection_path=B/('mechanism_inspection_v4_'+TAG)/'inspection.json';old_path=B/('mechanism_inspection_v4_'+OLD)/'inspection.json'
ledger_path=B/TAG/'verified_ledger.json';old_ledger_path=B/OLD/'verified_ledger.json'
for p,k in [(archive,'archive_sha256'),(receipt_path,'receipt_sha256'),(proof_path,'offserver_proof_sha256'),(inspection_path,'inspection_sha256'),(ledger_path,'ledger_sha256')]:assert sha(p)==root[k]
assert sha(old_ledger_path)==root['previous_ledger_sha256']=='cbd6693b86ba36e8b88f41816abc1a12d05e3be52fb9deefb9c1b4e84dbfc315'
assert sha(old_path)=='7f8702ee079eb4e3ea534745e8f98b29cb43613d5383808629b0bec50ccbcdab'
inspection,old=read(inspection_path),read(old_path)
rows={r['id']:r for r in inspection['records']};oldids={r['id'] for r in old['records']}
assert len(rows)==228 and len(oldids)==225 and set(rows)-oldids==set(IDS)
assert [r for r in inspection['records'] if r['id'] in oldids]==old['records']
def raw_records(p):
 text=p.read_text(encoding='utf-8-sig');pos=text.index('[',text.index('"records"'));pos+=1;out=[];decoder=json.JSONDecoder()
 while True:
  while text[pos].isspace() or text[pos]==',':pos+=1
  if text[pos]==']':return out
  value,end=decoder.raw_decode(text,pos);out.append((value['id'],text[pos:end]));pos=end
assert [(i,v) for i,v in raw_records(inspection_path) if i in oldids]==raw_records(old_path)
assert inspection['new_count']==128 and inspection['reused_count']==100 and not inspection['invalid']
assert len([r for r in rows.values() if r['variant']=='minus_U'])==100 and len([r for r in rows.values() if r['variant']=='minus_C'])==28
manifest_path=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/manifest.json';manifest=read(manifest_path)
base_path=R/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json';baseline=read(base_path)
assert sha(manifest_path)==inspection['manifest_sha256']=='498ed0e033ef6eb5532286820ec987af3b44ca6251d8a84c42ce9c3e5ff5ab2d'
assert sha(base_path)==inspection['full_inventory_sha256']=='3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
ep=R/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
assert sha(ep)==inspection['source_script_sha256']=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
spec=importlib.util.spec_from_file_location('native128_existing_evidence',ep);ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
receipt,proof=read(receipt_path),read(proof_path);verification=ev.verify_archive(archive,receipt)
assert verification['members_verified']==proof['members_verified']==54 and proof['pass'] and proof['different_host_observed']
assert receipt['accepted_new_ids']==proof['accepted_new_ids']==IDS and not receipt['failure_identities'] and receipt['reused_full_weights_repacked']==0
ledger,priorledger=read(ledger_path),read(old_ledger_path)
assert len(ledger['entries'])==19 and ledger['entries'][:-1]==priorledger['entries'] and len(priorledger['entries'])==18
previous=None;accepted=set()
for entry in ledger['entries']:
 p=B/Path(entry['receipt']).name;assert sha(p)==entry['receipt_sha256'];rc=read(p)
 assert rc['previous_receipt_sha256']==previous and rc['manifest_sha256']==sha(manifest_path)
 assert not accepted.intersection(rc['accepted_new_ids']);accepted.update(rc['accepted_new_ids']);previous=entry['receipt_sha256']
assert accepted=={r['id'] for r in rows.values() if r['role']=='new'} and len(accepted)==128
assert previous==sha(receipt_path)
full={(r['distribution'],r['attack'],r['seed']):r for r in baseline['records'] if r['method']=='GuardFed-AD2+'};entries={r['id']:r for r in manifest['jobs']}
identities=[];source_members=0
with tarfile.open(archive) as t:
 members=json.load(t.extractfile('backup_inventory.json'))['members']
 assert members['sourcefreeze/manifest.json']['sha256']==sha(manifest_path) and members['sourcefreeze/evidence_v4.py']['sha256']==sha(ep)
 assert members['sourcefreeze/inspection.json']['sha256']==sha(inspection_path)
 for rel,pin in manifest['source_hashes'].items():
  member='sourcefreeze/repo/'+rel
  if member in members:assert members[member]['sha256']==pin;source_members+=1
 for identity in IDS:
  job=json.load(t.extractfile('jobs/'+identity+'.json'));result=json.load(t.extractfile('runs/'+identity+'/result.json'));entry=entries[identity];row=rows[identity]
  control=full[job['distribution'],job['attack'],job['config']['seed']]
  ev.terminal_checks(result,job,manifest,job['variant']);ev.partition_identity(result,control)
  assert job['id']==identity and job['variant']=='minus_C' and job['config']['ablation_component']=='C'
  assert (job['distribution'],job['attack'],job['config']['seed'])==('IID','FedSA',int(identity[-5:]))
  assert job['source_hashes']==manifest['source_hashes'] and job['adapter_hashes']==manifest['adapter_hashes'] and job['protocol_sha256']==manifest['protocol_sha256']
  assert result['config']==job['config'] and {k:result['revision_job'][k] for k in job}==job
  assert result['revision_job']['torch_version']=='2.11.0+cu128'
  assert members['jobs/'+identity+'.json']['sha256']==entry['job_sha256']==row['files'][entry['job']]
  for kind,filename in [('checkpoint','model.pt'),('result','result.json')]:assert members['runs/'+identity+'/'+filename]['sha256']==row['files'][entry['output']+'/'+filename]
  assert row['checkpoint_sha256']==result['revision_job']['checkpoint_sha256']==members['runs/'+identity+'/model.pt']['sha256']
  assert row['accuracy_pct']==100*result['metrics']['accuracy'] and row['aeod']==result['metrics']['aeod'] and row['aspd']==result['metrics']['aspd']
  cfg={k:v for k,v in job['config'].items() if k not in ev.IGNORE_RECIPE};fcfg={k:v for k,v in control['config'].items() if k not in ev.IGNORE_RECIPE};assert cfg==fcfg
  accept=json.load(t.extractfile('runs/'+identity+'/mechanism_acceptance.json'))
  assert accept['pass'] and accept['variant']=='minus_C' and accept['rounds']==70 and accept['candidate_calls_verified']==accept['expected_candidate_calls']
  assert accept['job_sha256']==entry['job_sha256'] and accept['result_sha256']==members['runs/'+identity+'/result.json']['sha256'] and accept['checkpoint_sha256']==row['checkpoint_sha256']
  assert accept['adapter_hashes']==manifest['adapter_hashes'] and accept['candidate_audit_sha256']==members['runs/'+identity+'/candidate_mask_audit.json']['sha256']
  assert json.load(t.extractfile('runs/'+identity+'/config.json'))==result['config']
  identities.append({'id':identity,'checkpoint_sha256':row['checkpoint_sha256'],'job_sha256':entry['job_sha256'],'result_sha256':accept['result_sha256'],'root_image_ids_sha256':control['data_contract']['root_image_ids_sha256'],'paired_Full_id':control['id'],'complete_rounds':70,'valid_n':19867,'native_metrics_match_inspection':True})
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
target=B/TAG/'ROOT_INDEPENDENT_REVIEW.json'
report=dict(status='ROOT_NATIVE128_DELTA_MEMBERS_OLD225_RECORDS_AND_EXACT3_C_FEDSA_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_sha256=sha(archive),inspection_sha256=sha(inspection_path),ledger_sha256=sha(ledger_path),receipt_sha256=sha(receipt_path),offserver_proof_sha256=sha(proof_path),source_root_proof_sha256=sha(B/TAG/'ROOT_DELTA_VERIFICATION.json'),previous_inspection_sha256=sha(old_path),previous_ledger_sha256=sha(old_ledger_path),native_accepted=128,added3=IDS,archive_members_verified=54,ledger_entries_verified=19,ledger_accepted_unique_ids=128,ledger_previous18_entries_exact=True,receipt_chain_verified=True,original225_records_exact=True,original225_raw_json_record_bytes_exact=True,original225_record_order_preserved=True,inspection_records=228,full_reused=100,minus_U_complete_100=True,minus_C_partial_28=True,minus_C_IID_FedSA_partial_8=True,new_identity_checks=identities,frozen_repo_source_members_verified=source_members,strict_source_script_sha256=sha(ep),manifest_sha256=sha(manifest_path),full_inventory_sha256=sha(base_path),old_archive_members_rescanned=False,live_data_files_rehashed=False,independent_mean_table_generated=False,three_view_acceptance_changed=False,oldFull_models_repacked=0,new_inference=0,test=False,whole_rebuttal_complete=False,verification_script_sha256=sha(Path(__file__)))
with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'root_review_path':str(target),'root_review_sha256':sha(target),'native128':True,'new3':IDS,'members':54}))
