"""Record-only join of this fixed saved-array batch to the actual native288 archive."""
from pathlib import Path
import hashlib,json,tarfile,contextlib,datetime,traceback
H=Path(__file__).resolve().parent;R=H.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
canonical=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p,w):
 assert sha(p)==w,str(p)
 return read(p)
def save(n,x):
 with (H/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 E=read(H/'EXECUTION_INPUTS.json');P=read(H/'PREPARED.json');pre=read(H/'PREFLIGHT.stdout.json');raw=read(H/'RAW_STORAGE_INDEX.json');expected=pre['selected_ids'];N=len(expected)
 assert N==8 and expected==P['candidate_ids'] and raw['accepted_new_ids']==expected
 assert (raw['metrics'],raw['counts'],raw['rules'])==(9*N,24*N,3*N)
 prior_proof=pin(P['prior_root_path'],P['prior_root_sha256']);prior=pin(P['prior_index_path'],P['prior_index_sha256'])
 assert prior_proof['cumulative_accepted']==len(prior['all_ids'])==280 and not set(expected)&set(prior['all_ids'])
 currentproof=pin(E['native_root_path'],E['native_root_sha256']);inspection=pin(E['native_inspection_path'],E['native_inspection_sha256']);ledger=pin(E['native_ledger_path'],E['native_ledger_sha256'])
 assert currentproof['root_adopted'] and currentproof['total_new_strict_and_offserver']==288 and currentproof['inspection_sha256']==E['native_inspection_sha256']
 previousroot=Path(P['native280_root_path']);previousproof=pin(previousroot,P['native280_root_sha256']);oldinspection=pin(previousroot.parent/'inspection/inspection.json',previousproof['inspection_sha256']);oldledger=pin(previousroot.parent/'verified_ledger.json',previousproof['ledger_sha256'])
 assert len(oldinspection['records'])==380 and len(inspection['records'])==100+currentproof['total_new_strict_and_offserver']
 assert [row for row in inspection['records'] if row['id'] not in currentproof['new_ids']]==oldinspection['records']
 assert ledger['entries'][:-1]==oldledger['entries'] and len(ledger['entries'])==len(oldledger['entries'])+1
 native_rows={row['id']:row for row in inspection['records']}
 assert set(P['candidate_ids'])<=set(inspection['accepted_new_ids'])-set(prior['all_ids'])
 receipt=pin(raw['receipt'],raw['receipt_sha256']);off=pin(raw['offserver_verification'],raw['offserver_verification_sha256'])
 assert sha(raw['archive'])==raw['archive_sha256']==receipt['archive_sha256']==off['actual_transport_archive_sha256']
 assert receipt['previous_backup_receipt_sha256']==P['previous_receipt_sha256']==sha(P['previous_local_receipt'])
 assert raw['all_transported_ids']==receipt['all_transported_ids']==P['previous_all_transported_ids']+expected
 assert receipt['accepted_new_ids']==expected and off['accepted_offserver']==0 and off['root_adoption_pending']
 assert off['source_seal_sha256']==receipt['source_seal_sha256']=='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
 assert receipt['transport_source_seal_sha256']=='1a021b707575292c33959c19fcfa2fa1ee8c7f285d20576c562e4843d1488fb3'
 saved=off['original_saved_array_verification'];records=saved['records']
 assert saved['status']=='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS' and [x['id'] for x in records]==expected
 assert (saved['independent_metric_checks'],saved['independent_confusion_count_checks'],saved['prediction_rule_checks'])==(9*N,24*N,3*N)
 baseline=pin(R/'docs/server_deployment_20260923/training_20260923/final_evaluation_prepared_20261009/model_inventory.json','3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd');full_rows={x['id']:x for x in baseline['records']}
 extract=Path(raw['offserver_verification']).parent/'verified_extract';bindings={};binding_files={};artifacts={};member_checks=[];native_archives=[];archives_by_id={}
 with contextlib.ExitStack() as stack:
  for root_pin in E['native_archive_roots']:
   path=Path(root_pin['path']);proof=pin(path,root_pin['sha256'])
   ap=Path(proof['archive_local_path']);assert sha(ap)==proof['archive_sha256']
   archive=stack.enter_context(tarfile.open(ap,'r:gz'));inv=json.load(archive.extractfile('backup_inventory.json'));assert inv['accepted_new_ids']==proof['new_ids']
   native_archives.append(dict(path=str(ap),sha256=sha(ap),root_verification_path=str(path.relative_to(R)),root_verification_sha256=sha(path)))
   for identity in set(expected)&set(proof['new_ids']):assert identity not in archives_by_id;archives_by_id[identity]=(archive,inv)
  assert set(archives_by_id)==set(expected)
  for rec in records:
      identity = rec['id']; runtime = extract / 'runtime' / identity; run_dir = extract / 'runs' / identity
      archive,inv = archives_by_id[identity]
      binding = read(runtime / 'binding.json'); r = binding['record']; native = native_rows[identity]
      assert binding['status'] == 'IMMUTABLE_TERMINAL_ONCE_BOUND' and r['accepted_v4_row'] == binding['native_acceptance']['row'] == native
      assert binding['checkpoint_sha256'] == r['checkpoint']['sha256'] == rec['checkpoint_sha256'] == native['checkpoint_sha256']
      assert (r['variant'], r['distribution'], r['attack'], r['actual_alpha'], r['terminal_round'], r['original_split'], r['original_n_eval']) == ('minus_A', native['distribution'], native['attack'], 5000 if native['distribution']=='IID' else 5, 70, 'valid', 19867)
      assert r['seed'] == int(identity[-5:]) and r['config']['ablation_component'] == 'A' and canonical(r['config']) == r['config_canonical_sha256']
      payloads = {}
      for name, wanted in (('model.pt', r['checkpoint']['sha256']), ('result.json', r['result']['sha256'])):
          member = 'runs/' + identity + '/' + name; payload = archive.extractfile(member).read()
          actual = hashlib.sha256(payload).hexdigest()
          assert actual == wanted == inv['members'][member]['sha256'] and len(payload) == inv['members'][member]['bytes']
          member_checks.append(dict(member=member, sha256=actual, bytes=len(payload)))
          if name == 'result.json': payloads[name] = json.loads(payload)
      result = payloads['result.json']; revision = result['revision_job']
      assert result['config'] == r['config'] and result['rounds'] == 70 and result['seed'] == r['seed']
      assert revision['source_hashes'] == r['source_hashes'] and result['data_contract']['image_data_contract'] == r['data_contract']
      assert inv['members']['jobs/' + identity + '.json']['sha256'] == r['raw_job']['sha256']
      for item in (r['checkpoint'], r['result'], r['raw_job']):
          assert item['sha256'] in native['files'].values()
      paired = r['paired_full']; full = full_rows[paired['id']]
      assert (full['distribution'], full['attack'], full['seed']) == (r['distribution'], r['attack'], r['seed'])
      assert paired['baseline_record_canonical_sha256'] == canonical(full)
      assert all(paired[key + '_sha256'] == full[key]['sha256'] for key in ('checkpoint', 'result', 'raw_job'))
      assert not paired['replay_required_here'] and not paired['weights_repacked_here']
      strict = read(run_dir / 'strict_acceptance.json'); bridge = read(run_dir / 'bridge_receipt.json'); sci = read(run_dir / 'receipt.json')
      approval = read(runtime / 'APPROVED.json'); complete = read(runtime / 'REMOTE_COMPLETE.json'); inventory = read(runtime / 'inventory.json')
      assert strict['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and strict['id'] == identity
      assert strict['views'] == sci['views'] == rec['views']
      assert bridge['views'] == list(strict['views'])
      assert strict['checkpoint_sha256'] == sci['checkpoint_sha256'] == complete['checkpoint_sha256'] == r['checkpoint']['sha256']
      assert strict['native_comparison']['tolerance'] == bridge['native_tolerance'] == approval['native_tolerance'] == 1e-12
      assert strict['native_comparison']['accepted'] and strict['native_comparison']['max_abs_difference'] == rec['native_max_abs_difference'] == complete['native_difference'] == 0
      assert strict['native_comparison']['expected'] == r['prior_validation_metrics']
      assert strict['bridge_receipt_sha256'] == sha(run_dir / 'bridge_receipt.json')
      assert bridge['scientific_body_receipt_sha256'] == sha(run_dir / 'receipt.json')
      assert bridge['source_before'] == bridge['source_after'] and bridge['artifact_before'] == bridge['artifact_after']
      assert sha(runtime / 'binding.json') == complete['binding_sha256'] == approval['binding_sha256']
      assert sha(runtime / 'inventory.json') == complete['inventory_sha256'] == strict['inventory_sha256'] == bridge['inventory_sha256'] == approval['inventory_sha256']
      assert sha(runtime / 'APPROVED.json') == bridge['approval_sha256'] and sha(run_dir / 'strict_acceptance.json') == complete['strict_sha256']
      assert inventory['records'] == [r] and approval['selected_ids'] == [identity]
      assert approval['device'] == 'cpu' and approval['compute_threads'] == 8 and approval['max_processes'] == 1 and approval['allowed_cpus'] == list(range(112, 120))
      assert sci['original_result_sha256'] == r['result']['sha256'] and sci['original_job_sha256'] == r['raw_job']['sha256'] and sci['config_canonical_sha256'] == r['config_canonical_sha256']
      assert sci['weights_before'] == sci['weights_after'] and not sci['optimizer_created'] and not sci['gradients_created']
      assert sci['valid_n'] == 19867 and sci['valid_image_ids_sha256'] == r['data_contract']['evaluation_image_ids_sha256']
      assert sci['root_reconstruction']['root_n'] == 16277 and sci['root_reconstruction']['root_image_ids_sha256'] == r['data_contract']['root_image_ids_sha256']
      assert sci['root_reconstruction']['client_sample_counts'] == r['data_contract']['client_sample_counts']
      assert (rec['independent_metric_checks'], rec['independent_confusion_count_checks'], rec['prediction_rule_checks']) == (9, 24, 3)
      assert complete['accepted_offserver'] == complete['new_training'] == complete['new_Full_inference'] == strict['new_training'] == 0
      assert not complete['test'] and not strict['test_inference'] and not sci['test_inference_performed']
      assert sha(run_dir / 'validation_predictions.npz') == rec['prediction_arrays_sha256'] == sci['prediction_arrays_sha256'] == complete['prediction_arrays_sha256']
      bindings[identity] = binding; binding_files[identity] = dict(path=str(runtime / 'binding.json'), sha256=sha(runtime / 'binding.json'))
      artifacts[identity] = {name: dict(path=str(path), sha256=sha(path)) for name, path in (
          ('scientific_receipt', run_dir / 'receipt.json'), ('bridge_receipt', run_dir / 'bridge_receipt.json'), ('strict_json', run_dir / 'strict_acceptance.json'),
          ('delegated_approval', runtime / 'APPROVED.json'), ('bound_inventory', runtime / 'inventory.json'), ('remote_complete', runtime / 'REMOTE_COMPLETE.json'))}

 assert len(member_checks)==2*N
 allids=prior['all_ids']+expected;assert len(allids)==len(set(allids))==280+N and allids[:280]==prior['all_ids']
 index=dict(status='PROPOSED_REPLAY_ID_INDEX_PENDING_ROOT_ADOPTION',prior_index_path=str(Path(P['prior_index_path']).relative_to(R)),prior_index_sha256=P['prior_index_sha256'],prior_adoption_path=str(Path(P['prior_root_path']).relative_to(R)),prior_adoption_sha256=P['prior_root_sha256'],all_ids=allids,new_ids=expected,new_records=records,new_bindings=bindings,new_binding_files=binding_files,new_artifacts=artifacts,new_archive=raw,native_inspection_path=str(Path(E['native_inspection_path']).relative_to(R)),native_inspection_sha256=E['native_inspection_sha256'],native_archives=native_archives,native_members_rehashed=2*N,Full_inference=0,test=False,root_adopted=False)
 save('MECHANISM_INDEX.json',index)
 save('NATIVE_IDENTITY_JOIN.json',dict(status='SAVED_ARRAY_EXACT8_NATIVE_IDENTITY_JOIN_PASS_PENDING_ROOT',new_ids=expected,prior280_objects_and_order_unchanged=True,old380_native_records_exact=True,old_native_ledger_entries_exact=True,native_model_result_members_rehashed=2*N,native_member_checks=member_checks,new_source_config_data_checkpoint_receipt_identities_exact=N,Full_reference_joins=N,Full_weights_repacked=0,metrics=9*N,counts=24*N,rules=3*N,native_max_abs_difference=max(v['native_max_abs_difference'] for v in records),index_sha256=sha(H/'MECHANISM_INDEX.json'),root_adopted=False))
 print(json.dumps(dict(index_sha256=sha(H/'MECHANISM_INDEX.json'),join_sha256=sha(H/'NATIVE_IDENTITY_JOIN.json'),N=N,cumulative=280+N)))
if __name__=='__main__':
 try:main()
 except BaseException as e:save('JOIN_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True));raise
