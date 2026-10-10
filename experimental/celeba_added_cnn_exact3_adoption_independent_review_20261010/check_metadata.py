"""Read-only metadata/source review; no numpy, torch, fits, CNN, SSH or Git."""
from pathlib import Path
import ast, datetime, hashlib, json, math
R=Path(__file__).resolve().parents[2];O=R/'tmp/celeba_added_cnn_exact3_root_execution_20261010';HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_bytes())
adopt=O/'ROOT_SCIENTIFIC_ADOPTION.json'
assert sha(adopt.read_bytes())=='631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456'
a=read(adopt)
for name,h in a['proof_files_sha256'].items():assert sha((O/name).read_bytes())==h,name
assert sha((R/a['adoption_source_path']).read_bytes())==a['adoption_source_sha256']
source=R/'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009/chunk_039/verified_extract/sourcefreeze/062_saved_science.py'
text=source.read_text(encoding='utf8');assert sha(source.read_bytes())==a['original_saved_check_file_sha256']
fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='check_saved')
block=next(n for n in fn.body if isinstance(n,ast.With))
assert sha(ast.dump(block,include_attributes=False).encode())==a['exact_saved_array_AST_block_sha256']
assert sha(ast.get_source_segment(text,fn).encode())=='ed8dc5eb2332ef8df176166087a91fb5f2b5043cc0e4669f7c985ca5b7452cfc'
linux_path=R/'tmp/verify_added_cnn_exact3_linux_saved_root_20261010.py'
ltree=ast.parse(linux_path.read_text(encoding='utf8'))
remote=next(ast.literal_eval(n.value) for n in ltree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets))
assert remote==(O/'linux_saved_root_remote.py').read_text(encoding='utf8')
assert 'body=[node]' in remote and "ns['check_saved']" in remote
assert 'body=[block]' in (R/'tmp/verify_added_cnn_exact3_offserver_arrays_20261010.py').read_text(encoding='utf8')
for prefix in ['LINUX_EXACT_ROOT','SCIENCE']:
 e=read(O/(prefix+'_EXIT.json'))
 for suffix in ['STDOUT','STDERR']:assert sha((O/(prefix+'_'+suffix+'.txt')).read_bytes())==e[suffix.lower()+'_sha256']
assert read(O/'LINUX_EXACT_ROOT_EXIT.json')['exit']==0 and read(O/'SCIENCE_EXIT.json')['exit']==1
assert (O/'LINUX_EXACT_ROOT_STDOUT.txt').read_bytes()==(O/'LINUX_EXACT_ROOT_CHECK.json').read_bytes()
preserved=read(O/'PRESERVED_WINDOWS_FAILURE.json')
assert preserved['exit_record']==read(O/'SCIENCE_EXIT.json')
assert preserved['original_stderr_text']==(O/'SCIENCE_STDERR.txt').read_text(encoding='utf8')
assert preserved['tolerance_changed'] is False and preserved['whole_windows_check_retried'] is False
transport=read(O/'TRANSPORT_VERIFICATION.json');F=Path(transport['archive_path']).parent/'verified_extract'
# Only small JSON receipts are opened on F; original transport pins bind arrays.
gatepath=F/'bundle/GATE_RESULT.json';gate=read(gatepath)
assert sha(gatepath.read_bytes())==transport['members']['bundle/GATE_RESULT.json']['sha256']
m=read(R/'tmp/celeba_added_cnn_three_view_gate_preparation_20261010/MANIFEST.json')
linux=read(O/'LINUX_EXACT_ROOT_CHECK.json');local=read(O/'OFFSERVER_ARRAY_REFIT_CHECK.json')
ids=a['exact_ids'];assert len(ids)==len(set(ids))==3
for records in [gate['receipts'],m['records'],a['records'],linux['records'],local['records']]:assert [x['id'] for x in records]==ids
metric_count=base_count=rule_count=0;rows=[]
for receipt,row,ad,li,lo in zip(gate['receipts'],m['records'],a['records'],linux['records'],local['records']):
 pin=transport['members']['bundle/'+row['id']+'/receipt.json'];raw=(F/'bundle'/row['id']/'receipt.json').read_bytes()
 assert sha(raw)==pin['sha256']==ad['receipt_sha256']==li['receipt_sha256']==lo['receipt_sha256']
 assert json.loads(raw)==receipt
 assert receipt['checkpoint_sha256']==ad['checkpoint_sha256']==li['checkpoint_sha256']==row['identity']['checkpoint']['sha256']
 assert receipt['prediction_arrays_sha256']==ad['array_sha256']==li['array_sha256']==lo['array_sha256']==transport['members']['bundle/'+row['id']+'/validation_predictions.npz']['sha256']
 assert row['identity']['config']['rounds']==70 and row['identity']['config']['celeba_evaluation_split']=='valid'
 assert receipt['weights_before']==receipt['weights_after'] and receipt['valid_n']==19867
 assert receipt['root_reconstruction']['root_n']==16277
 assert not any(receipt[k] for k in ['optimizer_created','gradients_created','test_labels_accessed','test_inference_performed','final_dispatch_created'])
 assert receipt['native_comparison']['accepted'] and receipt['native_comparison']['max_abs_difference']==0.0 and receipt['native_comparison']['tolerance']==1e-12
 for view in ['native','raw','shared_calibration']:
  score=receipt['views'][view];c=score['group_confusion_counts'];metric_count+=3
  for g in ['0','1']:
   z=c[g]
   for key in ['tp','fp','tn','fn']:assert type(z[key]) is int;base_count+=1
   assert z['tp']+z['fn']==z['positives'] and z['tn']+z['fp']==z['negatives'] and z['positives']+z['negatives']==z['n']
  calc={'accuracy':sum(c[g]['tp']+c[g]['tn'] for g in c)/sum(c[g]['n'] for g in c),'aeod':abs(c['0']['tp']/c['0']['positives']-c['1']['tp']/c['1']['positives']),'aspd':abs((c['0']['tp']+c['0']['fp'])/c['0']['n']-(c['1']['tp']+c['1']['fp'])/c['1']['n'])}
  assert all(math.isclose(v,score[k],rel_tol=0,abs_tol=1e-15) for k,v in calc.items())
  fit=receipt['fits'][view];rule_count+=1
  assert fit['rule']==('group_margin_greater_equal' if view=='shared_calibration' else 'argmax_margin_strictly_positive')
  assert fit['fit_data']==('clean_train_root_only' if view=='shared_calibration' else 'none')
 rows.append({'id':row['id'],'checkpoint_sha256':ad['checkpoint_sha256'],'rounds':70,'split':'valid','native_delta':0.0,'metrics':9,'base_confusion_counts':24,'rules':3})
assert (metric_count,base_count,rule_count)==(27,72,9)==(a['metric_values_checked'],a['integer_base_counts_checked'],a['prediction_rules_checked'])
assert a['Windows_whole_saved_check_pass'] is False and a['Linux_whole_original_saved_check_pass'] is True
assert a['mechanism_three_view_cutoff_unchanged']==251 and a['interface_records_accepted']==3
assert not any(a[k] for k in ['final_test','full100_complete','full17_complete','final_primary_endpoint_decided'])
assert a['records'][0]['preserved_root_audit_differences'][0]['difference']==-2.168404344971009e-19
sources=[linux_path,R/'tmp/verify_added_cnn_exact3_offserver_arrays_20261010.py',O/'linux_saved_root_remote.py',R/'tmp/diagnose_added_cnn_exact3_offserver_root_20261010.py',source]
e={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'adoption_sha256':sha(adopt.read_bytes()),'adoption_proof_files_checked':len(a['proof_files_sha256']),'source_pins':{p.relative_to(R).as_posix():sha(p.read_bytes()) for p in sources},'original_saved_check_AST_unchanged_on_Linux':True,'Windows_original_saved_array_With_block_AST_hash_matches':True,'Linux_helper_literal_and_retained_script_equal':True,'Linux_exit_stdout_stderr_hashes_checked':True,'Windows_failed_exit_and_stderr_preserved':True,'records':rows,'metric_count':metric_count,'base_count':base_count,'rule_count':rule_count,'arrays_opened':False,'F_writes':0,'SSH':0,'new_fit':0,'new_inference':0,'Git_mutations':0,'STATE_writes':0,'limits':'Source and small receipt inspection; previous numeric refits are root-run evidence, not rerun here.'}
(HERE/'EVIDENCE.json').write_text(json.dumps(e,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'status':'PASS','proof_files':e['adoption_proof_files_checked'],'counts':[metric_count,base_count,rule_count],'records':3,'new_fit':0,'new_inference':0}))
