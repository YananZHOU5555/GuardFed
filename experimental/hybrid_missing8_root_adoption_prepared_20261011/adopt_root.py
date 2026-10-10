"""Prepared root-only fixed Hybrid8 + screen1 join; no fit, forward or training."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,subprocess,zipfile
H=Path(__file__).resolve().parent;R=H.parents[1]
C=R/'tmp/hybrid_missing8_pool32_runtime_prepared_20261011'
A=R/'tmp/hybrid_missing8_saved_root_execution_20261011/saved001';W=A.parent
F=Path('F:/YananResearchStorage/GuardFed/hybrid_missing8_saved_20261011/attempt001')
S=R/'tmp/hybrid_screen91001_single_replay_prepared_20261011'
SA=R/'tmp/hybrid_screen91001_saved_root_execution_20261011/saved001';SW=SA.parent
SCREEN_WRAPPER=R/'tmp/hybrid_screen91001_saved_runtime_prepared_20261011/windows_runtime_wrapper.py'
SF=Path('F:/YananResearchStorage/GuardFed/hybrid_screen91001_saved_20261011/attempt001')
SCREEN_PACKAGE='9be2ce96b4548144903e9e928567efa96565326917e2363f1ad11712ca44015d'
SCREEN_ID='CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91001_screen'
PACKAGE='f3b197ce786a3f391f820baf8767fa25fba2d2387002acf654bced7dc8e95239'
HELPERS='89bf8df9e575a627c726cfadbc449bafcca841832a9f1e49e09a6a375c818ce8'
IDS=['CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed'+str(s)+'_fullcoverage' for s in range(91003,91011)]
PRIOR_ID='CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage'
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def check_seal(directory,expected):
 need(sha(directory/'FILES_SHA256.json')==expected,'Source seal differs')
 for rel,pin in read(directory/'FILES_SHA256.json')['files'].items():
  p=(directory/rel).resolve();need(p.is_relative_to(directory.resolve()) and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'Source member differs: '+rel)
def expected_files():
 return {'candidate_seal':C/'FILES_SHA256.json','saved_helpers_seal':R/'tmp/hybrid_missing8_saved_runtime_prepared_20261011/FILES_SHA256.json','Linux_original_whole':A/'LINUX_SAVED_CHECK.json','F_transport':A/'TRANSPORT_VERIFICATION.json','Windows_zero_fit_audit':F/'WINDOWS002_SAVED_OUTPUT_CHECK.json','Linux_command':A/'LINUX_CHECK_COMMAND.json','Linux_exit':A/'LINUX_CHECK_EXIT.json','transport_command':A/'TRANSPORT_COMMAND.json','transport_exit':A/'TRANSPORT_SSH_EXIT.json','Windows_command':A/'WINDOWS002_AUDIT_COMMAND.json','Windows_exit':A/'WINDOWS002_AUDIT_EXIT.json','Windows_wrapper':W/'windows_runtime_wrapper.py','Windows_wrapper_review':W/'WINDOWS_REPAIR_SOURCE_REVIEW.json','original_Windows_command':A/'WINDOWS_AUDIT_COMMAND.json','original_Windows_exit':A/'WINDOWS_AUDIT_EXIT.json','original_Windows_failure':F/'WINDOWS_SAVED_OUTPUT_CHECK.failure.json','original_Windows_started':F/'WINDOWS_SAVED_OUTPUT_CHECK.started.json','prior_Hybrid_interface_root':R/'tmp/celeba_added_cnn_exact3_root_execution_20261010/ROOT_SCIENTIFIC_ADOPTION.json','native_exact8_root':R/'tmp/celeba_hybrid_native9_root_adoption_20261011/ROOT_ADOPTION.json','exact_diagnostic_map':A/'ROOT_EXACT_DIAGNOSTIC_DIFFERENCES.json','screen_candidate_seal':S/'FILES_SHA256.json','screen_Linux_original_whole':SA/'LINUX_SAVED_CHECK.json','screen_F_transport':SA/'TRANSPORT_VERIFICATION.json','screen_Windows_zero_fit_audit':SF/'WINDOWS_SAVED_OUTPUT_CHECK.json','screen_Linux_command':SA/'LINUX_CHECK_COMMAND.json','screen_Linux_exit':SA/'LINUX_CHECK_EXIT.json','screen_transport_command':SA/'TRANSPORT_COMMAND.json','screen_transport_exit':SA/'TRANSPORT_SSH_EXIT.json','screen_Windows_command':SA/'WINDOWS_AUDIT_COMMAND.json','screen_Windows_exit':SA/'WINDOWS_AUDIT_EXIT.json','screen_Windows_wrapper':SCREEN_WRAPPER,'screen_Windows_wrapper_review':SW/'WINDOWS_REPAIR_SOURCE_REVIEW.json','screen_exact_diagnostic_map':SA/'ROOT_EXACT_DIAGNOSTIC_DIFFERENCES.json'}
def pin_path(path):
 return path.relative_to(R).as_posix() if path.is_relative_to(R) else path.as_posix()
def validate_inputs(inputs):
 expected=expected_files()
 need(inputs['adoption_source_sha256']==sha(__file__),'Root join source changed')
 need(inputs['exact_ids']==IDS,'Exact8 root input order differs')
 need(inputs['screen_exact_ids']==[SCREEN_ID],'Only the one accepted screen checkpoint may be appended')
 need(set(inputs['files'])==set(expected) and all(inputs['files'][role]['path']==pin_path(path) for role,path in expected.items()),'Closed actual evidence roles/paths differ')
 diagnostic=read(A/'ROOT_EXACT_DIAGNOSTIC_DIFFERENCES.json')
 need(sha(A/'ROOT_EXACT_DIAGNOSTIC_DIFFERENCES.json')=='3928a665bffeecc9f26d73a4668d5637549c21375909d6dfb159d63c05cf52bb' and diagnostic['no_tolerance_substitution'] is True,'Root exact diagnostic map differs')
 need(inputs['exact_root_receipt_differences']==diagnostic['exact_differences'],'Root-reviewed exact diagnostic map required')
 exact_differences=inputs['exact_root_receipt_differences']
 need(set(exact_differences)==set(IDS),'Every exact ID needs a root-reviewed differences list')
 for rid,diffs in exact_differences.items():
  need(isinstance(diffs,list) and len(diffs)<=1,'At most the exact group-KL diagnostic may differ')
  for d in diffs:
   need(set(d)=={'path','local','saved','kind','local_hex','saved_hex','difference'} and d['path']=='/server_sampling_audit/group_kl' and d['kind']=='scalar','Only explicit group-KL diagnostic allowed')
   need(all(type(d[k]) is float and math.isfinite(d[k]) for k in ('local','saved','difference')),'Nonfinite/nonfloat diagnostic refused')
   need(d['local'].hex()==d['local_hex'] and d['saved'].hex()==d['saved_hex'] and d['local']-d['saved']==d['difference'],'Diagnostic hex/value mismatch; no tolerance substitution')
def command_arg(command,name):
 argv=command['argv'];need(argv.count(name)==1,'Missing/duplicate command argument: '+name);return argv[argv.index(name)+1]
def check_screen(inputs,volume):
 # Fixed second batch only. The original member/per-record checks below remain
 # the same; this is not a dynamic inventory or a future fullcoverage runner.
 check_seal(S,SCREEN_PACKAGE)
 need(sha(S/'check_saved.py')=='9408f64db4fb68d53eee4d882d69f3d0a3a0025baedbcb163823b3794a328934','Screen outer saved consumer differs')
 m=read(S/'MANIFEST.json');ids=[SCREEN_ID]
 need(m['exact_ids']==[r['id'] for r in m['records']]==ids and m['source_role']=='accepted_screen_reuse' and m['selection_seed'] is True and m['phase']=='screen' and m['new_formal_training']==0,'Screen identity/selection history differs')
 need(m['native_tolerance']==1e-12 and m['views']==['native','raw','shared_calibration'] and m['device']=='cpu' and m['dtype']=='float32' and m['threads']==8,'Screen scientific policy differs')
 native=R/m['native_root_pin']['path'];need(sha(native)==m['native_root_pin']['sha256']=='26bab727c431679b2913ac76cefe7dc28729c49d660bb3c8141489426bd28e6d','Accepted screen native proof differs')
 need(read(native)['accepted_total']==27 and read(native)['final_test'] is False and m['records'][0]['identity']['external_proof_sha256']['root']==sha(native) and m['records'][0]['identity']['checkpoint']['sha256']=='e8461cedfda69369be3e5fec4efb791fd2f5674f62554a1324f8ea8a87b7f85f','Screen native/checkpoint source differs')
 linux=read(SA/'LINUX_SAVED_CHECK.json');t=read(SA/'TRANSPORT_VERIFICATION.json');audit=read(SF/'WINDOWS_SAVED_OUTPUT_CHECK.json')
 need(sha(SA/'LINUX_SAVED_CHECK.json')=='927619163d07477436e92440d2a4689376fbdbd1f1706b2b0360fa384c17c890' and linux['status']=='HYBRID_SCREEN91001_LINUX_ORIGINAL_WHOLE_SAVED_PASS_NOT_ADOPTED' and linux['cached_root_refits']==linux['fit_calls']==1,'Screen original Linux whole proof differs')
 need(t['status']=='HYBRID_SCREEN91001_F_SAVED_MEMBERS_SHA_PASS' and t['member_count']==6 and t['linux_proof_sha256']==sha(SA/'LINUX_SAVED_CHECK.json'),'Screen F transport proof differs')
 need(audit['status']=='HYBRID_SCREEN91001_WINDOWS_SAVED_OUTPUT_ZERO_FIT_PASS_NOT_ADOPTED','Screen Windows saved-output proof differs')
 for proof in [linux,audit]:
  need(proof['package_sha256']==SCREEN_PACKAGE and proof['gate_result_sha256']==t['gate_result_sha256']=='03df21a0cdb677b614c149d7119ff3ef33591303e873749887c98abf87c2fc7a' and proof['original_check_saved_sha256']=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745' and proof['exact_ids']==ids and proof['native_tolerance']==1e-12 and proof['new_CNN']==proof['new_training']==0 and proof['test'] is False and proof['root_adopted'] is False,'Screen original checker identity/policy differs')
 need(audit['fit_calls']==audit['cached_root_refits']==0 and audit['Windows_whole_check_pass'] is None and audit['Windows_exact_recalibration_claimed'] is False and audit['prior_FL_Windows_whole_and_refit_failures_not_relabelled'] is True,'Screen Windows evidence relabelled')
 review=read(SW/'WINDOWS_REPAIR_SOURCE_REVIEW.json');need(review['status']=='ROOT_REVIEWED_BEFORE_ACTUAL_WINDOWS_SAVED_CHECK' and review['wrapper_sha256']==sha(SCREEN_WRAPPER) and review['original_checker_sha256']==sha(S/'check_saved.py') and review['package_sha256']==SCREEN_PACKAGE,'Screen wrapper review/source differs')
 for name in ['LINUX_CHECK_EXIT.json','TRANSPORT_SSH_EXIT.json','WINDOWS_AUDIT_EXIT.json']:need(read(SA/name)['exit']==0,'Successful screen actual exit required: '+name)
 command=read(SA/'WINDOWS_AUDIT_COMMAND.json');argv=command['argv']
 need(command['wrapper_sha256']==sha(SCREEN_WRAPPER) and command['source_review_sha256']==sha(SW/'WINDOWS_REPAIR_SOURCE_REVIEW.json'),'Screen Windows command source differs')
 need(command_arg(command,'--mode')=='windows-saved-output' and '--allow-saved-output-zero-fit' in argv and '--allow-original-cached-root-refit' not in argv and command_arg(command,'--package-sha256')==SCREEN_PACKAGE and command_arg(command,'--gate-result-sha256')==t['gate_result_sha256'] and command_arg(command,'--transport-proof-sha256')==sha(SA/'TRANSPORT_VERIFICATION.json') and command_arg(command,'--linux-proof-sha256')==sha(SA/'LINUX_SAVED_CHECK.json') and Path(command_arg(command,'--output')).resolve()==(SF/'WINDOWS_SAVED_OUTPUT_CHECK.json').resolve(),'Screen Windows zero-fit command links differ')
 need(read(SA/'LINUX_CHECK_COMMAND.json')['gate_result_sha256']==t['gate_result_sha256'] and read(SA/'LINUX_CHECK_COMMAND.json')['allow_original_cached_root_refit'] is True and read(SA/'TRANSPORT_COMMAND.json')['linux_proof_sha256']==sha(SA/'LINUX_SAVED_CHECK.json'),'Screen Linux/transport command link differs')
 froot=Path('F:/YananResearchStorage/GuardFed').resolve();archive=Path(t['archive_path']).resolve();extract=Path(t['verified_extract']).resolve()
 need(archive.is_relative_to(froot) and extract.is_relative_to(froot) and sha(archive)==t['archive_sha256'] and archive.stat().st_size==t['archive_bytes'],'F archive identity differs')
 members={'bundle/GATE_RESULT.json','bundle/metadata_receipt.json','LINUX_SAVED_CHECK.json'}|{'bundle/'+i+'/'+n for i in ids for n in ('receipt.json','validation_predictions.npz')}
 need(set(t['members'])==members,'Minimum exact6 screen scope differs')
 with zipfile.ZipFile(archive) as z:need(len(z.namelist())==6 and set(z.namelist())==members|{'TRANSPORT_MANIFEST.json'},'Screen F archive member scope differs')
 for rel,pin in t['members'].items():
  path=extract/rel;need(path.resolve().is_relative_to(extract) and sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes'],'Saved output member differs')
 gate=read(extract/'bundle/GATE_RESULT.json');need(sha(extract/'bundle/GATE_RESULT.json')==t['gate_result_sha256']==linux['gate_result_sha256'],'Gate identity differs')
 need(gate['status']=='HYBRID_SCREEN91001_THREE_VIEW_PASS_NOT_ROOT_ADOPTED' and gate['package_sha256']==linux['package_sha256']==SCREEN_PACKAGE,'Screen gate source differs')
 need([x['id'] for x in gate['receipts']]==[x['id'] for x in linux['records']]==[x['id'] for x in audit['records']]==ids,'Exact1 screen proof order differs')
 diagnostic=read(SA/'ROOT_EXACT_DIAGNOSTIC_DIFFERENCES.json');need(sha(SA/'ROOT_EXACT_DIAGNOSTIC_DIFFERENCES.json')=='6dbd06c5ca93d81e54388dbc09e2b9a464319fa2d074ce529e10d4dfe662bdbe' and diagnostic['windows_output_sha256']==sha(SF/'WINDOWS_SAVED_OUTPUT_CHECK.json') and diagnostic['no_tolerance_substitution'] is True,'Screen root diagnostic/report linkage differs')
 diffs=inputs['screen_exact_root_receipt_differences'];need(diffs==diagnostic['exact_differences'] and set(diffs)=={SCREEN_ID} and isinstance(diffs[SCREEN_ID],list) and len(diffs[SCREEN_ID])<=1,'Root-reviewed screen differences required')
 for d in diffs[SCREEN_ID]:
  need(set(d)=={'path','local','saved','kind','local_hex','saved_hex','difference'} and d['path']=='/server_sampling_audit/group_kl' and d['kind']=='scalar' and all(type(d[k]) is float and math.isfinite(d[k]) for k in ('local','saved','difference')) and d['local'].hex()==d['local_hex'] and d['saved'].hex()==d['saved_hex'] and d['local']-d['saved']==d['difference'],'Only exact root-reviewed screen group-KL difference allowed')
 rows=[]
 for row,g,l,w in zip(m['records'],gate['receipts'],linux['records'],audit['records']):
  rid=row['id'];need(row['terminal_round']==70 and row['split']=='valid' and row['n_eval']==19867,'Nonterminal/test/subset')
  need(g['checkpoint_sha256']==l['checkpoint_sha256']==w['checkpoint_sha256']==row['identity']['checkpoint']['sha256'],'Different checkpoint')
  need(g['weights_before']==g['weights_after'] and l['root_receipt_exact'] and l['cached_root_fit_exact'] and l['saved_predictions_metrics_counts_exact'],'Linux/weight check incomplete')
  need(w['saved_fit_payload_hashes_exact'] and w['saved_predictions_metrics_counts_exact'] and w['root_valid_ID_partition_exact'] and w['fit_calls']==0,'Windows saved checks incomplete')
  need(g['prediction_arrays_sha256']==l['array_sha256']==w['array_sha256']==t['members']['bundle/'+rid+'/validation_predictions.npz']['sha256'],'Array identity differs')
  need(l['receipt_sha256']==w['receipt_sha256']==t['members']['bundle/'+rid+'/receipt.json']['sha256'],'Receipt identity differs')
  need(g['native_comparison']==l['native_comparison']==w['native_comparison'] and g['native_comparison']['accepted'],'Native result differs')
  need(g['native_comparison']['tolerance']==1e-12 and g['native_comparison']['max_abs_difference']<=1e-12 and g['native_comparison']['expected']==row['identity']['prior_validation_metrics'],'Native tolerance/source metrics differ')
  need(w['root_receipt_differences']==diffs[rid] and w['local_full_root_receipt_exact']==(not w['root_receipt_differences']),'Screen diagnostic outside reviewed exact values')
  rows.append({'id':rid,'method':'CosineFairnessHybrid','distribution':row['distribution'],'attack':row['attack'],'seed':row['seed'],'checkpoint_sha256':g['checkpoint_sha256'],'array_sha256':w['array_sha256'],'receipt_sha256':w['receipt_sha256'],'native_comparison':w['native_comparison'],'native_max_abs_difference':w['native_comparison']['max_abs_difference'],'Windows_full_root_receipt_exact':not w['root_receipt_differences'],'preserved_root_audit_differences':w['root_receipt_differences'],'phase':'screen','selection_seed':True,'source_role':'accepted_screen_reuse'})
 return rows,t
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs-sha256',required=True);p.add_argument('--allow-complementary-root-adoption',action='store_true',required=True);args=p.parse_args()
 need(__debug__,'No -O');need(sha(H/'ROOT_INPUTS.json')==args.inputs_sha256,'Root inputs pin changed')
 inputs=read(H/'ROOT_INPUTS.json');validate_inputs(inputs)
 output=H/'ROOT_SCIENTIFIC_ADOPTION.json';need(not output.exists() and not (H/'ROOT_ADOPTION.failure.json').exists(),'Preserve original root adoption attempt')
 volume=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command','Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'],text=True));need(volume['FileSystemLabel']=='Yanan 2TB' and volume['HealthStatus']=='Healthy','F missing/unhealthy')
 for role,pin in inputs['files'].items():
  path=R/pin['path'];need(sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes'],'Actual evidence changed: '+role)
 check_seal(C,PACKAGE)
 check_seal(R/'tmp/hybrid_missing8_saved_runtime_prepared_20261011',HELPERS)
 oldpath=expected_files()['prior_Hybrid_interface_root'];need(sha(oldpath)=='631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456','Prior interface differs');old=read(oldpath)
 need(old['root_adoption'] is True and old['Linux_whole_original_saved_check_pass'] is True and old['Windows_whole_saved_check_pass'] is False and old['Windows_original_array_refit_block_pass'] is True and old['cross_platform_audit_failure_preserved'] is True,'Prior complementary failure roles differ')
 prior=[x for x in old['records'] if x['method']=='CosineFairnessHybrid'];need(len(prior)==1 and prior[0]['id']==PRIOR_ID and prior[0]['linux_whole_original_check_exact'] and prior[0]['offserver_array_fit_predictions_metrics_exact'] and prior[0]['windows_whole_root_receipt_exact'] and prior[0]['preserved_root_audit_differences']==[],'Prior Hybrid reuse differs')
 prior_failure=oldpath.parent/'PRESERVED_WINDOWS_FAILURE.json';need(sha(prior_failure)=='02984ace52d9bb9865941538c046f99335d0f70491d54e982742fc8777ec0e57' and read(prior_failure)['original_stderr_sha256']==old['proof_files_sha256']['SCIENCE_STDERR.txt'] and read(prior_failure)['tolerance_changed'] is False,'Historical exact3 Windows failure changed')
 reuse=read(C/'proofs/existing_replay_reuse_check.json');need(reuse['root_pin']['sha256']==sha(oldpath) and reuse['checkpoint_sha256']==prior[0]['checkpoint_sha256'] and reuse['Windows_combined_whole_checker_pass'] is False and reuse['preserved_Windows_failure_method']=='FLGMM','Existing Hybrid reuse source differs')
 for name,key in [('receipt.json','receipt_sha256'),('validation_predictions.npz','array_sha256')]:need(reuse['existing_saved_members'][name]['sha256']==prior[0][key],'Prior saved member differs')
 nativepath=expected_files()['native_exact8_root'];need(sha(nativepath)=='1434b40de5116bf3d53bc5a6ae2bd3b90f4e54222f23ad099ff72b1fd3ef1775','Native root differs');native=read(nativepath)
 need(native['accepted_new_ids']==IDS and native['accepted_ids']==[PRIOR_ID]+IDS and (native['prior_accepted'],native['new_accepted'],native['cumulative_accepted'])==(1,8,9) and native['final_test'] is False,'Native exact8 scope differs')
 m=read(C/'MANIFEST.json');need(m['exact_ids']==IDS==[x['id'] for x in m['records']] and m['previously_accepted_skip_ids']==[PRIOR_ID] and m['excluded_screen91001_identity_gap'] is True,'Exact8 manifest/prior exclusion differs')
 need(m['native_tolerance']==1e-12 and m['views']==['native','raw','shared_calibration'] and m['device']=='cpu' and m['dtype']=='float32' and m['threads']==8,'Scientific policy differs')
 need(m['native_root_pin']['sha256']==sha(nativepath) and m['existing_replay_root_pin']['sha256']==sha(oldpath),'Native/prior manifest pins differ')
 need([x['checkpoint_sha256'] for x in native['records']]==[x['identity']['checkpoint']['sha256'] for x in m['records']],'Manifest/native checkpoint join differs')
 linux=read(A/'LINUX_SAVED_CHECK.json');t=read(A/'TRANSPORT_VERIFICATION.json');audit=read(F/'WINDOWS002_SAVED_OUTPUT_CHECK.json')
 need(sha(A/'LINUX_SAVED_CHECK.json')=='7899be218a8b658cd6d9434382b9b4b9d8da4a65cdcff6ebbab491247195f0a0' and linux['status']=='HYBRID_MISSING8_LINUX_ORIGINAL_WHOLE_SAVED_PASS_NOT_ADOPTED' and linux['cached_root_refits']==linux['fit_calls']==8,'Original Linux whole proof differs')
 need(sha(A/'TRANSPORT_VERIFICATION.json')=='4293c2734d1f4ae592efd0bd9b5b1d27e16f4a147e8250dfb4611e0d46818833' and t['status']=='HYBRID_MISSING8_F_SAVED_MEMBERS_SHA_PASS' and t['member_count']==20 and t['linux_proof_sha256']==sha(A/'LINUX_SAVED_CHECK.json'),'F transport proof differs')
 need(sha(F/'WINDOWS002_SAVED_OUTPUT_CHECK.json')=='83528eef827603ca80b7c7bc33941156787911bb768ecd9abd1fce85143f0a82' and audit['status']=='HYBRID_MISSING8_WINDOWS_SAVED_OUTPUT_ZERO_FIT_PASS_NOT_ADOPTED','Windows saved-output proof differs')
 for proof in [linux,audit]:
  need(proof['package_sha256']==PACKAGE and proof['gate_result_sha256']==t['gate_result_sha256'] and proof['original_check_saved_sha256']=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745' and proof['exact_ids']==IDS and proof['native_tolerance']==1e-12 and proof['new_CNN']==proof['new_training']==0 and proof['test'] is False and proof['root_adopted'] is False,'Original checker identity/policy differs')
 need(audit['fit_calls']==audit['cached_root_refits']==0 and audit['Windows_whole_check_pass'] is None and audit['Windows_exact_recalibration_claimed'] is False and audit['prior_FL_Windows_whole_and_refit_failures_not_relabelled'] is True,'Windows evidence relabelled')
 need(read(A/'ROOT_EXACT_DIAGNOSTIC_DIFFERENCES.json')['windows_output_sha256']==sha(F/'WINDOWS002_SAVED_OUTPUT_CHECK.json'),'Diagnostic/report linkage differs')
 repair=read(W/'WINDOWS_REPAIR_SOURCE_REVIEW.json');failure=read(F/'WINDOWS_SAVED_OUTPUT_CHECK.failure.json')
 need(repair['status']=='ROOT_WINDOWS_RESOURCE_IMPORT_ONLY_REPAIR_SOURCE_REVIEW_PASS' and repair['wrapper_sha256']==sha(W/'windows_runtime_wrapper.py') and repair['original_consumer_sha256']==sha(C/'check_saved.py') and repair['original_package_sha256']==PACKAGE and repair['failed_original_attempt_sha256']==sha(F/'WINDOWS_SAVED_OUTPUT_CHECK.failure.json'),'Windows wrapper source review differs')
 need(repair['original_attempt_completed_records']==failure['completed']==0 and repair['original_failed_started_preserved'] is True and repair['fresh_output_required'] is True and repair['no_CNN_or_refit_in_wrapper'] is True and "No module named 'resource'" in failure['traceback'] and failure['root_adopted'] is False,'Original pre-record failure changed')
 start=read(F/'WINDOWS_SAVED_OUTPUT_CHECK.started.json');need(start['package_sha256']==PACKAGE and start['gate_result_sha256']==t['gate_result_sha256'] and start['mode']=='windows-saved-output' and start['root_adopted'] is False,'Original failed started record differs')
 for name in ['LINUX_CHECK_EXIT.json','TRANSPORT_SSH_EXIT.json','WINDOWS002_AUDIT_EXIT.json']:need(read(A/name)['exit']==0,'Successful actual exit required: '+name)
 need(read(A/'WINDOWS_AUDIT_EXIT.json')['exit']==1,'Original Windows failure exit changed')
 command=read(A/'WINDOWS002_AUDIT_COMMAND.json');argv=command['argv']
 need(command['wrapper_sha256']==sha(W/'windows_runtime_wrapper.py') and command['source_review_sha256']==sha(W/'WINDOWS_REPAIR_SOURCE_REVIEW.json') and command['prior_original_failed_attempt_preserved'] is True,'Windows002 command/source review differs')
 need(command_arg(command,'--mode')=='windows-saved-output' and '--allow-saved-output-zero-fit' in argv and '--allow-original-cached-root-refit' not in argv and command_arg(command,'--package-sha256')==PACKAGE and command_arg(command,'--gate-result-sha256')==t['gate_result_sha256'] and command_arg(command,'--transport-proof-sha256')==sha(A/'TRANSPORT_VERIFICATION.json') and command_arg(command,'--linux-proof-sha256')==sha(A/'LINUX_SAVED_CHECK.json') and Path(command_arg(command,'--output')).resolve()==(F/'WINDOWS002_SAVED_OUTPUT_CHECK.json').resolve(),'Actual Windows002 zero-fit evidence linkage differs')
 firstcommand=read(A/'WINDOWS_AUDIT_COMMAND.json');need(Path(command_arg(firstcommand,'--output')).resolve()==(F/'WINDOWS_SAVED_OUTPUT_CHECK.json').resolve() and command_arg(firstcommand,'--package-sha256')==PACKAGE,'Original failed command changed')
 need(read(A/'LINUX_CHECK_COMMAND.json')['gate_result_sha256']==t['gate_result_sha256'] and read(A/'LINUX_CHECK_COMMAND.json')['allow_original_cached_root_refit'] is True and read(A/'TRANSPORT_COMMAND.json')['linux_proof_sha256']==sha(A/'LINUX_SAVED_CHECK.json'),'Linux/transport command link differs')
 froot=Path('F:/YananResearchStorage/GuardFed').resolve();archive=Path(t['archive_path']).resolve();extract=Path(t['verified_extract']).resolve()
 need(archive.is_relative_to(froot) and extract.is_relative_to(froot) and sha(archive)==t['archive_sha256'] and archive.stat().st_size==t['archive_bytes'],'F archive identity differs')
 members={'bundle/GATE_RESULT.json','bundle/metadata_receipt.json','LINUX_SAVED_CHECK.json'}|{'bundle/'+i+'/'+n for i in IDS for n in ('receipt.json','validation_predictions.npz')}
 need(set(t['members'])==members,'Minimum exact20 archive scope differs')
 with zipfile.ZipFile(archive) as z:need(len(z.namelist())==20 and set(z.namelist())==members|{'TRANSPORT_MANIFEST.json'},'F archive member scope differs')
 for rel,pin in t['members'].items():
  path=extract/rel;need(path.resolve().is_relative_to(extract) and sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes'],'Saved output member differs')
 gate=read(extract/'bundle/GATE_RESULT.json');need(sha(extract/'bundle/GATE_RESULT.json')==t['gate_result_sha256']==linux['gate_result_sha256'],'Gate identity differs')
 need(gate['status']=='HYBRID_MISSING8_THREE_VIEW_PASS_NOT_ROOT_ADOPTED' and gate['package_sha256']==linux['package_sha256']==PACKAGE,'Gate source differs')
 need([x['id'] for x in gate['receipts']]==[x['id'] for x in linux['records']]==[x['id'] for x in audit['records']]==IDS,'Exact8 proof order differs')
 rows=[]
 for row,g,l,w in zip(m['records'],gate['receipts'],linux['records'],audit['records']):
  rid=row['id'];need(row['terminal_round']==70 and row['split']=='valid' and row['n_eval']==19867,'Nonterminal/test/subset')
  need(g['checkpoint_sha256']==l['checkpoint_sha256']==w['checkpoint_sha256']==row['identity']['checkpoint']['sha256'],'Different checkpoint')
  need(g['weights_before']==g['weights_after'] and l['root_receipt_exact'] and l['cached_root_fit_exact'] and l['saved_predictions_metrics_counts_exact'],'Linux/weight check incomplete')
  need(w['saved_fit_payload_hashes_exact'] and w['saved_predictions_metrics_counts_exact'] and w['root_valid_ID_partition_exact'] and w['fit_calls']==0,'Windows saved checks incomplete')
  need(g['prediction_arrays_sha256']==l['array_sha256']==w['array_sha256']==t['members']['bundle/'+rid+'/validation_predictions.npz']['sha256'],'Array identity differs')
  need(l['receipt_sha256']==w['receipt_sha256']==t['members']['bundle/'+rid+'/receipt.json']['sha256'],'Receipt identity differs')
  need(g['native_comparison']==l['native_comparison']==w['native_comparison'] and g['native_comparison']['accepted'],'Native result differs')
  need(g['native_comparison']['tolerance']==1e-12 and g['native_comparison']['max_abs_difference']<=1e-12 and g['native_comparison']['expected']==row['identity']['prior_validation_metrics'],'Native tolerance/source metrics differ')
  need(w['root_receipt_differences']==inputs['exact_root_receipt_differences'][rid] and w['local_full_root_receipt_exact']==(not w['root_receipt_differences']),'Root audit differences outside reviewed exact values')
  rows.append({'id':rid,'method':'CosineFairnessHybrid','distribution':row['distribution'],'attack':row['attack'],'seed':row['seed'],'checkpoint_sha256':g['checkpoint_sha256'],'array_sha256':w['array_sha256'],'receipt_sha256':w['receipt_sha256'],'native_comparison':w['native_comparison'],'native_max_abs_difference':w['native_comparison']['max_abs_difference'],'Windows_full_root_receipt_exact':not w['root_receipt_differences'],'preserved_root_audit_differences':w['root_receipt_differences']})
 for row in rows:row.update(phase='fullcoverage',selection_seed=False,source_role='actual_root_adopted_exact8')
 screen_rows,screen_transport=check_screen(inputs,volume);rows.extend(screen_rows)
 result={'status':'ROOT_HYBRID_FIXED8_PLUS_SCREEN1_COMPLEMENTARY_EVIDENCE_ADOPTED_PRIOR_INTERFACE1_REUSED_WINDOWS_FAILURE_PRESERVED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_adoption':True,'new_records':rows,'records':rows,'prior_interface_explicitly_reused':prior,'new_three_view_records_accepted':9,'formal_new_records':8,'screen_new_records':1,'Hybrid_total_three_view_records':10,'exact_ids':IDS,'accepted_ids':[SCREEN_ID,PRIOR_ID]+IDS,'screen91001_accepted_by_this_adoption':True,'Linux_whole_original_saved_check_pass':True,'Linux_root_fit_verified':True,'Linux_original_root_refit_records_new':9,'Windows_saved_outputs_audit_pass':True,'Windows_saved_outputs_audit_fit_calls':0,'Windows_new9_refit_executed':False,'Windows_whole_saved_check_pass':False,'Windows_exact_recalibration_claimed':False,'original_Windows_pre_record_failure_sha256':sha(F/'WINDOWS_SAVED_OUTPUT_CHECK.failure.json'),'original_Windows_failed_completed_records':0,'original_failure_preserved':True,'Windows_runtime_wrapper_sha256':sha(W/'windows_runtime_wrapper.py'),'Windows_runtime_wrapper_source_review_sha256':sha(W/'WINDOWS_REPAIR_SOURCE_REVIEW.json'),'prior_exact3_combined_Windows_whole_pass':False,'prior_exact3_failure_sha256':sha(prior_failure),'exact_diagnostic_map_sha256':sha(A/'ROOT_EXACT_DIAGNOSTIC_DIFFERENCES.json'),'exact_root_receipt_differences':inputs['exact_root_receipt_differences'],'screen_exact_root_receipt_differences':inputs['screen_exact_root_receipt_differences'],'no_tolerance_substitution':True,'cross_platform_bitwise_recalibration_claimed':False,'prior_root_sha256':sha(oldpath),'native_root_sha256':sha(nativepath),'proof_files':inputs['files'],'inputs_sha256':args.inputs_sha256,'adoption_source_sha256':sha(__file__),'metric_values_checked_new':81,'integer_base_counts_checked_new':216,'prediction_rules_checked_new':27,'counts_scope':'Fixed8 formal + fixed1 selected screen x original three views; linked original saved checks, not a new numerical checker','archive_sha256':t['archive_sha256'],'archive_members':20,'archives':[{'path':t['archive_path'],'sha256':t['archive_sha256'],'members':20},{'path':screen_transport['archive_path'],'sha256':screen_transport['archive_sha256'],'members':6}],'archive_members_total':26,'storage':volume,'mechanism_scope_modified':False,'new_fit':0,'new_CNN':0,'new_training':0,'test':False,'Benign10_complete':True,'scene_table_created':False,'full100_complete':False,'final_primary_endpoint_decided':False,'scope':'Fixed eight formal terminal-valid endpoints + one originally selected screen91001 + unchanged Hybrid91002 interface; ten unique IID Benign checkpoints. Linux whole supplies original root-fit evidence separately for 8+1; Windows audits saved outputs with zero fits and root-reviewed exact diagnostics. Original Windows import failure and historical exact3 FL failure remain preserved. Screen selection history is explicit; not full100, final test, or a scene performance table.'}
 with output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'new':9,'total':10,'root_sha256':sha(output)}))
if __name__=='__main__':main()
