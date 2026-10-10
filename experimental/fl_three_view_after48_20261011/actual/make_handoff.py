"""Compact actual-proof pins and unexecuted root-join handoff, no scientific work."""
from pathlib import Path
import datetime,hashlib,json
H=Path(__file__).resolve().parent;C=H.parent;R=C.parents[1];S=C/'saved';A=H/'saved001'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':p.relative_to(R).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def save(p,j):
 with p.open('x',encoding='utf8') as f:json.dump(j,f,indent=2,allow_nan=False);f.write('\n')
audit=read(A/'SAVED_OUTPUTS_AUDIT_NO_REFIT.json');linux=read(A/'LINUX_SAVED_CHECK.json');t=read(A/'TRANSPORT_VERIFICATION.json')
ids=read(C/'MANIFEST.json')['exact_ids']
assert audit['status']=='SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED' and audit['fit_calls']==0
assert [x['id'] for x in audit['records']]==[x['id'] for x in linux['records']]==ids and len(ids)==13
files={role:pin(p) for role,p in {
 'candidate_seal':C/'FILES_SHA256.json','candidate_manifest':C/'MANIFEST.json','candidate_root_review':C/'ROOT_SOURCE_REVIEW.json',
 'saved_static_seal':S/'STATIC_SOURCE_SHA256.json','saved_root_review':S/'ROOT_SOURCE_REVIEW.json','audit_source_seal':S/'FILES_SHA256.json','audit_source_pins':S/'SOURCE_PINS.json',
 'Linux_original_whole':A/'LINUX_SAVED_CHECK.json','F_transport':A/'TRANSPORT_VERIFICATION.json','Windows_zero_fit_audit':A/'SAVED_OUTPUTS_AUDIT_NO_REFIT.json',
 'Linux_command':A/'LINUX_CHECK_COMMAND.json','Linux_exit':A/'LINUX_CHECK_EXIT.json','transport_command':A/'TRANSPORT_COMMAND.json','transport_exit':A/'TRANSPORT_SSH_EXIT.json',
 'Windows_command':A/'WINDOWS_AUDIT_COMMAND.json','Windows_exit':A/'WINDOWS_AUDIT_EXIT.json',
 'prior48_root':R/'tmp/celeba_flgmm_closed47_root_execution_20261011/ROOT_SCIENTIFIC_ADOPTION.json',
 'native57_root':R/'tmp/fl_native_after54_20261011/ROOT_ADOPTION_REVIEW.json',
 'prior_Windows47_failure':R/'tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001/OFFSERVER_ARRAY_REFIT_CHECK.failure.json',
}.items()}
assert read(A/'LINUX_CHECK_EXIT.json')['exit']==read(A/'TRANSPORT_SSH_EXIT.json')['exit']==read(A/'WINDOWS_AUDIT_EXIT.json')['exit']==0
inputs={'scope':'Actual exact13 complementary proof join after unchanged48; source prepared, root not yet executed','exact_ids':ids,'files':files,'adoption_source_sha256':sha(C/'adopt_root.py'),'exact_root_receipt_differences':{x['id']:x['root_receipt_differences'] for x in audit['records']}}
save(C/'ROOT_INPUTS.json',inputs)
differences={x['id']:x['root_receipt_differences'] for x in audit['records'] if x['root_receipt_differences']}
report={'status':'EXACT13_LINUX_WHOLE_F30_WINDOWS_ZERO_FIT_PASS_ROOT_ADOPTION_PENDING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prior_accepted':48,'new_verified_not_yet_root_adopted':13,'potential_total_after_root_adoption':61,'exact_ids':ids,'Linux_whole_root_refits_actual':13,'Windows_fit_calls_actual':0,'saved_metric_values':117,'saved_integer_base_counts':312,'saved_prediction_rules':39,'native_max_abs_difference':max(x['native_comparison']['max_abs_difference'] for x in audit['records']),'root_receipt_differences':differences,'proofs':files,'root_inputs_sha256':sha(C/'ROOT_INPUTS.json'),'root_join_source_sha256':sha(C/'adopt_root.py'),'root_join_executed':False,'root_command':'python -B tmp/fl_three_view_after48_20261011/adopt_root.py --inputs-sha256 '+sha(C/'ROOT_INPUTS.json')+' --allow-complementary-root-adoption','archive':{'path':t['archive_path'],'sha256':t['archive_sha256'],'members':t['member_count']},'original_Windows47_refit_failed_preserved':True,'Windows_new13_refit_not_executed':True,'cross_platform_bitwise_recalibration_claimed':False,'new_training':0,'test':False,'STATE_modified':False,'Git_modified':False}
save(C/'HANDOFF.json',report)
print(json.dumps({k:report[k] for k in ['status','native_max_abs_difference','root_receipt_differences','root_inputs_sha256','root_join_source_sha256','root_command']}))
