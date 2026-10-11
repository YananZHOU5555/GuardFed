from pathlib import Path
import hashlib,json,datetime
H=Path(__file__).resolve().parent;P=H.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def pin(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def put(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
assert sha(P/'FILES_SHA256.json')=='40fa2578e9247d3bc746262915d4f54da470edcb980e364aa5a749babf94fd9e'
for n,p in read(P/'FILES_SHA256.json')['files'].items():assert sha(P/n)==p['sha256']
audit=P/'actual/SAVED_AUDIT.json';a=read(audit)
assert sha(audit)=='1fa17fec204178873cc422e567bfda93d8e493bad9d93f5733492dab63057d71'
assert a['status']=='LOGO100_ALL100_CUDA_SAVED_PREDICTIONS_NO_REFIT_PASS_NOT_ROOT_ADOPTED'
ids=read(P/'SCOPE.json')['ids'];assert a['ids']==ids==[v['post_id'] for v in a['records']] and len(set(ids))==100
assert a['max_metric_error']<=1e-12 and all(v['max_metric_error']<=1e-12 for v in a['records'])
assert (a['prediction_items_checked'],a['metric_values_checked'],a['integer_confusion_counts_recomputed'])==(1986700,300,800)
assert read(H/'AUDIT_COMMAND.json')['exit_code']==read(H/'TRANSPORT_COMMAND.json')['exit_code']==0
t=read(P/'LOGO100_DELTA98_TRANSPORT_VERIFICATION.json');inp=read(H/'ACTUAL_AUDIT_INPUTS.json')
assert t['remote']['new_arrays']==98 and t['remote']['reused_arrays']==2 and len(t['remote']['reused_member_pins'])==33
handoff=dict(status='ACTUAL_LOGO100_DELTA98_TRANSPORT_AND_ALL100_ZERO_FIT_AUDIT_PASS_NOT_ROOT_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal=pin(P/'FILES_SHA256.json'),source_review=inp['source_review'],transport=pin(P/'LOGO100_DELTA98_TRANSPORT_VERIFICATION.json'),audit=pin(audit),transport_inputs=pin(H/'ACTUAL_TRANSPORT_INPUTS.json'),audit_inputs=pin(H/'ACTUAL_AUDIT_INPUTS.json'),transport_command=pin(H/'TRANSPORT_COMMAND.json'),audit_command=pin(H/'AUDIT_COMMAND.json'),fresh_all100_binding=pin(H/'ACTUAL100_BINDING.json'),archive={k:t[k] for k in ['archive_path','archive_sha256','archive_bytes','archive_members','payload_members','member_index_sha256','verified_extract']},counts=dict(records=100,old_root_records_deduplicated=2,new_records=98,predictions=1986700,metrics=300,primitive_confusion_counts=800,new_scores_npz=98,reused_payload_members=33),source_members_unchanged=True,max_metric_error=a['max_metric_error'],constant_predictions_preserved=[r['post_id'] for r in a['records'] if r['constant_prediction'] is not None],new_CNN=0,new_fit=0,pickle_loads=0,test=False,root_adopted=False,offserver_whole_predict_reexecuted=False,historical_CPU_equality_required=False,local_wrapper_compile_rejection=pin(H/'WRAPPER_COMPILE_REJECTION.json'),wrapper_rejection_not_scientific_or_transport_failure=True,transport_executions=1,audit_executions=1,next='Root independently review and adopt100 while removing old2 overlap; no STATE/Git writes performed.')
put(H/'HANDOFF.json',handoff)
files={str(p.relative_to(P)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(H.rglob('*')) if p.is_file()}
for p in [audit,P/'actual/STARTED.json',P/'LOGO100_DELTA98_TRANSPORT_VERIFICATION.json',P/'LOGO100_DELTA98_BACKUP_COMMAND.json',P/'LOGO100_DELTA98_BACKUP.stdout',P/'LOGO100_DELTA98_BACKUP.stderr',P/'LOGO100_DELTA98_SCP_COMMAND.json',P/'LOGO100_DELTA98_SCP.stdout',P/'LOGO100_DELTA98_SCP.stderr']:files[str(p.relative_to(P)).replace('\\','/')]=dict(sha256=sha(p),bytes=p.stat().st_size)
put(H/'ACTUAL_FILES_SHA256.json',dict(status='ACTUAL_METADATA_SEAL_NOT_SCIENTIFIC_ADOPTION',files=files))
print(json.dumps({'handoff':pin(H/'HANDOFF.json'),'actual_seal':pin(H/'ACTUAL_FILES_SHA256.json'),'audit':pin(audit),'audit_inputs':pin(H/'ACTUAL_AUDIT_INPUTS.json'),'constant_predictors':len(handoff['constant_predictions_preserved']),'max_metric_error':a['max_metric_error']},ensure_ascii=False))
