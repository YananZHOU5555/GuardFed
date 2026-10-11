from pathlib import Path
import hashlib,json,datetime
H=Path(__file__).resolve().parent;P=H.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def pin(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def put(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
assert sha(P/'FILES_SHA256.json')=='eb08186667ef3ca700cd853aa4dfc919181c86f080b89e5b2b1484203051e3a3'
for n,p in read(P/'FILES_SHA256.json')['files'].items():assert sha(P/n)==p['sha256']
audit=P/'actual/SAVED_AUDIT.json';a=read(audit)
assert sha(audit)=='41364281d5735fe0b10bb50ab4b9309766a82420c0f8c53aeda63f26a7d134e6'
assert a['status']=='LOGO_FIXED2_SAVED_PREDICTIONS_NO_REFIT_PASS_NOT_ROOT_ADOPTED'
assert len(a['records'])==2 and all(v['max_metric_error']==0 for v in a['records'])
assert read(H/'AUDIT_COMMAND.json')['exit_code']==read(H/'TRANSPORT_COMMAND.json')['exit_code']==0
t=read(P/'LOGO_FIXED2_TRANSPORT_VERIFICATION.json')
handoff=dict(status='ACTUAL_LOGO_FIXED2_TRANSPORT_AND_ZERO_FIT_AUDIT_PASS_NOT_ROOT_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal=pin(P/'FILES_SHA256.json'),source_review=pin(P/'ROOT_SOURCE_REVIEW.json'),transport=pin(P/'LOGO_FIXED2_TRANSPORT_VERIFICATION.json'),audit=pin(audit),transport_inputs=pin(H/'ACTUAL_TRANSPORT_INPUTS.json'),audit_inputs=pin(H/'ACTUAL_AUDIT_INPUTS.json'),transport_command=pin(H/'TRANSPORT_COMMAND.json'),audit_command=pin(H/'AUDIT_COMMAND.json'),fresh_observation=pin(H/'OBSERVATION.json'),archive={k:t[k] for k in ['archive_path','archive_sha256','archive_bytes','archive_members','payload_members','member_index_sha256','verified_extract']},counts=dict(records=2,predictions=39734,metrics=6,primitive_confusion_counts=16,new_scores_npz=2),source_members_unchanged=True,max_metric_error=0.0,constant_predictions_preserved=[r['post_id'] for r in a['records'] if r['constant_prediction'] is not None],new_CNN=0,new_fit=0,pickle_loads=0,test=False,root_adopted=False,remaining98_started=False,offserver_whole_predict_reexecuted=False,historical_CPU_equality_required=False,prior_v1_never_executed_preserved=True,next='Root independently review and adopt these two; root alone gates remaining98.')
put(H/'HANDOFF.json',handoff)
files={str(p.relative_to(P)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(H.iterdir()) if p.is_file()}
for p in [audit,P/'actual/STARTED.json',P/'LOGO_FIXED2_TRANSPORT_VERIFICATION.json',P/'LOGO_FIXED2_BACKUP_COMMAND.json',P/'LOGO_FIXED2_BACKUP.stdout',P/'LOGO_FIXED2_BACKUP.stderr',P/'LOGO_FIXED2_SCP_COMMAND.json',P/'LOGO_FIXED2_SCP.stdout',P/'LOGO_FIXED2_SCP.stderr']:files[str(p.relative_to(P)).replace('\\','/')]=dict(sha256=sha(p),bytes=p.stat().st_size)
put(H/'ACTUAL_FILES_SHA256.json',dict(status='ACTUAL_METADATA_SEAL_NOT_SCIENTIFIC_ADOPTION',files=files))
print(json.dumps({'handoff':pin(H/'HANDOFF.json'),'actual_seal':pin(H/'ACTUAL_FILES_SHA256.json'),'audit':pin(audit)},ensure_ascii=False))
