"""Single authorized invocation of sealed Full100 transport, then original zero-fit audit."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,traceback
Q=Path(__file__).resolve().parent;H=Q.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SEAL='e6f6bd5ddedb3b1aa093d6de5b69e377b08cb10701a92dc66eb5b2db7246ed5b'
def put(n,v):
    with (Q/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def run(name,argv):
    put(name+'_COMMAND.json',{'argv':argv,'source_sha256':sha(H/argv[2]),'outer_source_sha256':sha(__file__),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'automatic_retry':False})
    result=None;error=None
    try:
        with (Q/(name+'.stdout')).open('xb') as out,(Q/(name+'.stderr')).open('xb') as err:result=subprocess.run(argv,cwd=H,stdout=out,stderr=err,timeout=900)
    except BaseException:
        error=traceback.format_exc();raise
    finally:
        put(name+'_EXIT.json',{'returncode':None if result is None else result.returncode,'exception':error,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stdout_sha256':sha(Q/(name+'.stdout')),'stderr_sha256':sha(Q/(name+'.stderr')),'automatic_retry':False})
    if result.returncode:raise RuntimeError(name+' failed; evidence preserved; stop without retry')
auth=Q/'ROOT_AUTHORIZED_TRANSPORT_INPUTS.json'
assert sha(auth)=='440989a8431921f64ee32cad20206c84b4791493e6803c70a08bacdb499cae4c'
assert sha(H/'FILES_SHA256.json')==SEAL
run('TRANSPORT',[sys.executable,'-B','transport_once.py','--root-inputs',str(auth),'--root-inputs-sha256',sha(auth),'--source-seal-sha256',SEAL])
tp=H/'ACTUAL_TRANSPORT_VERIFICATION.json';transport=json.loads(tp.read_bytes())
assert transport['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and transport['all_members_verified'] is True
index=json.loads((Path(transport['verified_extract'])/'ROOT_MEMBERS.json').read_bytes())
scope=json.loads((H/'SCOPE.json').read_bytes());rootauth=json.loads(auth.read_bytes())
assert index['exact_ids']==scope['ids'] and index['delta_ids']==scope['delta_ids'] and set(index['accepted_sha256'])==set(scope['ids'])
assert index['accepted_sha256']==rootauth['accepted_sha256']
inputs=json.loads((H/'ROOT_INPUTS.template.json').read_bytes())
inputs.update(root_authorized_audit=True,source_review=rootauth['source_review'],transport={'path':str(tp),'sha256':sha(tp)},archive_sha256=transport['archive_sha256'],accepted_sha256=index['accepted_sha256'])
put('ACTUAL_AUDIT_INPUTS.json',inputs)
run('AUDIT',[sys.executable,'-B','audit_saved.py','--inputs',str(Q/'ACTUAL_AUDIT_INPUTS.json'),'--inputs-sha256',sha(Q/'ACTUAL_AUDIT_INPUTS.json'),'--source-seal-sha256',SEAL])
audit=H/'actual/SAVED_OUTPUTS_AUDIT.json';o=json.loads(audit.read_bytes())
put('ACTUAL_COMPLETION.json',{'status':'TRANSPORT_AND_ORIGINAL_ZERO_FIT_AUDIT_EXIT0_ROOT_ADOPTION_PENDING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'transport':{'path':str(tp),'sha256':sha(tp)},'audit':{'path':str(audit),'sha256':sha(audit)},'archive':{'path':transport['archive_path'],'sha256':transport['archive_sha256'],'bytes':transport['archive_bytes'],'members':transport['archive_members']},'records':len(o['records']),'prior_adopted_reused':o['prior_adopted_reused'],'new_zero_fit_checked':o['new_zero_fit_checked'],'root_adopted':False,'STATE_Git_modified':False})
print((Q/'ACTUAL_COMPLETION.json').read_text(encoding='utf8'))
