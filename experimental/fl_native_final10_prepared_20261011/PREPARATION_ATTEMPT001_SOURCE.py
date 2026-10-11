"""Source-only final Sp-DFA10 derivation; no SSH, F access or scientific imports."""
from pathlib import Path
import ast, difflib, hashlib, json

H=Path(__file__).resolve().parent; R=H.parents[1]
P=R/'tmp/fl_native_after77_SDFA9_prepared_20261011'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(name, value):
    data=value if isinstance(value,bytes) else ((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode() if not isinstance(value,str) else value.encode())
    with (H/name).open('xb') as f:f.write(data)

latest=R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json'; prior=read(latest)
assert sha(latest)=='e1f0e65a0f842bb15b0d50b8ff3daae126d702e1abbbd87e74e673bdb1555884'
root=R/prior['root_adoption_path']; proof=read(root)
assert sha(root)==prior['root_adoption_sha256']=='904e3ee036dc4df66459f804db1903211b873ffb5de714ccb97f3900e39c824a'
assert prior['accepted_total']==proof['accepted_total']==len(proof['accepted_job_ids'])==86
oldscope=read(P/'FROZEN_SCOPE.json'); mp=R/oldscope['original_manifest_path']; manifest=read(mp)
assert sha(mp)==oldscope['original_manifest_sha256']=='09a4ddceb7f1d2a1d67280b7388aed959f76f66347fa5e2f1decb0996d08bb21'
jobs=[j for j in manifest['jobs'] if (j['distribution'],j['attack'])==('non-IID','Sp-DFA')]
ids=[j['id'] for j in jobs]
assert [j['seed'] for j in jobs]==list(range(91001,91011))
assert len(manifest['jobs'])==96 and len(manifest['reused_jobs'])==4
assert proof['accepted_job_ids']+ids==[j['id'] for j in manifest['jobs']]
for j in jobs:assert sha(mp.parent/'jobs'/j['job'])==j['job_sha256']
names=['run_once.py','collect_delta.py','preflight.py','close_observation.py','root_adopt.py']
parent_pins=read(P/'PREPARED_FILES_SHA256.json')['files']
for n in names:assert sha(P/n)==parent_pins[n]['sha256']
put('PREVIOUS_LATEST.json',latest.read_bytes())
put('PREVIOUS_ROOT_ADOPTION.json',root.read_bytes())

# Engineering-only service/queue gate, inlined into the two existing SSH readers.
# The original controller stores completed_new as a count, so the exact set comes
# from its frozen96 jobs plus each saved70-round terminal marker, not a fake field.
gate='''def service_contract(name, rc, state, exitstatus, worker_count, queue, expected_ids, completed_ids):
 assert name=='guardfed_celeba_flgmm_fullcoverage'
 assert not queue['failed']
 if state=='RUNNING':
  assert rc==0 and 1<=worker_count<=2
  return 'ORIGINAL_RUNNING_WITH_1_TO_2_STAGE_WORKERS'
 assert state=='EXITED' and rc==3 and exitstatus==0 and worker_count==0
 assert len(expected_ids)==len(set(expected_ids))==96
 assert queue['completed_new']==96 and queue['pending']==0 and not queue['active']
 assert len(completed_ids)==len(set(completed_ids))==96 and set(completed_ids)==set(expected_ids)
 return 'NORMAL_EXITED_ZERO_WORKER_EXACT96_TERMINAL_NOT_SCIENTIFIC_ACCEPTANCE'

def service_gate(service, workers, queue, manifest, stage):
 fields=service.stdout.split();assert len(fields)>=2
 name,state=fields[:2];completed=[];exitstatus=None
 expected=[item['id'] for item in manifest['jobs']]
 if state=='EXITED':
  # Query the existing Supervisor installation through supervisorctl's own
  # interpreter. Missing/mismatched RPC evidence fails closed; no restart.
  import shlex,shutil
  control=Path(shutil.which('supervisorctl'));shebang=control.read_text().splitlines()[0]
  assert shebang.startswith('#!')
  query="from supervisor.supervisorctl import ClientOptions; from supervisor.xmlrpc import SupervisorTransport; from xmlrpc.client import ServerProxy; import json; o=ClientOptions(); o.realize([]); p=ServerProxy('http://127.0.0.1',transport=SupervisorTransport(o.username,o.password,o.serverurl)).supervisor.getProcessInfo('guardfed_celeba_flgmm_fullcoverage'); print(json.dumps({k:p[k] for k in ('name','statename','pid','exitstatus','spawnerr')}))"
  info=json.loads(subprocess.check_output(shlex.split(shebang[2:])+['-c',query],text=True))
  assert info['name']==name and info['statename']=='EXITED' and info['pid']==0 and not info['spawnerr']
  exitstatus=info['exitstatus']
  for item in manifest['jobs']:
   out=stage/'runs'/item['id']
   assert all((out/n).is_file() for n in ('progress.json','result.json','acceptance.json','screen_identity.json','model.pt','job.json'))
   progress=read(out/'progress.json');accepted=read(out/'acceptance.json')
   assert progress['round']==accepted['rounds']==70
   assert (accepted['status'],accepted['evaluation_split'],accepted['train_rows'],accepted['evaluation_rows'])==('PASS','valid',162770,19867)
   assert sha(out/'job.json')==item['job_sha256']
   completed.append(item['id'])
 mode=service_contract(name,service.returncode,state,exitstatus,len(workers),queue,expected,completed)
 return dict(mode=mode,supervisor_status_returncode=service.returncode,supervisor_exitstatus=exitstatus,completed_ids_checked=completed,scientific_acceptance=False)

'''

diffs=[]; inverse={}; changes={}
for name in names:
    old=(P/name).read_bytes(); s=old.decode()
    common=[(repr(oldscope['ids']),repr(ids)),
            ('fl_native_after77_SDFA9_prepared_20261011',H.name),
            ('7c79ee6a2434b1a1c1df1977a9ac7091b386a78412c7ad838ed44c907c448ab0',prior['next_collector_previous_sha256']),
            ('5fdf135fd13665df6d528818db81765ad3567d5d754b18c538b1772f3994c80d',sha(root)),
            ('exact9 after77','exact10 after86'),('exact9 only','exact10 only'),('All nine authorized','All ten authorized'),
            ('frozen nine-ID','frozen ten-ID'),('frozen nine-ID','frozen ten-ID'),('EXACT9_AFTER_NATIVE77','EXACT10_AFTER_NATIVE86'),
            ('after77','after86'),('old77','old86'),('prior77','prior86'),('Parent77','Parent86')]
    for before,after in common:s=s.replace(before,after)
    if name=='run_once.py':
        for before,after in [("receipt['accepted_total']==86","receipt['accepted_total']==96"),("receipt['accepted_new']==9","receipt['accepted_new']==10"),
          ('{77+len(ids)}','{86+len(ids)}'),('total=77+n','total=86+n'),("prior['accepted_total']==77","prior['accepted_total']==86"),
          ("p['accepted_job_ids'][:77]","p['accepted_job_ids'][:86]"),('prior_accepted_new=77','prior_accepted_new=86'),
          ('accepted_before=77','accepted_before=86'),("a['prior_count']==77 and a['terminal_count']>=86","a['prior_count']==86 and a['terminal_count']==96")]:
            assert before in s,before;s=s.replace(before,after)
    if name=='root_adopt.py':
        for before,after in [('(9,86,96,4)','(10,96,96,4)'),("len(set(off['accepted_job_ids'])) == 86","len(set(off['accepted_job_ids'])) == 96"),
          ('accepted_before=77,accepted_new=9,accepted_total=86','accepted_before=86,accepted_new=10,accepted_total=96'),
          ('not_complete_fullcoverage_or_final_summary=True','native_coverage_complete=True,final_summary_generated=False'),
          ('accepted_total=86','accepted_total=96'),('LATEST_BACKUP.root86.tmp','LATEST_BACKUP.root96.tmp')]:
            assert before in s,before;s=s.replace(before,after)
    if name=='preflight.py':
        s=s.replace("sha=lambda p:",gate+"sha=lambda p:",1)
        before="service=subprocess.run(['supervisorctl','status','guardfed_celeba_flgmm_fullcoverage'],capture_output=True,text=True);assert service.stdout.split()[1]=='RUNNING'"
        assert before in s;s=s.replace(before,before.split(';assert')[0],1)
        s=s.replace("assert not list(stage.glob('*FAILURE*.json')) and 1<=len(workers)<=2", "assert not list(stage.glob('*FAILURE*.json'))",1)
        s=s.replace("q=read(stage/'queue_progress.json');active=", "q=read(stage/'queue_progress.json');service_check=service_gate(service,workers,q,manifest,stage);active=",1)
        s=s.replace('service=service.stdout,actual_workers=', 'service=service.stdout,service_terminal_gate=service_check,actual_workers=',1)
    if name=='close_observation.py':
        s=s.replace('import datetime,hashlib,json,os,subprocess','import datetime,hashlib,json,os,subprocess',1)
        s=s.replace('sha=lambda p:',"read=lambda p:json.loads(Path(p).read_bytes())\n"+gate+'sha=lambda p:',1)
        s=s.replace('busy=[];collectors=[]',"stage=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage')\nbusy=[];collectors=[];workers=[]",1)
        needle="  argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\\0')"
        assert needle in s;s=s.replace(needle,needle+"\n  if str(stage/'run_one.py') in argv:workers.append(dict(pid=int(p.name),argv=argv))",1)
        before="service=subprocess.check_output(['supervisorctl','status','guardfed_celeba_flgmm_fullcoverage'],text=True);assert service.split()[1]=='RUNNING' and not list(stage.glob('*FAILURE*.json'))"
        after="service=subprocess.run(['supervisorctl','status','guardfed_celeba_flgmm_fullcoverage'],capture_output=True,text=True);assert not list(stage.glob('*FAILURE*.json'))\nassert sha(stage/'PACKAGE_SHA256.json')=='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'\nmanifest=read(stage/'manifest.json');assert sha(stage/'manifest.json')==read(stage/'PACKAGE_SHA256.json')['files']['manifest.json']\nservice_check=service_gate(service,workers,q,manifest,stage)"
        assert before in s;s=s.replace(before,after,1)
        s=s.replace('service=service.strip(),queue=', 'service=service.stdout.strip(),service_terminal_gate=service_check,actual_workers=workers,queue=',1)
    new=s.encode();compile(new,str(H/name),'exec');put(name,new)
    delta=list(difflib.ndiff(old.decode().splitlines(keepends=True),s.splitlines(keepends=True)))
    assert ''.join(difflib.restore(delta,1)).encode()==old
    assert ''.join(difflib.restore(delta,2)).encode()==new
    inverse[name]=delta
    changes[name]={'parent_sha256':sha(P/name),'source_sha256':sha(H/name),'raw_inverse_parent_exact':True,'raw_forward_new_exact':True}
    diffs.extend(difflib.unified_diff(old.decode().splitlines(keepends=True),s.splitlines(keepends=True),fromfile='executed_SDFA9/'+name,tofile='final_SpDFA10/'+name))
put('SOURCE_DIFF.patch',''.join(diffs));put('SOURCE_INVERSE.json',inverse)
put('FROZEN_SCOPE.json',dict(status='PREPARED_FIXED_FINAL10_NOT_TERMINAL_OR_ACCEPTED_PROOF',ids=ids,jobs=jobs,original_manifest_path=oldscope['original_manifest_path'],original_manifest_sha256=sha(mp),source_package_sha256=oldscope['source_package_sha256'],prior_native_new=86,proposed_native_new_if_root_adopts=96,screen_reused_separately=4,proposed_native_plus_reused=100,prior_accepted_job_ids=proof['accepted_job_ids'],new_native_accepted_by_preparation=0,new_three_view_accepted=0))
put('PENDING_BINDING.json',dict(status='PENDING_ACTUAL_FINAL96_SNAPSHOT_AND_ROOT_EXECUTION_REVIEW',required_status='ROOT_AUTHORIZED_FIXED_EXACT10_AFTER_NATIVE86_NOT_ACCEPTANCE',snapshot_path=None,snapshot_sha256=None,snapshot_utc=None,terminal_count=None,prior_count=86,authorized_ids=ids,prior_root_path=prior['root_adoption_path'],prior_root_sha256=sha(root),prior_offserver_path=prior['next_collector_previous_path'],prior_offserver_sha256=prior['next_collector_previous_sha256'],previous_offserver_local_copy_pending=True,actual_root_execution_review_path=None,actual_root_execution_review_sha256=None,actual_archive_sha256=None,actual_offserver_sha256=None,actual_root_adoption_sha256=None,execution_authorized=False,scientific_execution_performed=False))
reuse=read(P/'SOURCE_REUSE.json')
reuse.update(thin_entry_sha256=sha(H/'collect_delta.py'),effective_collector_sha256=sha(H/'collect_delta.py'),sole_effective_change='Final fixed10 non-IID Sp-DFA seeds91001..10 after native86; engineering-only RUNNING-or-normal-EXITED gate in preflight/close.',predecessor_root_sha256=sha(root),predecessor_offserver_sha256=prior['next_collector_previous_sha256'],direct_parent_transport_path=(P/'run_once.py').relative_to(R).as_posix(),direct_parent_transport_sha256=sha(P/'run_once.py'),source_snapshot_sha256=None,source_snapshot_pending=True)
put('SOURCE_REUSE.json',reuse)
put('SOURCE_PARENT_PINS.json',dict(parent_directory=P.relative_to(R).as_posix(),parent_prepared_seal_sha256=sha(P/'PREPARED_FILES_SHA256.json'),files={(P/n).relative_to(R).as_posix():sha(P/n) for n in names},previous_latest_sha256=sha(latest),previous_root_path=prior['root_adoption_path'],previous_root_sha256=sha(root),previous_offserver_sha256=prior['next_collector_previous_sha256'],original_verifier_sha256=reuse['original_verifier_sha256'],source_changes=changes))
print(json.dumps(dict(status='FINAL10_SOURCE_DERIVED_ONLY',prior=86,proposed=96,reused_separately=4,new_accepted=0)))
