"""One read-only Full100 completion observation; no arrays, model or scientific execution."""
from pathlib import Path
import datetime,hashlib,json,subprocess,traceback
Q=Path(__file__).resolve().parent
H=Q.parent
R=H.parents[1]
sha=lambda b:hashlib.sha256(b).hexdigest()
def put(name,value):
    with (Q/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
scope=json.loads((H/'SCOPE.json').read_bytes())
code='SCOPE='+repr(scope)+'\n'+'''
from pathlib import Path
import datetime,hashlib,json,os
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(value,message):
 if not value:raise ValueError(message)
need(digest('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa','Guide changed')
siblings=set(map(int,Path('/sys/devices/system/cpu/cpu110/topology/thread_siblings_list').read_text().strip().split(',')))
affinity=set(os.sched_getaffinity(0))-siblings
need(affinity,'No eligible observation CPU after excluding CPU110/SMT')
os.sched_setaffinity(0,affinity)
groups=[];records={};missing=[]
for group in SCOPE['groups']:
 base=Path(group['base'])
 for key in ['cohort','deployment']:
  pin=group[key];need(digest(pin['path'])==pin['sha256'],'Producer '+key+' changed')
 fixed=group['queue_success_snapshot']
 snapshot_path=Path(fixed['path']) if fixed else base/'progress.json'
 raw=snapshot_path.read_bytes();snapshot=json.loads(raw)
 if fixed:need(hashlib.sha256(raw).hexdigest()==fixed['sha256'],'Historical parent snapshot changed')
 successful=snapshot[group['completed_field']]
 need(isinstance(successful,list) and all(isinstance(i,str) for i in successful) and len(successful)==len(set(successful)),'Malformed parent success list')
 selected=[i for i in group['full_ids'] if i in successful]
 active=snapshot.get('active')
 for identity in group['full_ids']:
  complete=identity in successful and active!=identity
  path=base/'tasks'/identity/'ACCEPTED.json'
  row={'producer':group['tag'],'parent_returncode0_success_list_contains_id':identity in successful,'selected_active':active==identity,'complete':complete,'accepted':None}
  if complete:
   a=json.loads(path.read_bytes())
   need(a['id']==identity and a['cohort']==group['cohort'] and a['root_offserver_adopted'] is False,'Wrong accepted identity')
   need(a['acceptance']['status']=='PER_RECORD_FULL_CUDA_SAVED_WHOLE_ACCEPTED','Missing saved whole acceptance')
   need(not (base/'tasks'/identity/'FAILURE.json').exists() and not (base/'runs'/(identity+'.formal_failure.json')).exists(),'Selected failure exists')
   actual=digest(path);expected=group['fixed_accepted'].get(identity)
   if expected:need(actual==expected,'Fixed historical ACCEPTED changed')
   row['accepted']={'path':str(path),'sha256':actual}
   row['receipt']=a['receipt'];row['arrays_reference_only']=a['arrays']
  else:missing.append(identity)
  records[identity]=row
 groups.append({'tag':group['tag'],'base':str(base),'cohort':group['cohort'],'deployment':group['deployment'],'snapshot':{'path':str(snapshot_path),'sha256':hashlib.sha256(raw).hexdigest()},'original_success_field':group['completed_field'],'original_success_ids':successful,'full_ids':group['full_ids'],'successful_selected_full_ids':selected,'active':active,'snapshot_utc':snapshot.get('utc'),'historical_parent_failure_retained':bool(fixed)})
need(set(records)==set(SCOPE['ids']) and len(records)==100,'Fixed100 grid changed')
print(json.dumps({'status':'READ_ONLY_FULL100_EXACT_ID_PARENT_SUCCESS_OBSERVATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'Full100_complete':not missing,'complete_full_records':100-len(missing),'missing_ids':missing,'ordered_ids':SCOPE['ids'],'groups':groups,'records':records,'scope_sha256':hashlib.sha256(json.dumps(SCOPE,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'CPU110_and_SMT_excluded':sorted(siblings),'affinity_excludes_CPU110_SMT':not (set(os.sched_getaffinity(0))&siblings),'arrays_models_read':False,'new_scientific_acceptance':0,'transport_authorized':False},allow_nan=False))
'''
payload=code.encode('utf8')
with (Q/'READ_ONLY_REMOTE.py').open('xb') as f:f.write(payload)
argv=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 ionice -c 3 python3 -B -']
started=datetime.datetime.now(datetime.timezone.utc).isoformat();result=None;error=None
try:
    with (Q/'OBSERVATION.stdout').open('xb') as out,(Q/'OBSERVATION.stderr').open('xb') as err:
        result=subprocess.run(argv,input=payload,stdout=out,stderr=err,timeout=120)
except BaseException:
    error=traceback.format_exc();raise
finally:
    put('OBSERVATION_COMMAND.json',{'argv':argv,'source_sha256':sha(Path(__file__).read_bytes()),'remote_source_sha256':sha(payload),'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':None if result is None else result.returncode,'exception':error,'stdout_sha256':sha((Q/'OBSERVATION.stdout').read_bytes()),'stderr_sha256':sha((Q/'OBSERVATION.stderr').read_bytes()),'automatic_retry':False,'science_executed':False,'transport_executed':False})
assert result.returncode==0,'Read-only observation failed; preserve evidence and stop'
o=json.loads((Q/'OBSERVATION.stdout').read_bytes());put('OBSERVATION.json',o)
review=H/'ROOT_SOURCE_REVIEW.json'
assert sha(review.read_bytes())=='ded413319bd7e9198db7b1710eef992e3f7faaf812c51febea4602d9287aee33'
assert sha((H/'FILES_SHA256.json').read_bytes())=='e6f6bd5ddedb3b1aa093d6de5b69e377b08cb10701a92dc66eb5b2db7246ed5b'
put('ACTUAL_TRANSPORT_INPUTS.json',{'execution_authorized':False,'source_review':{'path':str(review),'sha256':sha(review.read_bytes())},'actual_transport':None,'source_seal_sha256':'e6f6bd5ddedb3b1aa093d6de5b69e377b08cb10701a92dc66eb5b2db7246ed5b','completion_observation':{'path':str(Q/'OBSERVATION.json'),'sha256':sha((Q/'OBSERVATION.json').read_bytes())},'Full100_complete_observed':o['Full100_complete'],'fixed_ids':scope['ids'],'accepted_sha256':{i:r['accepted']['sha256'] if r['accepted'] else None for i,r in o['records'].items()},'transport_or_audit_executed':False})
print(json.dumps({'Full100_complete':o['Full100_complete'],'complete_full_records':o['complete_full_records'],'missing_ids':o['missing_ids'],'observation_sha256':sha((Q/'OBSERVATION.json').read_bytes()),'transport_inputs_sha256':sha((Q/'ACTUAL_TRANSPORT_INPUTS.json').read_bytes()),'execution_authorized':False}))
