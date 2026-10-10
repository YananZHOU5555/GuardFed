
import datetime,hashlib,io,json,os,pathlib,subprocess,sys,zipfile
base=pathlib.Path('/workspace/guardfed_checks/celeba_hybrid_three_view_missing8_20261011')
out=base/'outputs/attempt001'
assert hashlib.sha256(pathlib.Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert hashlib.sha256((base/'source/FILES_SHA256.json').read_bytes()).hexdigest()=='f3b197ce786a3f391f820baf8767fa25fba2d2387002acf654bced7dc8e95239'
assert hashlib.sha256((base/'source/check_saved.py').read_bytes()).hexdigest()=='1410e344f07cb7d5987dcf773de4dab968bb3f03d21b378f59704e67074bd924'
assert not (out/'FAILURE.json').exists() and not list(out.rglob('FAILURE.json'))
assert hashlib.sha256((out/'GATE_RESULT.json').read_bytes()).hexdigest()=='__GATE_RESULT_SHA256__'
gate=json.loads((out/'GATE_RESULT.json').read_text())
ids=['CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91003_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91004_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91005_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91006_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91007_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91008_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91009_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91010_fullcoverage']
assert gate['status']=='HYBRID_MISSING8_THREE_VIEW_PASS_NOT_ROOT_ADOPTED' and [r['id'] for r in gate['receipts']]==ids
assert gate['package_sha256']=='f3b197ce786a3f391f820baf8767fa25fba2d2387002acf654bced7dc8e95239' and gate['root_adopted_new']==0 and gate['test'] is False
status=subprocess.run(['supervisorctl','status','guardfed_hybrid_missing8_valid'],capture_output=True,text=True)
assert status.returncode==3 and status.stdout.split()[:2]==['guardfed_hybrid_missing8_valid','EXITED'],status.stdout
fl_status=subprocess.run(['supervisorctl','status','guardfed_flgmm_FFlip10_capacity_pool32_valid'],capture_output=True,text=True)
assert fl_status.returncode==3 and fl_status.stdout.split()[:2]==['guardfed_flgmm_FFlip10_capacity_pool32_valid','EXITED'],fl_status.stdout
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 try:
  command=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  assert not any(pathlib.Path(token).name=='candidate.py' and any(ns in token for ns in ('celeba_hybrid_three_view_missing8_20261011','fl_FFlip10_capacity_pool32_20261011')) for token in command),'Live replay worker'
 except (FileNotFoundError,ProcessLookupError):pass
 for task in (proc/'task').glob('*'):
  try:
   affinity=set(os.sched_getaffinity(int(task.name)))
   assert not (len(affinity)<=32 and affinity.intersection(os.sched_getaffinity(0))),'Restricted-thread CPU overlap'
  except ProcessLookupError:pass
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.getpriority(os.PRIO_PROCESS,0)>=10 and os.sched_getaffinity(0)=={110}
proof=base/'LINUX_SAVED_CHECK.json'
assert hashlib.sha256(proof.read_bytes()).hexdigest()=='__LINUX_PROOF_SHA256__'
linux=json.loads(proof.read_text())
assert linux['status']=='HYBRID_MISSING8_LINUX_ORIGINAL_WHOLE_SAVED_PASS_NOT_ADOPTED'
assert len(linux['records'])==8 and [r['id'] for r in linux['records']]==ids
assert linux['original_check_saved_sha256']=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
assert all(r['root_receipt_exact'] and r['cached_root_fit_exact'] and r['saved_predictions_metrics_counts_exact'] for r in linux['records'])
assert linux['new_CNN']==0 and linux['new_training']==0 and linux['test'] is False and linux['cached_root_refits']==8
assert linux['gate_result_sha256']=='__GATE_RESULT_SHA256__' and linux['package_sha256']=='f3b197ce786a3f391f820baf8767fa25fba2d2387002acf654bced7dc8e95239'
files={'bundle/GATE_RESULT.json':out/'GATE_RESULT.json','bundle/metadata_receipt.json':out/'metadata_receipt.json','LINUX_SAVED_CHECK.json':proof}
for ident in ids:
 for name in ['receipt.json','validation_predictions.npz']:files['bundle/'+ident+'/'+name]=out/ident/name
observed={}; payload={}
for rel,path in files.items():
 assert path.is_file() and path.stat().st_size<10_000_000
 b=path.read_bytes();h=hashlib.sha256(b).hexdigest();payload[rel]=b
 observed[rel]={'sha256':h,'bytes':len(b),'server_path':str(path),'resolved_path':str(path.resolve())}
assert sum(len(b) for b in payload.values())<128_000_000
for ident,receipt in zip(ids,gate['receipts']):
 assert json.loads(payload['bundle/'+ident+'/receipt.json'])==receipt
 assert receipt['status']=='NATIVE_VALID_REPLAY_PASS' and receipt['prediction_arrays_sha256']==observed['bundle/'+ident+'/validation_predictions.npz']['sha256']
for row in linux['records']:
 assert row['receipt_sha256']==observed['bundle/'+row['id']+'/receipt.json']['sha256'] and row['array_sha256']==observed['bundle/'+row['id']+'/validation_predictions.npz']['sha256']
for rel,path in files.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==observed[rel]['sha256']
report={'status':'HYBRID_MISSING8_SAVED_ARRAY_TRANSPORT_SOURCE_MEMBERS_UNCHANGED_NOT_ROOT_ADOPTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'members':observed,'exact_ids':ids,'service':status.stdout,'test':False,'new_CNN':0,'new_fit':0,'model_or_images_downloaded':False}
bio=io.BytesIO()
with zipfile.ZipFile(bio,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for rel,b in payload.items():z.writestr(rel,b)
 z.writestr('TRANSPORT_MANIFEST.json',json.dumps(report,indent=2)+'\n')
b=bio.getvalue();print(json.dumps({'archive_sha256':hashlib.sha256(b).hexdigest(),'archive_bytes':len(b),'members':len(payload)+1}),file=sys.stderr)
sys.stdout.buffer.write(b)
