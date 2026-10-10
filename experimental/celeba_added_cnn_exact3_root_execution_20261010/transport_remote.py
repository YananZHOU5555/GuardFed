
import datetime,hashlib,io,json,pathlib,subprocess,sys,zipfile
base=pathlib.Path('/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010')
out=base/'outputs/attempt001'
assert not (out/'FAILURE.json').exists()
gate=json.loads((out/'GATE_RESULT.json').read_text())
ids=['FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage','CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage','FedNGA_eta0.01_non-IID_Benign_seed91001_screen']
assert gate['status']=='EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED' and [r['id'] for r in gate['receipts']]==ids
assert gate['package_sha256']=='49c42b90214eae57d456f8ded1d2a4ff2de2762a78d6bcf5c4ae1a5766de9434'
status=subprocess.run(['supervisorctl','status','guardfed_added_cnn_exact3_gate'],capture_output=True,text=True)
assert 'EXITED' in status.stdout and 'RUNNING' not in status.stdout,status.stdout
files={'bundle/GATE_RESULT.json':out/'GATE_RESULT.json','bundle/metadata_receipt.json':out/'metadata_receipt.json','metadata.npz':pathlib.Path('/workspace/GuardFed-celeba-expanded/data/celeba/derived/rgb64_v1/metadata.npz')}
for ident in ids:
 for name in ['receipt.json','validation_predictions.npz']:files['bundle/'+ident+'/'+name]=out/ident/name
for name in ['stdout.log','stderr.log']:files['execution/'+name]=base/'execution'/name
observed={}; payload={}
for rel,path in files.items():
 assert path.is_file() and path.stat().st_size<10_000_000
 b=path.read_bytes();h=hashlib.sha256(b).hexdigest();payload[rel]=b
 observed[rel]={'sha256':h,'bytes':len(b),'server_path':str(path),'resolved_path':str(path.resolve())}
assert sum(len(b) for b in payload.values())<20_000_000
assert observed['metadata.npz']['sha256']=='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091'
for ident,receipt in zip(ids,gate['receipts']):
 assert json.loads(payload['bundle/'+ident+'/receipt.json'])==receipt
 assert receipt['status']=='NATIVE_VALID_REPLAY_PASS' and receipt['prediction_arrays_sha256']==observed['bundle/'+ident+'/validation_predictions.npz']['sha256']
for rel,path in files.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==observed[rel]['sha256']
report={'status':'EXACT3_SAVED_ARRAY_TRANSPORT_SOURCE_MEMBERS_UNCHANGED_NOT_SCIENTIFIC_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'members':observed,'exact_ids':ids,'service':status.stdout,'test':False,'new_CNN':0,'new_fit':0,'model_or_images_downloaded':False}
bio=io.BytesIO()
with zipfile.ZipFile(bio,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for rel,b in payload.items():z.writestr(rel,b)
 z.writestr('TRANSPORT_MANIFEST.json',json.dumps(report,indent=2)+'\n')
b=bio.getvalue();print(json.dumps({'archive_sha256':hashlib.sha256(b).hexdigest(),'archive_bytes':len(b),'members':len(payload)+1}),file=sys.stderr)
sys.stdout.buffer.write(b)
