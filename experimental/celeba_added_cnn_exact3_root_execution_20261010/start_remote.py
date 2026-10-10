
import base64,datetime,hashlib,json,pathlib,subprocess,sys
p=json.load(sys.stdin); base=pathlib.Path(p['base']); program=p['program']
assert str(base)=='/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010' and program=='guardfed_added_cnn_exact3_gate'
config=pathlib.Path('/etc/supervisor/conf.d')/(program+'.conf')
assert not config.exists() and not (base/'outputs/attempt001').exists() and not (base/'START_RECEIPT.json').exists(), 'Existing attempt; no blind retry'
assert hashlib.sha256((base/'source/FILES_SHA256.json').read_bytes()).hexdigest()==p['package']
assert hashlib.sha256((base/'ROOT_SOURCE_REVIEW.json').read_bytes()).hexdigest()==p['source_review_sha256']
for name,pin in p['files'].items():
 assert name in ('LINUX_PREFLIGHT.json','AUTHORIZATION.json')
 b=base64.b64decode(pin['base64'],validate=True); assert hashlib.sha256(b).hexdigest()==pin['sha256'] and len(b)==pin['bytes']
 assert not (base/name).exists(); (base/name).write_bytes(b)
pre=json.loads((base/'LINUX_PREFLIGHT.json').read_text()); age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(pre['utc'])).total_seconds()
assert 0<=age<=280, 'Preflight expired before supervisor launch'
assert hashlib.sha256(p['config'].encode()).hexdigest()==p['config_sha256']
(base/'execution').mkdir(exist_ok=False)
config.write_text(p['config'])
commands=[]
try:
 for argv in (['supervisorctl','reread'],['supervisorctl','add',program],['supervisorctl','start',program],['supervisorctl','status',program]):
  c=subprocess.run(argv,capture_output=True,text=True,timeout=25); commands.append({'argv':argv,'exit':c.returncode,'stdout':c.stdout,'stderr':c.stderr})
  assert c.returncode==0, commands[-1]
 r={'status':'ROOT_EXACT3_SUPERVISOR_STARTED_NOT_SCIENTIFIC_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'program':program,'commands':commands,'config_sha256':p['config_sha256'],'authorization_sha256':p['files']['AUTHORIZATION.json']['sha256'],'linux_preflight_sha256':p['files']['LINUX_PREFLIGHT.json']['sha256'],'package_sha256':p['package'],'autostart':False,'autorestart':False,'startretries':0,'new_scientific_acceptances':0}
except BaseException:
 import traceback
 r={'status':'ROOT_EXACT3_START_FAILURE_PRESERVED_NO_RETRY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':commands,'traceback':traceback.format_exc()}
(base/'START_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r))
if 'FAILURE' in r['status']: sys.exit(1)
