
import base64,datetime,hashlib,json,pathlib,sys
p=json.load(sys.stdin); base=pathlib.Path(p['base'])
assert str(base)=='/workspace/guardfed_checks/fl_three_view_after48_20261011'
assert not base.exists(), 'Fresh deployment path required; preserve existing attempt'
checked={}
for rel,pin in p['members'].items():
 target=base/rel
 assert target.is_relative_to(base) and '..' not in pathlib.PurePosixPath(rel).parts
 b=base64.b64decode(pin['base64'],validate=True)
 assert hashlib.sha256(b).hexdigest()==pin['sha256'] and len(b)==pin['bytes']
 checked[rel]=(target,b,pin)
base.mkdir(parents=True,exist_ok=False)
for rel,(target,b,pin) in checked.items():
 target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(b)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==pin['sha256']
r={'status':'ROOT_SMALL_SOURCE_DEPLOYED_BYTES_VERIFIED_NOT_STARTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':str(base),'members':{r:{'sha256':v[2]['sha256'],'bytes':v[2]['bytes']} for r,v in checked.items()},'new_inference':0,'new_fit':0,'new_training':0}
(base/'SOURCE_DEPLOYMENT.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
