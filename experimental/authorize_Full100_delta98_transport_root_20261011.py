"""Bind actual completed Full100 to reviewed transport; no transport execution here."""
from pathlib import Path
import datetime,hashlib,json
R=Path(__file__).resolve().parents[1]
H=R/'tmp/Full100_cuda_saved_closure_prepare_v2_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
op=H/'actual_metadata/OBSERVATION.json'
ip=H/'actual_metadata/ACTUAL_TRANSPORT_INPUTS.json'
rp=H/'ROOT_SOURCE_REVIEW.json'
assert sha(op)=='05d57edb3d8a1681a58459b113dd554c8ea883098e3c3a86e33b6235163f8e29'
assert sha(ip)=='edd9bc186c0cf73cbc380e67f17b736cddd5b310155522a56835939fe9c14dc3'
assert sha(rp)=='ded413319bd7e9198db7b1710eef992e3f7faaf812c51febea4602d9287aee33'
o=read(op);x=read(ip);scope=read(H/'SCOPE.json')
assert o['Full100_complete'] is True and o['complete_full_records']==100 and not o['missing_ids']
assert o['ordered_ids']==scope['ids']==x['fixed_ids']
assert set(o['records'])==set(scope['ids'])==set(x['accepted_sha256'])
assert x['execution_authorized'] is False and x['transport_or_audit_executed'] is False
assert x['source_review']==dict(path=str(rp),sha256=sha(rp)) and read(rp)['source_adoptable']
x['execution_authorized']=True
x['root_authorized_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
x['single_transport_only']=True
x['CPU110_exclusive_fresh_gate_required']=True
p=H/'actual_metadata/ROOT_AUTHORIZED_TRANSPORT_INPUTS.json'
with p.open('x',encoding='utf8',newline='\n') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps({'path':str(p),'sha256':sha(p),'execution_authorized':True,'transport_executed':False}))
