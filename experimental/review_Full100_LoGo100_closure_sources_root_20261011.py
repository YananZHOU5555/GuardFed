"""Root metadata/source review only; no F, SSH, arrays or scientific execution."""
from pathlib import Path
import ast, base64, datetime, hashlib, json
R=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def put(p,v):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
def verify_seal(h,expected):
    assert sha(h/'FILES_SHA256.json')==expected
    files=read(h/'FILES_SHA256.json')['files']
    for n,v in files.items():
        p=h/n;assert p.resolve().is_relative_to(h.resolve()) and sha(p)==v['sha256'] and p.stat().st_size==v['bytes']
    return len(files)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
H=R/'tmp/Full100_cuda_saved_closure_prepare_v2_20261011'
seal='e6f6bd5ddedb3b1aa093d6de5b69e377b08cb10701a92dc66eb5b2db7246ed5b'
n=verify_seal(H,seal)
review=R/'tmp/Full100_cuda_saved_closure_independent_review_20261011/REVIEW.json'
assert sha(review)=='4d344a031ce4ef881a94dc9eefc464ec004f7aed9c9d0aa8b74c99aed1c4503d'
assert read(review)['source_adoptable'] and read(review)['source_seal_sha256']==seal
der=read(H/'V2_DERIVATION.json');old=R/'tmp/Full100_cuda_saved_closure_prepare_20261011/collect_remote.py'
assert base64.b64decode(der['inverse_parent_bytes_base64'])==old.read_bytes()
for name in ['audit_saved.py','transport_once.py','SCOPE.json']:
    assert (H/name).read_bytes()==(old.parent/name).read_bytes()
for name,pin in read(H/'SOURCE_PINS.json').items():
    assert sha(R/pin['path'])==pin['sha256']
    if name.endswith('.txt'):assert (H/name).read_bytes()==(R/pin['path']).read_bytes()
scope=read(H/'SCOPE.json')
assert len(scope['ids'])==len(set(scope['ids']))==100 and len(scope['delta_ids'])==98 and len(scope['reused_adopted_ids'])==2
assert set(scope['delta_ids']).isdisjoint(scope['reused_adopted_ids'])
assert set(scope['delta_ids'])|set(scope['reused_adopted_ids'])==set(scope['ids'])
assert [len(g['full_ids']) for g in scope['groups']]==[5,47,48]
put(H/'ROOT_SOURCE_REVIEW.json',dict(status='ROOT_SOURCE_REVIEW_FULL100_V2_PASS_RUNTIME_PENDING',utc=now,source_adoptable=True,source_seal_sha256=seal,sealed_members_checked=n,independent_review=dict(path=str(review),sha256=sha(review)),v2_metadata_alias_only=True,original_science_and_transport_unchanged=True,fixed_full_records=100,new_arrays=98,reused_adopted=2,whole500_completion_not_required=True,exact_parent_success_required=True,actual_transport=False,actual_audit=False,scientific_adoption=False,test=False))
H=R/'tmp/LoGo100_cuda100_saved_closure_prepare_20261011'
seal='40fa2578e9247d3bc746262915d4f54da470edcb980e364aa5a749babf94fd9e'
n=verify_seal(H,seal)
review=R/'tmp/LoGo100_cuda100_saved_closure_independent_review_20261011/SOURCE_REVIEW.json'
assert sha(review)=='14d3975c479c64ccbec2f1b916c13135601b964369de26b83074f8720a9e7d0a'
assert read(review)['source_adoptable'] and read(review)['source_seal_sha256']==seal
inverses=[]
for p in H.glob('*.INVERSE.json'):
    v=read(p);new=H/p.name.removesuffix('.INVERSE.json');raw=new.read_text('utf8')
    assert sha(new)==v['derived_sha256'] and sha(v['parent'])==v['parent_sha256']
    for span in reversed(v['spans']):
        assert raw.count(span['after'])==span['count'];raw=raw.replace(span['after'],span['before'])
    assert raw.encode()==Path(v['parent']).read_bytes();inverses.append(new.name)
assert len(inverses)==5
for name in ['original_metric_functions.py.txt','original_prefix_functions.py.txt','original_saved_science.py.txt']:
    p=H/name
    if p.exists():assert p.read_bytes()==(R/'tmp/LoGo_cuda_fixed2_saved_audit_prepare_v2_20261011'/name).read_bytes()
gate=ast.parse((H/'metadata_gate.py').read_text('utf8'))
assigned={n.targets[0].id:ast.literal_eval(n.value) for n in gate.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['IDS','CANARY_IDS']}
assert len(assigned['IDS'])==len(set(assigned['IDS']))==100 and len(assigned['CANARY_IDS'])==2
put(H/'ROOT_SOURCE_REVIEW.json',dict(status='ROOT_SOURCE_REVIEW_LOGO100_DELTA98_PASS_RUNTIME_PENDING',utc=now,source_adoptable=True,source_seal_sha256=seal,sealed_members_checked=n,independent_review=dict(path=str(review),sha256=sha(review)),complete_raw_inverse_files=inverses,original_scientific_loop_unchanged=True,fixed_records=100,new_arrays=98,reused_adopted=2,prior33_F_members_direct_reference=True,both_phases_COMPLETE_and_normal_exit_required=True,actual_transport=False,actual_audit=False,scientific_adoption=False,new_fit=0,new_CNN=0,test=False))
print(json.dumps({'Full_source_review_sha256':sha(R/'tmp/Full100_cuda_saved_closure_prepare_v2_20261011/ROOT_SOURCE_REVIEW.json'),'LoGo_source_review_sha256':sha(H/'ROOT_SOURCE_REVIEW.json')}))
