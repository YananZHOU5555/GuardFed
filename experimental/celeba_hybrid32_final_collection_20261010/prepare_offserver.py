"""Bind the accepted original member/tensor/record readers to one actual final5 archive."""
from pathlib import Path
import difflib, hashlib, json, runpy
B=Path(__file__).resolve().parent/'attempt_v2'
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'tmp/celeba_hybrid_delta_after23_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(path,value):
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def adapt(parent,target,replacements):
    old=parent.read_text('utf8');text=old
    for a,b in replacements:assert a in text,(parent,a);text=text.replace(a,b)
    reverse=text
    for a,b in reversed(replacements):assert b in reverse;reverse=reverse.replace(b,a)
    assert reverse==old
    compile(text,str(target),'exec')
    with target.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    with (B/(target.name+'.diff')).open('x',encoding='utf8',newline='\n') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile=str(parent),tofile=str(target))))
    return dict(parent=str(parent),parent_sha256=sha(parent),current_sha256=sha(target),replacements=replacements,reverse_metadata_bindings_byte_exact=True)
backup=read(B/'BACKUP_SHA256.json');old=read(P/'BACKUP_SHA256.json')
F=Path(read(B/'RAW_STORAGE_LOCATION.json')['directory'])
assert len(backup['accepted_new_ids'])==5 and backup['member_count']==71
assert sha(F/'hybrid_final5_delta.tar.gz')==backup['archive_sha256']
volume=runpy.run_path(str(ROOT/'tmp/guardfed_local_storage.py'))['check_bulk_storage'](sum(r['size'] for r in read(B/'MEMBERS.json')['members'].values())+4*1024**2)
proof={}
proof['member_tensor']=adapt(P/'verify_offserver.py',B/'verify_offserver.py',[
    ('celeba_hybrid_delta_after23_20261010','celeba_hybrid32_final_collection_20261010'),
    ('hybrid_after23_delta.tar.gz','hybrid_final5_delta.tar.gz'),(old['archive_sha256'],backup['archive_sha256']),
    ("'old_accepted':23","'old_accepted':27"),('==59','==71'),("'member_count':59","'member_count':71")])
bridge=B/'local_record_bridge_v2_verified';bridge.mkdir()
proof['record_bridge']=adapt(P/'local_record_bridge_v2_verified/bridge.py',bridge/'bridge.py',[
    ('celeba_hybrid_delta_after23_20261010','celeba_hybrid32_final_collection_20261010'),
    ('hybrid_after23_delta.tar.gz','hybrid_final5_delta.tar.gz'),
    *[(old[k],backup[k]) for k in ('archive_sha256','acceptance_sha256','inventory_sha256')],
    ("r['CPU']==[106]","r['CPU']==[107]")])
for name in ('checked_record_body.py','SOURCE_REUSE.json'):
    with (bridge/name).open('xb') as f:f.write((P/'local_record_bridge_v2_verified'/name).read_bytes())
    assert sha(bridge/name)==sha(P/'local_record_bridge_v2_verified'/name)
save(bridge/'FILES_SHA256.json',dict(members=[dict(path=name,sha256=sha(bridge/name),size=(bridge/name).stat().st_size) for name in ('bridge.py','checked_record_body.py','SOURCE_REUSE.json')]))
proof['record_runner']=adapt(P/'run_record_checks_v2.py',B/'run_record_checks.py',[
    ('celeba_hybrid_delta_after23_20261010','celeba_hybrid32_final_collection_20261010'),
    ("len(proof['records'])==len(expected)==4","len(proof['records'])==len(expected)==5")])
with (B/'verify_once.py').open('xb') as f:f.write((P/'verify_once.py').read_bytes())
save(B/'SOURCE_REUSE_VERIFICATION.json',dict(status='ORIGINAL_STRICT_MEMBER_TENSOR_RECORD_BODIES_UNCHANGED',offserver=proof,checked_record_body_bytes_exact=True,source_reuse_json_bytes_exact=True,source_members_verified=69,F_write_guard=volume,CPU107_metadata_rebinding_only=True))
print(json.dumps(dict(status='FINAL5_OFFSERVER_READERS_READY',archive_sha256=backup['archive_sha256'],archive_members=71,raw_directory=str(F))))
