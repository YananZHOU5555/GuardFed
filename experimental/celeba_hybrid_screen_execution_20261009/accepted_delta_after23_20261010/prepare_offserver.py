"""Only rebind original member/tensor/record verifier paths and actual receipt/cardinality pins."""
from pathlib import Path
import difflib,hashlib,json,sys
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent;ROOT=B.parents[1];H=ROOT/'tmp/celeba_hybrid_screen_execution_20261009';P=H/'accepted_delta_after22_20261010'
sys.path.insert(0,str(ROOT/'tmp'));from guardfed_local_storage import check_bulk_storage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,value):
    with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n')
def adapted(parent,target,replacements):
    original=parent.read_text('utf8');text=original
    for before,after in replacements:assert before in text,(parent,before);text=text.replace(before,after)
    reversed_text=text
    for before,after in reversed(replacements):assert after in reversed_text;reversed_text=reversed_text.replace(after,before)
    assert reversed_text==original
    compile(text,str(target),'exec')
    with target.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    with (B/(target.name+'.diff')).open('x',encoding='utf8',newline='\n') as f:f.write(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(parent),tofile=str(target))))
    return dict(parent=str(parent),parent_sha256=sha(parent),current_sha256=sha(target),replacements=replacements,reverse_bindings_byte_exact=True)
raw=read(B/'RAW_STORAGE_LOCATION.json');F=Path(raw['directory']);backup=read(B/'BACKUP_SHA256.json');old=read(P/'BACKUP_SHA256.json')
assert backup['member_count']==59 and len(backup['accepted_new_ids'])==4
assert sha(F/'hybrid_after23_delta.tar.gz')==backup['archive_sha256']
members=read(B/'MEMBERS.json')['members'];volume=check_bulk_storage(sum(v['size'] for v in members.values())+4*1024**2)
proof={}
proof['member_tensor']=adapted(P/'verify_offserver.py',B/'verify_offserver.py',[
('B=Path(__file__).resolve().parent;H=','B=Path('+repr(F.as_posix())+');H='),
('hybrid_after22_delta.tar.gz','hybrid_after23_delta.tar.gz'),(old['archive_sha256'],backup['archive_sha256']),
('==23','==59'),("'old_accepted':22","'old_accepted':23"),("'member_count':23","'member_count':59")])
bridge=B/'local_record_bridge_v2';bridge.mkdir()
proof['record_bridge']=adapted(P/'local_record_bridge_v2/bridge.py',bridge/'bridge.py',[
('HERE=Path(__file__).resolve().parent;B=HERE.parent;H=','HERE=Path(__file__).resolve().parent;B=Path('+repr(F.as_posix())+');H='),
('hybrid_after22_delta.tar.gz','hybrid_after23_delta.tar.gz'),
*[(old[k],backup[k]) for k in ('archive_sha256','acceptance_sha256','inventory_sha256')]])
for name in ('checked_record_body.py','SOURCE_REUSE.json'):
    with (bridge/name).open('xb') as f:f.write((P/'local_record_bridge_v2'/name).read_bytes())
    assert sha(bridge/name)==sha(P/'local_record_bridge_v2'/name)
with (bridge/'FILES_SHA256.json').open('x',encoding='utf8',newline='\n') as f:json.dump(dict(members=[dict(path=p.name,sha256=sha(p),size=p.stat().st_size) for p in sorted(bridge.iterdir())]),f,indent=2);f.write('\n')
proof['record_runner']=adapted(P/'run_record_checks.py',B/'run_record_checks.py',[
("B=Path(__file__).resolve().parent;bridge_dir=B/'local_record_bridge_v2';","B=Path("+repr(F.as_posix())+");bridge_dir=Path(__file__).resolve().parent/'local_record_bridge_v2';"),
("len(proof['records'])==len(expected)==1","len(proof['records'])==len(expected)==4")])
with (B/'verify_once.py').open('xb') as f:f.write((P/'verify_once.py').read_bytes())
save('SOURCE_REUSE_VERIFICATION.json',dict(status='ORIGINAL_COLLECTOR_STRICT_RECORD_AND_TENSOR_BODY_UNCHANGED',collector=read(B/'SOURCE_RECEIPT.json'),offserver=proof,checked_record_body_bytes_exact=True,source_reuse_json_bytes_exact=True,source_members_verified=69,F_write_guard=volume,scope='Only frozen exact4 metadata/count/namespace/actual pins and F paths rebound; no scientific formula, CUDA metadata path or failure rule changed.'))
print(json.dumps(dict(status='ORIGINAL_VERIFIERS_BOUND_TO_F_NOT_EXECUTED',archive_sha256=backup['archive_sha256'],member_count=backup['member_count'],F_directory=str(F))))
