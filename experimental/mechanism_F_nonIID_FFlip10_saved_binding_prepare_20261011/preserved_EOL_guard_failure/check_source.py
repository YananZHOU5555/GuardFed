"""Static source checks and synthetic JSON metadata only; no science imports."""
from pathlib import Path
import ast,difflib,hashlib,importlib.util,json,textwrap
H=Path(__file__).resolve().parent;R=H.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name,value):
    (H/name).write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))

adapt=read(H/'SOURCE_ADAPTATIONS.json');checks={};diff=[]
for name,binding in adapt.items():
    parent=R/binding['parent_path'];before=parent.read_bytes();after=(H/name).read_bytes()
    assert sha(parent)==binding['parent_sha256']
    compile(after,str(H/name),'exec')
    spans=[dict(old_start=a,old_end=b,new_start=c,new_end=d,old_hex=before[a:b].hex(),new_hex=after[c:d].hex())
        for tag,a,b,c,d in difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if tag!='equal']
    reverse=after
    for span in reversed(spans):
        a,b=span['new_start'],span['new_end'];assert reverse[a:b].hex()==span['new_hex']
        reverse=reverse[:a]+bytes.fromhex(span['old_hex'])+reverse[b:]
    assert reverse==before
    binding.update(source_sha256=sha(H/name),raw_inverse_spans=spans)
    checks[name]=dict(compiled=True,full_raw_inverse_exact=True,parent_sha256=sha(parent),sha256=sha(H/name),inverse_spans=len(spans))
    diff.extend(difflib.unified_diff(before.decode('utf8').splitlines(keepends=True),after.decode('utf8').splitlines(keepends=True),fromfile=binding['parent_path'],tofile=(H/name).relative_to(R).as_posix()))

def loop(text,needle):
    lines=text.splitlines(keepends=True)
    nodes=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.For) and needle in lines[n.lineno-1]]
    assert len(nodes)==1;node=nodes[0]
    return ''.join(lines[node.lineno-1:node.end_lineno]).encode('utf8')

reuse={}
for name,needles in {'join_saved.py':['for rec in records:'],'adopt_root.py':['for na in index[','for rec in records:'],'verify_native_inputs.py':["for identity in p['candidate_ids']:"]}.items():
    before=(R/adapt[name]['parent_path']).read_bytes().decode('utf8');after=(H/name).read_bytes().decode('utf8')
    for needle in needles:
        a,b=loop(before,needle),loop(after,needle)
        allowed="row['attack']=='F Flip'" in b.decode()
        if allowed:b=b.replace(b"row['attack']=='F Flip'",b"row['attack']=='Benign'")
        assert a==b
        assert ast.dump(ast.parse(textwrap.dedent(a.decode())),include_attributes=False)==ast.dump(ast.parse(textwrap.dedent(b.decode())),include_attributes=False)
        reuse[name+':'+needle]=dict(raw_exact=True,AST_exact=True,scene_literal_inverse_only=allowed,original_sha256=hashlib.sha256(a).hexdigest())

spec=importlib.util.spec_from_file_location('pure_scope_metadata_only',H/'accepted_scope.py')
scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)
oldids=['old_native_'+str(i) for i in range(360)];full=['old_Full_'+str(i) for i in range(100)]
old=dict(accepted_new_ids=oldids,accepted_reused_ids=full,new_count=360,invalid=[],records=[dict(id=i,fixture=True) for i in oldids+full])
new=dict(accepted_new_ids=oldids+scope.FIXED+['extra_closed_1','extra_closed_2'],accepted_reused_ids=full,new_count=372,invalid=[],records=old['records']+[dict(id=i,fixture=True) for i in scope.FIXED+['extra_closed_1','extra_closed_2']])
encode=lambda obj:json.dumps(obj,ensure_ascii=False,indent=2).encode('utf8')
oldb=encode(old);rawb=encode(new);view=scope.derive_scope(oldb,rawb)
assert view['accepted_new_ids']==oldids+scope.FIXED and view['records'][:460]==old['records']
assert view['pending_closed_ids']==['extra_closed_1','extra_closed_2'] and view['observed_native_count']==372 and view['new_count']==370
assert len(view['records'])==470 and not set(view['pending_closed_ids'])&set(view['accepted_new_ids'])
tests={'extras_retained_not_accepted':True,'old460_order_and_row_bytes':True,'fixed10_only':True}
for label,mutate in [
    ('wrong_old_row_refused',lambda x:x['records'][0].update(fixture=False)),
    ('missing_fixed_id_refused',lambda x:x['accepted_new_ids'].remove(scope.FIXED[-1])),
    ('invalid_record_refused',lambda x:x['invalid'].append('fixture_failure')),
    ('duplicate_observed_id_refused',lambda x:x['accepted_new_ids'].append(x['accepted_new_ids'][0]))]:
    changed=json.loads(rawb);mutate(changed)
    try:scope.derive_scope(oldb,encode(changed))
    except AssertionError:tests[label]=True
    else:raise AssertionError(label)
changed_raw=rawb.replace(b'"fixture": true',b'"fixture" : true',1)
try:scope.derive_scope(oldb,changed_raw)
except AssertionError:tests['old_row_serialization_change_refused']=True
else:raise AssertionError('old raw row token altered')
for name in ['accepted_scope.py','check_source.py','prepare_source.py']:
    compile((H/name).read_bytes(),str(H/name),'exec')
save('SOURCE_ADAPTATIONS.json',adapt)
(H/'SOURCE_DIFF.patch').write_bytes(''.join(diff).encode('utf8'))
save('SOURCE_CHECK.json',dict(status='SOURCE_ONLY_COMPILE_RAW_INVERSE_CORE_REUSE_AND_SYNTHETIC_METADATA_PASS_NOT_SCIENCE',files=checks,
    scientific_loops=reuse,synthetic_JSON_checks=tests,future_execution=False,SSH=False,F_access=False,science_imports=False,
    accepted_scope='parent native360 ordered rows plus fixed10; Full100 remains, raw inspection untouched, extras pending',
    future_native_root_sha256=None,future_saved_offserver_sha256=None,future_root_adoption_sha256=None))
print(json.dumps(dict(status='SOURCE_ONLY_CHECK_PASS_NOT_SCIENCE',compiled=8,raw_inverses=len(checks),core_loops=len(reuse),synthetic_checks=len(tests))))
