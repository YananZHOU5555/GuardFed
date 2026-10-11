"""Derive only compact metadata source; no scientific imports or F access."""
from pathlib import Path
import ast,difflib,hashlib,io,json,textwrap,tokenize
R=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent
OLD=R/'tmp/celeba_mechanism_remaining_F_nonIID_Benign10_prepare_20261011'
NATIVE=R/'tmp/mechanism_F_nonIID_FFlip10_prepare_20261011'
ADOPT=R/'tmp/celeba_F_nonIID_Benign10_root_adopter_prepare_20261011/adopt_root.py'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name,value):
    with (H/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

assert sha(NATIVE/'FILES_SHA256.json')=='b213c893904be9b780381a4d27ef26976eb7a955321da01fecb58778706b79de'
for n,pin in read(NATIVE/'FILES_SHA256.json')['files'].items():
    assert sha(NATIVE/n)==pin['sha256'] and (NATIVE/n).stat().st_size==pin['bytes']
pins=read(NATIVE/'SOURCE_PINS.json');params=read(NATIVE/'SAVED_PARAMETERS.json')
for name in ('verify_native_inputs.py','join_saved.py','bind_native.py','finalize_actual_handoff.py'):
    if name in pins['saved_parent_entry_sources']:assert sha(OLD/name)==pins['saved_parent_entry_sources'][name]['sha256']
assert sha(ADOPT)==pins['saved_parent_adopter']['sha256']
sources={n:OLD/n for n in ('verify_native_inputs.py','join_saved.py','bind_native.py','finalize_actual_handoff.py')};sources['adopt_root.py']=ADOPT
adaptations={};diffs=[];checks={}
numbers={'350':'360','360':'370','450':'460','49':'50','50':'51','170':'180','180':'190'}
common=[('celeba_mechanism_remaining_F_nonIID_Benign10_prepare_20261011',H.name),
    ('celeba_mechanism_remaining_F_nonIID_Benign10_root_adoption_20261011','celeba_mechanism_remaining_F_nonIID_FFlip10_root_adoption_20261011'),
    ('minus_F_non-IID_Benign_seed','minus_F_non-IID_F Flip_seed'),('F_NONIID_BENIGN','F_NONIID_FFLIP'),
    ('native350','native360'),('prior350','prior360'),('old450','old460'),('old49','old50'),('old350','old360'),('original350','original360'),
    ('MECHANISM360','MECHANISM370'),('REPLAY360','REPLAY370'),('MIN360','MIN370'),('>=360','>=370'),
    ('exact10 F non-IID Benign','exact10 F non-IID F Flip'),('three-view360','three-view370'),
    ('tmp/celeba_remaining_F_SpDFA10_prepared_20261011/adopt_by_root.py','tmp/celeba_F_nonIID_Benign10_root_adopter_prepare_20261011/adopt_root.py'),
    ('5b24956b4f14707e2a284fe73ffb538545b4aad218c833e31427720402ff02b3',params['prior_index']['sha256']),
    ('3f7b436e0b2fec0401d7bf6117d6dca48aef21c185b6f170d37954e2cafee5a4',params['prior_root']['sha256']),
    ('d40f806a27db0512647edc2206b8881e5d407a4a655c8dde1a47b7b001ee75b9',params['previous_receipt_sha256'])]

for name,parent in sources.items():
    raw=parent.read_bytes();text=raw.decode('utf8');ops=[]
    def replace(old,new):
        global text
        assert old in text,(name,old)
        count=text.count(old);text=text.replace(old,new);ops.append(dict(old=old,new=new,count=count))
    # Numeric tokens only: hashes, strings and science constants are unaffected.
    lines=text.splitlines(keepends=True);starts=[0]
    for line in lines:starts.append(starts[-1]+len(line))
    edits=[]
    for t in tokenize.generate_tokens(io.StringIO(text).readline):
        if t.type==tokenize.NUMBER and t.string in numbers:
            a=starts[t.start[0]-1]+t.start[1];b=starts[t.end[0]-1]+t.end[1]
            edits.append((a,b,t.string,numbers[t.string]))
    for a,b,old,new in reversed(edits):
        text=text[:a]+new+text[b:]
    for old,new in common:
        if old in text:replace(old,new)
    nl='\r\n' if b'\r\n' in raw else '\n'
    if name=='verify_native_inputs.py':
        replace('    old=read(p[\'native360_inspection_path\']);oldledger=read(p[\'native360_ledger_path\'])',
            '    from accepted_scope import load_scope'+nl+'    inspection=load_scope(e,p)'+nl+'    old=read(p[\'native360_inspection_path\']);oldledger=read(p[\'native360_ledger_path\'])')
        replace("row['attack']=='Benign'","row['attack']=='F Flip'")
    elif name=='bind_native.py':
        replace('    validate(e,p)',"    from accepted_scope import write_scope"+nl+"    write_scope(e,p,H/'ACCEPTED_NATIVE_SCOPE.json')"+nl+'    validate(e,p)')
        replace("assert not (H/'EXECUTION_INPUTS.json').exists() and not (H/'EXECUTION_FAILURE.json').exists()",
            "assert not (H/'EXECUTION_INPUTS.json').exists() and not (H/'EXECUTION_FAILURE.json').exists() and not (H/'ACCEPTED_NATIVE_SCOPE.json').exists()")
    elif name=='join_saved.py':
        replace(" previousroot=Path(P['native360_root_path'])",' from accepted_scope import load_scope'+nl+' inspection=load_scope(E,P)'+nl+" previousroot=Path(P['native360_root_path'])")
        replace("native_inspection_sha256=E['native_inspection_sha256'],native_archives=", "native_inspection_sha256=E['native_inspection_sha256'],accepted_native_scope_path=str(Path(E['accepted_native_scope_path']).relative_to(R)),accepted_native_scope_sha256=E['accepted_native_scope_sha256'],pending_native_closed_ids=inspection['pending_closed_ids'],native_archives=")
    elif name=='adopt_root.py':
        replace("assert actual['native_root_sha256']=='396e40654b6dfc75029d4a7a42dbfafe4caaf111a872ae2ece5045f6d1203418'",
            "assert isinstance(actual['native_root_sha256'],str) and len(actual['native_root_sha256'])==64")
        replace("old_native=Path(read(S/'PREPARED.json')['native360_root_path']);old_inspection=", "from accepted_scope import load_scope"+nl+"inspection=load_scope(execution,read(S/'PREPARED.json'))"+nl+"assert index['accepted_native_scope_sha256']==execution['accepted_native_scope_sha256'] and R/index['accepted_native_scope_path']==Path(execution['accepted_native_scope_path'])"+nl+"old_native=Path(read(S/'PREPARED.json')['native360_root_path']);old_inspection=")
        replace("row['attack']=='Benign'","row['attack']=='F Flip'")
        replace("['non-IID','Benign']],partial_F_scenes=", "['non-IID','Benign'],['non-IID','F Flip']],partial_F_scenes=")
        replace("native_inspection_sha256=sha(ip),native_ledger_sha256=", "native_inspection_sha256=sha(ip),accepted_native_scope_path=index['accepted_native_scope_path'],accepted_native_scope_sha256=execution['accepted_native_scope_sha256'],pending_native_closed_ids=inspection['pending_closed_ids'],native_ledger_sha256=")
    elif name=='finalize_actual_handoff.py':
        replace("sha(H/'remote_cpu111_export.py')", "sha(R/'tmp/celeba_mechanism_remaining_F_nonIID_Benign10_prepare_20261011/remote_cpu111_export.py')")
        replace("native_ledger_sha256=E['native_ledger_sha256'],", "native_ledger_sha256=E['native_ledger_sha256'],accepted_native_scope_path=E['accepted_native_scope_path'],accepted_native_scope_sha256=E['accepted_native_scope_sha256'],")
    # A complete sequence of edit snapshots makes reversal exact without global inverse collisions.
    derived=text.encode('utf8');compile(derived,str(H/name),'exec')
    with (H/name).open('xb') as f:f.write(derived)
    matcher=difflib.SequenceMatcher(None,raw,derived,autojunk=False)
    spans=[dict(old_start=a,old_end=b,new_start=c,new_end=d,old_hex=raw[a:b].hex(),new_hex=derived[c:d].hex()) for tag,a,b,c,d in matcher.get_opcodes() if tag!='equal']
    restored=derived
    for span in reversed(spans):
        a,b=span['new_start'],span['new_end'];assert restored[a:b].hex()==span['new_hex'];restored=restored[:a]+bytes.fromhex(span['old_hex'])+restored[b:]
    assert restored==raw
    adaptations[name]=dict(parent_path=parent.relative_to(R).as_posix(),parent_sha256=sha(parent),source_sha256=sha(H/name),metadata_replacements=ops,raw_inverse_spans=spans)
    diffs.extend(difflib.unified_diff(raw.decode('utf8').splitlines(keepends=True),text.splitlines(keepends=True),fromfile=parent.relative_to(R).as_posix(),tofile=(H/name).relative_to(R).as_posix()))
    checks[name]=dict(compiled=True,full_raw_inverse_exact=True,parent_sha256=sha(parent),sha256=sha(H/name))

p=read(OLD/'PREPARED.json')
p.update(status='SOURCE_ONLY_FIXED_F_NONIID_FFLIP10_PENDING_ACTUAL_NATIVE370_AND_SAVED',candidate_ids=params['candidate_ids'],prior_accepted=360,prior_transported=180,
    prior_root_path=str(R/params['prior_root']['path']),prior_root_sha256=params['prior_root']['sha256'],prior_index_path=str(R/params['prior_index']['path']),prior_index_sha256=params['prior_index']['sha256'],
    previous_local_receipt=params['previous_local_receipt'],previous_remote_receipt=params['previous_remote_receipt'],previous_receipt_sha256=params['previous_receipt_sha256'],previous_all_transported_ids=params['previous_all_transported_ids'],
    required_new_native_candidate_ids=params['candidate_ids'],replay_target_total=370,native_target_minimum=370)
for suffix in ('root','inspection','ledger'):
    p.pop('native350_'+suffix+'_path');p.pop('native350_'+suffix+'_sha256')
    root=R/read(NATIVE/'FROZEN_SCOPE.json')['parent_root_path']
    path=root if suffix=='root' else root.parent/('inspection/inspection.json' if suffix=='inspection' else 'verified_ledger.json')
    p['native360_'+suffix+'_path']=str(path);p['native360_'+suffix+'_sha256']=sha(path)
save('PREPARED.json',p)
save('SOURCE_ADAPTATIONS.json',adaptations)
with (H/'SOURCE_DIFF.patch').open('x',encoding='utf8',newline='') as f:f.write(''.join(diffs))

# Scientific archive/record loops remain raw-identical; only the scene identity literal changes in the native adopter loop.
def loop(text,needle):
    lines=text.splitlines(keepends=True);tree=ast.parse(text)
    nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.For,ast.With)) and needle in lines[n.lineno-1]]
    assert len(nodes)==1,(needle,len(nodes));n=nodes[0]
    return ''.join(lines[n.lineno-1:n.end_lineno]).encode('utf8')
reuse={}
for name,needles in {'join_saved.py':['for rec in records:'], 'adopt_root.py':['for na in index[','for rec in records:']}.items():
    before=sources[name].read_bytes().decode('utf8')
    after=(H/name).read_bytes().decode('utf8')
    for needle in needles:
        a,b=loop(before,needle),loop(after,needle)
        if name=='adopt_root.py' and needle.startswith('for na'):b=b.replace(b"row['attack']=='F Flip'",b"row['attack']=='Benign'")
        assert a==b and ast.dump(ast.parse(textwrap.dedent(a.decode())),include_attributes=False)==ast.dump(ast.parse(textwrap.dedent(b.decode())),include_attributes=False)
        reuse[name+':'+needle]=dict(raw_exact_after_declared_scene_literal_inverse=True,AST_exact=True,original_sha256=hashlib.sha256(a).hexdigest())
save('SOURCE_CHECK.json',dict(status='SOURCE_ONLY_COMPILE_RAW_INVERSE_AND_CORE_REUSE_PASS_NOT_SCIENCE',files=checks,scientific_loops=reuse,future_execution=False))
print(json.dumps(dict(status='SOURCE_DERIVED_NOT_EXECUTED',files=checks)))
