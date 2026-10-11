"""Independent stdlib-only source/metadata review; no candidate imports or execution."""
from pathlib import Path
import ast, datetime, difflib, hashlib, json, textwrap

R=Path(__file__).resolve().parents[2]
H=Path(__file__).resolve().parent
B=R/'tmp/mechanism_F_nonIID_FedSA10_three_view_prepare_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
seal=read(B/'FILES_SHA256.json');source_seal=read(B/'SOURCE_FILES_SHA256.json')
for manifest in (seal,source_seal):
    for name,pin in manifest['files'].items():
        p=(B/name).resolve();assert p.is_relative_to(B.resolve()) and p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256']
adapt=read(B/'SOURCE_ADAPTATIONS.json');inverses={};patch=[]
assert len(adapt)==18
for name,pin in adapt.items():
    old=(R/pin['parent_path']).read_bytes();new=(B/name).read_bytes()
    assert hashlib.sha256(old).hexdigest()==pin['parent_sha256'] and sha(B/name)==pin['sha256']
    undo=new
    for span in reversed(pin['raw_inverse_spans']):
        assert undo[span['c']:span['d']]==bytes.fromhex(span['new'])
        undo=undo[:span['c']]+bytes.fromhex(span['old'])+undo[span['d']:]
    assert undo==old
    ast.parse(new.decode('utf8'))
    inverses[name]=dict(raw_inverse_exact=True,parent_sha256=pin['parent_sha256'],sha256=pin['sha256'])
    patch.extend(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile=pin['parent_path'],tofile=str((B/name).relative_to(R))))
assert ''.join(patch).encode()==(B/'SOURCE_DIFF.patch').read_bytes()

def loop(payload,needle):
    text=payload.decode();lines=text.splitlines(True)
    nodes=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.For) and needle in lines[n.lineno-1]]
    assert len(nodes)==1
    n=nodes[0];return ''.join(lines[n.lineno-1:n.end_lineno]).encode()

loops={}
for name,needles in {'join_saved.py':['for rec in records:'],'adopt_root.py':['for na in index[','for rec in records:'], 'verify_numeric.py':['for row in rows:','for record in records:'],'verify_saved.py':["for view in ('native','raw','shared_calibration'):","for panel in tables['panels']:"]}.items():
    for needle in needles:
        old=loop((R/adapt[name]['parent_path']).read_bytes(),needle);new=loop((B/name).read_bytes(),needle)
        normalized=new
        if name=='adopt_root.py' and needle=='for na in index[':
            assert new.count(b"row['attack']=='FedSA'")==1
            normalized=new.replace(b"row['attack']=='FedSA'",b"row['attack']=='F Flip'")
        assert old==normalized
        assert ast.dump(ast.parse(textwrap.dedent(old.decode())),include_attributes=False)==ast.dump(ast.parse(textwrap.dedent(normalized.decode())),include_attributes=False)
        loops[name+':'+needle]=dict(raw_exact_without_rebinding=(old==new),raw_and_AST_exact_after_only_allowed_scene_assertion_rebinding=True)
for name in ['execute_once.py','remote_cpu111_export.py','cpu_release_readonly.py']:
    assert (B/name).read_bytes()==(R/adapt[name]['parent_path']).read_bytes()
before=(R/adapt['remote_preflight.py']['parent_path']).read_bytes()
after=(B/'remote_preflight.py').read_bytes()
skip=(b'candidate_ids=',b"assert latest['receipt']==",b"assert latest['receipt_sha256']==",b"assert latest['all_transported_ids']==")
strip_metadata=lambda raw:b''.join(line for line in raw.splitlines(keepends=True) if not line.startswith(skip))
assert strip_metadata(before)==strip_metadata(after)

p=read(B/'PREPARED.json')
fixed=[f'minus_F_non-IID_FedSA_seed{s}' for s in range(91001,91011)]
assert p['candidate_ids']==fixed and p['prior_accepted']==370 and p['prior_transported']==190
for key in ['prior_root','prior_index','native370_root','native370_inspection','native370_raw_inspection','native370_ledger']:
    assert not str(p[key+'_path']).startswith(('F:','F/'))
    assert sha(p[key+'_path'])==p[key+'_sha256']
prior=read(p['prior_root_path']);scope=read(p['native370_inspection_path']);raw=read(p['native370_raw_inspection_path'])
assert prior['accepted_native_scope_sha256']==p['native370_inspection_sha256'] and prior['native_inspection_sha256']==p['native370_raw_inspection_sha256']
assert len(raw['records'])==471 and raw['new_count']==371 and len(scope['records'])==470 and scope['new_count']==370
assert set(r['id'] for r in raw['records'])-set(r['id'] for r in scope['records'])=={fixed[0]}
assert [r for r in raw['records'] if r['id']!=fixed[0]]==scope['records']
assert not set(fixed)&set(scope['accepted_new_ids'])
npath=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T024637Z/ROOT_DELTA_VERIFICATION.json'
native=read(npath)
assert native['root_adopted'] is True and native['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS'
assert native['new_ids']==fixed and native['total_new_strict_and_offserver']==380 and native['test'] is False
assert native['previous_ledger_sha256']==p['native370_ledger_sha256']
assert sha(npath.parent/'inspection/inspection.json')==native['inspection_sha256'] and sha(npath.parent/'verified_ledger.json')==native['ledger_sha256']
assert len(read(npath.parent/'verified_ledger.json')['entries'])==52
assert read(npath.parent/'verified_ledger.json')['entries'][:-1]==read(p['native370_ledger_path'])['entries']
new=read(npath.parent/'inspection/inspection.json')
assert len(new['records'])==480 and [r for r in new['records'] if r['id'] not in fixed]==scope['records']
template=read(B/'ROOT_ACTUAL_PINS.template.json');adopter_source=(B/'adopt_root.py').read_text('utf8')
assert template['source_adopter_sha256']!=sha(B/'adopt_root.py')
assert "actual['source_adopter_sha256']" not in adopter_source
assert "source_adopter_sha256=sha(__file__)" in adopter_source
for name in ['EXECUTION_INPUTS.json','ROOT_BINDING.json','ROOT_ACTUAL_PINS.json','ACTUAL_HANDOFF.json','records.json','tables.json']:
    assert not (B/name).exists(),name
review=dict(status='PASS_SOURCE_ONLY_WITH_ROOT_LATE_BINDING_METADATA_NOTES_NO_DISPATCH_AUTHORIZATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_adoptable=True,actual_dispatch_authorized_by_this_review=False,
    candidate_directory=B.relative_to(R).as_posix(),delivery_seal_sha256=sha(B/'FILES_SHA256.json'),delivery_members=len(seal['files']),source_seal_sha256=sha(B/'SOURCE_FILES_SHA256.json'),source_members=len(source_seal['files']),derived_source_files18=len(inverses),complete_raw_inverses=inverses,full_SOURCE_DIFF_exact=True,scientific_loops=loops,resource_guards_raw_exact_except_fixed_ID_and_previous_receipt_metadata=True,unchanged_execute_export_bridge_release=True,
    actual_native380_root_path=npath.relative_to(R).as_posix(),actual_native380_root_sha256=sha(npath),actual_native380_compatible=True,old_raw471_preserved=True,old_accepted470_excludes_only_pending_FedSA91001=True,old470_objects_and_order_exact=True,ledger51_prefix_and52_total_exact=True,exact_selected_ids=fixed,
    findings=[dict(kind='NONBLOCKING_TEMPLATE_METADATA_REQUIRES_ACTUAL_ROOT_BINDING',path='ROOT_ACTUAL_PINS.template.json',field='source_adopter_sha256',template_sha256=template['source_adopter_sha256'],actual_sha256=sha(B/'adopt_root.py'),detail='Template is stale. Root must bind the actual source SHA. Adopter verifies whole ROOT_ACTUAL_PINS via required CLI SHA but does not separately consume this field; final adopted proof recomputes sha(__file__). No claim of a per-field guard.'),dict(kind='PROOF_WORDING_BOUNDARY',path='SOURCE_CHECK.json',detail='Six science loops are literally byte/AST unchanged. The native archive identity loop is byte/AST identical only after reversing its one permitted F Flip→FedSA scene assertion. SOURCE_CHECK raw_exact is normalized for that one loop, not literal new-source equality.')],
    limits=['Current target source author is another agent; reviewer authored the earlier native380 adopter and some reused prior sources; this is an independent review of the current adaptation, not an independent validation of all historical evidence.','No candidate imports, fixtures, binder, transport, numerical verifier, scientific code, SSH, F access, arrays or models were run/read.','Runtime source/data/CPU111 gates, root snapshot and actual saved-array/receipt/member checks remain required. No saved380 or F80 results/adoption exist in this source package.','Template authorization/source-review fields are not all individually consumed; actual authority is root review plus the required complete ROOT_ACTUAL_PINS CLI hash.'])
H.mkdir(exist_ok=True)
with (H/'REVIEW.json').open('x',encoding='utf8',newline='\n') as f:json.dump(review,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
with (H/'FILES_SHA256.json').open('x',encoding='utf8',newline='\n') as f:json.dump(dict(files={n:dict(sha256=sha(H/n),bytes=(H/n).stat().st_size) for n in ['review_source.py','REVIEW.json']}),f,indent=2);f.write('\n')
print(json.dumps(dict(review_sha256=sha(H/'REVIEW.json'),seal_sha256=sha(H/'FILES_SHA256.json'),delivery_members=len(seal['files']),source_members=len(source_seal['files']),native380_root_sha256=sha(npath),source_adoptable=True)))
