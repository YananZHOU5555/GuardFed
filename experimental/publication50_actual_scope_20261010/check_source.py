"""Author's bounded metadata/source check; no F/Git/network/science execution."""
from pathlib import Path
import ast,copy,difflib,json,sys
import publish_increment50 as p
import finalize_spec as f
HERE=Path(__file__).resolve().parent


def main():
    for q in HERE.glob('*.py'):compile(q.read_text(encoding='utf-8'),str(q),'exec')
    prepared=p.read(HERE/'PREPARED_MANIFEST.json')
    try:p.plan(prepared)
    except AssertionError:pass
    else:raise AssertionError('Pending prepared manifest could execute')
    raw=p.raw
    original=ast.get_source_segment(raw,next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='stage'))
    restored=p.stage_text.replace('publication50_frozen_bytes_v1','publication43_frozen_bytes_v1').replace(
        "assert d['accepted'] == ACCEPTED and set(d['bindings']) == ROLES",
        "assert d['accepted'] == {'native': 188, 'three_view': 180, 'FL_new': 22} and set(d['bindings']) == ROLES").replace(
        '# Git50 sealed compact bytes','# Git43 sealed source/startup bytes')
    assert restored==original
    (HERE/'ORIGINAL43_STAGE_DIFF.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),p.stage_text.splitlines(True),
        fromfile='original43/stage',tofile='direct50/stage')),encoding='utf-8',newline='\n')
    import verify_increment50 as v
    q=p.ROOT/'tmp/publication_increment43_20261010/verify_increment43.py'
    original_code=next(c for c in compile(ast.parse(q.read_text(encoding='utf-8')),str(q),'exec').co_consts if hasattr(c,'co_name') and c.co_name=='verify')
    assert v.verify.__code__.co_code==original_code.co_code
    assert p.ns['ROOT']==p.ROOT and p.original_source.__globals__['ROOT']==p.ROOT
    assert p.ns['ROOT'].drive.upper()=='E:'
    assert len(prepared['files'])==330 and len(prepared['pending_mutable'])==8
    assert len(prepared['bindings'])==16 and prepared['bindings']['current_state'] is None
    byname={e['source']:e for e in prepared['files']}
    for role,pin in prepared['bindings'].items():
        if pin:
            assert pin['path'] in byname
            b=(p.ROOT/pin['path']).read_bytes();assert p.sha(b)==pin['sha256']
            for key,value in pin['expect'].items():assert p.pointer(json.loads(b),key)==value
    extras=p.read(HERE/'EXACT_EXTRA_FILES.json')
    old=p.read(p.ROOT/'tmp/publication50_source_preparation_20261010/EXTRA_CANDIDATES.json')
    assert extras['inherited_three_groups']==old['adopted_extras']
    extra_entries=[e for g in extras['inherited_three_groups'] for e in g['files']]+extras['scientific_files']+extras['last_root_named_files']
    for e in extra_entries:
        b=(p.ROOT/e['source']).read_bytes()
        assert byname[e['source']]==e and p.sha(b)==e['sha256'] and len(b)==e['bytes']
    template=p.read(HERE/'ROOT_INPUTS_TEMPLATE.json')
    actual=copy.deepcopy(template);actual['status']='ROOT_CLOSED50_MUTABLE_BINDINGS_READY'
    for pin in actual['mutable_files'].values():pin.update(sha256='a'*64,bytes=1)
    f.validate_actual(actual,prepared)
    failures=[]
    for name,change in [('arbitrary_extras',lambda d:d.update(adopted_extras=[{'files':[{'source':'tmp/unknown.py'}]}])),
                        ('arbitrary_mutable_name',lambda d:d['mutable_files'].update({'tmp/unknown.py':{'sha256':'a'*64,'bytes':1}})),
                        ('missing_mutable',lambda d:d['mutable_files'].pop(next(iter(d['mutable_files'])))),
                        ('pending_mutable_SHA',lambda d:next(iter(d['mutable_files'].values())).update(sha256=None))]:
        d=copy.deepcopy(actual);change(d)
        try:f.validate_actual(d,prepared)
        except (AssertionError,TypeError):failures.append(name)
        else:raise AssertionError('Unexpected metadata acceptance: '+name)
    # Exercise the actual scope guard with a tiny virtual self-seal; no disk or F fixture.
    dummy={'files':{'check_source.py':{'sha256':'b'*64,'bytes':1}}}
    actual_read=p.read
    p.read=lambda path:dummy if path.name=='FILES_SHA256.json' else actual_read(path)
    original_path=Path.read_bytes;original_stat=Path.stat
    class Stat:
        st_size=2
    Path.read_bytes=lambda path:b'{}' if path==HERE/'FILES_SHA256.json' else original_path(path)
    Path.stat=lambda path,*a,**k:Stat() if path==HERE/'FILES_SHA256.json' else original_stat(path,*a,**k)
    try:
        d=copy.deepcopy(prepared)
        d['bindings']['current_state']=dict(path=prepared['pending_mutable'][0],sha256='a'*64,expect=p.FACTS['current_state'])
        own=HERE.relative_to(p.ROOT).as_posix()+'/'
        d['files'] += [dict(source=name,destination=p.destination(name),sha256='a'*64,bytes=1) for name in prepared['pending_mutable']]
        d['files'] += [dict(source=own+'check_source.py',destination=p.destination(own+'check_source.py'),sha256='b'*64,bytes=1),
                       dict(source=own+'FILES_SHA256.json',destination=p.destination(own+'FILES_SHA256.json'),sha256=p.sha(b'{}'),bytes=2)]
        d['allowed_paths'] += [own+'check_source.py',own+'FILES_SHA256.json']
        p.closed_scope(d)
        for label,change in [('role_as_parent_ref',lambda x:x['parent_recovery_references'].append(x['files'].pop(next(i for i,e in enumerate(x['files']) if e['source']==x['bindings']['exact3_scientific']['path'])))),
                             ('extra_unknown_file',lambda x:x['files'].append(dict(source='tmp/unknown.py',destination='experimental/unknown.py',sha256='a'*64,bytes=1))),
                             ('changed_scientific_pin',lambda x:x['bindings']['exact3_scientific'].update(sha256='a'*64))]:
            changed=copy.deepcopy(d);change(changed)
            try:p.closed_scope(changed)
            except AssertionError:failures.append(label)
            else:raise AssertionError('Unexpected closed-scope acceptance: '+label)
    finally:
        p.read=actual_read;Path.read_bytes=original_path;Path.stat=original_stat
    assert 'torch' not in sys.modules and not list(HERE.glob('__pycache__'))
    result=dict(status='AUTHOR_SOURCE_AND_CLOSED_SCOPE_CHECK_PASS_NOT_FINALIZED_OR_PUBLISHED',
        scope=prepared['scope'],selected_files=330,selected_bytes=sum(e['bytes'] for e in prepared['files']),
        inherited_exact_extra_files37=True,new_scientific_exact_extra_files=30,role_proofs_required_as_actual_F_files=16,
        final_root_named_files=12,actual_role_pins_checked=15,pending_mutable=8,positive_mutable_fixture=True,positive_closed_scope_fixture=True,
        targeted_refusals=failures,original43_stage_inverse_source_exact=True,
        original44_storage_and_safe_directory_git_AST_reused=True,original43_committed_blob_verifier_bytecode_exact=True,
        no_import_of_previous50_wrapper=True,ROOT_storage_alternates_stays_real_E=True,
        F_snapshot_source_reader_private_globals_only=True,Windows_whole_failure_preserved=True,
        Git_commands=0,Git_mutations=0,F_files_written=0,network_requests=0,science_executed=0,
        source_review_is_independent=False,test_started=False,goal_complete=False)
    p.ns['save'](HERE/'SOURCE_CHECK_FINAL.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
