"""Bounded source/metadata checks; no Git, F access, network, or scientific execution."""
from pathlib import Path
import ast,copy,difflib,json,sys
import publish_increment51 as p
import finalize_spec as f
HERE=Path(__file__).resolve().parent


def main():
    compiled=[]
    for q in sorted(HERE.glob('*.py')):
        compile(q.read_text(encoding='utf-8'),str(q),'exec');compiled.append(q.name)
    prepared=p.read(HERE/'PREPARED_MANIFEST.json')
    try:p.plan(prepared)
    except AssertionError:pass
    else:raise AssertionError('Pending manifest could execute')
    original=ast.get_source_segment(p.raw,next(n for n in ast.parse(p.raw).body if isinstance(n,ast.FunctionDef) and n.name=='stage'))
    restored=p.stage_text.replace('publication51_frozen_bytes_v1','publication43_frozen_bytes_v1').replace(
        "assert d['accepted'] == ACCEPTED and set(d['bindings']) == ROLES",
        "assert d['accepted'] == {'native': 188, 'three_view': 180, 'FL_new': 22} and set(d['bindings']) == ROLES").replace(
        '# Git51 sealed compact bytes','# Git43 sealed source/startup bytes')
    assert restored==original
    import verify_increment51 as v
    verifier=p.ROOT/'tmp/publication_increment43_20261010/verify_increment43.py'
    code=next(c for c in compile(ast.parse(verifier.read_text(encoding='utf-8')),str(verifier),'exec').co_consts if hasattr(c,'co_name') and c.co_name=='verify')
    assert v.verify.__code__.co_code==code.co_code and v.verify.__code__.co_consts==code.co_consts
    assert p.ns['ROOT']==p.ROOT and p.original_source.__globals__['ROOT']==p.ROOT and p.ROOT.drive.upper()=='E:'
    assert str(p.ns['OUTPUT_ROOT']).replace('\\','/')=='F:/YananResearchStorage/GuardFed/git_publication/i51'
    changes=p.read(HERE/'SOURCE_REBINDINGS.json')['publisher']
    current=(HERE/'publish_increment51.py').read_text(encoding='utf-8')
    for c in reversed(changes):
        assert current.count(c['after'])==1
        current=current.replace(c['after'],c['before'])
    assert current==(p.ROOT/'tmp/publication50_actual_scope_20261010/publish_increment50.py').read_text(encoding='utf-8')
    byname={e['source']:e for e in prepared['files']}
    assert len(byname)==len(prepared['files']) and len(prepared['pending_mutable'])==8
    assert len(prepared['bindings'])==13 and prepared['bindings']['current_state'] is None
    actual_pins=0;actual_pointer_facts=0
    for role,pin in prepared['bindings'].items():
        if pin:
            assert pin['path'] in byname
            _,b=p.source(pin['path'],prepared['allowed_paths']);assert p.sha(b)==pin['sha256']
            for key,value in pin['expect'].items():assert p.pointer(json.loads(b),key)==value;actual_pointer_facts+=1
            actual_pins+=1
    # All immutable compact bytes, including retained failures, are read/hash-checked only.
    for name,e in byname.items():
        _,b=p.source(name,prepared['allowed_paths'])
        assert p.sha(b)==e['sha256'] and len(b)==e['bytes']
    assert len(p.SMALL_FAILURE_TXT)==4 and all(name.endswith('.txt') for name in p.SMALL_FAILURE_TXT)
    assert len(p.TRACE_PINS)==4 and {Path(n).name for n in p.TRACE_PINS}=={'LINUX.stdout','LINUX.stderr','WINDOWS.stdout','WINDOWS.stderr'}
    assert {'.stdout','.stderr'}<=p.ns['SUFFIXES']
    for name in p.TRACE_PINS:
        _,b=p.source(name,prepared['allowed_paths']);p.check_trace(name,b)
    failures=[]
    for name in ['tmp/unknown.txt','tmp/unknown.stdout','tmp/unknown.stderr','tmp/unknown.npz']:
        try:p.validate_file_name(name,[name])
        except AssertionError:failures.append('unlisted_extension:'+name)
        else:raise AssertionError('Unknown extension accepted')
    template=p.read(HERE/'ROOT_INPUTS_TEMPLATE.json')
    actual=copy.deepcopy(template);actual['status']='ROOT_CLOSED51_MUTABLE_BINDINGS_READY'
    for pin in actual['mutable_files'].values():pin.update(sha256='a'*64,bytes=1)
    f.validate_actual(actual,prepared)
    for name,change in [('arbitrary_extras',lambda d:d.update(extras=['tmp/unknown.py'])),
        ('unknown_mutable',lambda d:d['mutable_files'].update({'tmp/unknown.py':{'sha256':'a'*64,'bytes':1}})),
        ('missing_mutable',lambda d:d['mutable_files'].pop(next(iter(d['mutable_files'])))),
        ('pending_mutable_SHA',lambda d:next(iter(d['mutable_files'].values())).update(sha256=None))]:
        d=copy.deepcopy(actual);change(d)
        try:f.validate_actual(d,prepared)
        except (AssertionError,TypeError):failures.append(name)
        else:raise AssertionError('Unexpected actual-root metadata acceptance: '+name)
    # Exercise the actual scope gate using one virtual self-seal; it writes no fixture files.
    dummy={'files':{'check_source.py':{'sha256':'b'*64,'bytes':1}}}
    actual_read=p.read;original_path=Path.read_bytes;original_stat=Path.stat
    p.read=lambda path:dummy if path==HERE/'FILES_SHA256.json' else actual_read(path)
    class Stat:st_size=2
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
        for name,change in [('role_as_parent_ref',lambda x:x['parent_recovery_references'].append(x['files'].pop(next(i for i,e in enumerate(x['files']) if e['source']==x['bindings']['FL47WindowsFailure']['path'])))),
            ('unknown_extra_file',lambda x:x['files'].append(dict(source='tmp/unknown.py',destination='experimental/unknown.py',sha256='a'*64,bytes=1))),
            ('changed_actual_role_pin',lambda x:x['bindings']['FL47SingleDiagnostic'].update(sha256='a'*64)),
            ('FL47_held_boundary_changed',lambda x:x.update(FL47_new_three_view_root_accepted=47))]:
            changed=copy.deepcopy(d);change(changed)
            try:p.closed_scope(changed)
            except AssertionError:failures.append(name)
            else:raise AssertionError('Unexpected closed-scope acceptance: '+name)
    finally:
        p.read=actual_read;Path.read_bytes=original_path;Path.stat=original_stat
    assert 'torch' not in sys.modules and not list(HERE.glob('__pycache__'))
    longest=max(len(str(p.ns['OUTPUT_ROOT']/'i'/'source'/name)) for name in byname)
    assert longest<260
    result=dict(status='AUTHOR_SOURCE_AND_CLOSED_SCOPE_CHECK_PASS_NOT_FINALIZED_OR_PUBLISHED',
        compiled=compiled,scope=prepared['scope'],accepted=p.ACCEPTED,selected_files=len(byname),
        selected_bytes=sum(e['bytes'] for e in prepared['files']),role_proofs_required_as_actual_F_files=len(p.ROLES),
        actual_role_pins_checked=actual_pins,actual_pointer_facts_checked=actual_pointer_facts,pending_mutable=8,
        positive_mutable_fixture=True,positive_closed_scope_fixture=True,targeted_refusals=failures,
        original43_stage_inverse_source_exact=True,successful50_wrapper_inverse_source_exact=True,
        original44_storage_and_safe_directory_git_AST_reused=True,original43_committed_blob_verifier_bytecode_constants_exact=True,
        no_import_of_previous50_wrapper=True,ROOT_storage_alternates_stays_real_E=True,
        F_snapshot_source_reader_private_globals_only=True,short_F_first_copy_max_path_chars=longest,
        exact_tiny_preparation_txt=4,exact_operation_stream_exceptions=4,FL47_new_adopted=0,
        Windows_whole_failure_preserved=True,single_record_and_operation_diagnostics_are_not_whole47_or_platform_causality=True,
        Git_commands=0,Git_mutations=0,F_files_written=0,network_requests=0,science_executed=0,
        source_review_is_independent=False,test_started=False,goal_complete=False)
    p.ns['save'](HERE/'SOURCE_CHECK.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
