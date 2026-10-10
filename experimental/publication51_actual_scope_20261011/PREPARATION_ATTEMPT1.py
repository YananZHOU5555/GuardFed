"""One bounded source-only rebinding; never freeze, stage, or execute science."""
from pathlib import Path
import ast, copy, difflib, hashlib, json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'tmp/publication50_actual_scope_20261010'
PLAN = ROOT / 'tmp/guardfed_publication51_source_preparation_20261011'
SUPPLEMENT = PLAN / 'attempt_v2'
PARENT = '4dfc9403c182c8f192c374e396eb2f564970159c'
SCOPE = 'CLOSED_NATIVE272_MECHANISM260_A60_AUTHOR_REVIEW_AND_FL47_PRESERVED_WINDOWS_ARRAY_FAILURE_WITH_FINAL_ID_METADATA_ONLY'
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_bytes())


def save(path, d):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def replace(text, before, after, changes):
    assert text.count(before) == 1, repr(before)
    changes.append(dict(before=before, after=after))
    return text.replace(before, after)


def main():
    assert sha((PLAN/'FILES_SHA256.json').read_bytes()) == '47f3c8618159bd0a3cb85b4ea57494f0424dae25d5298d24b47369265187fbe0'
    assert not SUPPLEMENT.exists() and not (HERE/'publish_increment51.py').exists()
    SUPPLEMENT.mkdir()
    selection = read(PLAN/'EXACT_SELECTION.json')
    old_index = read(PLAN/'INPUT_CANDIDATES.json')
    byname = {e['source']:e for e in old_index['files']}
    assert len(byname) == 311
    # Exact named sealed sources, their pure-metadata fixtures, and actual diagnostic outputs only.
    diagnostic_sources = [
        'tmp/flgmm47_windows_fit_diagnostic_source_20261011',
        'tmp/flgmm47_windows_fit_diagnostic_source_20261011/attempt_v2',
        'tmp/flgmm47_lambda_operation_source_20261011']
    new_names = []
    for name in diagnostic_sources:
        seal = ROOT/name/'FILES_SHA256.json'
        packet = read(seal)
        for rel,pin in packet['files'].items():
            q = ROOT/name/rel
            assert sha(q.read_bytes()) == pin['sha256'] and q.stat().st_size == pin['bytes']
            new_names.append(name+'/'+rel)
        new_names.append(name+'/FILES_SHA256.json')
    actual_base = 'tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001/'
    actual_names = [actual_base+name for name in [
        'OFFSERVER_SINGLE_RECORD_DIAGNOSTIC.started.json',
        'OFFSERVER_SINGLE_RECORD_DIAGNOSTIC.failure.json',
        'OFFSERVER_SINGLE_RECORD_DIAGNOSTIC_V2.started.json',
        'OFFSERVER_SINGLE_RECORD_DIAGNOSTIC_V2.MEASURED.json',
        'OFFSERVER_SINGLE_RECORD_DIAGNOSTIC_V2.json',
        'ROOT_SINGLE_RECORD_DIAGNOSTIC_REVIEW.json',
        'operation_trace001/ACTUAL_COMMANDS.json',
        'operation_trace001/ROOT_OPERATION_TRACE_REVIEW.json']]
    new_names += actual_names
    new_names += read(HERE/'EXACT_TRACE_ADDITIONS.json')['paths']
    gradient_base = 'tmp/celeba_gradient64_delta_after23_closed32_20261011/'
    gradient_packet = read(ROOT/gradient_base/'SOURCE_FILES_SHA256.json')
    new_names += [gradient_base+name for name in gradient_packet['files']]
    new_names += [gradient_base+name for name in [
        'SOURCE_FILES_SHA256.json','DELIVERY_FILES_SHA256.json','ROOT_ADOPTION_REVIEW.json',
        'ROOT_SOURCE_REVIEW.json','ROOT_READY_HANDOFF.json','ROOT_READY_CHAIN_LINK.json',
        'OFFSERVER_ACCEPTANCE.json','SAVED_TENSOR_STATE_CHECK.json','RAW_STORAGE_INDEX.json',
        'SERVER_COLLECTOR_RECEIPT.json','DOWNLOAD.json','COLLECT_COMMAND.json','VERIFY_COMMAND.json',
        'SCP_COMMAND.json','COLLECTOR_RELEASE.json','F_VOLUME_BEFORE_COLLECT.json',
        'F_VOLUME_BEFORE_DOWNLOAD.json','F_VOLUME_BEFORE_RESTORE.json','README.md',
        'root_adopter_prepared/adopt_by_root.py','root_adopter_prepared/PREPARED.json',
        'root_adopter_prepared/SOURCE_DIFF.patch']]
    assert len(new_names) == len(set(new_names))
    for name in sorted(new_names):
        b = (ROOT/name).read_bytes()
        assert name not in byname and len(b) < 1_000_000
        byname[name] = dict(source=name,destination='experimental/'+name[4:],sha256=sha(b),bytes=len(b))
    report_name = actual_base+'OFFSERVER_SINGLE_RECORD_DIAGNOSTIC_V2.json'
    assert byname[report_name]['sha256'] == 'e2aee292f365bfdaedb1f9810baf1d530774da4c9d3edb317902ba5711bfe883'
    diagnostic = read(ROOT/report_name)
    assert diagnostic['scientific_acceptances'] == 0 and diagnostic['diagnostic_records'] == 1
    assert diagnostic['fit_field_differences'][0]['difference'] == -1.1102230246251565e-16
    bindings = copy.deepcopy(selection['closed_role_bindings'])
    bindings['FL47SingleDiagnostic'] = dict(path=report_name,sha256=byname[report_name]['sha256'],expect={
        '/status':'SINGLE_RECORD_WINDOWS_SAVED_FIT_DIAGNOSTIC_NOT_ACCEPTANCE',
        '/diagnostic_records':1,'/fit_views_calls':1,'/scientific_acceptances':0,
        '/root_adopted':False,'/test':False,'/original_failure_preserved':True,
        '/prediction_mismatch_counts':{'native':0,'raw':0,'shared_calibration':0},
        '/metric_and_count_differences':[],'/root_receipt_differences':[]})
    trace_name = actual_base+'operation_trace001/ROOT_OPERATION_TRACE_REVIEW.json'
    assert byname[trace_name]['sha256'] == '4cb8a6dea7df06e02fbf8d45acf8ad1db509c227743a2aafa9f1b3d698e680f6'
    bindings['FL47OperationDiagnostic'] = dict(path=trace_name,sha256=byname[trace_name]['sha256'],expect={
        '/status':'ROOT_STDLIB_TWO_RUNTIME_OPERATION_TRACE_DIAGNOSTIC_NOT_ACCEPTANCE',
        '/first_different_operation':'log1p','/fit_calls':0,'/scientific_acceptances':0,
        '/root_adopted':False,'/test':False})
    gradient_name = gradient_base+'ROOT_ADOPTION_REVIEW.json'
    assert byname[gradient_name]['sha256'] == '185f0fa292be27b6921776818d13871cb8a5b787a765867b802362991d1f9a81'
    bindings['gradient32'] = dict(path=gradient_name,sha256=byname[gradient_name]['sha256'],expect={
        '/accepted_before':23,'/accepted_new':9,'/accepted_total':32,
        '/screen64_complete':False,'/method_champion_claim':False,'/final_test':False,
        '/all_negative_results_retained':True,'/new_CNN':0,'/new_training':0})
    bindings['current_state'] = None
    accepted = dict(native=272,three_view=260,FL_new=44,FL_reuse=4,gradient_screen=32,
                    Hybrid_screen=32,Hybrid_formal_new_accepted=1,Hybrid_reuse=4)
    boundary = dict(FL47_new_three_view_root_accepted=0,whole_windows_array_block_pass=False,
                    single_record_diagnostic_records=1,single_record_diagnostic_scientific_acceptances=0,
                    platform_cause_established=False,final_partition_metadata_only=True)
    facts = {role:pin['expect'] for role,pin in bindings.items() if pin}
    facts['current_state'] = {'/celeba_mechanism_v1/scientific_results_strictly_accepted':272,
                              '/celeba_mechanism_v1/three_view_new_models_accepted':260}
    manifest = dict(status='PREPARED51_SOURCE_ONLY_PENDING_ROOT_EIGHT_MUTABLE_BINDINGS',
        parent=PARENT,branch=old_index['branch'],origin=old_index['origin'],scope=SCOPE,accepted=accepted,
        bindings=bindings,files=sorted(byname.values(),key=lambda e:e['source']),parent_recovery_references=[],
        pending_mutable=list(selection['pending_mutable']),allowed_paths=sorted(set(byname)|set(selection['pending_mutable'])),
        exact3_interface_records_separate=3,**boundary,test_started=False,goal_complete=False,
        compact_subset_not_full_original_seal_mirror=True)
    save(HERE/'PREPARED_MANIFEST.json',manifest)
    save(HERE/'EXACT_SMALL_FAILURE_TXT.json',dict(paths=selection['exact_small_failure_txt']))
    template = dict(status='PENDING_ROOT_FINAL_MUTABLE_BYTES_NOT_EXECUTABLE',parent=PARENT,accepted=accepted,
                    mutable_files=selection['pending_mutable'])
    save(HERE/'ROOT_INPUTS_TEMPLATE.json',template)
    supplement = dict(status='SOURCE_ONLY_DIAGNOSTIC_ALLOWLIST_SUPPLEMENT_NO_COPY_NO_SCIENCE',
        original_plan_seal_sha256=sha((PLAN/'FILES_SHA256.json').read_bytes()),
        added_files=[byname[n] for n in sorted(new_names)],new_diagnostic_role_bindings={
            role:bindings[role] for role in ['FL47SingleDiagnostic','FL47OperationDiagnostic','gradient32']},
        causal_test_pending_sources_included=False,**boundary)
    save(SUPPLEMENT/'DIAGNOSTIC_INPUT_SUPPLEMENT.json',supplement)
    save(SUPPLEMENT/'HANDOFF.json',dict(status='SOURCE_INDEX_SUPPLEMENT_NOT_FROZEN_OR_PUBLISHED',
        actual_scope='tmp/publication51_actual_scope_20261011',added_files=len(new_names),
        added_bytes=sum(byname[n]['bytes'] for n in new_names),original_plan_seal_preserved=True))
    save(SUPPLEMENT/'FILES_SHA256.json',dict(files={q.name:dict(sha256=sha(q.read_bytes()),bytes=q.stat().st_size)
        for q in sorted(SUPPLEMENT.iterdir()) if q.is_file()}))
    old_pins = read(OLD/'SOURCE_PINS.json')
    pins = {key:old_pins[key] for key in ['git43','git44','verifier43']}
    pins['reused_successful50'] = {q.name:dict(path=q.relative_to(ROOT).as_posix(),sha256=sha(q.read_bytes()),bytes=q.stat().st_size)
        for q in [OLD/'publish_increment50.py',OLD/'finalize_spec.py',OLD/'verify_increment50.py',OLD/'FILES_SHA256.json']}
    pins['plan_seal'] = dict(path=(PLAN/'FILES_SHA256.json').relative_to(ROOT).as_posix(),
        sha256=sha((PLAN/'FILES_SHA256.json').read_bytes()))
    save(HERE/'SOURCE_PINS.json',pins)
    old_text = (OLD/'publish_increment50.py').read_text(encoding='utf-8')
    text = old_text
    changes = []
    for before,after in [
        ('Git50 compact preparation.','Git51 compact preparation.'),
        ("PARENT = 'f83999f4e972cd2c7a13f21dd6bef7ce47e5e159'",'PARENT = '+repr(PARENT)),
        ("Path('F:/YananResearchStorage/GuardFed/git_publication/increment50')","Path('F:/YananResearchStorage/GuardFed/git_publication/i51')")]:
        text = replace(text,before,after,changes)
    start = text.index('ACCEPTED=')
    end = text.index('ns.update(ACCEPTED=',start)
    text = replace(text,text[start:end],
        'ACCEPTED='+repr(accepted)+'\nROLES='+repr(set(bindings))+'\nFACTS='+repr(facts)+'\nBOUNDARY='+repr(boundary)+'\n',changes)
    text = replace(text,"PDF='outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/celeba_ten_method_native.pdf'",
        "SMALL_FAILURE_TXT=set(read(HERE/'EXACT_SMALL_FAILURE_TXT.json')['paths'])\nTRACE_PINS=read(HERE/'EXACT_TRACE_ADDITIONS.json')['pins']\noriginal_relative=relative\ntrace_relative=types.FunctionType(original_relative.__code__,dict(original_relative.__globals__,SUFFIXES=ns['SUFFIXES']-{'.stdout','.stderr'}),original_relative.__name__)\n\ndef relative(name):\n    return trace_relative(name) if name in TRACE_PINS else original_relative(name)\n\nns['relative']=relative",changes)
    text = replace(text,"p.suffix.lower() in EXTENSIONS or name==PDF, 'Not a compact owned source/report or explicit PDF'",
        "p.suffix.lower() in EXTENSIONS or name in SMALL_FAILURE_TXT or name in TRACE_PINS, 'Not compact source/report or an exact small failure/trace'",changes)
    before = "    if p.parts[0]=='outputs':\n        assert name.startswith('outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/')\n        return name\n"
    text = replace(text,before,'    # Git51 uses exact tmp and docs sources only.\n',changes)
    text = replace(text,"    return original_source(name,allowed)",
        "    result=original_source(name,allowed)\n    check_trace(name,result[1])\n    return result",changes)
    trace_check = '''def check_trace(name,b):
    if name not in TRACE_PINS:return
    pin=TRACE_PINS[name]
    assert sha(b)==pin['sha256'] and len(b)==pin['bytes']<20_000
    if name.endswith('.stdout'):
        def reject_constant(value):raise ValueError('Non-finite JSON trace: '+value)
        value=json.loads(b,parse_constant=reject_constant)
        assert isinstance(value,dict)
    elif name.endswith('/WINDOWS.stderr'):assert b==b''
    else:
        assert name.endswith('/LINUX.stderr') and len(b)==324


'''
    text = replace(text,'def source(name,allowed):',trace_check+'def source(name,allowed):',changes)
    text = replace(text,"        return private(name,allowed)",
        "        result=private(name,allowed)\n        check_trace(name,result[1])\n        return result",changes)
    text = replace(text,"    assert d['exact3_interface_records_separate']==3",
        "    assert d['exact3_interface_records_separate']==3\n    assert all(d[k]==v for k,v in BOUNDARY.items()), 'Failure/diagnostic boundaries are closed'",changes)
    for before,after in [
        ("'ROOT_FINAL_CLOSED50_INPUTS_READY'","'ROOT_FINAL_CLOSED51_INPUTS_READY'"),
        ("'CLOSED251_COMPACT_EXACT3_INTERFACE3_WITH_PRESERVED_CROSS_PLATFORM_AUDIT_FAILURE'",repr(SCOPE)),
        ("schema='publication50_frozen_bytes_v1'","schema='publication51_frozen_bytes_v1'"),
        ("scope=d['scope'],exact3_interface_records_separate=3,accepted=ACCEPTED", "scope=d['scope'],exact3_interface_records_separate=3,**BOUNDARY,accepted=ACCEPTED"),
        ("'publication50_frozen_bytes_v1'","'publication51_frozen_bytes_v1'"),
        ("'# Git50 sealed compact bytes'","'# Git51 sealed compact bytes'"),
        ("'<direct-original43-stage-for50>'","'<direct-original43-stage-for51>'")]:
        text = replace(text,before,after,changes)
    text = replace(text,"    out=ns['output'](name,d['total_bytes']*3)\n    snapshot=out/'source';snapshot.mkdir()",
        "    prospective=ns['OUTPUT_ROOT']/name/'source'\n    assert all(len(str(prospective/e['source']))<260 for e in d['files']), 'Use short i51 layout before first copy'\n    out=ns['output'](name,d['total_bytes']*3)\n    snapshot=out/'source';snapshot.mkdir()",changes)
    # The restored 50 wrapper must be byte-equivalent as text; original scientific code is never edited here.
    restored = text
    for c in reversed(changes):
        assert restored.count(c['after']) == 1
        restored = restored.replace(c['after'],c['before'])
    assert restored == old_text
    with (HERE/'publish_increment51.py').open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    finalize = (OLD/'finalize_spec.py').read_text(encoding='utf-8').replace('publish_increment50','publish_increment51')
    finalize = finalize.replace('CLOSED50','CLOSED51').replace('All16 role proofs','All role proofs')
    with (HERE/'finalize_spec.py').open('x',encoding='utf-8',newline='\n') as f:f.write(finalize)
    verify = (OLD/'verify_increment50.py').read_text(encoding='utf-8').replace('publish_increment50','publish_increment51')
    with (HERE/'verify_increment51.py').open('x',encoding='utf-8',newline='\n') as f:f.write(verify)
    save(HERE/'SOURCE_REBINDINGS.json',dict(publisher=changes,successful50_inverse_text_exact=True,
        finalizer_only_import_status_comment_rebinding=True,verifier_only_module_name_rebinding=True))
    with (HERE/'SOURCE_DIFF.patch').open('x',encoding='utf-8',newline='\n') as f:
        for a,b,before,after in [(OLD/'publish_increment50.py',HERE/'publish_increment51.py',old_text,text),
              (OLD/'finalize_spec.py',HERE/'finalize_spec.py',(OLD/'finalize_spec.py').read_text(encoding='utf-8'),finalize),
              (OLD/'verify_increment50.py',HERE/'verify_increment51.py',(OLD/'verify_increment50.py').read_text(encoding='utf-8'),verify)]:
            f.write(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=a.name,tofile=b.name)))
    for q in [HERE/'publish_increment51.py',HERE/'finalize_spec.py',HERE/'verify_increment51.py']:
        compile(q.read_text(encoding='utf-8'),str(q),'exec')
    print(json.dumps(dict(status='SOURCE_PREPARED_NO_FREEZE_NO_GIT_NO_SCIENCE',files=len(byname),
        bytes=sum(e['bytes'] for e in byname.values()),closed_roles=len(bindings)-1,pending_mutable=8,
        diagnostic_added_files=len(new_names)),ensure_ascii=False))


if __name__ == '__main__':main()
