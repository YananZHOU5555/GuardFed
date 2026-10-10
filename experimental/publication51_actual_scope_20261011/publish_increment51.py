"""Git51 compact preparation. Explicit freeze/stage only; no commit/push/fetch."""
from pathlib import Path
import argparse, ast, datetime, hashlib, json, os, re, subprocess, sys, types
if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
    raise RuntimeError('Optimized Python would disable publication guards')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = '4dfc9403c182c8f192c374e396eb2f564970159c'
BASE = ROOT / 'tmp/publication_increment43_20261010/publish_increment43.py'
BASE_SHA = '16c8f226a8169fc046fcfb02d7d74e900e4bdd12835d3bece0a9f868ad03f5ea'
STORAGE_BASE = ROOT / 'tmp/publication_increment44_20261010/publish_increment44.py'
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(BASE.read_bytes()) == BASE_SHA
# The actual SHA of this original storage source is bound in SOURCE_PINS.json.
assert sha(STORAGE_BASE.read_bytes()) == json.loads((HERE / 'SOURCE_PINS.json').read_bytes())['git44']['sha256']
raw = BASE.read_text(encoding='utf-8')
tree = ast.parse(raw)
names = {'storage','git','relative','source','destination','pointer','save','output','freeze','stage'}
ns = dict(globals(), __name__='direct_git43_transport', __file__=str(BASE))
selected = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or
            isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
              {'CHECKOUT','BRANCH','ORIGIN','LIMIT','BANNED','SUFFIXES','SECRET','sha','read'} for t in n.targets) or
            isinstance(n,ast.FunctionDef) and n.name in names]
exec(compile(ast.Module(body=selected,type_ignores=[]),str(BASE),'exec'),ns)
ns.update(ROOT=ROOT,PARENT=PARENT,OUTPUT_ROOT=Path('F:/YananResearchStorage/GuardFed/git_publication/i51'))
ns['SUFFIXES'].update({'.tar','.bin','.safetensors','.ckpt','.pickle','.log','.stdout','.stderr','.npz','.npy'})
ns['BANNED'].update({'raw','rawlogs','models'})
CHECKOUT,BRANCH,ORIGIN = (ns[k] for k in ('CHECKOUT','BRANCH','ORIGIN'))
relative,pointer,read = (ns[k] for k in ('relative','pointer','read'))
ACCEPTED={'native': 272, 'three_view': 260, 'FL_new': 44, 'FL_reuse': 4, 'gradient_screen': 32, 'Hybrid_screen': 32, 'Hybrid_formal_new_accepted': 1, 'Hybrid_reuse': 4}
ROLES={'FL47LinuxWhole', 'current_state', 'A60', 'finalIDsMetadata', 'replyA60', 'FL47Transport', 'FL47SingleDiagnostic', 'FL47WindowsFailure', 'native264', 'gradient32', 'native272', 'FL47OperationDiagnostic', 'mechanism260'}
FACTS={'native264': {'/total_new_strict_and_offserver': 264}, 'native272': {'/total_new_strict_and_offserver': 272, '/root_adopted': True}, 'mechanism260': {'/cumulative_accepted': 260, '/new_accepted': 9, '/prior_accepted': 251}, 'A60': {'/paired_models': 60, '/complete_scenes': 6, '/root_adoption': True}, 'replyA60': {'/original_comments': 24, '/A60_incorporated': True, '/manuscript_applied': False, '/final_test': False}, 'finalIDsMetadata': {'/candidate_target/partition': 2, '/candidate_target/n': 19962, '/final_evaluation_started': False, '/protocol_frozen': False}, 'FL47LinuxWhole': {'/status': 'LINUX_ORIGINAL_FLGMM47_WHOLE_SAVED_CHECK_PASS_NOT_ROOT_ADOPTED', '/cached_root_refits': 47, '/new_CNN': 0, '/test': False}, 'FL47Transport': {'/member_count': 98, '/root_adopted': False, '/new_CNN': 0, '/test': False}, 'FL47WindowsFailure': {'/status': 'FLGMM47_WINDOWS_ARRAY_CHECK_FAILED_PRESERVED', '/completed': 4, '/root_adopted': False}, 'FL47SingleDiagnostic': {'/status': 'SINGLE_RECORD_WINDOWS_SAVED_FIT_DIAGNOSTIC_NOT_ACCEPTANCE', '/diagnostic_records': 1, '/fit_views_calls': 1, '/scientific_acceptances': 0, '/root_adopted': False, '/test': False, '/original_failure_preserved': True, '/prediction_mismatch_counts': {'native': 0, 'raw': 0, 'shared_calibration': 0}, '/metric_and_count_differences': [], '/root_receipt_differences': []}, 'FL47OperationDiagnostic': {'/status': 'ROOT_STDLIB_TWO_RUNTIME_OPERATION_TRACE_DIAGNOSTIC_NOT_ACCEPTANCE', '/first_different_operation': 'log1p', '/fit_calls': 0, '/scientific_acceptances': 0, '/root_adopted': False, '/test': False}, 'gradient32': {'/accepted_before': 23, '/accepted_new': 9, '/accepted_total': 32, '/screen64_complete': False, '/method_champion_claim': False, '/final_test': False, '/all_negative_results_retained': True, '/new_CNN': 0, '/new_training': 0}, 'current_state': {'/celeba_mechanism_v1/scientific_results_strictly_accepted': 272, '/celeba_mechanism_v1/three_view_new_models_accepted': 260}}
BOUNDARY={'FL47_new_three_view_root_accepted': 0, 'whole_windows_array_block_pass': False, 'single_record_diagnostic_records': 1, 'single_record_diagnostic_scientific_acceptances': 0, 'platform_cause_established': False, 'final_partition_metadata_only': True}
ns.update(ACCEPTED=ACCEPTED,ROLES=ROLES)
# Original Git44 guard keeps ROOT at the real E repository for the alternates check.
ns['_storage']=ns['storage']
tree44=ast.parse(STORAGE_BASE.read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree44.body if isinstance(n,ast.FunctionDef) and n.name in {'storage','git'}],type_ignores=[]),str(STORAGE_BASE),'exec'),ns)
storage,git=ns['storage'],ns['git']
EXTENSIONS={'.py','.json','.md','.tex','.csv','.patch','.sha256','.conf','.sh'}
SMALL_FAILURE_TXT=set(read(HERE/'EXACT_SMALL_FAILURE_TXT.json')['paths'])
TRACE_PINS=read(HERE/'EXACT_TRACE_ADDITIONS.json')['pins']
original_relative=relative
trace_relative=types.FunctionType(original_relative.__code__,dict(original_relative.__globals__,SUFFIXES=ns['SUFFIXES']-{'.stdout','.stderr'}),original_relative.__name__)

def relative(name):
    return trace_relative(name) if name in TRACE_PINS else original_relative(name)

ns['relative']=relative
original_source=ns['source']


def validate_file_name(name,allowed):
    assert name in allowed, 'Only exact reviewed file allowlist'
    p=relative(name)
    assert p.suffix.lower() in EXTENSIONS or name in SMALL_FAILURE_TXT or name in TRACE_PINS, 'Not compact source/report or an exact small failure/trace'
    assert p.name not in {'result.json','diagnostics.json','state.json','rng_final.json'}


def check_trace(name,b):
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


def source(name,allowed):
    validate_file_name(name,allowed)
    result=original_source(name,allowed)
    check_trace(name,result[1])
    return result


def destination(name):
    p=relative(name)
    # Git51 uses exact tmp and docs sources only.
    return ns['original_destination'](name)


def closed_scope(d):
    prepared=read(HERE/'PREPARED_MANIFEST.json')
    assert d['scope']==prepared['scope'] and d['accepted']==ACCEPTED
    assert d['test_started'] is False and d['goal_complete'] is False
    packet=read(HERE/'FILES_SHA256.json')['files']
    own=HERE.relative_to(ROOT).as_posix()+'/'
    fixed={e['source']:e for e in prepared['files']+prepared['parent_recovery_references']}
    for name,pin in packet.items():
        fixed[own+name]=dict(source=own+name,destination=destination(own+name),**pin)
    seal=HERE/'FILES_SHA256.json'
    fixed[own+seal.name]=dict(source=own+seal.name,destination=destination(own+seal.name),
                            sha256=sha(seal.read_bytes()),bytes=seal.stat().st_size)
    allowed=set(fixed)|set(prepared['pending_mutable'])
    files={e['source']:e for e in d['files']};refs={e['source']:e for e in d['parent_recovery_references']}
    assert not set(files)&set(refs) and set(files)|set(refs)==allowed
    assert len(files)==len(d['files']) and len(refs)==len(d['parent_recovery_references'])
    assert set(d['allowed_paths'])==allowed and len(d['allowed_paths'])==len(allowed)
    for name,e in fixed.items():
        actual=(files|refs)[name]
        assert all(actual[k]==e[k] for k in ('source','destination','sha256','bytes')), 'Closed file differs: '+name
    assert set(d['bindings'])==ROLES
    for role,pin in d['bindings'].items():
        assert pin and pin['path'] in files, 'Every role proof must be a real snapshot file: '+role
        assert files[pin['path']]['sha256']==pin['sha256']
        if role!='current_state':assert pin==prepared['bindings'][role], 'Closed role binding differs: '+role
        else:assert pin['path']==prepared['pending_mutable'][0], 'Exact current-state file required'
    assert d['exact3_interface_records_separate']==3
    assert all(d[k]==v for k,v in BOUNDARY.items()), 'Failure/diagnostic boundaries are closed'


def plan(d):
    assert d['status']=='ROOT_FINAL_CLOSED51_INPUTS_READY' and not d['pending_mutable'], 'Root final mutable freeze required'
    assert d['parent']==PARENT and d['branch']==BRANCH and d['accepted']==ACCEPTED
    assert d['scope']=='CLOSED_NATIVE272_MECHANISM260_A60_AUTHOR_REVIEW_AND_FL47_PRESERVED_WINDOWS_ARRAY_FAILURE_WITH_FINAL_ID_METADATA_ONLY'
    assert set(d['bindings'])==ROLES and d['test_started'] is False and d['goal_complete'] is False
    allowed=d['allowed_paths']; entries=d['files']; refs=d['parent_recovery_references']
    assert len(allowed)==len(set(allowed)) and len({e['source'] for e in entries})==len(entries)
    closed_scope(d)
    available={e['source']:e for e in entries}
    for role,pin in d['bindings'].items():
        assert pin and pin['sha256'] and pin['expect'], 'Missing actual role: '+role
        _,b=source(pin['path'],allowed)
        assert sha(b)==pin['sha256'] and available[pin['path']]['sha256']==pin['sha256']
        assert all(pin['expect'].get(k)==v for k,v in FACTS[role].items())
        for k,v in pin['expect'].items():
            assert pointer(json.loads(b),k)==v, 'Actual role differs: '+role+k
    files=[]
    for e in entries:
        _,b=source(e['source'],allowed)
        assert sha(b)==e['sha256'] and len(b)==e['bytes']
        assert destination(e['source'])==e['destination']
        files.append(e)
    storage()
    for r in refs:
        assert r['parent_commit']==PARENT and destination(r['source'])==r['destination']
        b=git('show',PARENT+':'+r['destination'])
        assert sha(b)==r['sha256'] and len(b)==r['bytes']
    assert len({e['destination'] for e in files})==len(files)
    assert sum(e['bytes'] for e in files)<ns['LIMIT']
    return dict(schema='publication51_frozen_bytes_v1',parent=PARENT,branch=BRANCH,origin=ORIGIN,
        scope=d['scope'],exact3_interface_records_separate=3,**BOUNDARY,accepted=ACCEPTED,test_started=False,goal_complete=False,bindings=d['bindings'],
        allowed_paths=allowed,files=files,total_bytes=sum(e['bytes'] for e in files),parent_recovery_references=refs)


def freeze(path,expected,name):
    b=path.read_bytes(); assert sha(b)==expected
    d=plan(json.loads(b)); d['spec_sha256']=expected
    prospective=ns['OUTPUT_ROOT']/name/'source'
    assert all(len(str(prospective/e['source']))<260 for e in d['files']), 'Use short i51 layout before first copy'
    out=ns['output'](name,d['total_bytes']*3)
    snapshot=out/'source';snapshot.mkdir()
    for e in d['files']:
        _,content=source(e['source'],d['allowed_paths'])
        assert sha(content)==e['sha256']
        q=snapshot/e['source'];q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(content)
        assert sha(q.read_bytes())==e['sha256']
    assert all((snapshot/pin['path']).is_file() for pin in d['bindings'].values())
    ns['save'](out/'SOURCE_SNAPSHOT.json',{'source_root':str(snapshot),'files':d['files'],'parent':PARENT})
    d.update(source_root=str(snapshot),source_snapshot_sha256=sha((out/'SOURCE_SNAPSHOT.json').read_bytes()))
    ns['save'](out/'FROZEN_INPUTS.json',d)
    return d


# Reuse the original stage body directly, without importing 48/49 wrappers.
stage_text=ast.get_source_segment(raw,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage'))
stage_text=stage_text.replace('publication43_frozen_bytes_v1','publication51_frozen_bytes_v1')
stage_text=stage_text.replace("assert d['accepted'] == {'native': 188, 'three_view': 180, 'FL_new': 22} and set(d['bindings']) == ROLES",
                            "assert d['accepted'] == ACCEPTED and set(d['bindings']) == ROLES")
stage_text=stage_text.replace('# Git43 sealed source/startup bytes','# Git51 sealed compact bytes')
ns['original_destination']=ns['destination']
ns.update(source=source,destination=destination,plan=plan)
exec(compile(stage_text,'<direct-original43-stage-for51>','exec'),ns)
original_stage=ns['stage']


def stage(path,expected,name):
    d=read(path); assert sha(path.read_bytes())==expected
    closed_scope(d)
    storage(d['total_bytes']*3)
    root=Path(d['source_root']).resolve()
    assert root.is_relative_to(ns['OUTPUT_ROOT'].resolve()) and root.name=='source' and not root.is_symlink()
    receipt=root.parent/'SOURCE_SNAPSHOT.json'; assert sha(receipt.read_bytes())==d['source_snapshot_sha256']
    snap=read(receipt); assert snap['parent']==PARENT and snap['files']==d['files'] and Path(snap['source_root']).resolve()==root
    # Only the source-reader has private F globals; storage/alternates still use real E ROOT.
    private=types.FunctionType(original_source.__code__,dict(original_source.__globals__,ROOT=root),original_source.__name__)
    def frozen_source(name,allowed):
        validate_file_name(name,allowed)
        result=private(name,allowed)
        check_trace(name,result[1])
        return result
    prior=ns['source'];ns['source']=frozen_source
    try:
        attr=CHECKOUT/'.gitattributes'
        assert attr.is_file() and attr.read_bytes()==git('show',PARENT+':.gitattributes'), 'Parent attributes must be present byte-exact'
        for role,pin in d['bindings'].items():
            assert pin and pin['expect'] and all(pin['expect'].get(k)==v for k,v in FACTS[role].items())
            assert (root/pin['path']).is_file(), 'Role must exist in F source snapshot: '+role
            _,b=frozen_source(pin['path'],d['allowed_paths'])
            assert sha(b)==pin['sha256']
            for k,v in pin['expect'].items():assert pointer(json.loads(b),k)==v
        return original_stage(path,expected,name)
    finally:
        ns['source']=prior


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','freeze','stage'])
    p.add_argument('--input',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output-name')
    a=p.parse_args();assert sha(a.input.read_bytes())==a.sha256
    if a.action=='plan':print(json.dumps(plan(read(a.input)),ensure_ascii=False))
    else:
        assert a.output_name
        result=(freeze if a.action=='freeze' else stage)(a.input,a.sha256,a.output_name)
        if result:print(json.dumps({'files':len(result['files']),'bytes':result['total_bytes']}))
