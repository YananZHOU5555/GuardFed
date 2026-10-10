"""Git50 compact preparation. Explicit freeze/stage only; no commit/push/fetch."""
from pathlib import Path
import argparse, ast, datetime, hashlib, json, os, re, subprocess, sys, types
if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
    raise RuntimeError('Optimized Python would disable publication guards')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = 'f83999f4e972cd2c7a13f21dd6bef7ce47e5e159'
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
ns.update(ROOT=ROOT,PARENT=PARENT,OUTPUT_ROOT=Path('F:/YananResearchStorage/GuardFed/git_publication/increment50'))
ns['SUFFIXES'].update({'.tar','.bin','.safetensors','.ckpt','.pickle','.log','.stdout','.stderr','.npz','.npy'})
ns['BANNED'].update({'raw','rawlogs','models'})
CHECKOUT,BRANCH,ORIGIN = (ns[k] for k in ('CHECKOUT','BRANCH','ORIGIN'))
relative,pointer,read = (ns[k] for k in ('relative','pointer','read'))
ACCEPTED={'native':251,'three_view':251,'FL_new':44,'FL_reuse':4,'gradient_screen':23,
          'Hybrid_screen':32,'Hybrid_formal_new_accepted':1,'Hybrid_reuse':4}
ROLES={'native243','native251','replay240','replay251','A40','A50','reader_clarity','Hybrid1',
       'FL44','gradient23','nativePDF','cnn_bridge','exact3_metadata','exact3_independent','current_state','exact3_scientific'}
FACTS={'native243':{'/total_new_strict_and_offserver':243},'native251':{'/total_new_strict_and_offserver':251},
 'replay240':{'/cumulative_accepted':240},'replay251':{'/cumulative_accepted':251},
 'A40':{'/paired_models':40,'/complete_scenes':4},'A50':{'/paired_models':50,'/complete_scenes':5},
 'reader_clarity':{'/original_comments':24,'/A50_incorporated':False,'/manuscript_applied':False},
 'Hybrid1':{'/cumulative_accepted':1,'/rounds':70},'FL44':{'/accepted_total':44,'/reused_separately':4},
 'gradient23':{'/accepted_total':23,'/screen64_complete':False},'nativePDF':{'/pages':3,'/new_statistics':0},
 'cnn_bridge':{'/source_adopted':True,'/new_three_view_scientific_results':0,'/science_dispatch_authorized':False},
 'exact3_metadata':{'/source_members_verified':38,'/new_CNN':0,'/new_fit':0,'/new_acceptance':0,'/SSH':0},
 'exact3_independent':{'/source_adoptable':True,'/runtime_pass_claimed':False,'/science_dispatch_authorized':False},
 'exact3_scientific':{'/interface_records_accepted': 3, '/mechanism_three_view_cutoff_unchanged': 251, '/final_test': False, '/cross_platform_audit_failure_preserved': True, '/root_adoption': True, '/Linux_whole_original_saved_check_pass': True, '/Windows_whole_saved_check_pass': False, '/Windows_original_array_refit_block_pass': True},
 'current_state':{'/celeba_mechanism_v1/scientific_results_strictly_accepted':251,
                  '/celeba_mechanism_v1/three_view_new_models_accepted':251}}
ns.update(ACCEPTED=ACCEPTED,ROLES=ROLES)
# Original Git44 guard keeps ROOT at the real E repository for the alternates check.
ns['_storage']=ns['storage']
tree44=ast.parse(STORAGE_BASE.read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree44.body if isinstance(n,ast.FunctionDef) and n.name in {'storage','git'}],type_ignores=[]),str(STORAGE_BASE),'exec'),ns)
storage,git=ns['storage'],ns['git']
EXTENSIONS={'.py','.json','.md','.tex','.csv','.patch','.sha256','.conf','.sh'}
PDF='outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/celeba_ten_method_native.pdf'
original_source=ns['source']


def validate_file_name(name,allowed):
    assert name in allowed, 'Only exact reviewed file allowlist'
    p=relative(name)
    assert p.suffix.lower() in EXTENSIONS or name==PDF, 'Not a compact owned source/report or explicit PDF'
    assert p.name not in {'result.json','diagnostics.json','state.json','rng_final.json'}


def source(name,allowed):
    validate_file_name(name,allowed)
    return original_source(name,allowed)


def destination(name):
    p=relative(name)
    if p.parts[0]=='outputs':
        assert name.startswith('outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/')
        return name
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


def plan(d):
    assert d['status']=='ROOT_FINAL_CLOSED50_INPUTS_READY' and not d['pending_mutable'], 'Root final mutable freeze required'
    assert d['parent']==PARENT and d['branch']==BRANCH and d['accepted']==ACCEPTED
    assert d['scope']=='CLOSED251_COMPACT_EXACT3_INTERFACE3_WITH_PRESERVED_CROSS_PLATFORM_AUDIT_FAILURE'
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
    return dict(schema='publication50_frozen_bytes_v1',parent=PARENT,branch=BRANCH,origin=ORIGIN,
        scope=d['scope'],exact3_interface_records_separate=3,accepted=ACCEPTED,test_started=False,goal_complete=False,bindings=d['bindings'],
        allowed_paths=allowed,files=files,total_bytes=sum(e['bytes'] for e in files),parent_recovery_references=refs)


def freeze(path,expected,name):
    b=path.read_bytes(); assert sha(b)==expected
    d=plan(json.loads(b)); d['spec_sha256']=expected
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
stage_text=stage_text.replace('publication43_frozen_bytes_v1','publication50_frozen_bytes_v1')
stage_text=stage_text.replace("assert d['accepted'] == {'native': 188, 'three_view': 180, 'FL_new': 22} and set(d['bindings']) == ROLES",
                            "assert d['accepted'] == ACCEPTED and set(d['bindings']) == ROLES")
stage_text=stage_text.replace('# Git43 sealed source/startup bytes','# Git50 sealed compact bytes')
ns['original_destination']=ns['destination']
ns.update(source=source,destination=destination,plan=plan)
exec(compile(stage_text,'<direct-original43-stage-for50>','exec'),ns)
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
        return private(name,allowed)
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
