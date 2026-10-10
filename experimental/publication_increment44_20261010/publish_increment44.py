"""Git44 curated metadata; original Git43 byte transport, no commit/push implementation."""
from pathlib import Path
import argparse, ast, hashlib, json, os, subprocess, sys
if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
    raise RuntimeError('Assertions are publication guards; optimized Python is refused')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / 'tmp/publication_increment43_20261010/publish_increment43.py'
PRIOR_SHA = '16c8f226a8169fc046fcfb02d7d74e900e4bdd12835d3bece0a9f868ad03f5ea'
raw = PRIOR.read_bytes()
assert hashlib.sha256(raw).hexdigest() == PRIOR_SHA
ns = {'__name__': 'git43_transport_reused', '__file__': str(PRIOR)}
exec(compile(raw, str(PRIOR), 'exec'), ns)
PARENT = '7ef54a00a3ef0af32e0805b6a16a4e21f081dcb3'
ACCEPTED = {'native': 200, 'three_view': 200}
ROLES = {'native200', 'replay200', 'tableC100', 'rebuttalC100', 'logofair32', 'current_state'}
OPTIONAL = {'hybrid27', 'logofair100_startup'}
REQUIRED_FACTS = {
    'native200': {'/total_new_strict_and_offserver': 200},
    'replay200': {'/cumulative_accepted': 200},
    'tableC100': {'/paired_models': 100, '/complete_scenes': 10},
    'rebuttalC100': {'/complete_C_scenes': 10, '/author_review_only': True, '/manuscript_applied': False},
    'logofair32': {'/accepted_count': 32, '/selected_recipe/id': 'LoGoFair-DP_07'},
    'current_state': {'/celeba_mechanism_v1/scientific_results_strictly_accepted': 200,
                      '/celeba_mechanism_v1/three_view_new_models_accepted': 200}}
ns.update(PARENT=PARENT, ROOT=ROOT, ACCEPTED=ACCEPTED, ROLES=ROLES,
          OUTPUT_ROOT=Path('F:/YananResearchStorage/GuardFed/git_publication/increment44'))
ns['SUFFIXES'].update({'.tar','.bin','.safetensors','.ckpt','.pickle','.log'})
CHECKOUT, BRANCH, ORIGIN = (ns[k] for k in ('CHECKOUT', 'BRANCH', 'ORIGIN'))
source, sha, read, relative, destination, pointer = (ns[k] for k in ('source','sha','read','relative','destination','pointer'))
_storage = ns['storage']

def storage(required=0):
    result = _storage(required)
    for key in ('GIT_DIR','GIT_WORK_TREE','GIT_COMMON_DIR','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES','GIT_INDEX_FILE'):
        assert not os.environ.get(key), 'Git path redirect is refused: ' + key
    objects = CHECKOUT / '.git/objects'
    assert objects.is_dir() and not objects.is_symlink() and objects.resolve().drive.upper() == 'F:'
    alt = (objects / 'info/alternates').read_text(encoding='utf8').strip()
    assert Path(alt).resolve() == (ROOT / '.git/objects').resolve()
    return result

def git(*args):
    return subprocess.run(['git','-c','core.longpaths=true','-c','safe.directory='+CHECKOUT.as_posix(),
        '-C',str(CHECKOUT),*args],capture_output=True,check=True,timeout=120).stdout

def plan(spec):
    assert spec['parent'] == PARENT and spec['branch'] == BRANCH
    assert spec['scope'] == 'CLOSED200_COMPACT_EVIDENCE_AND_AUTHOR_REVIEW_NO_BULK'
    assert spec['accepted'] == ACCEPTED and set(spec['bindings']) == ROLES
    assert set(spec['optional_bindings']) == OPTIONAL
    assert spec['test_started'] is False and spec['goal_complete'] is False
    allowed = spec['allowed_paths']; entries = spec['files']
    assert len({e['source'] for e in entries}) == len(entries)
    selected = {e['source']: e for e in entries}
    for role, pin in {**spec['bindings'], **spec['optional_bindings']}.items():
        if pin is None and role in OPTIONAL:
            continue
        assert pin and pin.get('sha256') and pin.get('expect'), 'Actual root binding missing: ' + role
        _, b = source(pin['path'], allowed); assert sha(b) == pin['sha256']
        facts = REQUIRED_FACTS.get(role, {})
        assert all(pin['expect'].get(k) == v for k,v in facts.items())
        for key,value in pin['expect'].items():
            assert pointer(json.loads(b),key) == value, 'Root fact differs: '+role+key
        assert selected[pin['path']]['sha256'] == pin['sha256']
    files=[]
    for name,e in sorted(selected.items()):
        _, b=source(name,allowed)
        assert e['sha256'] and e['bytes'] is not None, 'Unfrozen input: '+name
        assert sha(b)==e['sha256'] and len(b)==e['bytes']
        files.append(dict(source=name,destination=destination(name),sha256=sha(b),bytes=len(b)))
    assert len({e['destination'] for e in files})==len(files)
    assert sum(e['bytes'] for e in files)<ns['LIMIT']
    if spec['parent_recovery_references']:
        storage()
        for ref in spec['parent_recovery_references']:
            assert ref['parent_commit']==PARENT and destination(ref['source'])==ref['destination']
            original=git('show',PARENT+':'+ref['destination'])
            assert sha(original)==ref['sha256'] and len(original)==ref['bytes']
    return dict(schema='publication44_frozen_bytes_v1',parent=PARENT,branch=BRANCH,origin=ORIGIN,
        scope=spec['scope'],accepted=ACCEPTED,test_started=False,goal_complete=False,
        bindings=spec['bindings'],optional_bindings=spec['optional_bindings'],allowed_paths=allowed,
        files=files,total_bytes=sum(e['bytes'] for e in files),parent_recovery_references=spec['parent_recovery_references'])

# Exactly three stage text changes: schema, count contract, human-readable attribute comment.
tree=ast.parse(raw)
stage_text=ast.get_source_segment(raw.decode('utf8'),next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage'))
old="assert d['accepted'] == {'native': 188, 'three_view': 180, 'FL_new': 22} and set(d['bindings']) == ROLES"
assert stage_text.count(old)==1
stage_text=stage_text.replace(old,"assert d['accepted'] == ACCEPTED and set(d['bindings']) == ROLES")
stage_text=stage_text.replace('publication43_frozen_bytes_v1','publication44_frozen_bytes_v1').replace('# Git43 sealed source/startup bytes','# Git44 sealed evidence/source bytes')
ns.update(git=git,storage=storage,plan=plan)
exec(compile(stage_text,str(HERE/'publish_increment44.py')+':stage_reuse','exec'),ns)
freeze,stage=ns['freeze'],ns['stage']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','freeze','stage']);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--output-name');a=p.parse_args()
    assert sha(a.input.read_bytes())==a.sha256
    if a.action=='plan':
        d=plan(read(a.input));print(json.dumps(dict(status='PLAN_ONLY_NO_WRITES',files=len(d['files']),bytes=d['total_bytes'])))
    else:
        assert a.output_name
        (freeze if a.action=='freeze' else stage)(a.input,a.sha256,a.output_name)
