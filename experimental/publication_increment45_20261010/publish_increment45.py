"""Git45 metadata binding around the exact Git44 byte transport; no commit/push."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'tmp/publication_increment44_20261010'
H=lambda b:hashlib.sha256(b).hexdigest()
assert H((OLD/'FILES_SHA256.json').read_bytes())=='1abced4226020698b083d56dc5ddff4f772a24a8c7136c3d1c58988f088eb939'
for name,pin in json.loads((OLD/'FILES_SHA256.json').read_bytes())['files'].items():
    b=(OLD/name).read_bytes();assert H(b)==pin['sha256'] and len(b)==pin['bytes']
sp=importlib.util.spec_from_file_location('publication44_reused',OLD/'publish_increment44.py');base=importlib.util.module_from_spec(sp);sp.loader.exec_module(base)
PARENT='3fc72d336d80a7e51d1f0682de269fca22902938'
ROLES={'current_state','gradient5','LoGo100startup'};OPTIONAL={'native_delta'}
base.PARENT=PARENT;base.ROLES=ROLES;base.OPTIONAL=OPTIONAL
base.ns.update(PARENT=PARENT,ROLES=ROLES,OUTPUT_ROOT=Path('F:/YananResearchStorage/GuardFed/git_publication/increment45'))
CHECKOUT,BRANCH,ORIGIN,git,storage,relative=(getattr(base,n) for n in ('CHECKOUT','BRANCH','ORIGIN','git','storage','relative'))
old_text=(OLD/'publish_increment44.py').read_text(encoding='utf8')
node=next(n for n in ast.parse(old_text).body if isinstance(n,ast.FunctionDef) and n.name=='plan')
plan_source=ast.get_source_segment(old_text,node).replace('CLOSED200_COMPACT_EVIDENCE_AND_AUTHOR_REVIEW_NO_BULK','CLOSED_DELTA_AND_ACTUAL_LOGO100_STARTUP_NO_BULK').replace('publication44_frozen_bytes_v1','publication45_frozen_bytes_v1')
exec(compile(plan_source,str(HERE/'publish_increment45.py')+':original_plan_metadata','exec'),base.__dict__)
original_plan=base.plan

def configure(accepted):
    assert accepted == {'native':208,'three_view':200}, 'Git45 exact accepted cutoff is 208/200'
    assert accepted['three_view']==200, 'New replay acceptance is outside this increment scope'
    base.ACCEPTED=accepted;base.ns['ACCEPTED']=accepted
    base.REQUIRED_FACTS={
        'current_state':{'/celeba_mechanism_v1/scientific_results_strictly_accepted':accepted['native'],
                         '/celeba_mechanism_v1/three_view_new_models_accepted':200},
        'gradient5':{'/status':'ROOT_GRADIENT64_EXACT4_ORIGINAL_STRICT_OFFSERVER_ADOPTED','/accepted_before':1,'/accepted_new':4,'/accepted_total':5,'/screen64_complete':False},
        'LoGo100startup':{'/status':'ROOT_LOGOFAIR100_FIXED_RECIPE_ACTUAL_STARTUP_AND_FIRST_STRICT_FIT_PASS','/new_fits_planned':96,'/reused':4,'/fit_seed':1719,'/offserver_accepted':0,'/root_adopted':0,'/final_test':False}}

def plan(spec):
    configure(spec['accepted'])
    pin=spec['optional_bindings']['native_delta']
    if spec['accepted']['native']>200:
        assert pin and pin['expect'].get('/total_new_strict_and_offserver')==spec['accepted']['native'], 'Actual native delta root proof required'
    elif pin is not None:
        assert pin['expect'].get('/total_new_strict_and_offserver')==200
    return original_plan(spec)

base.ns['plan']=plan
stage_text=base.stage_text.replace('publication44_frozen_bytes_v1','publication45_frozen_bytes_v1').replace('# Git44 sealed evidence/source bytes','# Git45 sealed evidence/source bytes')
exec(compile(stage_text,str(HERE/'publish_increment45.py')+':original_stage','exec'),base.ns)
freeze=base.ns['freeze']
def stage(path,expected,name):
    data=json.loads(Path(path).read_bytes());configure(data['accepted'])
    return base.ns['stage'](path,expected,name)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','freeze','stage']);a.add_argument('--input',type=Path,required=True)
    a.add_argument('--sha256',required=True);a.add_argument('--output-name');x=a.parse_args();assert H(x.input.read_bytes())==x.sha256
    if x.action=='plan':
        d=plan(json.loads(x.input.read_bytes()));print(json.dumps(dict(status='PLAN_ONLY_NO_WRITES',files=len(d['files']),bytes=d['total_bytes'])))
    else:
        assert x.output_name;(freeze if x.action=='freeze' else stage)(x.input,x.sha256,x.output_name)
