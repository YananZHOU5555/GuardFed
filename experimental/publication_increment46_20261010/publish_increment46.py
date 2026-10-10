"""Git46 metadata binding around the exact Git44 byte transport; no commit/push."""
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
PARENT='e7ea15c5d2e77c40be932e20f0efe941e584fb13'
ROLES={'current_state','native212','replay212','tableA10'};OPTIONAL=set()
base.PARENT=PARENT;base.ROLES=ROLES;base.OPTIONAL=OPTIONAL
base.ns.update(PARENT=PARENT,ROLES=ROLES,OUTPUT_ROOT=Path('F:/YananResearchStorage/GuardFed/git_publication/increment46'))
CHECKOUT,BRANCH,ORIGIN,git,storage,relative=(getattr(base,n) for n in ('CHECKOUT','BRANCH','ORIGIN','git','storage','relative'))
old_text=(OLD/'publish_increment44.py').read_text(encoding='utf8')
node=next(n for n in ast.parse(old_text).body if isinstance(n,ast.FunctionDef) and n.name=='plan')
plan_source=ast.get_source_segment(old_text,node).replace('CLOSED200_COMPACT_EVIDENCE_AND_AUTHOR_REVIEW_NO_BULK','CLOSED212_A12_AND_ADOPTED_A_TABLE_NO_BULK').replace('publication44_frozen_bytes_v1','publication46_frozen_bytes_v1')
exec(compile(plan_source,str(HERE/'publish_increment46.py')+':original_plan_metadata','exec'),base.__dict__)
original_plan=base.plan

def configure(accepted):
    assert accepted == {'native':212,'three_view':212}, 'Git46 exact accepted cutoff is 212/212'
    base.ACCEPTED=accepted;base.ns['ACCEPTED']=accepted
    base.REQUIRED_FACTS={
      'current_state':{'/celeba_mechanism_v1/scientific_results_strictly_accepted':212,
        '/celeba_mechanism_v1/three_view_new_models_accepted':212,
        '/gradient64_validation_search_20261010/offserver_accepted':5,
        '/logofair100_fullcoverage_20261010/root_startup_sha256':'1ee8a6b002320e6dae552f816772f298d1ecf15e085a90fbecec775dbece8694'},
      'native212':{'/total_new_strict_and_offserver':212},
      'replay212':{'/status':'ROOT_A12_SAVED_ARRAYS_AND_NATIVE212_RESTORE_CHAIN_ADOPTED',
        '/new_accepted':12,'/prior_accepted':200,'/cumulative_accepted':212,'/test':False},
      'tableA10':{'/paired_models':10,'/complete_scenes':1}}


def plan(spec):
    configure(spec['accepted'])
    return original_plan(spec)

base.ns['plan']=plan
stage_text=base.stage_text.replace('publication44_frozen_bytes_v1','publication46_frozen_bytes_v1').replace('# Git44 sealed evidence/source bytes','# Git46 sealed evidence/source bytes')
exec(compile(stage_text,str(HERE/'publish_increment46.py')+':original_stage','exec'),base.ns)
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
