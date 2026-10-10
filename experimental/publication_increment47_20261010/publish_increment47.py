"""Git47 metadata binding around the exact Git44 byte transport; no commit/push."""
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
PARENT='d93f87a0f267e4d3d9c79a89c0853020c25c91ba'
ROLES={'current_state','native218','FL32','gatesreview'};OPTIONAL=set()
base.PARENT=PARENT;base.ROLES=ROLES;base.OPTIONAL=OPTIONAL
base.ns.update(PARENT=PARENT,ROLES=ROLES,OUTPUT_ROOT=Path('F:/YananResearchStorage/GuardFed/git_publication/increment47'))
CHECKOUT,BRANCH,ORIGIN,git,storage,relative=(getattr(base,n) for n in ('CHECKOUT','BRANCH','ORIGIN','git','storage','relative'))
old_text=(OLD/'publish_increment44.py').read_text(encoding='utf8')
node=next(n for n in ast.parse(old_text).body if isinstance(n,ast.FunctionDef) and n.name=='plan')
plan_source=ast.get_source_segment(old_text,node).replace('CLOSED200_COMPACT_EVIDENCE_AND_AUTHOR_REVIEW_NO_BULK','CLOSED218_REPLAY212_FL32_AND_SOURCE_REVIEWS_NO_BULK').replace('publication44_frozen_bytes_v1','publication47_frozen_bytes_v1')
exec(compile(plan_source,str(HERE/'publish_increment47.py')+':original_plan_metadata','exec'),base.__dict__)
original_plan=base.plan

def configure(accepted):
    assert accepted == {'native':218,'three_view':212}, 'Git47 exact accepted cutoff is 218/212'
    base.ACCEPTED=accepted;base.ns['ACCEPTED']=accepted
    base.REQUIRED_FACTS={
      'current_state':{'/celeba_mechanism_v1/scientific_results_strictly_accepted':218,
        '/celeba_mechanism_v1/three_view_new_models_accepted':212,
        '/flgmm_fullcoverage_v2_20261009/new_accepted':32,
        '/gradient64_validation_search_20261010/offserver_accepted':5,
        '/logofair100_fullcoverage_20261010/offserver_accepted':0,
        '/logofair100_fullcoverage_20261010/root_startup_sha256':'1ee8a6b002320e6dae552f816772f298d1ecf15e085a90fbecec775dbece8694'},
      'native218':{'/total_new_strict_and_offserver':218,'/test':False},
      'FL32':{'/accepted_before':28,'/accepted_new':4,'/accepted_total':32,'/planned_new':96,'/reused_separately':4,'/final_test':False},
      'gatesreview':{'/status':'INDEPENDENT_SOURCE_REVIEW_PASS_NOT_RUNTIME_OR_DISPATCH_APPROVAL',
        '/source_adoptable':True,'/actual_execution_authorized':False,
        '/source_seal_sha256':'f068ad7f51fd6981b2211725d39009a5cb5ebbecfa1de680cf0d46aa52ef7760'}}


def plan(spec):
    configure(spec['accepted'])
    return original_plan(spec)

base.ns['plan']=plan
stage_text=base.stage_text.replace('publication44_frozen_bytes_v1','publication47_frozen_bytes_v1').replace('# Git44 sealed evidence/source bytes','# Git47 sealed evidence/source bytes')
exec(compile(stage_text,str(HERE/'publish_increment47.py')+':original_stage','exec'),base.ns)
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
