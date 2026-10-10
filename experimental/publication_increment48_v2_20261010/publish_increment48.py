"""Git48 metadata binding around the exact Git44 byte transport; no commit/push."""
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
PARENT='d7ecf9f266f4d7ff6824b8cf1efd6483768b7281'
ROLES={'current_state','native220','replay220','gradient10','authorpatch','logofair100','A20table'};OPTIONAL=set()
base.PARENT=PARENT;base.ROLES=ROLES;base.OPTIONAL=OPTIONAL
base.ns.update(PARENT=PARENT,ROLES=ROLES,OUTPUT_ROOT=Path('F:/YananResearchStorage/GuardFed/git_publication/increment48'))
CHECKOUT,BRANCH,ORIGIN,git,storage,relative=(getattr(base,n) for n in ('CHECKOUT','BRANCH','ORIGIN','git','storage','relative'))
old_text=(OLD/'publish_increment44.py').read_text(encoding='utf8')
node=next(n for n in ast.parse(old_text).body if isinstance(n,ast.FunctionDef) and n.name=='plan')
plan_source=ast.get_source_segment(old_text,node).replace('CLOSED200_COMPACT_EVIDENCE_AND_AUTHOR_REVIEW_NO_BULK','CLOSED220_REPLAY220_GRADIENT10_LOGO100_A20_NO_BULK').replace('publication44_frozen_bytes_v1','publication48_v2_frozen_bytes_v1')
exec(compile(plan_source,str(HERE/'publish_increment48.py')+':original_plan_metadata','exec'),base.__dict__)
original_plan=base.plan

def configure(accepted):
    assert accepted == {'native':220,'three_view':220}, 'Git48 exact accepted cutoff is 220/220'
    base.ACCEPTED=accepted;base.ns['ACCEPTED']=accepted
    base.REQUIRED_FACTS={
      'current_state':{'/celeba_mechanism_v1/scientific_results_strictly_accepted':220,
        '/celeba_mechanism_v1/three_view_new_models_accepted':220,
        '/flgmm_fullcoverage_v2_20261009/new_accepted':32,
        '/gradient64_validation_search_20261010/offserver_accepted':10,
        '/logofair100_fullcoverage_20261010/offserver_accepted':100,
        '/logofair100_fullcoverage_20261010/root_startup_sha256':'1ee8a6b002320e6dae552f816772f298d1ecf15e085a90fbecec775dbece8694'},
      'native220':{'/total_new_strict_and_offserver':220,'/test':False},
      'replay220':{'/status':'ROOT_A20_SAVED_ARRAYS_AND_NATIVE220_RESTORE_CHAIN_ADOPTED',
        '/new_accepted':8,'/prior_accepted':212,'/cumulative_accepted':220,'/remaining620_new_accepted':40},
      'gradient10':{'/status':'ROOT_GRADIENT64_EXACT5_ORIGINAL_STRICT_OFFSERVER_ADOPTED',
        '/accepted_before':5,'/accepted_new':5,'/accepted_total':10,'/final_test':False,'/screen64_complete':False},
      'authorpatch':{'/status':'ROOT_TEXT_REVIEW_AND_ORIGINAL_READONLY_PATCH_CHECK_PASS_NOT_APPLIED',
        '/original_comments_preserved':24,'/new_scientific_acceptance':0,'/manuscript_applied':False,'/test':False},
      'logofair100':{'/status':'ROOT_LOGOFAIR_FIXED_RECIPE100_STRICT_SAVED_PREDICTION_AND_TABLES_ADOPTED',
        '/accepted_count':100,'/root_adopted':100,'/new_accepted':96,'/reused':4,'/final_test':False,
        '/records_sha256':'fdc7c4f2402e26fdaa7b34bbfa792ceccafc5e3d32759d77940def2ed1fbc98d'},
      'A20table':{'/status':'ROOT_A20_TWO_COMPLETE_IID_SCENE_THREE_VIEW_TABLE_ADOPTED',
        '/paired_models':20,'/complete_scenes':2,'/preserved_records':40,'/root_adoption':True,'/test':False}}



def plan(spec):
    configure(spec['accepted'])
    return original_plan(spec)

base.ns['plan']=plan
stage_text=base.stage_text.replace('publication44_frozen_bytes_v1','publication48_v2_frozen_bytes_v1').replace('# Git44 sealed evidence/source bytes','# Git48 sealed evidence/source bytes')
exec(compile(stage_text,str(HERE/'publish_increment48.py')+':original_stage','exec'),base.ns)
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
