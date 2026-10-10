"""Git49 metadata binding; exact original Git48/44 transport, no commit/push."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
H=lambda b:hashlib.sha256(b).hexdigest()
OLD=ROOT/'tmp/publication_increment48_v2_20261010'
assert H((OLD/'publish_increment48.py').read_bytes())=='e2641bf80e0e81c100adde49cca8de01a2e5e39d143fda72ac18583d1fa1cc56'
sp=importlib.util.spec_from_file_location('publication48_reused_for49',OLD/'publish_increment48.py')
old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old);base=old.base
PARENT='ebf8436126687e044262fabb90a0848f39275f52'
ACCEPTED={'native':236,'three_view':236}
SCOPE='CLOSED236_REPLAY236_FL38_GRADIENT18_HYBRID32_STARTUP_NO_BULK'
TRAIN='docs/server_deployment_20260923/training_20260923/'
HYBRID='tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010/'
FIXED_BINDINGS={
 'native236':(TRAIN+'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T091747Z/ROOT_DELTA_VERIFICATION.json','6d9594230da5f11b26a3df74db1b698811e8f6cdbb3368a9b467cdf92b2467cd'),
 'gradient18':('tmp/celeba_gradient64_delta_after10_20261010/ROOT_ADOPTION_REVIEW.json','83d22e5833fecddb398625893c457d2dfadb3b745bc4c5613d59cf87b6c02d39'),
 'FL38':('tmp/celeba_flgmm_fullcoverage_delta_after32_20261010/ROOT_ADOPTION_REVIEW.json','5e95872f3216a9842e95c534ba87629b2211c298d72a95d810bdaf9428021051'),
 'Hybrid32':('tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after27_20261010/ROOT32_SUMMARY_ADOPTION.json','6fcbdc41c7af01815e15995e0cb3688672404bf3b96844dc8d9bffa21028587f'),
 'Hybrid100_bound':(HYBRID+'ROOT_BOUND_ADOPTION.json','41e9d664e17105b84b1266824fffec0cfef7b7e66a4fa7cfca0cfc168c3f10cf'),
 'Hybrid7_gates':(HYBRID+'ROOT_SEVEN_CANARY_CLOSURE.json','eda03511ec037be8d06401886355b3c49929de3523bed272ca8df94e5799b05a'),
 'Hybrid96_startup':(HYBRID+'ROOT_COVERAGE_STARTUP.json','136205c2b101c05f3fd0bbfa1fe16df4a364a5b4f64a140aa51ee754ceb50045')}
REQUIRED_FACTS={
 'native236':{'/total_new_strict_and_offserver':236,'/test':False},
 'replay236':{'/prior_accepted':228,'/new_accepted':8,'/cumulative_accepted':236,'/remaining620_new_accepted':56,'/test':False},
 'gradient18':{'/accepted_before':10,'/accepted_new':8,'/accepted_total':18,'/screen64_complete':False,'/final_test':False},
 'FL38':{'/accepted_before':32,'/accepted_new':6,'/accepted_total':38,'/reused_separately':4,'/final_test':False},
 'Hybrid32':{'/accepted_total':32,'/all32_offserver_verified':True,'/selected_candidate/id':'CosineFairness_lam20.0_tau0.1_lr0.001','/final_test':False},
 'Hybrid100_bound':{'/new':96,'/reused':4,'/canaries':7,'/received_members':147,'/final_test':False},
 'Hybrid7_gates':{'/total_canary_runs':7,'/rounds':3,'/formal_table_samples':0,'/final_test':False},
 'Hybrid96_startup':{'/new_accepted':0,'/scientific70_records':0,'/planned_new':96,'/reused':4,'/actual_round':1,'/formal100_started':True,'/final_test':False},
 'current_state':{'/celeba_mechanism_v1/scientific_results_strictly_accepted':236,
  '/celeba_mechanism_v1/three_view_new_models_accepted':236,
  '/flgmm_fullcoverage_v2_20261009/new_accepted':38,
  '/gradient64_validation_search_20261010/offserver_accepted':18,
  '/hybrid_screen32_20261009/offserver_accepted70round_jobs':32,
  '/hybrid100_fullcoverage_20261010/canaries_offserver_adopted':7,
  '/hybrid100_fullcoverage_20261010/new_accepted':0,
  '/hybrid100_fullcoverage_20261010/formal100_started':True,
  '/hybrid100_fullcoverage_20261010/coverage_start_sha256':'136205c2b101c05f3fd0bbfa1fe16df4a364a5b4f64a140aa51ee754ceb50045'}}
ROLES=set(REQUIRED_FACTS);OPTIONAL=set()
base.PARENT=PARENT;base.ROLES=ROLES;base.OPTIONAL=OPTIONAL;base.ACCEPTED=ACCEPTED;base.REQUIRED_FACTS=REQUIRED_FACTS
base.ns.update(PARENT=PARENT,ROLES=ROLES,ACCEPTED=ACCEPTED,OUTPUT_ROOT=Path('F:/YananResearchStorage/GuardFed/git_publication/increment49'))
CHECKOUT,BRANCH,ORIGIN,git,storage,relative=(getattr(base,n) for n in ('CHECKOUT','BRANCH','ORIGIN','git','storage','relative'))
ORIGINAL_SOURCE=base.source
plan_source=old.plan_source.replace('CLOSED220_REPLAY220_GRADIENT10_LOGO100_A20_NO_BULK',SCOPE).replace('publication48_v2_frozen_bytes_v1','publication49_frozen_bytes_v1')
exec(compile(plan_source,str(HERE/'publish_increment49.py')+':original_plan_metadata','exec'),base.__dict__)
original_plan=base.plan


def configure(accepted):
    assert accepted==ACCEPTED,'Git49 requires actual root native236/replay236'


def bind_snapshot(pin):
    assert pin and pin.get('path') and pin.get('sha256'),'Frozen F source snapshot is required'
    p=Path(pin['path']);storage()
    assert p.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed/git_publication/increment49').resolve())
    b=p.read_bytes();assert H(b)==pin['sha256'];d=json.loads(b)
    assert d['status']=='COMPACT_SOURCE_BYTES_FROZEN_NOT_STAGED' and d['parent']==PARENT
    source_root=Path(d['source_root']);assert source_root.resolve().is_relative_to(p.parent.resolve()) and not source_root.is_symlink()
    assert base.ROOT==ROOT and base.ns['ROOT']==ROOT,'Original Git storage authority must remain on the real repository'
    frozen_source=types.FunctionType(ORIGINAL_SOURCE.__code__,dict(ORIGINAL_SOURCE.__globals__,ROOT=source_root),ORIGINAL_SOURCE.__name__,ORIGINAL_SOURCE.__defaults__,ORIGINAL_SOURCE.__closure__)
    base.source=frozen_source;base.ns['source']=frozen_source
    return source_root


def validate_bindings(d):
    configure(d['accepted']);assert set(d['bindings'])==ROLES
    for role,pin in d['bindings'].items():
        assert pin and pin.get('path') and isinstance(pin.get('sha256'),str) and len(pin['sha256'])==64,'Actual root binding missing: '+role
        assert all(c in '0123456789abcdef' for c in pin['sha256'])
        if role in FIXED_BINDINGS:assert (pin['path'],pin['sha256'])==FIXED_BINDINGS[role],role
        if role=='current_state':assert pin['path']==TRAIN+'TRAINING_STATE.json'
        assert all(pin.get('expect',{}).get(k)==v for k,v in REQUIRED_FACTS[role].items()),role


def plan(spec):
    validate_bindings(spec)
    bind_snapshot(spec['source_snapshot'])
    d=original_plan(spec);d['source_snapshot']=spec['source_snapshot']
    return d


base.ns['plan']=plan
stage_text=old.stage_text.replace('publication48_v2_frozen_bytes_v1','publication49_frozen_bytes_v1').replace('# Git48 sealed evidence/source bytes','# Git49 sealed evidence/source bytes')
exec(compile(stage_text,str(HERE/'publish_increment49.py')+':original_stage','exec'),base.ns)
freeze=base.ns['freeze']


def stage(path,expected,name):
    d=json.loads(Path(path).read_bytes());validate_bindings(d);bind_snapshot(d['source_snapshot'])
    return base.ns['stage'](path,expected,name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','freeze','stage']);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--output-name');a=p.parse_args();assert H(a.input.read_bytes())==a.sha256
    if a.action=='plan':
        d=plan(json.loads(a.input.read_bytes()));print(json.dumps(dict(status='PLAN_ONLY_NO_WRITES',files=len(d['files']),bytes=d['total_bytes'])))
    else:
        assert a.output_name;(freeze if a.action=='freeze' else stage)(a.input,a.sha256,a.output_name)
