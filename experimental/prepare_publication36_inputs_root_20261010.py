"""Bind actual closed increment36 evidence; no Git, SSH, science or shared-state writes."""
from pathlib import Path
import hashlib, json, sys

if sys.flags.optimize:raise RuntimeError('Optimized Python is forbidden for evidence guards')

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
CHECKS=TRAIN/'server_reactivation_20261009'
PREP=ROOT/'tmp/publication_increment36_prepared_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
rel=lambda p:p.relative_to(ROOT).as_posix()
closed=read(PREP/'ROOT_CLOSED_INPUTS_TEMPLATE.json')
assert sha(PREP/'FILES_SHA256.json')=='2e22fb9c4ed8de3d27f439a310263f5a02932f548c508d49becf136a668dcaca'
review=ROOT/'tmp/publication_increment36_root_review_20261010/ROOT_REVIEW.json'
assert sha(review)=='d64e6167301b268a2e284b2ebd58ce0ff26598d34412646e7c5abfe40b8cff4f'
assert read(review)['status']=='PASS_SOURCE_ONLY_INCREMENT36_READY_FOR_ROOT_CLOSED_INPUT_BINDING'
state=read(TRAIN/'TRAINING_STATE.json')
main=state['celeba_mechanism_v1']
assert main['scientific_results_offserver_verified']==main['three_view_new_models_offserver_verified']==156
assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==11
assert state['hybrid_screen32_20261009']['offserver_accepted70round_jobs']==19
C6_stage=main['C_after50_valid_replay']
assert C6_stage['offserver_new_accepted']==6 and C6_stage['service_terminal']=='EXITED'
C6=ROOT/C6_stage['root_adoption_path']
assert sha(C6)==C6_stage['root_adoption_sha256']
assert (read(C6)['prior_three_view_models'],read(C6)['accepted_new'],read(C6)['cumulative_three_view_models'])==(150,6,156)
assert state['latest_rebuttal_draft']['complete_C_scenes']==5
assert state['latest_rebuttal_draft']['root_proof_sha256']=='64b083fb61cb8bff17304031af01a0d692d03fdc16e956de0dcd154f5a524691'
live=next(p for p in CHECKS.glob('root_live_*.json') if sha(p)==sha(CHECKS/'latest_formal_live.json'))
pins=dict(C6_adoption=C6,
    C6_source_review=ROOT/'tmp/celeba_mechanism_C_after50_source_review_20261010/ROOT_INDEPENDENT_REVIEW.json',
    FL_adoption=ROOT/'tmp/celeba_flgmm_fullcoverage_delta_after9_20261010/ROOT_ADOPTION_REVIEW.json',
    Hybrid_adoption=ROOT/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after18_20261010/ROOT_ADOPTION_REVIEW.json',
    state=TRAIN/'TRAINING_STATE.json',formal_live=live,
    previous_publication=TRAIN/'publication_closed_increment35_verified_20261010.json')
assert sha(pins['C6_source_review'])=='3c0b8fb2b37647c2b410e3c10abdcb1064f161e839ec1495e9d03aa1953075c8'
assert sha(pins['FL_adoption'])=='6feb41c9f2f06980d29865ca03d59e5d2cffeeb2a6f0d6f0209f065cf80caf80'
closed.update(status='ROOT_CLOSED_INCREMENT36_INPUTS',counts=closed['required_closed_counts'],
    closure_pins={name:dict(path=rel(path),sha256=sha(path)) for name,path in pins.items()})
extra={name:sha(ROOT/name) for name in closed['extra_pins']}
def add(path):
    assert path.is_file() and path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink()
    assert path.suffix not in ('.pt','.pth')
    assert not path.is_relative_to(ROOT/'tmp/celeba_hybrid_screen_execution_20261009')
    extra[rel(path)]=sha(path)
def directory(path):
    for file in path.rglob('*'):
        if file.is_file() and not {'__pycache__','restored','verified','verified_extract'}&set(file.parts):add(file)
for folder in ('tmp/celeba_mechanism_C_after50_root_operations_20261010',
    'tmp/celeba_mechanism_C_after50_source_review_20261010',
    'tmp/rebuttal_integrated_C50_root_review_20261010',
    'tmp/current_C50_state_wiring_review_20261010',
    'tmp/publication_increment36_root_review_20261010'):
    directory(ROOT/folder)
for name in ('prepare_C_after50_root_operations_20261010.py','adopt_native156_review_root_20261010.py',
    'prepare_FL96_after9_adopter_root_20261010.py','adopt_FL96_after9_delta_root_20261010.py',
    'adopt_rebuttal_integrated_C50_root_20261010.py',
    'update_reactivation_state_20261009.py','update_completion_current_20261009.py',
    'update_overview_closure100_root_20261009.py','prepare_publication36_inputs_root_20261010.py',
    'adopt_publication36_root_20261010.py'):
    add(ROOT/'tmp'/name)
for key in ('flgmm_screen32_20261009','hybrid_screen32_20261009'):
    path=ROOT/state[key]['latest_readonly_terminal_observation']['entry'];add(path)
    raw=path.with_name(path.stem+'.RAW.json')
    if raw.exists():add(raw)
directory((ROOT/state['flgmm_fullcoverage_v2_20261009']['latest_readonly_observation_path']).parent)
directory(ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010')
closed['extra_pins']=extra
closed['C50_table']=None
closed['note']='Root actual closure156native/156views/11FL/19Hybrid. Already-published Hybrid archive and C50 table are excluded; complete C50 author-review text is pinned, not manuscript-applied or final-test evidence.'
target=ROOT/'tmp/publication_increment36_closed_root_20261010.json'
with target.open('x',encoding='utf8') as stream:json.dump(closed,stream,indent=2);stream.write('\n')
print(json.dumps(dict(path=rel(target),sha256=sha(target),extra_pins=len(extra),C50_table_included=False)))
