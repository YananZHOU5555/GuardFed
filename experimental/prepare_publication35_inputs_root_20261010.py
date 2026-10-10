"""Pin actual closed increment35 evidence; does not stage, commit, push or call SSH."""
from pathlib import Path
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
CHECKS=TRAIN/'server_reactivation_20261009'
PREP=ROOT/'tmp/publication_increment35_prepared_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
rel=lambda p:p.relative_to(ROOT).as_posix()
closed=read(PREP/'ROOT_CLOSED_INPUTS_TEMPLATE.json')
state=read(TRAIN/'TRAINING_STATE.json')
assert state['celeba_mechanism_v1']['scientific_results_offserver_verified']==state['celeba_mechanism_v1']['three_view_new_models_offserver_verified']==150
C3=ROOT/'tmp/celeba_mechanism_valid_C_after47_20261010/execution_candidate/backups/incremental_20261009T235311Z/ROOT_ADOPTION_REVIEW.json'
assert sha(C3)=='3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8'
C50=TRAIN/'celeba_mechanism_v1/three_view_C_five_scenes_20261010'
assert sha(C50/'ROOT_VERIFICATION.json')=='811ad551c1398f6e57681f59046c316b05c609ef30d4897158460b8121b1a28d'
live=ROOT/state['celeba_mechanism_v1']['latest_real_observation']['entry'] if 'latest_real_observation' in state['celeba_mechanism_v1'] else None
if live is None:
    live=next(p for p in CHECKS.glob('root_live_*.json') if sha(p)==sha(CHECKS/'latest_formal_live.json'))
pins=dict(C3_adoption=C3,
    C3_source_review=ROOT/'tmp/celeba_mechanism_C_after47_source_review_20261010/ROOT_INDEPENDENT_REVIEW.json',
    FL_adoption=ROOT/'tmp/celeba_flgmm_fullcoverage_delta_after7_20261010/ROOT_ADOPTION_REVIEW.json',
    Hybrid_adoption=ROOT/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after18_20261010/ROOT_ADOPTION_REVIEW.json',
    state=TRAIN/'TRAINING_STATE.json',formal_live=live,
    previous_publication=TRAIN/'publication_closed_increment34_verified_20261010.json')
closed.update(status='ROOT_CLOSED_INCREMENT35_INPUTS',counts=closed['required_closed_counts'],
    closure_pins={name:dict(path=rel(path),sha256=sha(path)) for name,path in pins.items()})
extra={name:sha(ROOT/name) for name in closed['extra_pins']}
def add(path):
    assert path.is_file() and path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink()
    assert path.suffix not in ('.pt','.pth')
    extra[rel(path)]=sha(path)
def directory(path):
    for file in path.rglob('*'):
        if file.is_file() and not {'__pycache__','restored','verified','verified_extract'}&set(file.parts):
            add(file)
for folder in ('tmp/celeba_mechanism_C_after47_root_operations_20261010',
    'tmp/celeba_mechanism_C_after47_source_review_20261010',
    'tmp/celeba_mechanism_C50_root_arithmetic_review_20261010',
    'tmp/publication_increment35_root_review_20261010'):
    directory(ROOT/folder)
for name in ('prepare_C_after47_root_operations_20261010.py','adopt_native150_review_root_20261010.py',
    'prepare_FL96_after7_adopter_root_20261010.py','adopt_FL96_after7_delta_root_20261010.py',
    'prepare_Hybrid_after18_adopter_root_20261010.py','adopt_Hybrid_after18_root_20261010.py',
    'seal_actual_C50_table_root_20261010.py','adopt_C_five_scene_table_root_20261010.py',
    'update_reactivation_state_20261009.py','update_completion_current_20261009.py',
    'update_overview_closure100_root_20261009.py','prepare_publication35_inputs_root_20261010.py'):
    add(ROOT/'tmp'/name)
add(ROOT/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after18_20261010/ROOT_DELIVERY_COPY.json')
for key in ('flgmm_screen32_20261009','hybrid_screen32_20261009'):
    path=ROOT/state[key]['latest_readonly_terminal_observation']['entry'];add(path)
    raw=path.with_name(path.stem+'.RAW.json')
    if raw.exists():add(raw)
directory((ROOT/state['flgmm_fullcoverage_v2_20261009']['latest_readonly_observation_path']).parent)
reply=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_C50_update_20261010'
if reply.exists():
    assert (reply/'ROOT_REVIEW.json').exists()
    directory(reply)
    add(ROOT/'tmp/adopt_rebuttal_C50_update_root_20261010.py')
closed['extra_pins']=extra
closed['C50_table']=dict(directory=rel(C50),root_proof=dict(path=rel(C50/'ROOT_VERIFICATION.json'),sha256=sha(C50/'ROOT_VERIFICATION.json')),
    seal=dict(filename='ACTUAL_FILES_SHA256.json',sha256=sha(C50/'ACTUAL_FILES_SHA256.json')),C3_adoption_field='C3_root_adoption_sha256')
target=ROOT/'tmp/publication_increment35_closed_root_20261010.json'
with target.open('x',encoding='utf8') as stream:
    json.dump(closed,stream,indent=2);stream.write('\n')
print(json.dumps(dict(path=rel(target),sha256=sha(target),extra_pins=len(extra),C50_included=True)))
