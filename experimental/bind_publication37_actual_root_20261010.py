"""Bind existing adopted evidence and current documents; no staging or experiment execution."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
B=R/'tmp/publication_increment37_prepared_20261010'
T=R/'docs/server_deployment_20260923/training_20260923'
C=T/'server_reactivation_20261009'
E=R/'tmp/celeba_mechanism_valid_C_after56_20261010/execution_candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
prep=read(B/'PREPARED_INPUTS.json')
x=read(B/'ROOT_CLOSED_INPUTS_TEMPLATE.json')
adoption=E/'backups/incremental_20261010T005530Z/ROOT_ADOPTION_REVIEW.json'
assert sha(adoption)=='21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e'
table=T/'celeba_mechanism_v1/three_view_C_six_scenes_20261010'
assert sha(table/'ROOT_VERIFICATION.json')=='f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742'
live=C/'root_live_20261010T005703Z.json'
assert sha(live)==sha(C/'latest_formal_live.json')=='1189f2bb4188529128737f139b2cf64142043181cd09d2f9cb1d510339dc3bde'
paths=dict(C4_adoption=adoption,C60_root=table/'ROOT_VERIFICATION.json',
           C60_seal=table/'ACTUAL_FILES_SHA256.json',state=T/'TRAINING_STATE.json',
           formal_live=live,previous_publication=T/'publication_closed_increment36_verified_20261010.json')
x['status']='ROOT_CLOSED_INCREMENT37_INPUTS'
x['counts']=dict(native=160,three_view=160,FL_new=12,Hybrid=19,baseline_valid=900)
x['closure_pins']={k:dict(path=p.relative_to(R).as_posix(),sha256=sha(p)) for k,p in paths.items()}
x['extra_pins']={p:sha(R/p) for p in prep['required_extra_paths']}
names=prep['required_C4_runtime_names']+['ROOT_PROGRESS_20261010T005458Z.json','ROOT_PROGRESS_20261010T005458Z.RAW.json']
x['C4_runtime_pins']={(E/n).relative_to(R).as_posix():sha(E/n) for n in names}
helpers=['tmp/bind_publication37_actual_root_20261010.py','tmp/review_C_after56_source_root_20261010.py',
         'tmp/review_C_after56_transport_root_20261010.py','tmp/seal_C60_actual_root_20261010.py',
         'tmp/adopt_C60_table_root_20261010.py','tmp/C60_actual_C4_binding_root_20261010.json',
         'tmp/check_C60_rebuttal_addendum_root_20261010.py']
x['root_helper_pins']={n:sha(R/n) for n in helpers}
out=R/'tmp/publication37_actual_closed_inputs_root_20261010.json'
with out.open('x',encoding='utf8') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps(dict(path=out.relative_to(R).as_posix(),sha256=sha(out),runtime_members=len(names))))
