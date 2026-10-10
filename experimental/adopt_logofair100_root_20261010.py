"""Adopt the actual independently checked LoGoFair100; copy compact reports only."""
from pathlib import Path
import datetime, hashlib, json

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
FROOT=Path('F:/YananResearchStorage/GuardFed/logofair_fullcoverage100_20261010')
review_path=ROOT/'tmp/celeba_logofair100_root_review_20261010/ROOT_REVIEW.json'
assert sha(review_path)=='b4fc447ba86c2fd40b54a1100c1ca2cdba890546b383feb5ddfa6bae19958610'
r=read(review_path)
assert r['status']=='INDEPENDENT_LOGOFAIR100_HASH_PROVENANCE_AND_ARITHMETIC_PASS_NO_ADOPTION'
assert (r['accepted_records'],r['new96'],r['reused4'],r['complete_scenes'])==(100,96,4,10)
assert (r['artifact_hashes_recomputed'],r['mean_SD_scalars_recomputed'],r['display_cells'])==(1027,198,99)
assert r['max_abs_statistical_difference']<=1e-12 and len(r['constant_prediction_ids'])==1
assert not r['test'] and r['new_CNN']==r['new_fits']==0
assert not r['original_strict_reexecuted'] and not r['arrays_loaded'] and not r['Torch_loaded']
index=FROOT/'attempt001/STRICT100_INDEX.json'
assert sha(index)==r['index_sha256']=='1d2cbb94e47716dd21854c704f5d78506cf6f075774d38e06d89ec71d3be8ba7'
original=FROOT/'summary001'
out=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_logofair100_accepted_20261010'
assert not out.exists()
payload={}
for name,digest in r['summary_files_sha256'].items():
    p=original/name
    assert p.suffix in ('.json','.md') and p.stat().st_size<1024**2 and sha(p)==digest
    payload[name]=p.read_bytes()
records=read(original/'records100.json'); a=read(original/'ACCEPTANCE100.json')
assert len(records)==len({x['id'] for x in records})==100
assert a['saved_prediction_items']==1986700 and a['saved_metric_checks']==300 and a['saved_metric_max_difference']==0
old=ROOT/'tmp/celeba_logofair32_root_adoption_20261010/ROOT_ADOPTION.json'
assert sha(old)==r['screen32_root_adoption_sha256']
recipe=read(old)['selected_recipe']
assert recipe['id']=='LoGoFair-DP_07'
out.mkdir()
for name,data in payload.items():
    with (out/name).open('xb') as f:f.write(data)
    assert sha(out/name)==r['summary_files_sha256'][name]
storage=dict(status='ORIGINAL_F_ARTIFACT_HASH_INVENTORY_REFERENCED_NO_BULK_COPY',
    F_root=FROOT.as_posix(),original_acceptance_path=(original/'ACCEPTANCE100.json').as_posix(),
    original_acceptance_sha256=sha(original/'ACCEPTANCE100.json'),artifact_count=1027,
    complete_index_path=index.as_posix(),complete_index_sha256=sha(index),
    original_summary_paths={name:dict(path=(original/name).as_posix(),sha256=digest) for name,digest in r['summary_files_sha256'].items()},
    storage_rule='All models, mappings, fitted states, predictions and raw fitting evidence remain on checked Yanan 2TB F storage; E contains only these four compact reports and restore references.',
    models_or_arrays_copied=0,archive_created=False)
with (out/'RAW_STORAGE_INDEX.json').open('x',encoding='utf8') as f:json.dump(storage,f,indent=2);f.write('\n')
proof=dict(status='ROOT_LOGOFAIR_FIXED_RECIPE100_STRICT_SAVED_PREDICTION_AND_TABLES_ADOPTED',
    adopted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_count=100,root_adopted=100,new_accepted=96,reused=4,
    selected_recipe=recipe,model_seeds=list(range(91001,91011)),fit_seed=1719,virtual_cohorts=20,true_client_fairness=False,
    records_path=(out/'records100.json').as_posix(),records_sha256=sha(out/'records100.json'),
    independent_review_path=review_path.relative_to(ROOT).as_posix(),independent_review_sha256=sha(review_path),
    original_acceptance_path=(original/'ACCEPTANCE100.json').as_posix(),original_acceptance_sha256=sha(original/'ACCEPTANCE100.json'),
    original_summary_command_exit=0,strict_index_path=index.as_posix(),strict_index_sha256=sha(index),
    artifact_hashes_recomputed=1027,saved_prediction_items=1986700,saved_metric_checks=300,saved_metric_max_difference=0,
    mean_SD_scalars_recomputed=198,display_cells=99,cross_scene_seed_first_metric_checks=75,
    constant_prediction_ids=r['constant_prediction_ids'],all_negative_and_constant_results_retained=True,
    files_sha256={name:sha(out/name) for name in (*payload,'RAW_STORAGE_INDEX.json')},
    canonical_table=(out/'TABLES.md').relative_to(ROOT).as_posix(),new_CNN=0,new_fits_during_acceptance=0,final_test=False,
    incorporated_into_full_rebuttal=False,primary_endpoint_selected=False,whole_rebuttal_complete=False,
    limitations=r['limitations'])
target=out/'ROOT_ADOPTION.json'
with target.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],path=str(target),sha256=sha(target),accepted=100,new=96,reused=4)))
