"""Root adopts the already checked F50 table; no science is rerun."""
from pathlib import Path
import json, hashlib, datetime, shutil

b=Path(__file__).resolve().parent
ROOT=b.parents[1]
T=ROOT/'docs/server_deployment_20260923/training_20260923'
d=T/'celeba_mechanism_v1/three_view_F_IID_five_scenes50_20261011'
old=T/'celeba_mechanism_v1/three_view_F_IID_four_scenes40_20261011'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=lambda p:json.loads(Path(p).read_bytes())
assert not d.exists()
v=R(b/'ACTUAL_VERIFY.stdout.json')
assert v['status']=='PASS_ORIGINAL_FSUM_COUNTS_AND_SAVED_DISPLAY_SCOPE'
assert (v['mean_sd_scalars'],v['display_cells'],v['receipt_metrics_from_group_counts'],v['base_confusion_counts_structurally_checked'])==(810,405,900,2400)
records=R(b/'records.json')['records'];prior=R(old/'records.json')['records'];tables=R(b/'tables.json')
assert records[:80]==prior and len(records)==100
for current,previous in zip(tables['panels'],R(old/'tables.json')['panels']):
    assert [r for r in current['rows'] if r['attack']!='Sp-DFA']==previous['rows']
assert H(b/'ROOT_BINDING.json')=='3c259365f013aa9bf23ee279438c95b2e64644a7c8e0f20d81921734b4fed440'
assert H(ROOT/'tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/ROOT_ADOPTION.json')=='3f7b436e0b2fec0401d7bf6117d6dca48aef21c185b6f170d37954e2cafee5a4'
names=['records.json','tables.json','coverage.json','paired_per_seed.json','checks.json','SOURCE_BINDINGS.json','TABLES.md']
d.mkdir()
for n in names:
    shutil.copyfile(b/n,d/n);assert H(d/n)==H(b/n)
newrows=[r for p in tables['panels'] if p['view']=='native' and len(p['seeds'])==10 for r in p['rows'] if r['attack']=='Sp-DFA']
proof=dict(status='ROOT_F50_FIVE_IID_SCENES_COUNTS_PAIRED_STATISTICS_DISPLAY_AND_PRIOR_ROWS_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_scenes=5,paired_models=50,records=100,preserved_pairs=50,old80_record_objects_order_exact=True,old648_statistical_scalars_exact=True,old324_display_cells_preserved=True,original_fsum_verification=v,
    source_prepared_path=b.relative_to(ROOT).as_posix(),source_seal_sha256=H(b/'FILES_SHA256.json'),source_root_review_sha256=H(b/'ROOT_SOURCE_REVIEW.json'),binding_sha256=H(b/'ROOT_BINDING.json'),original_parent_root_sha256=H(old/'ROOT_VERIFICATION.json'),actual_views350_root_sha256=H(ROOT/'tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/ROOT_ADOPTION.json'),actual_views350_index_sha256=H(ROOT/'tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/MECHANISM350_INDEX.json'),root_executed_saved_verifier_once=True,table_path=(d/'TABLES.md').relative_to(ROOT).as_posix(),table_sha256=H(d/'TABLES.md'),files={n:dict(sha256=H(d/n),bytes=(d/n).stat().st_size) for n in names},views=['raw','native','shared_calibration'],seed_panels=[10,9,6],sampleSD_ddof=1,Full_replay_devices={'cpu':3,'cuda:0':47},minus_F_replay_devices={'cpu':50},native_shared_metrics_and_counts_exact=True,nonIID_F_scenes_remaining=5,other_control_runs_remaining=400,new_SpDFA_native10_rows=newrows,
    scope='Five IID scenes only. SpDFA native/shared minusF improves ACC and AEOD means but worsens ASPD; raw both disparities worsen while ACC slightly rises. Retain all directions; no significance, universal necessity or isolated-causality claim. Mixed replay devices/selection history retained.',primary_endpoint_selected=False,new_threshold_fits=0,new_CNN=0,new_training=0,final_test=False,whole_mechanism_complete=False,whole_rebuttal_complete=False,
    root_adopter_path=Path(__file__).relative_to(ROOT).as_posix(),root_adopter_sha256=H(__file__),root_reader_note='An earlier inline metadata adopter failed to parse before executing any statement. No outputs or scientific results were affected; this saved adopter is the actual adoption source.')
with (d/'ROOT_VERIFICATION.json').open('x',encoding='utf8',newline='\n') as f:
    json.dump(proof,f,indent=2,ensure_ascii=False);f.write('\n')
(d/'README.md').write_text('Accepted five-IID-scene Full versus minus_F table. See TABLES.md and ROOT_VERIFICATION.json. Mean ± sampleSD(ddof1), paired seeds91001..91010 plus fixed9/6 panels. Native/shared identical here. Five non-IID F scenes and four other controls remain incomplete. Validation only; no final-test or universal necessity claim.\n',encoding='utf8',newline='\n')
manifest=dict(status='ROOT_COMPACT_F50_TABLE_DELIVERY',files={p.name:dict(sha256=H(p),bytes=p.stat().st_size) for p in sorted(d.iterdir()) if p.is_file()})
with (d/'FILES_SHA256.json').open('x',encoding='utf8',newline='\n') as f:
    json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],root_sha256=H(d/'ROOT_VERIFICATION.json'),table_sha256=H(d/'TABLES.md'),records=100,paired=50)))
