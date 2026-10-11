"""Root adopts the already checked F60 table; no science is rerun."""
from pathlib import Path
import json, hashlib, datetime, shutil, argparse

parser=argparse.ArgumentParser()
parser.add_argument("--binding-sha256",required=True)
parser.add_argument("--verifier-sha256",required=True)
parser.add_argument("--root-source-review-sha256",required=True)
args=parser.parse_args()

b=Path(__file__).resolve().parent
ROOT=b.parents[1]
T=ROOT/'docs/server_deployment_20260923/training_20260923'
d=T/'celeba_mechanism_v1/three_view_F_six_scenes60_20261011'
old=T/'celeba_mechanism_v1/three_view_F_six_scenes60_20261011'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=lambda p:json.loads(Path(p).read_bytes())
assert not d.exists()
assert H(b/'ROOT_BINDING.json')==args.binding_sha256
assert H(b/'ACTUAL_VERIFY.stdout.json')==args.verifier_sha256
assert H(b/'ROOT_SOURCE_REVIEW.json')==args.root_source_review_sha256
binding=R(b/'ROOT_BINDING.json')
assert binding['root_adopted'] is True and binding['status']=='ACTUAL_REPLAY360_BOUND_FOR_F_SIX_SCENES60'
assert H(b/'FILES_SHA256.json')==binding['source_seal_sha256']
for name,pin in R(b/'FILES_SHA256.json')['files'].items():
    assert H(b/name)==pin['sha256'] and (b/name).stat().st_size==pin['bytes']
for key in ('adoption','index','native_root'):
    assert H(ROOT/binding[key])==binding[key+'_sha256']
actualroot=R(ROOT/binding['adoption'])
assert actualroot['records_index_sha256']==binding['index_sha256'] and actualroot['native_root_sha256']==binding['native_root_sha256']
assert (actualroot['prior_accepted'],actualroot['new_accepted'],actualroot['cumulative_accepted'])==(350,10,360) and actualroot['original350_unchanged'] is True
v=R(b/'ACTUAL_VERIFY.stdout.json')
assert v['status']=='PASS_ORIGINAL_FSUM_COUNTS_AND_SAVED_DISPLAY_SCOPE'
assert (v['mean_sd_scalars'],v['display_cells'],v['receipt_metrics_from_group_counts'],v['base_confusion_counts_structurally_checked'])==(972,486,1080,2880)
records=R(b/'records.json')['records'];prior=R(old/'records.json')['records'];tables=R(b/'tables.json')
assert records[:100]==prior and len(records)==120
for current,previous in zip(tables['panels'],R(old/'tables.json')['panels']):
    assert [r for r in current['rows'] if r['distribution']=='IID' and r['attack'] in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')]==previous['rows']
assert H(old/'ROOT_VERIFICATION.json')=='cf89d438b80f48df617c994084a69d8d913cf7dc35c65c762d8800517b89590b'
assert H(old/'tables.json')=='c7a6e30ef60485ee4955052ec26abfa2b709e9bcde92ff6e4a18fa871f9da44f'
names=['records.json','tables.json','coverage.json','paired_per_seed.json','checks.json','SOURCE_BINDINGS.json','TABLES.md']
d.mkdir()
for n in names:
    shutil.copyfile(b/n,d/n);assert H(d/n)==H(b/n)
newrows=[r for p in tables['panels'] if p['view']=='native' and len(p['seeds'])==10 for r in p['rows'] if (r['distribution'],r['attack'])==('non-IID','Benign')]
proof=dict(status='ROOT_F60_SIX_SCENES_COUNTS_PAIRED_STATISTICS_DISPLAY_AND_PRIOR_ROWS_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_scenes=6,paired_models=60,records=120,preserved_pairs=60,old100_record_objects_order_exact=True,old810_statistical_scalars_exact=True,old405_display_cells_preserved=True,original_fsum_verification=v,
    source_prepared_path=b.relative_to(ROOT).as_posix(),source_seal_sha256=H(b/'FILES_SHA256.json'),source_root_review_sha256=H(b/'ROOT_SOURCE_REVIEW.json'),binding_sha256=H(b/'ROOT_BINDING.json'),original_parent_root_sha256=H(old/'ROOT_VERIFICATION.json'),actual_views360_root_sha256=binding['adoption_sha256'],actual_views360_index_sha256=binding['index_sha256'],root_executed_saved_verifier_once=True,table_path=(d/'TABLES.md').relative_to(ROOT).as_posix(),table_sha256=H(d/'TABLES.md'),files={n:dict(sha256=H(d/n),bytes=(d/n).stat().st_size) for n in names},views=['raw','native','shared_calibration'],seed_panels=[10,9,6],sampleSD_ddof=1,Full_replay_devices=tables['replay_devices']['Full'],minus_F_replay_devices=tables['replay_devices']['minus_F'],native_shared_metrics_and_counts_exact=R(b/'checks.json')['native_shared_metrics_and_counts_exact'],nonIID_F_scenes_remaining=4,other_control_runs_remaining=400,new_nonIID_Benign_native10_rows=newrows,
    scope='Five IID scenes plus non-IID Benign only. All fixed10/9/6 panels and all signed paired differences retained; no cross-scene aggregate, significance, universal necessity or isolated-causality claim. Mixed replay devices/selection history retained.',primary_endpoint_selected=False,new_threshold_fits=0,new_CNN=0,new_training=0,final_test=False,whole_mechanism_complete=False,whole_rebuttal_complete=False,
    root_adopter_path=Path(__file__).relative_to(ROOT).as_posix(),root_adopter_sha256=H(__file__),root_reader_note='Adopts only outputs pinned to the actual future views360 root and independently saved verifier; source preparation itself is not statistical acceptance.')
with (d/'ROOT_VERIFICATION.json').open('x',encoding='utf8',newline='\n') as f:
    json.dump(proof,f,indent=2,ensure_ascii=False);f.write('\n')
(d/'README.md').write_text('Accepted five IID scenes plus non-IID Benign Full versus minus_F table. See TABLES.md and ROOT_VERIFICATION.json. Mean ± sampleSD(ddof1), paired seeds91001..91010 plus fixed9/6 panels. See actual checks for native/shared equality. Four non-IID F scenes and four other controls remain incomplete. Validation only; no final-test or universal necessity claim.\n',encoding='utf8',newline='\n')
manifest=dict(status='ROOT_COMPACT_F60_TABLE_DELIVERY',files={p.name:dict(sha256=H(p),bytes=p.stat().st_size) for p in sorted(d.iterdir()) if p.is_file()})
with (d/'FILES_SHA256.json').open('x',encoding='utf8',newline='\n') as f:
    json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],root_sha256=H(d/'ROOT_VERIFICATION.json'),table_sha256=H(d/'TABLES.md'),records=120,paired=60)))
