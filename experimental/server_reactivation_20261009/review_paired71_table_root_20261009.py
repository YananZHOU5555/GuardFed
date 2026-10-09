"""Accept the seven-scene saved-receipt join, independently checking its statistics."""
from pathlib import Path
import datetime,hashlib,json,math,shutil,tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_three_view_paired71_20261009'
SNAP=BASE/'snapshot_724_full100_mechanism71'
DEST=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim71_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
assert sha(BASE/'FILES_SHA256.json')=='00b362769ebe7158711d8f5897c4c41547e153f98004d56957e06001fbcc5990'
for row in read(BASE/'FILES_SHA256.json')['members']:
 assert sha(BASE/row['path'])==row['sha256'] and (BASE/row['path']).stat().st_size==row['size']
for name,digest in read(SNAP/'INPUTS_SHA256.json')['files'].items():assert sha(Path(name))==digest
tables=read(SNAP/'tables.json');records=read(SNAP/'records.json')['records'];archives={}
assert len(records)==len({r['id'] for r in records})==142
try:
 for row in records:
  p=row['provenance']
  if 'receipt_path' in p:
   assert sha(Path(p['offserver_proof_path']))==p['offserver_proof_sha256']
   assert sha(Path(p['strict_acceptance_path']))==p['strict_acceptance_sha256']
   assert sha(Path(p['prediction_array_path']))==p['prediction_arrays_sha256']
   raw=Path(p['receipt_path']).read_bytes()
  else:
   path=Path(p['archive_path'] if 'archive_path' in p else p['archive'])
   if not path.is_absolute():path=ROOT/path
   if str(path) not in archives:
    assert sha(path)==p['archive_sha256'];archives[str(path)]=tarfile.open(path)
   bundle=archives[str(path)];raw=bundle.extractfile(p['receipt_member']).read()
   assert hashlib.sha256(bundle.extractfile(p['strict_acceptance_member']).read()).hexdigest()==p['strict_acceptance_sha256']
  assert hashlib.sha256(raw).hexdigest()==p['receipt_sha256'];receipt=json.loads(raw)
  assert receipt['id']==row['id'] and receipt['checkpoint_sha256']==row['checkpoint_sha256']
  assert receipt['views']==row['views'] and receipt['fits']==row['fits'] and receipt['runtime']==row['replay_runtime']
  assert receipt['native_comparison']['accepted'] and receipt['native_comparison']['max_abs_difference']<=1e-12
  assert receipt['valid_n']==19867 and receipt['root_reconstruction']['root_n']==16277
finally:
 for bundle in archives.values():bundle.close()
by_cell={(r['distribution'],r['attack'],r['seed'],r['variant']):r for r in records}
assert len(by_cell)==142
scenes={('IID',a) for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')}|{('non-IID','Benign'),('non-IID','F Flip')}
checks=0;max_delta=0.0
for panel in tables['panels']:
 seeds=panel['seeds'];assert seeds in [list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
 assert len(panel['rows'])==21
 assert {(r['distribution'],r['attack']) for r in panel['rows']}==scenes
 for row in panel['rows']:
  assert row['n']==row['expected_n']==len(seeds) and row['seeds']==seeds and row['complete']
  for key,metric,unit in [('accuracy_pct','accuracy',100),('aeod','aeod',1),('aspd','aspd',1)]:
   values=[]
   for seed in seeds:
    cell=(row['distribution'],row['attack'],seed)
    if row['variant']=='minus_U minus Full':
     value=unit*(by_cell[cell+('minus_U',)]['views'][panel['view']][metric]-by_cell[cell+('Full',)]['views'][panel['view']][metric])
    else:value=unit*by_cell[cell+(row['variant'],)]['views'][panel['view']][metric]
    assert math.isfinite(value);values.append(value)
   mean=math.fsum(values)/len(values);sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1))
   for name,value in [('mean',mean),('sample_sd_ddof1',sd)]:
    difference=abs(row[key][name]-value);assert difference<=1e-12;max_delta=max(max_delta,difference);checks+=1
assert checks==1134 and tables['complete_scenes']==7 and tables['table_model_records']==140
assert tables['complete_paired_checkpoints']==70 and tables['incomplete_pair_count']==1
assert not tables['primary_endpoint_selected'] and not tables['mechanism900_complete'] and not tables['final_test']
assert tables['new_inference']==tables['new_training']==0
old=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim_20261009T145900Z'
old_records=read(old/'records.json')['records'];lookup={r['id']:r for r in records}
assert all(r==lookup[r['id']] for r in old_records) and len(old_records)==120
for before,after in zip(read(old/'tables.json')['panels'],tables['panels']):
 assert all(row in after['rows'] for row in before['rows'])
DEST.mkdir(exist_ok=False)
for path in SNAP.iterdir():
 if path.is_file():shutil.copyfile(path,DEST/path.name);assert sha(path)==sha(DEST/path.name)
proof=dict(status='ROOT_SEVEN_SCENE_THREE_VIEW_PAIRED_SAVED_RECEIPTS_AND_STATISTICS_PASS',
 checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal_sha256=sha(BASE/'FILES_SHA256.json'),
 complete_paired_scenes=7,paired_checkpoints=70,preserved_pairs=71,displayed_records=140,
 actual_saved_receipts_verified=142,root_independent_mean_sampleSD_checks=checks,max_abs_statistic_difference=max_delta,
 original_six_scene_records_unchanged=120,original_six_scene_summary_rows_unchanged=162,
 table_json_sha256=sha(DEST/'tables.json'),records_sha256=sha(DEST/'records.json'),
 source_inputs_sha256=sha(DEST/'INPUTS_SHA256.json'),agent_numeric_checks_sha256=sha(DEST/'NUMERIC_CHECKS.json'),
 primary_endpoint_selected=False,uniform_device_comparison=False,new_inference=0,new_training=0,final_test=False)
with (DEST/'ROOT_REVIEW.json').open('x',encoding='utf8') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(proof|{'root_proof_sha256':sha(DEST/'ROOT_REVIEW.json')}))
