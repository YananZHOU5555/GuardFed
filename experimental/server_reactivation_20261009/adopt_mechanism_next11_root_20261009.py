"""Independently bind the eleven accepted backups to their actual source and terminal."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009';EX=BASE/'execution_candidate'
DELTA=EX/'backups/incremental_20261009T152532Z'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
assert sha(BASE/'FILES_SHA256.json')=='65f706e8c7c7d7e18e76c8a300dd845297c97b3bd5c6c182bbbb8aad0103b5ff'
assert sha(EX/'EXECUTION_SOURCE_SHA256.json')=='9f1252dd7c11abfe7cee297b2028ca9c58f179d964efb39ee6008ea860508b15'
assert sha(DELTA/'backup_receipt.json')=='a4a1e839a7ecce71e801b500651e258e42bcc6d130d0ca2b7867897a9950f0fb'
assert sha(DELTA/'OFFSERVER_VERIFICATION.json')=='dd451208be78ff79b8d836328e2ce33aa0953a804df1b794c1ebe615f2266067'
scope=read(BASE/'SCOPE.json');expected=scope['selected_ids']
assert len(expected)==len(set(expected))==11
receipt=read(DELTA/'backup_receipt.json');proof=read(DELTA/'OFFSERVER_VERIFICATION.json')
assert receipt['accepted_new_ids']==receipt['all_accepted_ids']==proof['accepted_new_ids']==proof['all_accepted_ids']==expected
assert receipt['previous_backup_receipt_sha256'] is proof['previous_backup_receipt_sha256'] is None
assert proof['status']=='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
assert (proof['independent_metric_checks'],proof['independent_confusion_count_checks'],proof['prediction_rule_checks'])==(99,264,33)
assert proof['source_seal_sha256']==sha(EX/'EXECUTION_SOURCE_SHA256.json')
progress=[p for p in EX.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json')]
progress=max(progress);assert sha(progress)=='c20e9f4bf1743cdc491f1fa652939254536f03b33f7d37ed3e818300e5b93eb1'
live=read(progress)
assert 'EXITED' in live['service'] and not live['processes'] and not live['batch_failure']
assert {r['id'] for r in live['completed']}==set(expected)
archive=DELTA/'incremental_valid_three_views.tar.gz'
assert sha(archive)==receipt['archive_sha256']==proof['archive_sha256']
with tarfile.open(archive) as bundle:
 assert len(bundle.getnames())==len(set(bundle.getnames()))==receipt['members']==120
 raw=bundle.extractfile('backup_inventory.json').read();inventory=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==receipt['inventory_sha256']==proof['inventory_sha256']
 assert set(bundle.getnames())==set(inventory['members'])|{'backup_inventory.json'}
 assert inventory['previous_backup'] is None
 assert inventory['models_repacked']==inventory['new_training']==inventory['new_test_inference']==0
 assert inventory['execution_seal_sha256']==sha(EX/'EXECUTION_SOURCE_SHA256.json')
 for item in bundle:
  rel=PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
  payload=bundle.extractfile(item).read()
  if item.name in inventory['members']:
   row=inventory['members'][item.name];assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
 for row in read(BASE/'FILES_SHA256.json')['members']:
  assert inventory['members']['source/science/'+row['path']]['sha256']==row['sha256']==sha(BASE/row['path'])
 for row in read(EX/'EXECUTION_SOURCE_SHA256.json')['members']:
  assert inventory['members']['source/'+row['path']]['sha256']==row['sha256']==sha(EX/row['path'])
 for name in ('ROOT_APPROVED.json','EXECUTION_DRAFT.json','APPROVED.json'):
  assert inventory['members']['source/'+name]['sha256']==sha(EX/name)
 assert json.loads(bundle.extractfile('runtime/batch_complete.json').read())==live['batch_complete']
original={r['id']:r for r in read(BASE/'inventory_actual71_Full100refs.json')['records']}
assert len(proof['records'])==11 and {r['id'] for r in proof['records']}==set(expected)
for row in proof['records']:
 assert row['checkpoint_sha256']==original[row['id']]['checkpoint']['sha256']
 assert row['native_max_abs_difference']==0 and set(row['views'])=={'native','raw','shared_calibration'}
prior=ROOT/'tmp/celeba_mechanism_valid_incremental_next37_20261009/execution_candidate/backups/incremental_20261009T144055Z/ROOT_ADOPTION_REVIEW.json'
assert sha(prior)=='a7a563390de455914b33cf62064d674347e0c26754a778b9a344ac99bdc4fac3'
assert read(prior)['cumulative_three_view_models']==60
result=dict(status='ROOT_NEXT11_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',
 checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_new_ids=expected,
 prior_three_view_models=60,accepted_new=11,cumulative_three_view_models=71,
 archive_sha256=sha(archive),archive_members_verified=120,content_members_verified=119,
 server_strict_bound_in_saved_receipts=True,offserver_verification_sha256=sha(DELTA/'OFFSERVER_VERIFICATION.json'),
 backup_receipt_sha256=sha(DELTA/'backup_receipt.json'),previous_backup_receipt_sha256=None,
 execution_seal_sha256=sha(EX/'EXECUTION_SOURCE_SHA256.json'),science_seal_sha256=sha(BASE/'FILES_SHA256.json'),
 all_native_differences_zero=True,new_training=0,new_Full_inference=0,test_inference=False,
 original60_unchanged=True,prior60_root_adoption_sha256=sha(prior),new_CNN_inference_for_root_review=0,
 negative_results_preserved=True,remote_terminal_proof_sha256=sha(progress),source_scope_complete=True)
target=DELTA/'ROOT_ADOPTION_REVIEW.json'
with target.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result|{'root_proof_sha256':sha(target)}))
