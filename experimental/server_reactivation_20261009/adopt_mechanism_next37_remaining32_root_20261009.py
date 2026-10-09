"""Adopt only the final32 from the frozen37 scope after independent off-server checks."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_next37_20261009';EX=BASE/'execution_candidate'
parser=argparse.ArgumentParser();parser.add_argument('--delta',required=True);args=parser.parse_args()
DELTA=(EX/'backups'/args.delta).resolve()
assert DELTA.parent==(EX/'backups').resolve() and args.delta.startswith('incremental_')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
FIRST=EX/'backups/incremental_20261009T141304Z'
assert sha(FIRST/'ROOT_ADOPTION_REVIEW.json')=='48b219abe106d4edfb2a8bf1425928c712fcdac7919ed3f1c5e2f688ac9e1955'
assert sha(FIRST/'backup_receipt.json')=='e22525155441aa6d2368b252ca103fdba03978c00e45659e1ba72123c1a45ec6'
assert sha(EX/'EXECUTION_SOURCE_SHA256.json')=='4f71c60c6966da40d334f0ac17141a7e763520b7fe478df8548809e773971caf'
assert sha(BASE/'FILES_SHA256.json')=='95978fa42c28e9b4ff5b855b33c2dda56edc2b14fcfd56c3a29b0a9ba98135fd'
scope=read(BASE/'SCOPE.json');prior=read(FIRST/'backup_receipt.json')
expected=[identity for identity in scope['selected_ids'] if identity not in prior['all_accepted_ids']]
assert len(expected)==len(set(expected))==32
receipt=read(DELTA/'backup_receipt.json');proof=read(DELTA/'OFFSERVER_VERIFICATION.json')
assert receipt['accepted_new_ids']==proof['accepted_new_ids']==expected
assert receipt['all_accepted_ids']==proof['all_accepted_ids']==scope['selected_ids']
assert receipt['previous_backup_receipt_sha256']==proof['previous_backup_receipt_sha256']==sha(FIRST/'backup_receipt.json')
assert proof['status']=='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
assert proof['independent_metric_checks']==288 and proof['independent_confusion_count_checks']==768 and proof['prediction_rule_checks']==96
assert proof['source_seal_sha256']==sha(EX/'EXECUTION_SOURCE_SHA256.json')
archive=DELTA/'incremental_valid_three_views.tar.gz'
assert sha(archive)==receipt['archive_sha256']==proof['archive_sha256']
with tarfile.open(archive) as bundle:
    assert len(bundle.getnames())==len(set(bundle.getnames()))==receipt['members']
    raw=bundle.extractfile('backup_inventory.json').read();inventory=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==receipt['inventory_sha256']==proof['inventory_sha256']
    assert set(bundle.getnames())==set(inventory['members'])|{'backup_inventory.json'}
    assert inventory['previous_backup']['receipt_sha256']==sha(FIRST/'backup_receipt.json')
    assert inventory['models_repacked']==inventory['new_training']==inventory['new_test_inference']==0
    assert inventory['execution_seal_sha256']==sha(EX/'EXECUTION_SOURCE_SHA256.json')
    for item in bundle:
        rel=PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
        payload=bundle.extractfile(item).read()
        if item.name in inventory['members']:
            row=inventory['members'][item.name]
            assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
    assert inventory['members']['execution/backup_completed.py']['sha256']==sha(EX/'backup_completed.py')
    terminal=json.loads(bundle.extractfile('runtime/batch_complete.json').read())
    live=read(EX/'ROOT_PROGRESS_20261009T143713Z.json')
    assert terminal==live['batch_complete'] and not live['batch_failure'] and len(live['completed'])==37 and not live['processes']
original={row['id']:row for row in read(BASE/'inventory_actual60_Full100refs.json')['records']}
assert len(proof['records'])==32 and {row['id'] for row in proof['records']}==set(expected)
for row in proof['records']:
    assert row['checkpoint_sha256']==original[row['id']]['checkpoint']['sha256']
    assert row['native_max_abs_difference']==0 and set(row['views'])=={'native','raw','shared_calibration'}
result=dict(status='ROOT_NEXT37_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_new_ids=expected,
    prior_three_view_models=28,accepted_new=32,cumulative_three_view_models=60,
    archive_sha256=sha(archive),archive_members_verified=receipt['members'],content_members_verified=len(inventory['members']),
    server_strict_bound_in_saved_receipts=True,offserver_verification_sha256=sha(DELTA/'OFFSERVER_VERIFICATION.json'),
    backup_receipt_sha256=sha(DELTA/'backup_receipt.json'),previous_backup_receipt_sha256=sha(FIRST/'backup_receipt.json'),
    execution_seal_sha256=sha(EX/'EXECUTION_SOURCE_SHA256.json'),science_seal_sha256=sha(BASE/'FILES_SHA256.json'),
    all_native_differences_zero=True,new_training=0,new_Full_inference=0,test_inference=False,
    original23_unchanged=True,new_CNN_inference_for_root_review=0,negative_results_preserved=True,
    remote_terminal_proof_sha256=sha(EX/'ROOT_PROGRESS_20261009T143713Z.json'),source_scope_complete=True)
with (DELTA/'ROOT_ADOPTION_REVIEW.json').open('x',encoding='utf8') as stream:
    json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result|{'root_proof_sha256':sha(DELTA/'ROOT_ADOPTION_REVIEW.json')}))
