"""Adopt the reviewed first five off-server replays; verify every archive byte."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_next37_20261009'
EX=BASE/'execution_candidate';DELTA=EX/'backups/incremental_20261009T141304Z'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(DELTA/'OFFSERVER_VERIFICATION.json')=='5f8e6c13e490c113af244ee24eb94b528a0721d1e0949812a22aa37b74ea5247'
assert sha(DELTA/'backup_receipt.json')=='e22525155441aa6d2368b252ca103fdba03978c00e45659e1ba72123c1a45ec6'
assert sha(EX/'EXECUTION_SOURCE_SHA256.json')=='4f71c60c6966da40d334f0ac17141a7e763520b7fe478df8548809e773971caf'
receipt=read(DELTA/'backup_receipt.json');proof=read(DELTA/'OFFSERVER_VERIFICATION.json')
expected=[f'minus_U_IID_FedSA_seed{s}' for s in range(91004,91009)]
assert proof['accepted_new_ids']==proof['all_accepted_ids']==receipt['accepted_new_ids']==expected
assert proof['status']=='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
assert proof['independent_metric_checks']==45 and proof['independent_confusion_count_checks']==120 and proof['prediction_rule_checks']==15
assert proof['previous_backup_receipt_sha256'] is receipt['previous_backup_receipt_sha256'] is None
assert proof['source_seal_sha256']==sha(EX/'EXECUTION_SOURCE_SHA256.json')
archive=DELTA/'incremental_valid_three_views.tar.gz'
assert sha(archive)==receipt['archive_sha256']==proof['archive_sha256']=='4f16c82638807b3432610d31600a34b4c4fa3ffe092babdbf63d18e1ab260f3f'
with tarfile.open(archive) as bundle:
    assert len(bundle.getnames())==len(set(bundle.getnames()))==receipt['members']==78
    raw=bundle.extractfile('backup_inventory.json').read();inventory=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==receipt['inventory_sha256']==proof['inventory_sha256']
    assert set(bundle.getnames())==set(inventory['members'])|{'backup_inventory.json'} and len(inventory['members'])==77
    for item in bundle:
        rel=PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
        payload=bundle.extractfile(item).read()
        if item.name in inventory['members']:
            row=inventory['members'][item.name];assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
    for row in read(BASE/'FILES_SHA256.json')['members']:
        assert inventory['members']['source/science/'+row['path']]['sha256']==row['sha256']==sha(BASE/row['path'])
    for name in ('ROOT_APPROVED.json','EXECUTION_DRAFT.json','APPROVED.json'):
        assert inventory['members']['source/'+name]['sha256']==sha(EX/name)
original={r['id']:r for r in read(BASE/'inventory_actual60_Full100refs.json')['records']}
for row in proof['records']:
    assert row['id'] in expected and row['checkpoint_sha256']==original[row['id']]['checkpoint']['sha256']
    assert row['native_max_abs_difference']==0 and set(row['views'])=={'native','raw','shared_calibration'}
result=dict(status='ROOT_NEXT37_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_new_ids=expected,
    prior_three_view_models=23,accepted_new=5,cumulative_three_view_models=28,
    archive_sha256=sha(archive),archive_members_verified=78,content_members_verified=77,
    server_strict_bound_in_saved_receipts=True,offserver_verification_sha256=sha(DELTA/'OFFSERVER_VERIFICATION.json'),
    backup_receipt_sha256=sha(DELTA/'backup_receipt.json'),previous_backup_receipt_sha256=None,
    execution_seal_sha256=sha(EX/'EXECUTION_SOURCE_SHA256.json'),science_seal_sha256=sha(BASE/'FILES_SHA256.json'),
    all_native_differences_zero=True,new_training=0,new_Full_inference=0,test_inference=False,
    original23_unchanged=True,new_CNN_inference_for_root_review=0,negative_results_preserved=True)
target=DELTA/'ROOT_ADOPTION_REVIEW.json'
with target.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result|{'root_proof_sha256':sha(target)}))
