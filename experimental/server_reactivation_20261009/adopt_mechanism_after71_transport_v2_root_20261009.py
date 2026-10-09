"""ROOT-only after71 transport-v2 adoption; original scientific guards and prior71."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,sys,tarfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_after71_20261009';EX=BASE/'execution_candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())

def check_terminal(live,expected):
    assert 'EXITED' in live['service'] and not live['processes'] and not live['batch_failure']
    assert live['batch_complete'] and len(live['completed'])==11
    ids=[row['id'] for row in live['completed']]
    assert len(ids)==len(set(ids))==len(expected)==11 and set(ids)==set(expected)

def expected_archive_names(expected,science,execution):
    names={'backup_inventory.json','source/science/FILES_SHA256.json','runtime/batch_complete.json','execution/backup_completed.py'}
    names.update('source/science/'+row['path'] for row in science['members'])
    names.update('source/'+row['path'] for row in execution['members'])
    names.update('source/'+name for name in ('EXECUTION_SOURCE_SHA256.json','ROOT_APPROVED.json','EXECUTION_DRAFT.json',
        'APPROVED.json','APPROVED.sha256','preflight.json','start_receipt.json','batch_resource_before.json'))
    for identity in expected:
        names.update('runs/'+identity+'/'+name for name in ('receipt.json','bridge_receipt.json','validation_predictions.npz','strict_acceptance.json'))
        names.update(('logs/'+identity+'.log','approvals/'+identity+'.json','runtime/completed_'+identity+'.json'))
    assert len(names)==7*len(expected)+len(science['members'])+len(execution['members'])+12
    return names

def recovery_review():
    old=ROOT/'tmp/backup_mechanism_after71_root_20261009.py'
    assert sha(old)=='09f4f9d015dc2a745edc8a82f54043130928fe730a42b226f9da6f8ffd64e6dd'
    assert sha(EX/'ROOT_BACKUP_ATTEMPT.json')=='cd01f77444fd21c40e6565dfc026e83593b88ca5d94d0459506e3287d2041058'
    path=EX/'ROOT_TRANSPORT_RECOVERY_REVIEW.json'
    assert sha(path)=='7c9cf002a689d8d418cfa5694e1d5751e4ffdec6d158050380cce297f7cadefe'
    review=read(path)
    assert review['status']=='ROOT_WINDOWS_ARGV206_PRE_SSH_FAILURE_AND_EMPTY_REMOTE_BACKUP_CONFIRMED'
    assert review['old_helper_sha256']==sha(old) and review['old_attempt_sha256']==sha(EX/'ROOT_BACKUP_ATTEMPT.json')
    assert review['old_command_result_exists'] is review['old_command_stdout_exists'] is False
    remote=review['remote_readonly']
    assert remote['backup_latest_exists'] is remote['batch_failure_exists'] is False
    assert remote['backup_directories']==[] and remote['completed']==11
    assert remote['batch_complete_sha256']=='1a66d99505715ae501b9fb080c22f026fcd910436a36b0b986bede068329770a'
    assert review['new_inference']==0 and review['automatic_retry'] is False
    return review


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delta',required=True,type=Path)
    parser.add_argument('--receipt-sha256',required=True)
    parser.add_argument('--proof-sha256',required=True)
    parser.add_argument('--terminal-progress-sha256',required=True)
    args=parser.parse_args();assert not sys.flags.optimize
    recovery=recovery_review()
    assert sha(ROOT/'tmp/adopt_mechanism_after71_root_20261009.py')=='2e9c639b7ff46f7f0387dffc5a7c83e6d0d54afc8fee7fc5f8d017135dbdf8a8'
    DELTA=args.delta.resolve();assert not args.delta.is_symlink()
    assert DELTA.parent==(EX/'backups').resolve() and DELTA.name.startswith('incremental_')
    assert not (DELTA/'ROOT_ADOPTION_REVIEW.json').exists(), 'Existing adoption must be preserved'
    assert sha(BASE/'FILES_SHA256.json')=='d05a0b81620d1791858a9f71405252443e593600f390359175ff961dde0e2bec'
    assert sha(EX/'EXECUTION_SOURCE_SHA256.json')=='0fe232aab75a870b7840fd6ef0bba85b3d12c541ca49a0c0976f3f8a7095352f'
    assert sha(BASE/'inventory_actual82_Full100refs.json')=='72aca76f626580faf465c61f35b2274068e76782eb1bcef543e98da23c2e6d1e'
    assert sha(EX/'ROOT_STARTUP_OBSERVATION.json')=='a95018b7ccc62472d7b1f796e7a2d1307a72bb4bd4905397c9f623ca6a4d2f60'
    assert sha(DELTA/'backup_receipt.json')==args.receipt_sha256
    assert sha(DELTA/'OFFSERVER_VERIFICATION.json')==args.proof_sha256
    scope=read(BASE/'SCOPE.json');expected=scope['selected_ids']
    assert len(expected)==len(set(expected))==11
    receipt=read(DELTA/'backup_receipt.json');proof=read(DELTA/'OFFSERVER_VERIFICATION.json')
    assert receipt['accepted_new_ids']==receipt['all_accepted_ids']==proof['accepted_new_ids']==proof['all_accepted_ids']==expected
    assert receipt['previous_backup_receipt_sha256'] is proof['previous_backup_receipt_sha256'] is None
    assert proof['status']=='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
    assert (proof['independent_metric_checks'],proof['independent_confusion_count_checks'],proof['prediction_rule_checks'])==(99,264,33)
    assert proof['source_seal_sha256']==sha(EX/'EXECUTION_SOURCE_SHA256.json')
    progress=[p for p in EX.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json')]
    assert progress, 'Actual terminal observation is required'
    progress=max(progress);assert sha(progress)==args.terminal_progress_sha256
    live=read(progress);check_terminal(live,expected)
    attempt=read(EX/'ROOT_BACKUP_TRANSPORT_ATTEMPT.json')
    assert attempt['terminal_progress_sha256']==sha(progress) and attempt['selected_ids']==expected
    assert attempt['recovery_review_sha256']==sha(EX/'ROOT_TRANSPORT_RECOVERY_REVIEW.json')
    assert attempt['original_attempt_sha256']==sha(EX/'ROOT_BACKUP_ATTEMPT.json')
    assert attempt['old_helper_sha256']==recovery['old_helper_sha256']
    assert read(EX/'ROOT_BACKUP_TRANSPORT_COMMAND_RESULT.json')['returncode']==0
    assert read(EX/'ROOT_BACKUP_COMMAND_STDOUT.json')==receipt
    archive=DELTA/'incremental_valid_three_views.tar.gz'
    assert sha(archive)==receipt['archive_sha256']==proof['archive_sha256']
    expected_names=expected_archive_names(expected,read(BASE/'FILES_SHA256.json'),read(EX/'EXECUTION_SOURCE_SHA256.json'))
    expected_members=len(expected_names)
    with tarfile.open(archive) as bundle:
        assert len(bundle.getnames())==len(set(bundle.getnames()))==receipt['members']==expected_members
        assert set(bundle.getnames())==expected_names
        raw=bundle.extractfile('backup_inventory.json').read();inventory=json.loads(raw)
        assert hashlib.sha256(raw).hexdigest()==receipt['inventory_sha256']==proof['inventory_sha256']
        assert set(bundle.getnames())==set(inventory['members'])|{'backup_inventory.json'}
        assert inventory['previous_backup'] is None
        assert inventory['models_repacked']==inventory['new_training']==inventory['new_test_inference']==0
        assert inventory['execution_seal_sha256']==sha(EX/'EXECUTION_SOURCE_SHA256.json')
        assert inventory['accepted_new_ids']==inventory['all_accepted_ids']==expected
        assert inventory['original_inventory_sha256']==sha(BASE/'inventory_actual82_Full100refs.json')
        assert inventory['original_prepared_seal_sha256']==sha(BASE/'FILES_SHA256.json')
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
        deployment=read(EX/'deployment_receipt.json')
        assert sha(EX/'ROOT_APPROVED.json')==deployment['root_approval_sha256']=='0438a11351de81103f52983dd86b5eee5f5ecbabdef4585dbb44d01602af838d'
        assert sha(EX/'EXECUTION_DRAFT.json')==deployment['external_draft_sha256']
        assert inventory['approval_sha256']==sha(EX/'APPROVED.json')
        assert json.loads(bundle.extractfile('runtime/batch_complete.json').read())==live['batch_complete']
    original={r['id']:r for r in read(BASE/'inventory_actual82_Full100refs.json')['records']}
    assert len(proof['records'])==11 and {r['id'] for r in proof['records']}==set(expected)
    for row in proof['records']:
        assert row['checkpoint_sha256']==original[row['id']]['checkpoint']['sha256']
        assert row['native_max_abs_difference']==0 and set(row['views'])=={'native','raw','shared_calibration'}
    prior=ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009/execution_candidate/backups/incremental_20261009T152532Z/ROOT_ADOPTION_REVIEW.json'
    assert sha(prior)=='692ecd168ecab0b9c960965decb68424ce2ad80a5cf7ca452d2739da6b0a768a'
    assert read(prior)['cumulative_three_view_models']==71
    assert len(scope['excluded_prior_ids'])==len(set(scope['excluded_prior_ids']))==71
    assert set(original)==set(scope['excluded_prior_ids'])|set(expected) and not set(expected)&set(scope['excluded_prior_ids'])
    result=dict(status='ROOT_AFTER71_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_new_ids=expected,
        prior_three_view_models=71,accepted_new=11,cumulative_three_view_models=82,
        archive_sha256=sha(archive),archive_members_verified=expected_members,content_members_verified=expected_members-1,
        server_strict_bound_in_saved_receipts=True,offserver_verification_sha256=sha(DELTA/'OFFSERVER_VERIFICATION.json'),
        backup_receipt_sha256=sha(DELTA/'backup_receipt.json'),previous_backup_receipt_sha256=None,
        execution_seal_sha256=sha(EX/'EXECUTION_SOURCE_SHA256.json'),science_seal_sha256=sha(BASE/'FILES_SHA256.json'),
        all_native_differences_zero=True,new_training=0,new_Full_inference=0,test_inference=False,
        original71_unchanged=True,prior71_root_adoption_sha256=sha(prior),new_CNN_inference_for_root_review=0,
        negative_results_preserved=True,remote_terminal_proof_sha256=sha(progress),source_scope_complete=True,
        startup_observation_sha256=sha(EX/'ROOT_STARTUP_OBSERVATION.json'),backup_transport_attempt_sha256=sha(EX/'ROOT_BACKUP_TRANSPORT_ATTEMPT.json'),
        original_failed_attempt_sha256=sha(EX/'ROOT_BACKUP_ATTEMPT.json'),
        transport_recovery_review_sha256=sha(EX/'ROOT_TRANSPORT_RECOVERY_REVIEW.json'))
    target=DELTA/'ROOT_ADOPTION_REVIEW.json'
    with target.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result|{'root_proof_sha256':sha(target)}))

if __name__=='__main__':main()
