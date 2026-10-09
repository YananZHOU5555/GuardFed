"""ROOT-only bind actual after92 exact8 archive to source, latest terminal and prior92."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,sys,tarfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_after92_20261009';EX=BASE/'execution_candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())

def check_terminal(live,expected):
    assert 'EXITED' in live['service'] and not live['processes'] and not live['batch_failure']
    assert live['batch_complete'] and len(live['completed'])==8
    ids=[row['id'] for row in live['completed']]
    assert len(ids)==len(set(ids))==len(expected)==8 and set(ids)==set(expected)

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

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delta',required=True,type=Path)
    parser.add_argument('--receipt-sha256',required=True)
    parser.add_argument('--proof-sha256',required=True)
    parser.add_argument('--terminal-progress-sha256',required=True)
    args=parser.parse_args();assert not sys.flags.optimize
    DELTA=args.delta.resolve();assert not args.delta.is_symlink()
    assert DELTA.parent==(EX/'backups').resolve() and DELTA.name.startswith('incremental_')
    assert not (DELTA/'ROOT_ADOPTION_REVIEW.json').exists(), 'Existing adoption must be preserved'
    assert sha(BASE/'FILES_SHA256.json')=='832e02a7ab0bc58dd22373c1793c39fc7926a4e961d4fba0c750964eb7b7a94a'
    assert sha(EX/'EXECUTION_SOURCE_SHA256.json')=='68b11d80698d1073250736ad73695a79b29ed26d0bbc133441876a56f6f5c5bf'
    assert sha(BASE/'inventory_actual100_Full100refs.json')=='156cfea40a65e47f122b96a2b08369f387bf5b7397a0c82e05fe616888ef61bd'
    assert sha(EX/'ROOT_STARTUP_OBSERVATION.json')=='c773018542610620d1640f21bf8b8c443b7bff442898a922fc56ea93ee228030'
    assert sha(DELTA/'backup_receipt.json')==args.receipt_sha256
    assert sha(DELTA/'OFFSERVER_VERIFICATION.json')==args.proof_sha256
    scope=read(BASE/'SCOPE.json');expected=scope['selected_ids']
    assert len(expected)==len(set(expected))==8
    receipt=read(DELTA/'backup_receipt.json');proof=read(DELTA/'OFFSERVER_VERIFICATION.json')
    assert receipt['accepted_new_ids']==receipt['all_accepted_ids']==proof['accepted_new_ids']==proof['all_accepted_ids']==expected
    assert receipt['previous_backup_receipt_sha256'] is proof['previous_backup_receipt_sha256'] is None
    assert proof['status']=='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
    assert (proof['independent_metric_checks'],proof['independent_confusion_count_checks'],proof['prediction_rule_checks'])==(72,192,24)
    assert proof['source_seal_sha256']==sha(EX/'EXECUTION_SOURCE_SHA256.json')
    progress=[p for p in EX.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json')]
    assert progress, 'Actual terminal observation is required'
    progress=max(progress);assert sha(progress)==args.terminal_progress_sha256
    live=read(progress);check_terminal(live,expected)
    attempt=read(EX/'ROOT_BACKUP_ATTEMPT.json')
    assert attempt['terminal_progress_sha256']==sha(progress) and attempt['selected_ids']==expected
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
        assert inventory['original_inventory_sha256']==sha(BASE/'inventory_actual100_Full100refs.json')
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
        assert sha(EX/'ROOT_APPROVED.json')==deployment['root_approval_sha256']=='90f2b12d1c966ba9f3818f23ca5d8563ec9e3ea64e57c142c29c9f3aaf3a1560'
        assert sha(EX/'EXECUTION_DRAFT.json')==deployment['external_draft_sha256']
        assert inventory['approval_sha256']==sha(EX/'APPROVED.json')
        assert json.loads(bundle.extractfile('runtime/batch_complete.json').read())==live['batch_complete']
    original={r['id']:r for r in read(BASE/'inventory_actual100_Full100refs.json')['records']}
    assert len(proof['records'])==8 and {r['id'] for r in proof['records']}==set(expected)
    for row in proof['records']:
        assert row['checkpoint_sha256']==original[row['id']]['checkpoint']['sha256']
        assert row['native_max_abs_difference']==0 and set(row['views'])=={'native','raw','shared_calibration'}
    prior=ROOT/'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009/execution_candidate/backups/incremental_20261009T174922Z/ROOT_ADOPTION_REVIEW.json'
    assert sha(prior)=='b9e40d1ca565c0bcf146058433ff3e037ab4e824aa6972d1a3f3f47a088e8683'
    assert read(prior)['cumulative_three_view_models']==92
    assert len(scope['excluded_prior_ids'])==len(set(scope['excluded_prior_ids']))==92
    assert set(original)==set(scope['excluded_prior_ids'])|set(expected) and not set(expected)&set(scope['excluded_prior_ids'])
    result=dict(status='ROOT_AFTER92_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_new_ids=expected,
        prior_three_view_models=92,accepted_new=8,cumulative_three_view_models=100,
        archive_sha256=sha(archive),archive_members_verified=expected_members,content_members_verified=expected_members-1,
        server_strict_bound_in_saved_receipts=True,offserver_verification_sha256=sha(DELTA/'OFFSERVER_VERIFICATION.json'),
        backup_receipt_sha256=sha(DELTA/'backup_receipt.json'),previous_backup_receipt_sha256=None,
        execution_seal_sha256=sha(EX/'EXECUTION_SOURCE_SHA256.json'),science_seal_sha256=sha(BASE/'FILES_SHA256.json'),
        all_native_differences_zero=True,new_training=0,new_Full_inference=0,test_inference=False,
        original92_unchanged=True,prior92_root_adoption_sha256=sha(prior),new_CNN_inference_for_root_review=0,
        negative_results_preserved=True,remote_terminal_proof_sha256=sha(progress),source_scope_complete=True,
        startup_observation_sha256=sha(EX/'ROOT_STARTUP_OBSERVATION.json'),backup_attempt_sha256=sha(EX/'ROOT_BACKUP_ATTEMPT.json'))
    target=DELTA/'ROOT_ADOPTION_REVIEW.json'
    with target.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result|{'root_proof_sha256':sha(target)}))

if __name__=='__main__':main()
