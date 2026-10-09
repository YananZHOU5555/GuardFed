"""ROOT-only bind actual C_after36 exact4 archive to source, actual terminal and prior136."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,sys,tarfile
sys.dont_write_bytecode=True
if sys.flags.optimize:
    raise RuntimeError('Optimized Python is forbidden for root operation guards')
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp/celeba_mechanism_valid_C_after36_20261010';EX=BASE/'execution_candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())

def check_terminal(live,expected):
    assert 'EXITED' in live['service'] and not live['processes'] and not live['batch_failure']
    assert live['batch_complete'] and len(live['completed'])==4
    ids=[row['id'] for row in live['completed']]
    assert len(ids)==len(set(ids))==len(expected)==4 and set(ids)==set(expected)

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
    parser.add_argument('--terminal-progress',required=True,type=Path)
    parser.add_argument('--terminal-progress-sha256',required=True)
    parser.add_argument('--startup-sha256',required=True)
    parser.add_argument('--deployment-sha256',required=True)
    args=parser.parse_args();assert not sys.flags.optimize
    DELTA=args.delta.resolve();assert not args.delta.is_symlink()
    assert DELTA.parent==(EX/'backups').resolve() and DELTA.name.startswith('incremental_')
    assert not (DELTA/'ROOT_ADOPTION_REVIEW.json').exists(), 'Existing adoption must be preserved'
    assert sha(BASE/'FILES_SHA256.json')=='450ae61e37432f6651da1a594ed5b9f701c465282435a3d8242664ae228d4509'
    assert sha(EX/'EXECUTION_SOURCE_SHA256.json')=='b1e6467eac5b7218cda6af189c2ae2b655fb80780d48db05b45463aa9bdb578f'
    assert sha(BASE/'inventory_actual140_Full100refs.json')=='ddda34996da47f176520e782ca03ebf536bda7b1a361b6e34b70092e660299c1'
    assert sha(EX/'ROOT_STARTUP_OBSERVATION.json')==args.startup_sha256
    assert sha(EX/'deployment_receipt.json')==args.deployment_sha256
    assert sha(DELTA/'backup_receipt.json')==args.receipt_sha256
    assert sha(DELTA/'OFFSERVER_VERIFICATION.json')==args.proof_sha256
    scope=read(BASE/'SCOPE.json');expected=scope['selected_ids']
    assert len(expected)==len(set(expected))==4
    receipt=read(DELTA/'backup_receipt.json');proof=read(DELTA/'OFFSERVER_VERIFICATION.json')
    assert receipt['accepted_new_ids']==receipt['all_accepted_ids']==proof['accepted_new_ids']==proof['all_accepted_ids']==expected
    assert receipt['previous_backup_receipt_sha256'] is proof['previous_backup_receipt_sha256'] is None
    assert proof['status']=='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
    assert (proof['independent_metric_checks'],proof['independent_confusion_count_checks'],proof['prediction_rule_checks'])==(36,96,12)
    assert proof['source_seal_sha256']==sha(EX/'EXECUTION_SOURCE_SHA256.json')
    progress=args.terminal_progress.resolve()
    assert progress.parent==EX.resolve() and progress.name.startswith('ROOT_PROGRESS_') and not progress.name.endswith('.RAW.json')
    assert sha(progress)==args.terminal_progress_sha256
    live=read(progress);check_terminal(live,expected)
    attempt=read(EX/'ROOT_BACKUP_ATTEMPT.json')
    assert attempt['terminal_progress_sha256']==sha(progress) and attempt['selected_ids']==expected
    assert attempt['startup_sha256']==args.startup_sha256 and attempt['deployment_sha256']==args.deployment_sha256
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
        assert inventory['original_inventory_sha256']==sha(BASE/'inventory_actual140_Full100refs.json')
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
        assert sha(EX/'ROOT_APPROVED.json')==deployment['root_approval_sha256']
        assert sha(EX/'EXECUTION_DRAFT.json')==deployment['external_draft_sha256']
        assert inventory['approval_sha256']==sha(EX/'APPROVED.json')
        assert json.loads(bundle.extractfile('runtime/batch_complete.json').read())==live['batch_complete']
    original={r['id']:r for r in read(BASE/'inventory_actual140_Full100refs.json')['records']}
    assert len(proof['records'])==4 and {r['id'] for r in proof['records']}==set(expected)
    for row in proof['records']:
        assert row['checkpoint_sha256']==original[row['id']]['checkpoint']['sha256']
        assert row['native_max_abs_difference']==0 and set(row['views'])=={'native','raw','shared_calibration'}
    prior=ROOT/'tmp/celeba_mechanism_valid_C_after28_20261009/execution_candidate/backups/incremental_20261009T221150Z/ROOT_ADOPTION_REVIEW.json'
    assert sha(prior)=='cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0'
    assert read(prior)['cumulative_three_view_models']==136
    assert len(scope['excluded_prior_ids'])==len(set(scope['excluded_prior_ids']))==136
    assert set(original)==set(scope['excluded_prior_ids'])|set(expected) and not set(expected)&set(scope['excluded_prior_ids'])
    result=dict(status='ROOT_C_AFTER36_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_new_ids=expected,
        prior_three_view_models=136,accepted_new=4,cumulative_three_view_models=140,
        archive_sha256=sha(archive),archive_members_verified=expected_members,content_members_verified=expected_members-1,
        server_strict_bound_in_saved_receipts=True,offserver_verification_sha256=sha(DELTA/'OFFSERVER_VERIFICATION.json'),
        backup_receipt_sha256=sha(DELTA/'backup_receipt.json'),previous_backup_receipt_sha256=None,
        execution_seal_sha256=sha(EX/'EXECUTION_SOURCE_SHA256.json'),science_seal_sha256=sha(BASE/'FILES_SHA256.json'),
        all_native_differences_zero=True,new_training=0,new_Full_inference=0,test_inference=False,
        original136_unchanged=True,prior136_root_adoption_sha256=sha(prior),new_CNN_inference_for_root_review=0,
        negative_results_preserved=True,remote_terminal_proof_sha256=sha(progress),source_scope_complete=True,
        startup_observation_sha256=sha(EX/'ROOT_STARTUP_OBSERVATION.json'),backup_attempt_sha256=sha(EX/'ROOT_BACKUP_ATTEMPT.json'))
    target=DELTA/'ROOT_ADOPTION_REVIEW.json'
    with target.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result|{'root_proof_sha256':sha(target)}))

if __name__=='__main__':main()
