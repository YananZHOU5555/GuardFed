"""ROOT-only one-shot C_after36 deployment. Preparing/importing this file does not run it."""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, shlex, subprocess, sys, tarfile, traceback

sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Optimized Python is forbidden for root operation guards')
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'tmp/celeba_mechanism_valid_C_after36_20261010'
EX = BASE / 'execution_candidate'
REMOTE = '/workspace/guardfed_checks/celeba_mechanism_valid_C_after36_20261010'
SCIENCE = '450ae61e37432f6651da1a594ed5b9f701c465282435a3d8242664ae228d4509'
EXECUTION = 'b1e6467eac5b7218cda6af189c2ae2b655fb80780d48db05b45463aa9bdb578f'
PACKAGE = '8dbcafb42cd1c67ead623feccc8d332aebdbf49a8ba820d95864ee63eb0e4d83'
LINEAGE = 'b1ff1fad6f5fe08cebec3008b6df861396c2e11c289b8f05815debb78dc3ed14'
GUIDE = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
HOST = 'root@89.22.197.55'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')



def validate_source_review(path, expected_sha):
    assert not sys.flags.optimize, 'Optimized Python is forbidden'
    assert sha(path) == expected_sha, 'External source review SHA differs'
    review = read(path)
    assert review['status'] == 'PASS_SOURCE_READY_FOR_ROOT_LINUX_PREFLIGHT_AND_EXACT4_APPROVAL'
    assert review['source_adoptable'] is True and review['actual_dispatch_authorized_by_this_review'] is False
    assert review['science_seal_sha256'] == SCIENCE and review['execution_seal_sha256'] == EXECUTION
    assert review['package_sha256'] == '647409f863e37d0031fcd1794d6f5f128c481ee2e71d156720c69b06376228e5'
    assert review['native_accepted_snapshot'] == 140 and review['excluded_prior_three_view_ids'] == 136
    assert review['old136_records_exact'] and review['Full100_references_exact'] and review['Full100_actual900_source_records_exact']
    assert review['new_three_view_accepted'] == 0 and review['positive_approval_exact4'] is True
    assert review['exact_selected_ids'] == ['minus_C_IID_S-DFA_seed91007', 'minus_C_IID_S-DFA_seed91008', 'minus_C_IID_S-DFA_seed91009', 'minus_C_IID_S-DFA_seed91010']
    assert review['actual_worker_pre_science_bind_ids'] == review['exact_selected_ids']
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-seal', required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    assert not sys.flags.optimize and args.execution_seal == EXECUTION
    review_path = args.review.resolve()
    review = validate_source_review(review_path, args.review_sha256)
    assert sha(BASE / 'FILES_SHA256.json') == SCIENCE and sha(EX / 'EXECUTION_SOURCE_SHA256.json') == EXECUTION
    assert sha(BASE / 'PACKAGE_RECEIPT.json') == PACKAGE
    assert sha(BASE / 'PACKAGE_SHA256.json') == review['package_sha256']
    assert sha(ROOT / 'tmp/celeba_mechanism_valid_incremental_next11_20261009/root_source_review/ROOT_REVIEW.json') == LINEAGE
    scope = read(BASE / 'SCOPE.json')
    assert scope['selected_ids'] == review['exact_selected_ids'] and len(scope['excluded_prior_ids']) == review['excluded_prior_three_view_ids'] == 136
    assert len(scope['selected_ids']) == 4 and not set(scope['selected_ids']).intersection(scope['excluded_prior_ids'])
    for name in ('ROOT_APPROVED.json', 'EXECUTION_DRAFT.json', 'root_deployment_source.tar.gz', 'deployment_receipt.json', 'ROOT_DEPLOYMENT_FAILURE.json'):
        assert not (EX / name).exists(), 'Existing root attempt preserved: ' + name
    members = {}
    for folder, seal, relative in ((BASE, 'FILES_SHA256.json', Path('.')), (EX, 'EXECUTION_SOURCE_SHA256.json', Path('execution_candidate'))):
        for row in read(folder / seal)['members']:
            path = folder / row['path']
            assert sha(path) == row['sha256'] and path.stat().st_size == row['size']
            members[(relative / row['path']).as_posix()] = path
        members[(relative / seal).as_posix()] = folder / seal
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350', HOST]
    preflight = """from pathlib import Path
import hashlib,subprocess
guide=Path('/etc/vast-agents-guide.md').read_bytes()
assert hashlib.sha256(guide).hexdigest()==%r
assert not Path(%r).exists()
for service,expected in [('sglang','STOPPED'),('guardfed_celeba_mechanism_valid_C_after28','EXITED')]:
 s=subprocess.run(['supervisorctl','status',service],capture_output=True,text=True).stdout.strip()
 assert len(s.split())>=2 and s.split()[0]==service and s.split()[1]==expected,s
print('FRESH_C_AFTER36_NAMESPACE_GUIDE_SHA_SGLANG_STOPPED_PRIOR_C_AFTER28_EXITED')
""" % (GUIDE, REMOTE)
    pre = subprocess.run(ssh + ['python -B -'], input=preflight.encode(), capture_output=True, check=True, timeout=30)
    # ROOT invocation creates fresh authority; b1ff is historical science lineage only.
    authority = read(EX / 'ROOT_REVIEW_TEMPLATE.json')
    authority.update(status='ROOT_REVIEW_PASS_BOUNDED_C_AFTER36_VALID_REPLAY', execution_authorized_within_existing_user_request=True,
        reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), execution_seal_sha256=EXECUTION,
        independent_C_after36_review_sha256=args.review_sha256, execution_review_sha256=args.review_sha256,
        scientific_source_review_sha256=LINEAGE, authorization='ROOT invocation: exact4 accepted-terminal valid-only replay under existing user request; original science and pending endpoint retained.')
    assert authority['source_only_root_review_sha256'] == LINEAGE
    save(EX / 'ROOT_APPROVED.json', authority)
    draft = read(EX / 'APPROVED_TEMPLATE.json')
    draft.update(status='APPROVED_C_AFTER36_MECHANISM_VALID_REPLAY_ONLY', root_approval_sha256=sha(EX / 'ROOT_APPROVED.json'), execution_seal_sha256=EXECUTION)
    save(EX / 'EXECUTION_DRAFT.json', draft)
    members['root_independent_review/ROOT_INDEPENDENT_REVIEW.json'] = review_path
    for name in ('ROOT_APPROVED.json', 'EXECUTION_DRAFT.json'):
        members['execution_candidate/' + name] = EX / name
    archive = EX / 'root_deployment_source.tar.gz'
    with tarfile.open(archive, 'x:gz') as bundle:
        for name, path in sorted(members.items()):
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and '..' not in rel.parts and path.is_file() and not path.is_symlink()
            bundle.add(path, arcname=BASE.name + '/' + name, recursive=False)
    archive_sha = sha(archive)
    remote_archive = '/workspace/guardfed_checks/C_after36_source_root_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.tar.gz'
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-P', '60350', str(archive), HOST + ':' + remote_archive], check=True, timeout=90)
    code = """from pathlib import Path,PurePosixPath
import hashlib,json,subprocess,tarfile
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()==%r
archive=Path(%r);assert hashlib.sha256(archive.read_bytes()).hexdigest()==%r
target=Path(%r);assert not target.exists()
with tarfile.open(archive) as bundle:
 for item in bundle:
  rel=PurePosixPath(item.name)
  assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]==%r
 bundle.extractall(target.parent,filter='data')
cmd=['taskset','-c','112-119','ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(target/'execution_candidate/install_once.py'),'--draft-sha256',%r]
result=subprocess.run(cmd,capture_output=True,text=True)
print(json.dumps(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)))
""" % (GUIDE, remote_archive, archive_sha, REMOTE, BASE.name, sha(EX / 'EXECUTION_DRAFT.json'))
    result = subprocess.run(ssh + ['python -B -'], input=code.encode(), capture_output=True, check=True, timeout=180)
    installed = json.loads(result.stdout)
    receipt = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_archive_sha256=archive_sha,
        source_archive_members=len(members), source_archive_remote=remote_archive, science_seal_sha256=SCIENCE,
        execution_seal_sha256=EXECUTION, independent_C_after36_review_sha256=args.review_sha256, historical_science_lineage_sha256=LINEAGE,
        root_approval_sha256=sha(EX / 'ROOT_APPROVED.json'), external_draft_sha256=sha(EX / 'EXECUTION_DRAFT.json'),
        empty_namespace_prerequisite=pre.stdout.decode().strip(), remote_installation=installed,
        new_independent_source_review_path=str(review_path), new_independent_source_review_sha256=args.review_sha256, root_remote_stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), new_training=0, new_Full_inference=0, test_inference=False)
    save(EX / 'deployment_receipt.json', receipt)
    assert installed['returncode'] == 0, installed
    print(json.dumps(dict(status='C_AFTER36_INSTALLED_ACTUAL_STARTUP_OBSERVATION_PENDING', deployment_receipt_sha256=sha(EX / 'deployment_receipt.json'))))


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as error:
        failure = EX / 'ROOT_DEPLOYMENT_FAILURE.json'
        if EX.exists() and not failure.exists():
            save(failure, dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False, existing_outputs_preserved=True))
        raise
