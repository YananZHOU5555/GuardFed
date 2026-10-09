"""ROOT-only one-shot after82_v2 deployment. Preparing/importing this file does not run it."""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, shlex, subprocess, sys, tarfile, traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009'
EX = BASE / 'execution_candidate'
REVIEW = BASE / 'root_independent_review/ROOT_READY_REVIEW.json'
REMOTE = '/workspace/guardfed_checks/celeba_mechanism_valid_incremental_after82_v2_20261009'
SCIENCE = 'b95801ac039bb39276e79012393792adfde290b92091ff45c0a4323bd4fdd8f0'
EXECUTION = '94d99842b334346ae8a6715b84f7fddea35574a77fbf51f9460b25301c6ccae6'
PACKAGE = '4f2232933fcf474548dba28724afb8dbb8b39c12ae42169c91e1440f51934d28'
LINEAGE = 'b1ff1fad6f5fe08cebec3008b6df861396c2e11c289b8f05815debb78dc3ed14'
GUIDE = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
HOST = 'root@89.22.197.55'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-seal', required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    assert not sys.flags.optimize and args.execution_seal == EXECUTION
    assert sha(REVIEW) == args.review_sha256
    review = read(REVIEW)
    assert review['status'] == 'ROOT_READY_AFTER82_V2_SOURCE_REVIEW_PASS_NOT_DISPATCHED'
    assert review['science_seal_sha256'] == SCIENCE and review['execution_seal_sha256'] == EXECUTION
    assert review['package_seal_sha256'] == PACKAGE and review['helper_sha256'] == sha(Path(__file__))
    assert review['scientific_functions_unchanged'] and not review['CNN_executed'] and not review['dispatch_performed']
    assert sha(BASE / 'FILES_SHA256.json') == SCIENCE and sha(EX / 'EXECUTION_SOURCE_SHA256.json') == EXECUTION
    assert sha(BASE / 'PACKAGE_SHA256.json') == PACKAGE
    assert sha(ROOT / 'tmp/celeba_mechanism_valid_incremental_next11_20261009/root_source_review/ROOT_REVIEW.json') == LINEAGE
    scope = read(BASE / 'SCOPE.json')
    assert scope['selected_ids'] == review['selected_ids'] and scope['excluded_prior_ids'] == review['excluded82_ids']
    assert len(scope['selected_ids']) == 10 and not set(scope['selected_ids']).intersection(scope['excluded_prior_ids'])
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
for service,expected in [('sglang','STOPPED'),('guardfed_celeba_mechanism_valid_after82','EXITED')]:
 s=subprocess.run(['supervisorctl','status',service],capture_output=True,text=True).stdout.strip()
 assert len(s.split())>=2 and s.split()[0]==service and s.split()[1]==expected,s
print('FRESH_AFTER82_V2_NAMESPACE_GUIDE_SHA_SGLANG_STOPPED_PRIOR_NEXT11_EXITED')
""" % (GUIDE, REMOTE)
    pre = subprocess.run(ssh + ['python -B -'], input=preflight.encode(), capture_output=True, check=True, timeout=30)
    # ROOT invocation creates fresh authority; b1ff is historical science lineage only.
    authority = read(EX / 'ROOT_REVIEW_TEMPLATE.json')
    authority.update(status='ROOT_REVIEW_PASS_BOUNDED_AFTER82_V2_VALID_REPLAY', execution_authorized_within_existing_user_request=True,
        reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), execution_seal_sha256=EXECUTION,
        independent_after82_v2_review_sha256=args.review_sha256, execution_review_sha256=args.review_sha256,
        scientific_source_review_sha256=LINEAGE, authorization='ROOT invocation: exact10 accepted-terminal valid-only replay under existing user request; original science and pending endpoint retained.')
    assert authority['source_only_root_review_sha256'] == LINEAGE
    save(EX / 'ROOT_APPROVED.json', authority)
    draft = read(EX / 'APPROVED_TEMPLATE.json')
    draft.update(status='APPROVED_AFTER82_V2_MECHANISM_VALID_REPLAY_ONLY', root_approval_sha256=sha(EX / 'ROOT_APPROVED.json'), execution_seal_sha256=EXECUTION)
    save(EX / 'EXECUTION_DRAFT.json', draft)
    members['root_independent_review/ROOT_READY_REVIEW.json'] = REVIEW
    for name in ('ROOT_APPROVED.json', 'EXECUTION_DRAFT.json'):
        members['execution_candidate/' + name] = EX / name
    archive = EX / 'root_deployment_source.tar.gz'
    with tarfile.open(archive, 'x:gz') as bundle:
        for name, path in sorted(members.items()):
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and '..' not in rel.parts and path.is_file() and not path.is_symlink()
            bundle.add(path, arcname=BASE.name + '/' + name, recursive=False)
    archive_sha = sha(archive)
    remote_archive = '/workspace/guardfed_checks/after82_v2_source_root_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.tar.gz'
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
cmd=['ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(target/'execution_candidate/install_once.py'),'--draft-sha256',%r]
result=subprocess.run(cmd,capture_output=True,text=True)
print(json.dumps(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)))
""" % (GUIDE, remote_archive, archive_sha, REMOTE, BASE.name, sha(EX / 'EXECUTION_DRAFT.json'))
    result = subprocess.run(ssh + ['python -B -'], input=code.encode(), capture_output=True, check=True, timeout=180)
    installed = json.loads(result.stdout)
    receipt = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_archive_sha256=archive_sha,
        source_archive_members=len(members), source_archive_remote=remote_archive, science_seal_sha256=SCIENCE,
        execution_seal_sha256=EXECUTION, independent_after82_v2_review_sha256=args.review_sha256, historical_science_lineage_sha256=LINEAGE,
        root_approval_sha256=sha(EX / 'ROOT_APPROVED.json'), external_draft_sha256=sha(EX / 'EXECUTION_DRAFT.json'),
        empty_namespace_prerequisite=pre.stdout.decode().strip(), remote_installation=installed,
        root_remote_stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), new_training=0, new_Full_inference=0, test_inference=False)
    save(EX / 'deployment_receipt.json', receipt)
    assert installed['returncode'] == 0, installed
    print(json.dumps(dict(status='AFTER82_V2_INSTALLED_ACTUAL_STARTUP_OBSERVATION_PENDING', deployment_receipt_sha256=sha(EX / 'deployment_receipt.json'))))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        failure = EX / 'ROOT_DEPLOYMENT_FAILURE.json'
        if EX.exists() and not failure.exists():
            save(failure, dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False, existing_outputs_preserved=True))
        raise
