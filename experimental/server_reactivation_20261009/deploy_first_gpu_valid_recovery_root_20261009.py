"""Verify and deploy one reviewed GPU valid replay; never launch or extend scope."""
from pathlib import Path, PurePosixPath
import argparse
import datetime
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / 'tmp/celeba_valid_gpu_recovery_implementation_20261009'
PROPOSAL = ROOT / 'tmp/celeba_valid_recovery_prepared_20261009'
EXEC = ROOT / 'tmp/celeba_valid_gpu_recovery_execution_20261009'
REMOTE = '/workspace/guardfed_checks'
GUIDE = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
OLD424 = ROOT / 'tmp/celeba_final_valid_replay_20261009/v4/remaining872_execution_20261009/cumulative_424_accepted.json'
FIRST = 'FairGuard_IID_FedSA_seed91003'
SHA = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_bytes())


def save(path, value):
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


parser = argparse.ArgumentParser()
parser.add_argument('--package-sha256', required=True)
args = parser.parse_args()
assert SHA((IMPL / 'PACKAGE_SHA256.json').read_bytes()) == args.package_sha256
assert not EXEC.exists(), 'Never overwrite a prior deployment/attempt'
payload = {}
for folder, expected in [(IMPL, args.package_sha256), (PROPOSAL, '586d4443ae6f074e686f872400dd90b91356f0e504d29faa8c8cb0ff3ea8e26a')]:
    seal = folder / 'PACKAGE_SHA256.json'
    assert SHA(seal.read_bytes()) == expected
    rows = read(seal)['members']
    for name, row in rows.items():
        p = folder / name
        rel = PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts and p.resolve().is_relative_to(folder.resolve()) and not p.is_symlink()
        data = p.read_bytes()
        assert SHA(data) == row['sha256'] and len(data) == row['bytes'], name
        payload[folder.name + '/' + name] = data
    payload[folder.name + '/PACKAGE_SHA256.json'] = seal.read_bytes()
assert SHA(OLD424.read_bytes()) == '75ec99bc1eb9fb2c5e20ead68aad7af867d11cb51f2cab991d01712568985fdd'
assert SHA((IMPL / 'gpu_replay_body.py').read_bytes()) == '2aafb0d08b2a0c84bbcfc224638dc216329f72d450facf37fd2e50d1cb0f0ad5'
check = subprocess.run([sys.executable, str(IMPL / 'selfcheck.py')], capture_output=True, check=True)
checked = json.loads(check.stdout)
assert checked['status'] == 'PASS_LOCAL_NO_CNN_NO_COHORT_ACCEPTANCE' and checked['new_inference'] == 0 and checked['Torch_imported'] is False
assert checked['native_one_sample_mismatch_accepted'] is False
EXEC.mkdir()
(EXEC / 'ROOT_LOCAL_CHECK.log').write_bytes(check.stdout + check.stderr)
approval = read(IMPL / 'APPROVAL_TEMPLATE.json')
approval.update(status='ROOT_APPROVED_GPU_VALID_RECOVERY_V1', implementation_package_sha256=args.package_sha256,
                approved_ids=[FIRST], execute_new465=True,
                gpu_uuid='GPU-da357477-30a7-fddc-344b-a20513b9a2d0')
approval.update(approved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                review_scope='Only first previously unexecuted valid replay, then strict/offserver review. No remaining464/import11 authorization.',
                authority='Existing user authority to complete rebuttal experiments; operational recovery preserving frozen science.',
                independent_source_review='Root reviewed exact v3 science except status/device/added guards; known duplicate interop setter absent; no numerical tolerance change.')
review = EXEC / 'ROOT_REVIEW_FIRST1.json'
save(review, approval)
review_sha = SHA(review.read_bytes())
payload['celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_FIRST1.json'] = review.read_bytes()
specs = {name: {'sha256': SHA(data), 'bytes': len(data)} for name, data in sorted(payload.items())}
manifest = {'members': specs, 'first_id': FIRST, 'package_sha256': args.package_sha256,
            'review_sha256': review_sha, 'no_launch': True}
save(EXEC / 'DEPLOYMENT_PAYLOAD.json', manifest)
archive = EXEC / 'deployment.tar.gz'
with tarfile.open(archive, 'w:gz') as bundle:
    for name, data in sorted(payload.items()):
        item = tarfile.TarInfo(name); item.size = len(data); item.mode = 0o644
        bundle.addfile(item, io.BytesIO(data))
archive_sha = SHA(archive.read_bytes())
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350', 'root@89.22.197.55']
code = """from pathlib import Path
import hashlib,json,subprocess
base=Path('/workspace/guardfed_checks')
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()==GUIDE
for name in NAMES: assert not (base/name).exists(), 'Fresh target required: '+name
status=subprocess.run(['supervisorctl','status','sglang'],capture_output=True,text=True)
assert 'STOPPED' in status.stdout, status.stdout
print(json.dumps({'status':'FRESH_DEPLOYMENT_TARGETS_AND_SGLANG_STOPPED','sglang':status.stdout.strip()}))
"""
pre = 'GUIDE=' + repr(GUIDE) + '\nNAMES=' + repr([IMPL.name, PROPOSAL.name, EXEC.name]) + '\n' + code
result = subprocess.run(ssh + ['python -c ' + shlex.quote(pre)], capture_output=True, timeout=45, check=True)
(EXEC / 'REMOTE_PREDEPLOY.json').write_bytes(result.stdout)
remote_tar = REMOTE + '/gpu_valid_recovery_first1_' + archive_sha + '.tar.gz'
subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-P', '60350', str(archive), 'root@89.22.197.55:' + remote_tar], check=True, timeout=120)
install = """from pathlib import Path,PurePosixPath
import hashlib,json,tarfile
base=Path('/workspace/guardfed_checks'); archive=Path(ARCHIVE)
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()==GUIDE
assert hashlib.sha256(archive.read_bytes()).hexdigest()==ARCHIVE_SHA
for name in NAMES: assert not (base/name).exists()
with tarfile.open(archive) as bundle:
    assert len(bundle.getnames())==len(set(bundle.getnames()))==len(SPECS)
    assert set(bundle.getnames())==set(SPECS)
    for item in bundle.getmembers():
        name=PurePosixPath(item.name)
        assert item.isfile() and not name.is_absolute() and '..' not in name.parts and name.parts[0] in NAMES
        data=bundle.extractfile(item).read(); row=SPECS[item.name]
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    for name in NAMES: (base/name).mkdir()
    for item in bundle.getmembers():
        target=base/item.name; target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as out: out.write(bundle.extractfile(item).read())
    (base/'celeba_valid_gpu_recovery_execution_20261009/attempt1').mkdir()
for name,row in SPECS.items(): assert hashlib.sha256((base/name).read_bytes()).hexdigest()==row['sha256']
print(json.dumps({'status':'EXACT_SEALED_DEPLOYMENT_COMPLETE_NOT_LAUNCHED','archive_sha256':ARCHIVE_SHA,'members_verified':len(SPECS),'first_id':FIRST}))
"""
bound = '\n'.join(k + '=' + repr(v) for k, v in dict(ARCHIVE=remote_tar, ARCHIVE_SHA=archive_sha, GUIDE=GUIDE,
                  NAMES=[IMPL.name, PROPOSAL.name, EXEC.name], SPECS=specs, FIRST=FIRST).items()) + '\n' + install
result = subprocess.run(ssh + ['python -c ' + shlex.quote(bound)], capture_output=True, timeout=60, check=True)
(EXEC / 'REMOTE_DEPLOYMENT.json').write_bytes(result.stdout)
assert SHA(OLD424.read_bytes()) == approval['accepted424_collector_sha256']
proof = {'status': 'ROOT_FIRST1_SEALED_REVIEW_AND_DEPLOYMENT_PASS_NOT_LAUNCHED',
         'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'package_sha256': args.package_sha256,
         'review_sha256': review_sha, 'archive_sha256': archive_sha, 'members': len(specs),
         'approved_ids': [FIRST], 'remaining464_authorized': False, 'imports11_authorized': False,
         'old424_unchanged': True, 'test': False, 'training': False}
save(EXEC / 'ROOT_DEPLOYMENT_VERIFICATION.json', proof)
print(json.dumps(proof))
