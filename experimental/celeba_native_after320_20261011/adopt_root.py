"""Adopt one already verified delta; do not rerun training or scientific checks."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json, os, shlex, shutil, subprocess, tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
TAG = 'root_delta_20261010T212808Z'
SRC = BASE / TAG
CANON = ROOT / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
PRIOR = CANON / 'root_delta_20261010T194755Z'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
def save_new(p, value):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')

assert sha(PRIOR / 'ROOT_DELTA_VERIFICATION.json') == '749577f73d151c958f6dc646a1e15714db7c72a8b099a75f22298b58b8c728bb'
assert sha(SRC / 'DELTA_VERIFICATION.json') == '3af125650ef28aaf29712340f245650554e5b09013d5a840bd4bbca958343b60'
assert sha(BASE / 'run_once.py') == '23bbf768f223b1c43dca34a8b03fc2c8e5fe8c75ec7fc519935980d665054dee'
assert sha(BASE / 'COLLECTOR_SOURCE_REVIEW.json') == 'a21a3bb6d03eb5b5b3e9689c0a1428263c4d19a59a953b63b29f1c1ae28d17c4'
assert sha(BASE / 'PARENT_LEDGER.json') == '0674614906f4667f03fae0033aa5f6dfdd87b8275db5d38a7dcd90643dc3fb32'
proof = read(SRC / 'DELTA_VERIFICATION.json')
assert proof['status'] == 'ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS_ROOT_PENDING'
assert proof['total_new_strict_and_offserver'] == 333 and len(proof['new_ids']) == 13
assert proof['new_ids'] == ['minus_F_IID_FedSA_seed91001', 'minus_F_IID_FedSA_seed91002', 'minus_F_IID_FedSA_seed91003', 'minus_F_IID_FedSA_seed91004', 'minus_F_IID_FedSA_seed91005', 'minus_F_IID_FedSA_seed91006', 'minus_F_IID_FedSA_seed91007', 'minus_F_IID_FedSA_seed91008', 'minus_F_IID_FedSA_seed91009', 'minus_F_IID_FedSA_seed91010', 'minus_F_IID_S-DFA_seed91002', 'minus_F_IID_S-DFA_seed91003', 'minus_F_IID_S-DFA_seed91004']
assert sha(CANON / 'verified_ledger.json') == sha(BASE / 'PARENT_LEDGER.json') == proof['previous_ledger_sha256']
old_ledger = read(BASE / 'PARENT_LEDGER.json'); new_ledger = read(SRC / 'verified_ledger.json')
assert new_ledger['entries'][:-1] == old_ledger['entries'] and len(old_ledger['entries']) == 46
assert sha(SRC / 'verified_ledger.json') == proof['ledger_sha256']
receipt_path = SRC / (TAG + '.tar.gz.receipt.json')
receipt = read(receipt_path); off = read(SRC / 'OFFSERVER_VERIFICATION.json')
assert sha(receipt_path) == proof['receipt_sha256'] == new_ledger['entries'][-1]['receipt_sha256']
assert sha(SRC / 'OFFSERVER_VERIFICATION.json') == proof['offserver_proof_sha256']
assert off['pass'] and off['different_host_observed'] and off['accepted_new_ids'] == proof['new_ids']
assert receipt['accepted_new_ids'] == proof['new_ids'] and not receipt['failure_identities']
assert receipt['previous_receipt_sha256'] == old_ledger['entries'][-1]['receipt_sha256']
report = read(SRC / 'inspection/inspection.json'); old_report = read(PRIOR / 'inspection/inspection.json')
assert sha(SRC / 'inspection/inspection.json') == proof['inspection_sha256'] == receipt['inspection_sha256']
assert report['new_count'] == 333 and report['reused_count'] == 100 and not report['invalid']
assert report['source_script_sha256'] == sha(ROOT / 'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py') == '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
for key in ('frozen_source_hashes', 'adapter_hashes', 'manifest_sha256', 'full_inventory_sha256', 'accepted_reused_ids', 'reused_full_restore_chain'):
    assert report[key] == old_report[key], key
old_ids = {row['id'] for row in old_report['records']}
assert len(old_ids) == 420 and [row for row in report['records'] if row['id'] in old_ids] == old_report['records']
assert set(report['accepted_new_ids']) - set(old_report['accepted_new_ids']) == set(proof['new_ids'])
commands = read(SRC / 'REMOTE_COMMANDS.json')
assert commands['canonical_ledger_unchanged'] and len(commands['logs']) == 2
assert all(item['returncode'] == 0 for item in commands['logs'])
assert not commands['preflight']['CPU107_restricted_owners'] and not commands['preflight']['queue']['failed']
archive = Path(proof['archive_local_path'])
assert sha(archive) == proof['archive_sha256'] == off['archive_sha256'] == receipt['archive_sha256']
with tarfile.open(archive, 'r:gz') as tf:
    listed = tf.getmembers(); assert len({m.name for m in listed}) == len(listed)
    inv_bytes = tf.extractfile('backup_inventory.json').read(); inv = json.loads(inv_bytes)
    assert hashlib.sha256(inv_bytes).hexdigest() == receipt['inventory_sha256']
    assert set(m.name for m in listed) == {'backup_inventory.json', *inv['members']}
    for m in listed:
        assert m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
        data = tf.extractfile(m).read()
        if m.name != 'backup_inventory.json':
            expected = inv['members'][m.name]
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], m.name
    assert len(listed) == off['members_verified'] == 134
    assert inv['accepted_new_ids'] == proof['new_ids'] and inv['reused_full_weights_repacked'] == 0
    assert {name.split('/')[1] for name in inv['members'] if name.startswith('runs/')} == set(proof['new_ids'])
    for row in report['records']:
        if row['id'] not in proof['new_ids']: continue
        for source, expected in row['files'].items():
            name = 'jobs/' + row['id'] + '.json' if source == row['job'] else 'runs/' + row['id'] + '/' + PurePosixPath(source).name
            assert inv['members'][name]['sha256'] == expected, name
        assert inv['members']['runs/' + row['id'] + '/model.pt']['sha256'] == row['checkpoint_sha256']
    assert tf.extractfile('sourcefreeze/inspection.json').read() == (SRC / 'inspection/inspection.json').read_bytes()

dest = CANON / TAG
assert not dest.exists(); shutil.copytree(SRC, dest)
remote = '/workspace/guardfed_checks/server_reactivation_20261009'
code = """from pathlib import Path
import hashlib,json,os
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
canonical=Path(REMOTE+'/mechanism_science_backups_20261009/verified_ledger.json')
source=Path(REMOTE+'/native_delta_after320_20261011/'+TAG+'/verified_ledger.json')
assert sha(canonical)==BEFORE and sha(source)==AFTER
temp=canonical.with_name('verified_ledger.root333.tmp');assert not temp.exists()
temp.write_bytes(source.read_bytes());assert sha(temp)==AFTER
os.replace(temp,canonical)
print(json.dumps({'status':'CANONICAL_LEDGER_PROMOTED','previous_sha256':BEFORE,'sha256':sha(canonical)}))
"""
bound = '\n'.join(k + '=' + repr(v) for k,v in dict(REMOTE=remote,TAG=TAG,BEFORE=proof['previous_ledger_sha256'],AFTER=proof['ledger_sha256']).items()) + '\n' + code
run = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -c ' + shlex.quote(bound)], capture_output=True)
(dest / 'ROOT_REMOTE_LEDGER_PROMOTION.stdout.json').write_bytes(run.stdout)
(dest / 'ROOT_REMOTE_LEDGER_PROMOTION.stderr.txt').write_bytes(run.stderr)
assert run.returncode == 0, 'Preserve promotion evidence; do not blindly repeat'
assert json.loads(run.stdout)['sha256'] == proof['ledger_sha256']
assert sha(CANON / 'verified_ledger.json') == proof['previous_ledger_sha256']
temp = CANON / 'verified_ledger.root333.tmp'; assert not temp.exists()
temp.write_bytes((dest / 'verified_ledger.json').read_bytes()); os.replace(temp, CANON / 'verified_ledger.json')
proof.update(status='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_adopted=True,root_checked_archive_members=134,old420_records_exact=True,canonical_local_and_remote_ledger_promoted=True,source_review_path=(BASE/'COLLECTOR_SOURCE_REVIEW.json').relative_to(ROOT).as_posix(),source_review_sha256=sha(BASE/'COLLECTOR_SOURCE_REVIEW.json'),prior_root_sha256=sha(PRIOR/'ROOT_DELTA_VERIFICATION.json'),adopter_path=Path(__file__).relative_to(ROOT).as_posix(),adopter_sha256=sha(__file__))
save_new(dest / 'ROOT_DELTA_VERIFICATION.json',proof)
print(json.dumps({'status':'ROOT_ADOPTED','accepted_native':333,'new':13,'root_path':(dest/'ROOT_DELTA_VERIFICATION.json').relative_to(ROOT).as_posix(),'root_sha256':sha(dest/'ROOT_DELTA_VERIFICATION.json')}))
