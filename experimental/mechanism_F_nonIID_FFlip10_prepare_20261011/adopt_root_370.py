"""Adopt one already verified delta; do not rerun training or scientific checks."""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, os, re, shlex, shutil, subprocess, tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--actual-inputs-sha256',required=True)
args=parser.parse_args()
actual_path=BASE/'ACTUAL_INPUTS.json'
assert hashlib.sha256(actual_path.read_bytes()).hexdigest()==args.actual_inputs_sha256
ACTUAL=json.loads(actual_path.read_bytes())
assert ACTUAL['status']=='ROOT_REVIEWED_ACTUAL_NATIVE_DELTA_AFTER360_NONIID_FFLIP10_READY_TO_ADOPT'
assert ACTUAL['root_execution_authorized'] is True
TAG=ACTUAL['tag'];assert isinstance(TAG,str) and re.fullmatch(r'root_delta_\d{8}T\d{6}Z',TAG)
TOTAL=ACTUAL['accepted_total'];NEW_IDS=ACTUAL['new_ids'];MEMBERS=ACTUAL['archive_members']
assert type(TOTAL) is int and type(MEMBERS) is int and MEMBERS>0
assert isinstance(NEW_IDS,list) and len(NEW_IDS)==len(set(NEW_IDS))==TOTAL-360 and TOTAL==370
REQUIRED_SCENE=[f'minus_F_non-IID_F Flip_seed{s}' for s in range(91001,91011)]
assert NEW_IDS==REQUIRED_SCENE and len(NEW_IDS)==10
for key in ('delta_sha256','source_review_sha256','archive_sha256','inspection_sha256','offserver_proof_sha256','receipt_sha256','ledger_sha256','source_seal_sha256'):
    assert isinstance(ACTUAL[key],str) and re.fullmatch('[0-9a-f]{64}',ACTUAL[key])
assert hashlib.sha256((BASE/'FILES_SHA256.json').read_bytes()).hexdigest()==ACTUAL['source_seal_sha256']
for name,pin in json.loads((BASE/'FILES_SHA256.json').read_bytes())['files'].items():
    file=(BASE/name).resolve();assert file.is_relative_to(BASE.resolve())
    assert file.stat().st_size==pin['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==pin['sha256']
SRC = BASE / TAG
CANON = ROOT / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
PRIOR = CANON / 'root_delta_20261011T002530Z'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
def save_new(p, value):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')

assert sha(PRIOR / 'ROOT_DELTA_VERIFICATION.json') == '396e40654b6dfc75029d4a7a42dbfafe4caaf111a872ae2ece5045f6d1203418'
assert sha(SRC / 'DELTA_VERIFICATION.json') == ACTUAL['delta_sha256']
assert sha(BASE / 'run_native_once.py') == 'c5dcabc47c5dd2c930413a296c26bf27aae01286833c3001390f66935b36ceff'
assert sha(BASE / 'COLLECTOR_SOURCE_REVIEW.json') == ACTUAL['source_review_sha256']
assert sha(BASE / 'PARENT_LEDGER.json') == '67c32aced70438bf7bf614c5862792f1887ddeb4fbede5e4da06cc157fd992ef'
proof = read(SRC / 'DELTA_VERIFICATION.json')
assert proof['status'] == 'ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS_ROOT_PENDING'
assert proof['tag']==TAG and proof['total_new_strict_and_offserver']==TOTAL and len(proof['new_ids'])==len(NEW_IDS)
for key in ('archive_sha256','inspection_sha256','offserver_proof_sha256','receipt_sha256','ledger_sha256'):assert proof[key]==ACTUAL[key]
assert proof['new_ids'] == NEW_IDS
assert sha(CANON / 'verified_ledger.json') == sha(BASE / 'PARENT_LEDGER.json') == proof['previous_ledger_sha256']
old_ledger = read(BASE / 'PARENT_LEDGER.json'); new_ledger = read(SRC / 'verified_ledger.json')
assert new_ledger['entries'][:-1] == old_ledger['entries'] and len(old_ledger['entries']) == 50
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
assert report['new_count'] == ACTUAL['observed_native_count'] >= TOTAL and report['reused_count'] == 100 and not report['invalid']
assert len(report['records'])==ACTUAL['observed_native_count']+100 and set(REQUIRED_SCENE)<=set(report['accepted_new_ids'])
assert report['source_script_sha256'] == sha(ROOT / 'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py') == '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
for key in ('frozen_source_hashes', 'adapter_hashes', 'manifest_sha256', 'full_inventory_sha256', 'accepted_reused_ids', 'reused_full_restore_chain'):
    assert report[key] == old_report[key], key
old_ids = {row['id'] for row in old_report['records']}
assert len(old_ids) == 460 and [row for row in report['records'] if row['id'] in old_ids] == old_report['records']
assert set(proof['new_ids']) <= set(report['accepted_new_ids']) - set(old_report['accepted_new_ids'])
assert sorted(set(report['accepted_new_ids']) - set(old_report['accepted_new_ids']) - set(proof['new_ids'])) == proof['observed_not_backed_up_ids'] == ACTUAL['pending_closed_ids']
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
    assert len(listed) == off['members_verified'] == MEMBERS  # Actual inventory cardinality, never a forecast.
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
source=Path(REMOTE+'/native_delta_after360_nonIID_FFlip10_20261011/'+TAG+'/verified_ledger.json')
assert sha(canonical)==BEFORE and sha(source)==AFTER
temp=canonical.with_name('verified_ledger.after360_nonIID_FFlip10.root.tmp');assert not temp.exists()
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
temp = CANON / 'verified_ledger.after360_nonIID_FFlip10.root.tmp'; assert not temp.exists()
temp.write_bytes((dest / 'verified_ledger.json').read_bytes()); os.replace(temp, CANON / 'verified_ledger.json')
proof.update(status='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_adopted=True,root_checked_archive_members=MEMBERS,old460_records_exact=True,canonical_local_and_remote_ledger_promoted=True,source_review_path=(BASE/'COLLECTOR_SOURCE_REVIEW.json').relative_to(ROOT).as_posix(),source_review_sha256=sha(BASE/'COLLECTOR_SOURCE_REVIEW.json'),prior_root_sha256=sha(PRIOR/'ROOT_DELTA_VERIFICATION.json'),adopter_path=Path(__file__).relative_to(ROOT).as_posix(),adopter_sha256=sha(__file__))
save_new(dest / 'ROOT_DELTA_VERIFICATION.json',proof)
print(json.dumps({'status':'ROOT_ADOPTED','accepted_native':TOTAL,'new':len(NEW_IDS),'root_path':(dest/'ROOT_DELTA_VERIFICATION.json').relative_to(ROOT).as_posix(),'root_sha256':sha(dest/'ROOT_DELTA_VERIFICATION.json')}))
