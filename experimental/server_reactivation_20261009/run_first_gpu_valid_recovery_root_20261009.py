"""Execute exactly one explicitly chosen operation for the reviewed first ID."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_valid_gpu_recovery_execution_20261009'
REMOTE = '/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009'
PKG = '/workspace/guardfed_checks/celeba_valid_gpu_recovery_implementation_20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
parser = argparse.ArgumentParser()
parser.add_argument('operation', choices=['run', 'accept', 'backup'])
parser.add_argument('--strict-sha256')
args = parser.parse_args()
proof = read(BASE / 'ROOT_DEPLOYMENT_VERIFICATION.json')
assert proof['status'] == 'ROOT_FIRST1_SEALED_REVIEW_AND_DEPLOYMENT_PASS_NOT_LAUNCHED'
assert sha(BASE / 'ROOT_REVIEW_FIRST1.json') == proof['review_sha256']
approved = read(BASE / 'ROOT_REVIEW_FIRST1.json')
assert approved['approved_ids'] == ['FairGuard_IID_FedSA_seed91003']
assert not approved['import_cpu_partial10'] and not approved['import_gpu_diagnostic1']
out = REMOTE + '/attempt1/chunk_000'
command = ['nice', '-n', '10', 'ionice', '-c', '3', '/workspace/guardfed_envs/celeba-cu128-20261009/bin/python', PKG + '/recovery.py',
           {'run': 'run-chunk', 'accept': 'accept', 'backup': 'backup'}[args.operation],
           '--review', REMOTE + '/ROOT_REVIEW_FIRST1.json', '--review-sha256', proof['review_sha256'],
           '--package-sha256', proof['package_sha256']]
if args.operation == 'run':
    command += ['--ids', approved['approved_ids'][0], '--output', out]
elif args.operation == 'accept':
    assert read(BASE / 'ROOT_run_EXIT.json')['returncode'] == 0
    command += ['--batch', out + '/batch', '--output', out + '/strict_acceptance.json']
else:
    assert read(BASE / 'ROOT_accept_EXIT.json')['returncode'] == 0
    assert args.strict_sha256 and len(args.strict_sha256) == 64
    command += ['--stage', out, '--chunk-index', '0', '--strict-sha256', args.strict_sha256]
log = BASE / ('ROOT_' + args.operation + '.log')
start = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': command,
         'review_sha256': proof['review_sha256'], 'package_sha256': proof['package_sha256']}
with (BASE / ('ROOT_' + args.operation + '_START.json')).open('x', encoding='utf-8') as stream:
    json.dump(start, stream, indent=2); stream.write('\n')
with log.open('xb') as stream:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350',
                             'root@89.22.197.55', shlex.join(command)], stdout=stream, stderr=subprocess.STDOUT)
receipt = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'returncode': result.returncode,
           'log_sha256': sha(log), 'accepted_or_registered_by_exit_alone': False}
with (BASE / ('ROOT_' + args.operation + '_EXIT.json')).open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
print(json.dumps(receipt))
raise SystemExit(result.returncode)
