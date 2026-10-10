"""Copy actual increment36 commit/remote verification unchanged into canonical evidence."""
from pathlib import Path
import argparse, hashlib, json, sys

if sys.flags.optimize:raise RuntimeError('Optimized Python is forbidden for evidence guards')

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
SOURCE=ROOT/'tmp/publication_increment36_prepared_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--proof-sha256',required=True)
parser.add_argument('--commit',required=True)
args=parser.parse_args()
source=SOURCE/'publication_closed_increment36_verified_20261010.json'
assert sha(source)==args.proof_sha256
proof=json.loads(source.read_bytes())
assert proof['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and proof['commit']==args.commit
assert proof['branch']=='codex/revision-evidence-baselines-20260928'
assert proof['committed_blobs_sha256_verified']>=proof['changed_paths']>0
assert [proof[k] for k in ('baseline_valid_replays_accepted','mechanism_offserver_verified','mechanism_three_view_offserver_verified','FLGMM_fullcoverage_new_offserver_verified','Hybrid_offserver_verified')]==[900,156,156,11,19]
assert proof['test_started'] is False and proof['scientific_goal_complete'] is False
receipt=SOURCE/'actual_stage_once/publication_closed_increment36_20261010.json'
assert sha(receipt)==proof['publication_receipt_sha256']
assert json.loads(receipt.read_bytes())['previous_commit']=='fbe6027794ce045bdf79254b995c8d2b1de2fb56'
rows=[]
for file in (receipt,source):
    target=TRAIN/file.name
    with target.open('xb') as stream:stream.write(file.read_bytes())
    assert sha(target)==sha(file)
    rows.append(dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target)))
print(json.dumps(dict(status='ACTUAL_REMOTE_VERIFICATION_CANONICAL_COPY_PASS',files=rows)))
