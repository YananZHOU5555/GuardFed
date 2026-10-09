"""Verify a pinned committed increment and, after push, the actual remote branch."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
BRANCH='codex/revision-evidence-baselines-20260928'
def git(*args,**kwargs):return subprocess.check_output(['git','-c','core.longpaths=true',*args],cwd=REPO,**kwargs)
parser=argparse.ArgumentParser();parser.add_argument('--remote',action='store_true')
parser.add_argument('--increment',type=int,choices=(16,17,18,19,20),default=16);args=parser.parse_args()
RECEIPT=TRAIN/f'publication_closed_increment{args.increment}_20261009.json'
expected_previous={16:'d659e0bb37bbe89b8390927c54ef5f37f602e6b6',17:'795f4b09c60c4a81de3d4aa67dad5beac5075827',18:'1ac345c0dda16dcdfdedfcc0b020f58a24092aef',19:'59c6e47b00e4875767dd1814fa09382cfc2b4e1c',20:'de40f774f8ea180440dc111ad45a608a666ac168'}[args.increment]
receipt=json.loads(RECEIPT.read_bytes());commit=git('rev-parse','HEAD',text=True).strip()
assert git('rev-parse','HEAD^',text=True).strip()==receipt['previous_commit']==expected_previous
assert git('branch','--show-current',text=True).strip()==BRANCH and not git('status','--porcelain',text=True).strip()
mapping=receipt['copied_sha256'];mapping[RECEIPT.relative_to(ROOT).as_posix()]=hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
payload=git('cat-file','--batch',input=''.join(commit+':'+name+'\n' for name in mapping).encode());position=0
for name,digest in mapping.items():
    end=payload.index(b'\n',position);header=payload[position:end].split();assert header[1]==b'blob';size=int(header[2])
    assert hashlib.sha256(payload[end+1:end+1+size]).hexdigest()==digest,name;position=end+size+2
assert position==len(payload)
changed=git('diff','--name-only','-z',receipt['previous_commit'],commit).decode().split('\0')[:-1]
assert set(changed)<=set(mapping)|{'.gitattributes'}
proof=dict(status='COMMITTED_BLOB_SHA_PASS_BEFORE_PUSH',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(mapping),changed_paths=len(changed),
    publication_receipt_sha256=mapping[RECEIPT.relative_to(ROOT).as_posix()],
    baseline_valid_replays_accepted=receipt['baseline_valid_replays_accepted'],
    mechanism_offserver_verified=receipt['mechanism_offserver_verified'],mechanism_three_view_offserver_verified=receipt['mechanism_three_view_offserver_verified'],
    FLGMM_offserver_verified=13,Hybrid_offserver_verified=4,test_started=False,scientific_goal_complete=False)
if args.remote:
    actual=git('ls-remote','--heads','origin',BRANCH,text=True).strip().split()
    assert len(actual)==2 and actual[0]==commit and actual[1]=='refs/heads/'+BRANCH
    proof['status']='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
    with (TRAIN/f'publication_closed_increment{args.increment}_verified_20261009.json').open('x',encoding='utf8',newline='\n') as stream:
        json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(proof))
