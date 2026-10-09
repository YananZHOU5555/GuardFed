"""Verify committed source/evidence bytes and the pushed branch independently."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
RECEIPT=TRAIN/'publication_closed_increment14_20261009.json'
PREVIOUS='510e96a55911f01f7f17b7976aa09833087ea924'
BRANCH='codex/revision-evidence-baselines-20260928'
def git(*args,**kwargs):return subprocess.check_output(['git','-c','core.longpaths=true',*args],cwd=REPO,**kwargs)
parser=argparse.ArgumentParser();parser.add_argument('--remote',action='store_true');args=parser.parse_args()
receipt=json.loads(RECEIPT.read_bytes());commit=git('rev-parse','HEAD',text=True).strip()
assert git('rev-parse','HEAD^',text=True).strip()==receipt['previous_commit']==PREVIOUS
assert git('branch','--show-current',text=True).strip()==BRANCH and not git('status','--porcelain',text=True).strip()
mapping=receipt['copied_sha256'];mapping[RECEIPT.relative_to(ROOT).as_posix()]=hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
payload=git('cat-file','--batch',input=''.join(commit+':'+name+'\n' for name in mapping).encode());position=0
for name,digest in mapping.items():
    end=payload.index(b'\n',position);header=payload[position:end].split();assert header[1]==b'blob',name;size=int(header[2])
    assert hashlib.sha256(payload[end+1:end+1+size]).hexdigest()==digest,name;position=end+size+2
assert position==len(payload)
changed=git('diff','--name-only','-z',PREVIOUS,commit).decode().split('\0')[:-1]
assert set(changed)<=set(mapping)|{'.gitattributes'}
proof=dict(status='COMMITTED_BLOB_SHA_PASS_BEFORE_PUSH',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(mapping),changed_paths=len(changed),
    publication_receipt_sha256=mapping[RECEIPT.relative_to(ROOT).as_posix()],
    mechanism_offserver_verified=receipt['mechanism_offserver_verified'],FLGMM_offserver_verified=receipt['FLGMM_offserver_verified'],
    baseline_valid_replays_accepted=receipt['baseline_valid_replays_accepted'],new_V2_queue_started=True,
    original464_queue_stopped=True,precontract_filename_failure_preserved=True,scientific_goal_complete=False,test_started=False,
    frozen_byte_whitespace_preserved=True)
if args.remote:
    actual=git('ls-remote','--heads','origin',BRANCH,text=True).strip().split()
    assert len(actual)==2 and actual[0]==commit and actual[1]=='refs/heads/'+BRANCH
    proof['status']='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
    with (TRAIN/'publication_closed_increment14_verified_20261009.json').open('x',encoding='utf8',newline='\n') as stream:
        json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(proof))
