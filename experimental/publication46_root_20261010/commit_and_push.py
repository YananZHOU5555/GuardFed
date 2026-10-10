"""Publish only the original transport's SHA-verified compact Git46 staging receipt."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys, traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PACKAGE=ROOT/'tmp/publication_increment46_20261010';sys.path.insert(0,str(PACKAGE))
from publish_increment46 import CHECKOUT,BRANCH,ORIGIN,PARENT,git,storage
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name,value):
    with (HERE/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')

def run(a):
    assert H(a.receipt)==a.sha256
    d=json.loads(a.receipt.read_bytes())
    assert d['status']=='STAGED_INDEX_BYTES_PASS_NOT_COMMITTED_OR_PUSHED'
    assert d['accepted']=={'native':212,'three_view':212} and d['blobs']
    assert not (HERE/'COMMIT_RECEIPT.json').exists()
    storage(20*1024**2)
    assert git('rev-parse','HEAD').decode().strip()==PARENT
    assert git('branch','--show-current').decode().strip()==BRANCH
    assert git('remote','get-url','origin').decode().strip()==ORIGIN
    assert git('ls-remote','--exit-code','origin','refs/heads/'+BRANCH).decode().split()[0]==PARENT
    changed=set(git('diff','--cached','--name-only','-z').decode().split('\0'))-{''}
    assert changed and changed<={r['path'] for r in d['blobs']}
    assert not git('diff','--name-only').strip()
    message=HERE/'COMMIT_MESSAGE.txt'
    with message.open('x',encoding='utf8') as f:
        f.write('Record mechanism212 and paired A-ablation validation table\n\n'
            'Join twelve new saved three-view results to the accepted native checkpoint restore chain. '
            'Add the complete ten-seed IID Benign Full/minus-A comparison, retaining raw/native/shared '
            'views, paired differences, partial F Flip records and negative outcomes.\n\n'
            'Prepare the two gradient methods for future fixed-recipe coverage without selecting '
            'a recipe or launching the prepared jobs. Large artifacts remain on F/server. '
            'Remaining methods, mechanisms, final evaluation and manuscript integration are incomplete.\n')
    (HERE/'commit.stdout').write_bytes(git('commit','-F',str(message)))
    commit=git('rev-parse','HEAD').decode().strip();assert commit!=PARENT
    save('COMMIT_RECEIPT.json',dict(commit=commit,parent=PARENT,branch=BRANCH,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage_receipt_sha256=H(a.receipt),
        staged_blobs=len(d['blobs']),changed_paths=len(changed),accepted=d['accepted'],test=False,goal_complete=False))
    pushed=subprocess.run(['git','-c','core.longpaths=true','-c','safe.directory='+CHECKOUT.as_posix(),
        '-C',str(CHECKOUT),'push','origin',BRANCH],capture_output=True,timeout=180)
    (HERE/'push.stdout').write_bytes(pushed.stdout);(HERE/'push.stderr').write_bytes(pushed.stderr)
    save('PUSH_COMMAND_EXIT.json',dict(commit=commit,returncode=pushed.returncode,automatic_retry=False))
    pushed.check_returncode()
    verified=subprocess.run([sys.executable,'-B',str(PACKAGE/'verify_increment46.py'),
        '--receipt',str(a.receipt),'--receipt-sha256',H(a.receipt),'--commit',commit,'--remote'],
        capture_output=True,timeout=600)
    (HERE/'verify.stdout').write_bytes(verified.stdout);(HERE/'verify.stderr').write_bytes(verified.stderr)
    save('VERIFY_COMMAND_EXIT.json',dict(commit=commit,returncode=verified.returncode,automatic_retry=False))
    verified.check_returncode()
    proof=json.loads(verified.stdout)
    proof.update(root_review_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),external_checkout=CHECKOUT.as_posix())
    save('REMOTE_VERIFICATION.json',proof);print(json.dumps(proof))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--sha256',required=True)
    try:run(p.parse_args())
    except BaseException as exc:
        save('FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False));raise
