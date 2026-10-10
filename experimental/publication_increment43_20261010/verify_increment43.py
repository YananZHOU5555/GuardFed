"""Read committed local blobs and optional remote branch identity; never fetch."""
from pathlib import Path
import argparse, hashlib, json, subprocess
from publish_increment43 import CHECKOUT, BRANCH, ORIGIN, PARENT, git, storage, relative


def verify(receipt, receipt_sha, commit, remote):
    storage()
    b = receipt.read_bytes(); assert hashlib.sha256(b).hexdigest() == receipt_sha
    d = json.loads(b)
    assert d['status'] == 'STAGED_INDEX_BYTES_PASS_NOT_COMMITTED_OR_PUSHED' and d['parent'] == PARENT and d['branch'] == BRANCH
    assert git('remote', 'get-url', 'origin').decode().strip() == ORIGIN
    assert git('show', '-s', '--format=%P', commit).decode().strip() == PARENT
    changed = set(git('diff-tree', '--no-commit-id', '--name-only', '-r', '-z', commit).decode().split('\0')) - {''}
    assert changed <= {x['path'] for x in d['blobs']}, 'Commit includes unreviewed paths'
    for x in d['blobs']:
        relative(x['path'])
        content = git('show', commit + ':' + x['path'])
        assert hashlib.sha256(content).hexdigest() == x['sha256'] and len(content) == x['bytes']
    if remote:
        assert git('ls-remote', '--exit-code', 'origin', 'refs/heads/' + BRANCH).decode().split()[0] == commit
    return dict(status='COMMITTED_BLOB_BYTES_PASS' + ('_AND_REMOTE_BRANCH_PASS' if remote else ''), commit=commit,
        branch=BRANCH, parent=PARENT, blobs_verified=len(d['blobs']), changed_paths=len(changed),
        receipt_sha256=receipt_sha, accepted=d['accepted'], test_started=False, goal_complete=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True); p.add_argument('--receipt-sha256', required=True)
    p.add_argument('--commit', required=True); p.add_argument('--remote', action='store_true')
    a = p.parse_args(); assert len(a.commit) == 40 and all(c in '0123456789abcdef' for c in a.commit)
    print(json.dumps(verify(a.receipt, a.receipt_sha256, a.commit, a.remote), indent=2))
