"""Verify increment40 index receipt against committed bytes and optionally the remote branch. No publishing."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / 'tmp/revision-publish-20260928'
TRAIN = Path('docs/server_deployment_20260923/training_20260923')
PARENT = '3601c9dfca63dc1c6203aceb2c7fc066faa630fa'
BRANCH = 'codex/revision-evidence-baselines-20260928'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args, input=None): return subprocess.check_output(['git', '-c', 'core.longpaths=true', *args], cwd=REPO, input=input)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True); parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--remote', action='store_true')
    args = parser.parse_args(); assert not sys.flags.optimize
    assert sha(args.receipt) == args.receipt_sha256
    receipt = json.loads(args.receipt.read_bytes()); assert receipt['previous_commit'] == PARENT
    assert [receipt[k] for k in ('baseline_valid_replays_accepted', 'mechanism_offserver_verified', 'mechanism_three_view_offserver_verified', 'FLGMM_fullcoverage_new_offserver_verified', 'Hybrid_offserver_verified')] == [900, 170, 170, 16, 22]
    assert receipt['test_started'] is False and receipt['scientific_goal_complete'] is False
    assert receipt['C70_table_included'] is True and receipt['C70_in_full_rebuttal'] is False
    assert receipt['C60_full_reply_republished'] is False and receipt['third_party_source_bodies_redistributed'] is False
    assert receipt['duplicated_old_models']==0 and receipt['manuscript_applied'] is False
    commit = git('rev-parse', 'HEAD').decode().strip()
    assert git('rev-parse', 'HEAD^').decode().strip() == PARENT
    assert git('branch', '--show-current').decode().strip() == BRANCH
    assert not git('status', '--porcelain', '--untracked-files=all').strip()
    mapping = receipt['copied_sha256'].copy(); mapping[(TRAIN / 'publication_closed_increment40_20261010.json').as_posix()] = sha(args.receipt)
    index = json.loads((args.receipt.parent / 'INDEX_VERIFICATION.json').read_bytes())
    assert index['status'] == 'INDEX_BLOB_SHA_PASS_NOT_COMMITTED_OR_PUSHED' and index['receipt_sha256'] == sha(args.receipt)
    assert {k: v for k, v in index['index_sha256'].items() if k != '.gitattributes'} == mapping
    mapping['.gitattributes'] = index['index_sha256']['.gitattributes']
    payload = git('cat-file', '--batch', input=''.join(commit + ':' + n + '\n' for n in mapping).encode()); offset = 0
    for name, digest in mapping.items():
        stop = payload.index(b'\n', offset); header = payload[offset:stop].split(); assert len(header) == 3 and header[1] == b'blob', name
        size = int(header[2]); start = stop + 1; data = payload[start:start + size]
        assert len(data) == size and hashlib.sha256(data).hexdigest() == digest, name
        assert payload[start + size:start + size + 1] == b'\n'; offset = start + size + 1
    assert offset == len(payload)
    changed = set(filter(None, git('diff', '--name-only', '-z', PARENT, commit).decode().split('\0'))); assert changed <= set(mapping)
    result = dict(status='COMMITTED_BLOB_SHA_PASS_BEFORE_PUSH', verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), commit=commit, branch=BRANCH,
        committed_blobs_sha256_verified=len(mapping), changed_paths=len(changed), publication_receipt_sha256=sha(args.receipt),
        baseline_valid_replays_accepted=900, mechanism_offserver_verified=170, mechanism_three_view_offserver_verified=170,
        FLGMM_offserver_verified=32, FLGMM_fullcoverage_new_offserver_verified=16, Hybrid_offserver_verified=22, test_started=False, scientific_goal_complete=False)
    if args.remote:
        remote = git('ls-remote', '--heads', 'origin', BRANCH).decode().split()
        assert remote == [commit, 'refs/heads/' + BRANCH]; result['status'] = 'COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
    assert args.output.resolve().is_relative_to(HERE) and not args.output.exists()
    with args.output.open('x', encoding='utf-8', newline='\n') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))

if __name__ == '__main__': main()
