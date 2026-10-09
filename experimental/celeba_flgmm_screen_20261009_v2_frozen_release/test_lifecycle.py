"""CPU unit boundaries only: no real images, GPU training, or result acceptance."""
import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import run_screen
from screen_common import HERE, authorized, checked_repo_path, digest, local_identity, read, repo_identity


def rejects(call):
    try:
        call()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Expected rejection')


def main():
    protocol, manifest = local_identity(require_frozen=False)
    assert protocol['status'] == 'PREPARED_NOT_FROZEN'
    rejects(local_identity)
    rejects(authorized)
    assert len(manifest['jobs']) == 32
    old = read(HERE / 'evidence/original_protocol.json')
    for key in ['base_config', 'source_hashes', 'candidates', 'distributions', 'attacks', 'selection_rule']:
        assert protocol[key] == old[key], key
    source = (HERE / 'evidence/original_verify_rank.py').read_text(encoding='utf8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'score')
    assert (HERE / 'frozen_score.py').read_text() == ast.get_source_segment(source, node) + '\n'
    assert run_screen.score(dict(accuracy=.8, aeod=0., aspd=0.)) == .8
    assert run_screen.score(dict(accuracy=.8, aeod=.1, aspd=.2)) < .8
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        fixture = root / 'content'; fixture.write_bytes(b'fixture only')
        assert repo_identity(root, {'source_hashes': {'content': digest(fixture)}})
        rejects(lambda: repo_identity(root, {'source_hashes': {'content': '0' * 64}}))
        rejects(lambda: repo_identity(root, {'source_hashes': {'../escape': '0' * 64}}))
        rejects(lambda: repo_identity(root, {'source_hashes': {str(fixture.resolve()): '0' * 64}}))
        rule = {'content': {'resolved_target': str(fixture.resolve()), 'sha256': digest(fixture)}}
        assert checked_repo_path(root, 'content', digest(fixture), rule) == fixture.resolve()
        rejects(lambda: checked_repo_path(root, 'content', '0' * 64, rule))
        wrong_target = {'content': dict(rule['content'], resolved_target=str(root / 'other'))}
        rejects(lambda: checked_repo_path(root, 'content', digest(fixture), wrong_target))
        item = {'id': 'fixture_only'}
        with patch.object(run_screen, 'HERE', root):
            assert run_screen.inspect_output(item) == 'pending'
            (root / 'runs' / item['id']).mkdir(parents=True)
            with patch.object(run_screen, 'accepted', return_value=None):
                rejects(lambda: run_screen.inspect_output(item))
            with patch.object(run_screen, 'accepted', return_value={'fixture': True}):
                assert run_screen.inspect_output(item) == 'accepted'
            with patch.object(run_screen, 'accepted', side_effect=ValueError('tampered/failure')):
                rejects(lambda: run_screen.inspect_output(item))
    # Candidate selection checks the actual function on complete unit fixture rows.
    fake = dict(jobs=[dict(id=str(i), tuning_candidate=f'c{i//4}', distribution=str(i%2), attack=str(i%4))
                     for i in range(32)])
    result = dict(metrics=dict(accuracy=.5, aeod=.1, aspd=.1))
    with patch.object(run_screen, 'accepted', return_value=result), patch.object(run_screen, 'digest', return_value='fixture'), patch.object(run_screen, 'write_json'):
        summary = run_screen.summarize(fake)
        assert summary['selected']['candidate'] == 'c0' and len(summary['records']) == 32
    with patch.object(run_screen, 'accepted', return_value=None):
        rejects(lambda: run_screen.summarize(fake))
    with tempfile.TemporaryDirectory(prefix='flgmm_freeze_unit_') as folder:
        out = Path(folder) / 'release'
        subprocess.run([sys.executable, str(HERE / 'freeze_release.py'), '--reviewed-sha',
                        digest(HERE / 'PACKAGE_SHA256.json'), '--out', str(out)], check=True,
                       capture_output=True, text=True)
        check = "from screen_common import *; p,m=local_identity(); assert p['status']=='FROZEN' and len(m['jobs'])==32; authorized()"
        process = subprocess.run([sys.executable, '-c', check], cwd=out, capture_output=True, text=True)
        assert process.returncode != 0 and 'EXECUTION_AUTHORIZATION' in process.stderr
        job = next((out / 'jobs').glob('*_screen.json'))
        job.write_bytes(job.read_bytes() + b' ')
        process = subprocess.run([sys.executable, '-c', 'from screen_common import local_identity; local_identity()'],
                                 cwd=out, capture_output=True, text=True)
        assert process.returncode != 0 and 'Package changed' in process.stderr
    print('PASS: prepared launch rejection, 32-grid/config identity, verbatim score, strict skip/partial/failure boundaries, tie/incomplete selection, separate freeze/regeneration, missing authorization and altered-job rejection. Unit fixtures only; real training count=0.')


if __name__ == '__main__':
    main()
