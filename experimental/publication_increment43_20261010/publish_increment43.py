"""Git43: freeze curated bytes, then explicitly stage on the external sparse checkout.

No commit, push, fetch, checkout, reset, or automatic retry is implemented.
"""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, re, subprocess, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKOUT = Path('F:/YananResearchStorage/GuardFed/git_publication/current')
OUTPUT_ROOT = Path('F:/YananResearchStorage/GuardFed/git_publication/increment43')
PARENT = 'e3618635c01d22e01cbf17db4c0ae19ad750dc3b'
BRANCH = 'codex/revision-evidence-baselines-20260928'
ORIGIN = 'https://github.com/YananZHOU5555/GuardFed.git'
LIMIT = 100_000_000
ROLES = {'native188', 'FL22', 'gradient_startup', 'logofair_startup', 'remaining620_startup', 'current_state'}
SOURCES = {'celeba_gradient_screen64_20261010', 'celeba_gradient_screen64_v2_20261010', 'celeba_logofair_screen32_20261010', 'celeba_mechanism_remaining_evaluation_20261010', 'celeba_mechanism_remaining_evaluation_v2_20261010'}
BANNED = {'__pycache__', 'restored', 'verified_extract', 'verified', '.git', 'attempt001', 'attempt002'}
SUFFIXES = {'.pt', '.pth', '.npz', '.npy', '.pkl', '.gz', '.zip', '.7z', '.pem', '.key', '.env'}
SECRET = re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}')
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def storage(required=0):
    # Keep the publication tool standalone; identical volume/one-GiB reserve contract.
    r = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command',
        "$ErrorActionPreference='Stop'; Get-Volume -DriveLetter F | Select-Object DriveLetter,FileSystemLabel,SizeRemaining,HealthStatus | ConvertTo-Json -Compress"],
        capture_output=True, text=True, check=True, timeout=30)
    d = json.loads(r.stdout.lstrip('\ufeff'))
    assert d['DriveLetter'] == 'F' and d['FileSystemLabel'] == 'Yanan 2TB'
    assert d['HealthStatus'] in ('Healthy', 0) and d['SizeRemaining'] >= required + 1024**3
    assert CHECKOUT.resolve().drive.upper() == 'F:' and OUTPUT_ROOT.resolve().drive.upper() == 'F:'
    return d


def git(*args):
    return subprocess.run(['git', '-c', 'safe.directory=' + CHECKOUT.as_posix(), '-C', str(CHECKOUT), *args],
                          capture_output=True, check=True, timeout=120).stdout


def relative(name):
    p = PurePosixPath(name)
    assert name == p.as_posix() and not p.is_absolute() and '..' not in p.parts and p.parts
    assert not any(x in BANNED for x in p.parts) and p.suffix.lower() not in SUFFIXES
    assert all(not any(c in x for c in '\n\r\t"\\*?[]') for x in p.parts)
    return p


def source(name, allowed):
    p = relative(name)
    assert any(name == a or name.startswith(a.rstrip('/') + '/') for a in allowed), 'Outside reviewed scope: ' + name
    q = ROOT.joinpath(*p.parts)
    assert q.resolve().is_relative_to(ROOT.resolve()) and q.is_file()
    assert not any(x.is_symlink() for x in [q, *q.parents] if x.is_relative_to(ROOT))
    b = q.read_bytes()
    assert len(b) < LIMIT and not SECRET.search(b), 'Oversize or credential material: ' + name
    return q, b


def destination(name):
    p = relative(name)
    if p.parts[0] == 'tmp':
        return 'experimental/' + PurePosixPath(*p.parts[1:]).as_posix()
    assert p.parts[0] == 'docs', 'Only curated tmp and docs sources are publishable'
    return name


def pointer(d, key):
    assert key.startswith('/')
    for part in key[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        d = d[int(part)] if isinstance(d, list) else d[part]
    return d


def plan(spec):
    assert spec['parent'] == PARENT and spec['branch'] == BRANCH
    assert spec['scope'] == 'SOURCE_AND_ACTUAL_STARTUP_NO_FUTURE_RESULTS'
    assert set(spec['bindings']) == ROLES
    assert {e['directory'] for e in spec['sealed_sources']} == {'tmp/' + n for n in SOURCES} and len(spec['sealed_sources']) == 5
    assert spec['accepted'] == {'native': 188, 'three_view': 180, 'FL_new': 22}
    assert spec['test_started'] is False and spec['goal_complete'] is False
    for role, facts in {'native188': {'/cumulative_new': 188, '/three_view_accepted_unchanged': 180},
                        'FL22': {'/accepted_total': 22},
                        'current_state': {'/celeba_mechanism_v1/scientific_results_strictly_accepted': 188,
                                          '/celeba_mechanism_v1/three_view_new_models_accepted': 180}}.items():
        assert spec['bindings'][role] and all(spec['bindings'][role]['expect'].get(k) == v for k, v in facts.items())
    allowed = spec['allowed_paths']
    selected = set(spec['files'])
    for entry in spec['sealed_sources']:
        base = entry['directory']; seal = base + '/' + entry.get('seal', 'FILES_SHA256.json')
        _, b = source(seal, allowed)
        assert sha(b) == entry['sha256'], 'Source seal changed: ' + seal
        members = json.loads(b)['files']
        for name, info in members.items():
            _, content = source(base + '/' + name, allowed)
            assert sha(content) == (info if isinstance(info, str) else info['sha256'])
            if isinstance(info, dict):
                assert len(content) == info['bytes']
            selected.add(base + '/' + name)
        selected.add(seal)
        selected.add(base + '/HANDOFF.json')
    for role, pin in spec['bindings'].items():
        assert pin and pin.get('sha256') and pin.get('expect'), 'Missing actual binding: ' + role
        _, b = source(pin['path'], allowed)
        assert sha(b) == pin['sha256'], 'Binding changed: ' + role
        d = json.loads(b)
        for key, expected in pin['expect'].items():
            assert pointer(d, key) == expected, 'Root fact mismatch: ' + role + key
        selected.add(pin['path'])
    files = []
    for name in sorted(selected):
        _, b = source(name, allowed)
        files.append(dict(source=name, destination=destination(name), sha256=sha(b), bytes=len(b)))
    assert len({x['destination'] for x in files}) == len(files)
    assert sum(x['bytes'] for x in files) < LIMIT, 'Curated increment must be under 100 MB'
    return dict(schema='publication43_frozen_bytes_v1', parent=PARENT, branch=BRANCH, origin=ORIGIN,
        scope=spec['scope'], accepted=spec['accepted'], test_started=False, goal_complete=False,
        bindings=spec['bindings'], allowed_paths=allowed, files=files, total_bytes=sum(x['bytes'] for x in files))


def save(path, d):
    with path.open('x', encoding='utf8', newline='\n') as f:
        f.write(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def output(name, required):
    relative(name)
    assert '/' not in name
    storage(required)
    p = OUTPUT_ROOT / name
    p.mkdir(parents=True, exist_ok=False)
    return p


def freeze(spec_path, expected, name):
    b = spec_path.read_bytes(); assert sha(b) == expected
    d = plan(json.loads(b)); d['spec_sha256'] = expected
    out = output(name, d['total_bytes'] * 3)
    save(out / 'FROZEN_INPUTS.json', d)
    print(json.dumps(dict(path=str(out / 'FROZEN_INPUTS.json'), sha256=sha((out / 'FROZEN_INPUTS.json').read_bytes()), files=len(d['files']), bytes=d['total_bytes'])))


def stage(path, expected, name):
    b = path.read_bytes(); assert sha(b) == expected
    d = json.loads(b)
    assert d['schema'] == 'publication43_frozen_bytes_v1' and d['parent'] == PARENT and d['branch'] == BRANCH and d['origin'] == ORIGIN
    assert d['accepted'] == {'native': 188, 'three_view': 180, 'FL_new': 22} and set(d['bindings']) == ROLES
    assert d['test_started'] is False and d['goal_complete'] is False
    assert d['total_bytes'] == sum(x['bytes'] for x in d['files']) < LIMIT
    assert len({x['destination'] for x in d['files']}) == len(d['files'])
    storage(d['total_bytes'] * 3)
    assert git('rev-parse', 'HEAD').decode().strip() == PARENT
    assert git('branch', '--show-current').decode().strip() == BRANCH
    assert git('remote', 'get-url', 'origin').decode().strip() == ORIGIN
    assert not git('status', '--porcelain', '--untracked-files=all').strip(), 'Preserve dirty checkout; do not overwrite'
    # Revalidate all bytes before the first worktree mutation.
    for x in d['files']:
        _, content = source(x['source'], d['allowed_paths'])
        assert x['destination'] == destination(x['source']) and sha(content) == x['sha256'] and len(content) == x['bytes']
    out = output(name, d['total_bytes'] * 3)
    save(out / 'FROZEN_INPUTS.json', d)
    try:
        for x in d['files']:
            _, content = source(x['source'], d['allowed_paths'])
            assert sha(content) == x['sha256'], 'Concurrent source update; stop and preserve partial stage'
            target = CHECKOUT / x['destination']; target.parent.mkdir(parents=True, exist_ok=True)
            assert target.resolve().is_relative_to(CHECKOUT.resolve())
            target.write_bytes(content)
        attr = CHECKOUT / '.gitattributes'
        prior = attr.read_bytes() if attr.exists() else b''
        patterns = ('\n# Git43 sealed source/startup bytes\n' + ''.join(json.dumps('/' + x['destination'], ensure_ascii=False) + ' -text\n' for x in d['files'])).encode('utf8')
        attr.write_bytes(prior + patterns)
        paths = [x['destination'] for x in d['files']] + ['.gitattributes']
        for start in range(0, len(paths), 40):
            storage(d['total_bytes'])
            git('add', '--sparse', '-f', '--', *paths[start:start+40])
        blobs = [dict(path=x['destination'], sha256=x['sha256'], bytes=x['bytes']) for x in d['files']]
        blobs.append(dict(path='.gitattributes', sha256=sha(attr.read_bytes()), bytes=attr.stat().st_size))
        for x in blobs:
            content = git('show', ':' + x['path'])
            assert sha(content) == x['sha256'] and len(content) == x['bytes']
        receipt = dict(status='STAGED_INDEX_BYTES_PASS_NOT_COMMITTED_OR_PUSHED', parent=PARENT, branch=BRANCH,
            frozen_inputs_sha256=expected, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), blobs=blobs,
            accepted=d['accepted'], test_started=False, goal_complete=False)
        save(out / 'STAGE_RECEIPT.json', receipt)
        print(json.dumps(dict(receipt=str(out / 'STAGE_RECEIPT.json'), sha256=sha((out / 'STAGE_RECEIPT.json').read_bytes()), blobs=len(blobs))))
    except Exception as e:
        save(out / 'FAILURE.json', dict(error=repr(e), no_automatic_retry=True, preserve_partial_worktree_and_index=True))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'freeze', 'stage'])
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output-name')
    a = p.parse_args()
    assert sha(a.input.read_bytes()) == a.sha256
    if a.action == 'plan':
        d = plan(read(a.input)); print(json.dumps(dict(files=len(d['files']), bytes=d['total_bytes'], status='PLAN_ONLY_NO_WRITES')))
    else:
        assert a.output_name, 'A fresh F output directory name is required'
        (freeze if a.action == 'freeze' else stage)(a.input, a.sha256, a.output_name)
