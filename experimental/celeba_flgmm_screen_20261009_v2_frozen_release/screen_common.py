"""Identity checks around the unchanged FLGMM worker and terminal checker."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))
from worker import digest, validate_job, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def under(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes root: ' + name)
    return path


def local_identity(require_frozen=True):
    seal = read(HERE / 'PACKAGE_SHA256.json')
    for name, expected in seal['files'].items():
        if digest(under(HERE, name)) != expected:
            raise ValueError('Package changed: ' + name)
    protocol = read(HERE / 'source/protocol.json')
    manifest = read(HERE / 'jobs/manifest.json')
    if require_frozen:
        if seal['status'] != 'FROZEN_PENDING_EXECUTION' or protocol['status'] != 'FROZEN':
            raise ValueError('Prepared package cannot run')
    if manifest['protocol_sha256'] != digest(HERE / 'source/protocol.json'):
        raise ValueError('Manifest/protocol identity mismatch')
    coverage = set()
    for item in manifest['jobs']:
        path = under(HERE / 'jobs', item['job'])
        if digest(path) != item['job_sha256']:
            raise ValueError('Job SHA mismatch')
        job = read(path)
        validate_job(job, protocol, require_frozen=require_frozen)
        if any(item[key] != job[key] for key in ['id', 'method', 'tuning_candidate', 'distribution', 'attack']):
            raise ValueError('Manifest/job mismatch')
        for name, expected in job['adapter_source_hashes'].items():
            if digest(under(HERE / 'source', name)) != expected:
                raise ValueError('Adapter changed: ' + name)
        key = (job['tuning_candidate'], job['distribution'], job['attack'])
        if key in coverage:
            raise ValueError('Duplicate condition')
        coverage.add(key)
    expected = {(c['id'], d, a) for c in protocol['candidates']
                for d in protocol['distributions'] for a in protocol['attacks']}
    if len(manifest['jobs']) != 32 or len(expected) != 32 or coverage != expected:
        raise ValueError('Expected exactly eight candidates x four conditions')
    return protocol, manifest


def checked_repo_path(repo, name, expected, mapping):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid frozen repository-relative path: ' + name)
    if name not in mapping:
        return under(repo, name)
    rule = mapping[name]
    actual = (repo / relative).resolve()
    if actual != Path(rule['resolved_target']) or expected != rule['sha256']:
        raise ValueError('Frozen data symlink target/SHA declaration changed: ' + name)
    return actual


def repo_identity(repo, protocol):
    mapping = read(HERE / 'REPO_SYMLINK_TARGETS.json')['entries']
    permitted = {'data/celeba/' + name for name in [
        'derived/rgb64_v1/manifest.json', 'derived/rgb64_v1/metadata.npz',
        'derived/rgb64_v1/images.npy', 'derived/rgb64_v1/available.npy',
        'list_attr_celeba.txt', 'list_eval_partition.txt']}
    if set(mapping) != permitted:
        raise ValueError('Only the six explicitly reviewed CelebA data targets are permitted')
    for name, expected in protocol['source_hashes'].items():
        if digest(checked_repo_path(repo, name, expected, mapping)) != expected:
            raise ValueError('Repository source/data changed: ' + name)
    return dict(protocol['source_hashes'])


def authorized():
    receipt = read(HERE / 'EXECUTION_AUTHORIZATION.json')
    if (receipt['status'], receipt['scope'], receipt['package_sha256']) != (
            'AUTHORIZED', '32_valid_only_screen', digest(HERE / 'PACKAGE_SHA256.json')):
        raise ValueError('Missing or mismatched final execution authorization')
    if not receipt.get('resource_review_utc') or not receipt.get('no_duplicate_workers_verified'):
        raise ValueError('Final live resource/duplicate review required')
    return receipt


def accepted(item, out):
    from accept_result import checked_result
    result = checked_result(HERE / 'jobs' / item['job'], out)
    if result is None:
        return None
    proof = read(out / 'screen_identity.json')
    if proof['package_sha256'] != digest(HERE / 'PACKAGE_SHA256.json'):
        raise ValueError('Result package identity mismatch')
    if proof['before'] != proof['after'] or proof['before'] != result['revision_job']['source_hashes']:
        raise ValueError('Missing matching before/after source and data checks')
    if proof['acceptance_sha256'] != digest(out / 'acceptance.json'):
        raise ValueError('Acceptance receipt changed')
    if result['data_contract']['root_clean_rows'] != 16277:
        raise ValueError('Root row count mismatch')
    return result
