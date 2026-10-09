"""Independently verify the declared population, without scores or target labels."""
from pathlib import Path
import datetime
import hashlib
import io
import json
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_logofair_population_proposal_20261009'
CHECKS = ROOT / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
assert sha(BASE / 'FILES_SHA256.json') == 'f90f8adf938ba65d0ed5ce3205d8ec731da69d87b72e904254bc1b3d551eb28b'
for name, spec in read(BASE / 'FILES_SHA256.json')['members'].items():
    assert sha(BASE / name) == spec['sha256'] and (BASE / name).stat().st_size == spec['bytes']
meta = read(BASE / 'mapping_metadata.json')
assert not meta['approved'] and not meta['execution_authorized'] and not meta['true_training_client_identity']
assert meta['status'] == 'PREPARED_NOT_APPROVED'
with np.load(BASE / 'mapping.npz', allow_pickle=False) as handle:
    mapping = {name: handle[name] for name in handle.files}
assert set(mapping) == {'root_image_id', 'root_client_id', 'valid_image_id', 'valid_client_id'}
domain = b'GuardFed/LoGoFair/virtual-cohort/v1\0'
counts = {}
for split, length in [('root', 16277), ('valid', 19867)]:
    ids, assigned = mapping[split + '_image_id'], mapping[split + '_client_id']
    assert ids.dtype == assigned.dtype == np.int64 and ids.shape == assigned.shape == (length,)
    assert len(set(map(int, ids))) == length and (ids > 0).all()
    expected = np.array([int(hashlib.sha256(domain + str(int(identity)).encode('ascii')).hexdigest(), 16) % 20 for identity in ids])
    assert np.array_equal(assigned, expected)
    assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest() == meta[split + '_image_ids_sha256']
    counts[split] = [int(np.sum(assigned == group)) for group in range(20)]
assert not np.intersect1d(mapping['root_image_id'], mapping['valid_image_id']).size
bindings = read(BASE / 'INPUT_BINDINGS.json')
assert len(bindings['bindings']) == 4 and bindings['cache_count'] == 4 and not bindings['missing_caches']
archives = {row['cache_archive']: row['cache_archive_sha256'] for row in bindings['bindings']}
assert len(archives) == 1
filename, expected = next(iter(archives.items()))
assert sha(filename) == expected
supports = read(BASE / 'population_support.json')
minimum = 16277
with tarfile.open(filename) as archive:
    for row in bindings['bindings']:
        data = archive.extractfile(row['cache_member']).read()
        assert hashlib.sha256(data).hexdigest() == row['cache_sha256']
        with np.load(io.BytesIO(data), allow_pickle=False) as cache:
            y, sensitive = cache['root_y'], cache['root_sensitive']
        assert y.shape == sensitive.shape == (16277,)
        report = next(item for item in supports if item['id'] == row['id'])
        for group in range(20):
            observed = report['rows'][group]
            assert observed['cohort'] == group and observed['root_n'] == counts['root'][group]
            assert observed['valid_n'] == counts['valid'][group]
            for target in (0, 1):
                for attribute in (0, 1):
                    n = int(np.sum((mapping['root_client_id'] == group) & (y == target) & (sensitive == attribute)))
                    assert n == observed[f'y{target}_Male{attribute}'] and n > 0
                    minimum = min(minimum, n)
assert minimum == 116
for filename, digest in bindings['original_bridge_files'].items():
    assert sha(ROOT / filename) == digest
proof = dict(status='ROOT_POPULATION_PROPOSAL_IDENTITY_SUPPORT_PASS_NOT_APPROVED',
             verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             seal_sha256=sha(BASE / 'FILES_SHA256.json'), mapping_sha256=sha(BASE / 'mapping.npz'),
             metadata_sha256=sha(BASE / 'mapping_metadata.json'), conditions=4, root=16277, valid=19867,
             cohorts=20, root_n_range=[min(counts['root']), max(counts['root'])],
             valid_n_range=[min(counts['valid']), max(counts['valid'])], min_root_label_sensitive_cell=minimum,
             original_bridge_unchanged=True, valid_labels_or_scores_decoded=False,
             CNN_or_Beta_fit_or_performance=False, formal_approval=False, unique_population_definition='virtual only')
with (CHECKS / 'LOGOFAIR_POPULATION_PROPOSAL_ROOT_VERIFICATION.json').open('x', encoding='utf8', newline='\n') as stream:
    json.dump(proof, stream, indent=2)
    stream.write('\n')
print(json.dumps(proof))
