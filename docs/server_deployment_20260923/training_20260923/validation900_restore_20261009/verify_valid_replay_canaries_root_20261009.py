"""Independent, inference-free verification of the two sealed image replays."""
from pathlib import Path
import datetime
import hashlib
import json
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_final_valid_replay_20261009'
OUT = ROOT / 'docs/server_deployment_20260923/training_20260923/validation900_restore_20261009'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((BASE / 'remote_receipts_inventory.json').read_text())
archive = BASE / manifest['archive']
assert sha(archive) == manifest['sha256'] and archive.stat().st_size == manifest['bytes']
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == len({m.name for m in members}) == 27
    assert {m.name for m in members} == set(manifest['members'])
    for member in members:
        assert member.isfile()
        data = stream.extractfile(member).read()
        expected = manifest['members'][member.name]
        assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256']

sealed = BASE / 'FILES_SHA256'
assert sha(sealed) == '39fa64bf9762d6975511d154d51cbaf3e7c292010f266b68dd8c98e96f35f402'
files = []
for line in sealed.read_text().splitlines():
    digest, name = line.split('  ', 1)
    path = BASE / name
    assert path.resolve().is_relative_to(BASE.resolve()) and sha(path) == digest
    files.append(name)
assert len(files) == 52

with np.load(BASE / 'verification_inputs/original_valid_cache.npz', allow_pickle=False) as values:
    y, sensitive = values['valid_y'], values['valid_sensitive']
assert len(y) == 19867
checks = []
for path in sorted((BASE / 'remote_receipts/canary2_cpu_v2').glob('*/receipt.json')):
    receipt = json.loads(path.read_text())
    prediction_path = path.parent / 'validation_predictions.npz'
    assert sha(prediction_path) == receipt['prediction_arrays_sha256']
    assert receipt['weights_before'] == receipt['weights_after']
    assert receipt['native_comparison']['max_abs_difference'] == 0
    assert not any(receipt[key] for key in ('optimizer_created', 'gradients_created',
        'test_labels_accessed', 'test_inference_performed', 'final_dispatch_created'))
    with np.load(prediction_path, allow_pickle=False) as values:
        assert len(values['valid_margins']) == 19867 and len(values['root_margins']) == 16277
        for view in ('raw', 'native', 'shared_calibration'):
            prediction = values['prediction_' + view]
            if view == 'raw':
                expected = values['valid_margins'] > 0
            else:
                thresholds = receipt['fits'][view]['thresholds']
                expected = np.where(sensitive == 0, values['valid_margins'] >= thresholds['0'],
                    values['valid_margins'] >= thresholds['1'])
            assert np.array_equal(prediction, expected)
            groups = []
            for group in (0, 1):
                selected = sensitive == group
                counts = {
                    'tp': int((selected & (y == 1) & (prediction == 1)).sum()),
                    'fp': int((selected & (y == 0) & (prediction == 1)).sum()),
                    'tn': int((selected & (y == 0) & (prediction == 0)).sum()),
                    'fn': int((selected & (y == 1) & (prediction == 0)).sum()),
                }
                stored = receipt['views'][view]['group_confusion_counts'][str(group)]
                assert all(stored[key] == value for key, value in counts.items())
                assert stored['n'] == sum(counts.values())
                assert stored['positives'] == counts['tp'] + counts['fn']
                assert stored['negatives'] == counts['tn'] + counts['fp']
                groups.append(counts)
            first, second = groups
            direct = {
                'accuracy': int((prediction == y).sum()) / len(y),
                'aeod': abs(first['tp'] / (first['tp'] + first['fn']) -
                    second['tp'] / (second['tp'] + second['fn'])),
                'aspd': abs((first['tp'] + first['fp']) / sum(first.values()) -
                    (second['tp'] + second['fp']) / sum(second.values())),
            }
            assert all(direct[key] == receipt['views'][view][key] for key in direct)
    checks.append({'id': receipt['id'], 'receipt_sha256': sha(path),
        'native_max_abs_difference': 0, 'views_metrics_recomputed': 9,
        'confusion_counts_recomputed': 24})
assert len(checks) == 2
report = {'status': 'PASS', 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'verifier_sha256': sha(Path(__file__)), 'sealed_manifest_sha256': sha(sealed),
    'sealed_files_verified': len(files), 'archive_sha256': sha(archive),
    'archive_members_verified': len(members), 'canaries': checks,
    'all900_replayed': False, 'test_accessed': False,
    'scope': 'Root independent archive bytes, threshold/tie reconstruction, group counts; no new inference',
    'prior_helper_failure': 'An inline helper compared a four-count dictionary against the whole stored group dictionary including support/rates. It stopped at that assertion before writing any receipt. This verifier compares the declared counts and separately checks support. Original evidence/inference/tolerances unchanged.'}
with (OUT / 'root_canary_independent_verification.json').open('x', encoding='utf-8') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
print(json.dumps(report))
