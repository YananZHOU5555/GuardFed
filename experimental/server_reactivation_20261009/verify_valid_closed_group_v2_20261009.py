"""Root check of a new sealed chunk group; no inference or training."""
from pathlib import Path
import argparse
import datetime
import hashlib
import io
import json
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_final_valid_replay_20261009'
CONTROL = BASE / 'v4/remaining872_execution_20261009'

def read(path): return json.loads(path.read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--first', type=int, required=True)
parser.add_argument('--last', type=int, required=True)
parser.add_argument('--previous-n', type=int, required=True)
parser.add_argument('--total-n', type=int, required=True)
parser.add_argument('--group-sha', required=True)
parser.add_argument('--collection-sha', required=True)
parser.add_argument('--previous-sha', required=True)
args = parser.parse_args()
assert 0 <= args.first <= args.last < 80 and 28 <= args.previous_n < args.total_n <= 900
group_label = f'{args.first:03d}_{args.last:03d}'
group_path = CONTROL / f'closed_group_{group_label}_offserver_receipt.json'
assert sha(group_path) == args.group_sha
group = read(group_path)
assert group['chunk_indices'] == list(range(args.first, args.last + 1))
assert group['new_n'] == args.total_n - args.previous_n
cache = BASE / 'verification_inputs/original_valid_cache.npz'
assert sha(cache) == '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
with np.load(cache, allow_pickle=False) as arrays:
    y, sensitive = arrays['valid_y'], arrays['valid_sensitive']
assert len(y) == len(sensitive) == 19867
ids, archive_proofs = [], []
for index in group['chunk_indices']:
    folder = BASE / f'v4/remaining872_attempt1/chunk_{index:03d}'
    assert sha(folder / 'FILES_SHA256') == group['chunk_files_sha256'][str(index)]
    for line in (folder / 'FILES_SHA256').read_text().splitlines():
        value, name = line.split(None, 1)
        assert sha(folder / name.strip().lstrip('*')) == value
    inventory = read(folder / 'remote_archive_inventory.json')
    archive = folder / inventory['archive']
    assert sha(archive) == inventory['sha256']
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        assert len(members) == len({m.name for m in members}) == len(inventory['members'])
        data = {}
        for member in members:
            row = inventory['members'][member.name]
            assert member.isfile() and member.size == row['bytes']
            data[member.name] = bundle.extractfile(member).read()
            assert hashlib.sha256(data[member.name]).hexdigest() == row['sha256']
    manifest_bytes = data['execution/manifest.json']
    assert hashlib.sha256(manifest_bytes).hexdigest() == 'ad6eebf517f534fb8489acb241c51a9ec5328bb285406e55275f7dd9c0c3ed43'
    manifest = json.loads(manifest_bytes)
    assert inventory['accepted_ids'] == manifest['chunks'][index]['ids']
    for identity in inventory['accepted_ids']:
        assert identity not in ids
        ids.append(identity)
        prefix = 'batch/runs/' + identity
        receipt = json.loads(data[prefix + '/receipt.json'])
        assert receipt['id'] == identity and receipt['native_comparison']['max_abs_difference'] == 0
        assert receipt['weights_before'] == receipt['weights_after'] and not receipt['test_inference_performed']
        with np.load(io.BytesIO(data[prefix + '/validation_predictions.npz']), allow_pickle=False) as predictions:
            for view in ('native', 'raw', 'shared_calibration'):
                prediction = predictions['prediction_' + view]
                assert prediction.shape == y.shape
                fit = receipt['fits'][view]
                if fit['rule'] == 'argmax_margin_strictly_positive':
                    rule = predictions['valid_margins'] > 0
                else:
                    assert fit['rule'] == 'group_margin_greater_equal'
                    rule = np.where(sensitive == 0, predictions['valid_margins'] >= fit['thresholds']['0'],
                                    predictions['valid_margins'] >= fit['thresholds']['1'])
                assert np.array_equal(prediction, rule)
                tpr, rate = [], []
                for group_id in (0, 1):
                    mask = sensitive == group_id
                    counts = dict(tp=int(np.sum(mask & (y == 1) & (prediction == 1))),
                                  fp=int(np.sum(mask & (y == 0) & (prediction == 1))),
                                  tn=int(np.sum(mask & (y == 0) & (prediction == 0))),
                                  fn=int(np.sum(mask & (y == 1) & (prediction == 0))))
                    assert all(receipt['views'][view]['group_confusion_counts'][str(group_id)][k] == v for k, v in counts.items())
                    tpr.append(counts['tp'] / (counts['tp'] + counts['fn']))
                    rate.append((counts['tp'] + counts['fp']) / int(mask.sum()))
                computed = dict(accuracy=int(np.sum(prediction == y)) / len(y),
                                aeod=abs(tpr[0] - tpr[1]), aspd=abs(rate[0] - rate[1]))
                assert all(receipt['views'][view][k] == v for k, v in computed.items())
    proof = read(folder / 'offserver_verification.json')
    assert proof['status'] == 'PASS' and proof['accepted_n'] == len(inventory['accepted_ids'])
    archive_proofs.append(dict(chunk=index, archive_sha256=sha(archive), members=len(members),
                               offserver_proof_sha256=sha(folder / 'offserver_verification.json')))

collection_path = CONTROL / f'cumulative_{args.total_n}_accepted.json'
assert sha(collection_path) == group['cumulative_sha256'] == args.collection_sha
collection = read(collection_path)
previous_path = CONTROL / f'cumulative_{args.previous_n}_accepted.json'
assert sha(previous_path) == args.previous_sha
old = read(previous_path)
assert old['accepted_n'] == args.previous_n
assert set(ids) == set(group['new_ids']) and len(ids) == group['new_n']
assert set(old['accepted_ids']).isdisjoint(ids)
assert set(old['accepted_ids']) | set(ids) == set(collection['accepted_ids'])
assert collection['accepted_n'] == len(set(collection['accepted_ids'])) == len(collection['accepted']) == args.total_n
assert len(collection['missing_ids']) == 900 - args.total_n
assert collection['all900_three_views_valid_replayed'] == (args.total_n == 900)
assert collection['collector_source_sha256'] == '19066f63c341b9ee23b7c6f491802cfdde0c1c0c833c2fe64724a16de9bb2234'
assert sha(CONTROL / f'collection_inputs_{args.total_n}.json') == collection['collection_inputs_sha256'] == group['collection_inputs_sha256']
report = dict(status='ROOT_ARCHIVES_AND_INDEPENDENT_ARRAY_CHECKS_PASS',
              verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), archives=archive_proofs,
              accepted_new_models=len(ids), direct_metric_checks=9*len(ids), confusion_count_checks=24*len(ids),
              prediction_rule_checks=3*len(ids), native_max_abs_difference=0, cumulative_actual_models=args.total_n,
              remaining=900-args.total_n, cumulative_sha256=sha(collection_path), original_models_repacked=0,
              test_started=False, goal_complete=False)
out = ROOT / f'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/VALID_CHUNKS{group_label}_ROOT_VERIFICATION.json'
with out.open('x', encoding='utf8') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(report))
