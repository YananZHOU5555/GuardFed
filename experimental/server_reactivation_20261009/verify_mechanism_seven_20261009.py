"""Independent saved-array and archive check for seven mechanism terminals."""
from pathlib import Path
import datetime
import hashlib
import io
import json
import tarfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_mechanism_valid_replay_20261009'
FOLDER = BASE / 'remaining_seven_completed_backup_20261009'
def read(path): return json.loads(path.read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(FOLDER / 'FILES_SHA256.json') == '96856e1847a5cc535c84c63ab7916ac54efb834cca59d9d0110bece9db1a652e'
for name, row in read(FOLDER / 'FILES_SHA256.json')['members'].items():
    assert sha(FOLDER / name) == row['sha256'] and (FOLDER / name).stat().st_size == row['bytes']
archive = FOLDER / 'remaining_seven_valid_three_views.tar.gz'
assert sha(archive) == '68dba401dcf4ec43b5ef12ea47c21ccdc2125603e4d2ec6326b8e9cb299de9b6'
receipt = read(FOLDER / 'backup_receipt.json')
with tarfile.open(archive) as bundle:
    members = bundle.getmembers()
    assert len(members) == len({m.name for m in members}) == 74
    data = {m.name: bundle.extractfile(m).read() for m in members if m.isfile()}
    assert len(data) == 74
    assert hashlib.sha256(data['backup_inventory.json']).hexdigest() == receipt['inventory_sha256']
    inventory = json.loads(data['backup_inventory.json'])
    assert set(data) == set(inventory['members']) | {'backup_inventory.json'}
    for name, row in inventory['members'].items():
        assert len(data[name]) == row['bytes'] and hashlib.sha256(data[name]).hexdigest() == row['sha256']
original_path = BASE / 'inventory_actual8_Full100refs.json'
assert sha(original_path) == inventory['original_inventory_sha256'] == 'be4d1d34b21443572a25e6a189710e8d75b108a8fcbdb58896a71f2ba1d88cc8'
originals = {row['id']: row for row in read(original_path)['records']}
ids = inventory['accepted_new_ids']
assert ids == ['minus_U_IID_Benign_seed' + str(s) for s in (91001,91003,91004,91005,91006,91007,91008)]
cache = ROOT / 'tmp/celeba_final_valid_replay_20261009/verification_inputs/original_valid_cache.npz'
assert sha(cache) == '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
with np.load(cache, allow_pickle=False) as arrays:
    y, sensitive = arrays['valid_y'], arrays['valid_sensitive']
assert len(y) == len(sensitive) == 19867
for identity in ids:
    prefix = 'runs/' + identity
    r = json.loads(data[prefix + '/receipt.json'])
    assert r['id'] == identity and r['native_comparison']['max_abs_difference'] == 0
    assert r['checkpoint_sha256'] == originals[identity]['checkpoint']['sha256']
    assert r['original_result_sha256'] == originals[identity]['result']['sha256']
    assert r['original_job_sha256'] == originals[identity]['raw_job']['sha256']
    assert r['weights_before'] == r['weights_after'] and not r['test_inference_performed']
    with np.load(io.BytesIO(data[prefix + '/validation_predictions.npz']), allow_pickle=False) as predictions:
        for view in ('native', 'raw', 'shared_calibration'):
            prediction = predictions['prediction_' + view]
            assert prediction.shape == y.shape
            fit = r['fits'][view]
            if fit['rule'] == 'argmax_margin_strictly_positive':
                rule = predictions['valid_margins'] > 0
            else:
                assert fit['rule'] == 'group_margin_greater_equal'
                rule = np.where(sensitive == 0, predictions['valid_margins'] >= fit['thresholds']['0'],
                                predictions['valid_margins'] >= fit['thresholds']['1'])
            assert np.array_equal(prediction, rule)
            tpr, rate = [], []
            for group in (0,1):
                mask = sensitive == group
                counts = dict(tp=int(np.sum(mask & (y==1) & (prediction==1))), fp=int(np.sum(mask & (y==0) & (prediction==1))),
                              tn=int(np.sum(mask & (y==0) & (prediction==0))), fn=int(np.sum(mask & (y==1) & (prediction==0))))
                assert all(r['views'][view]['group_confusion_counts'][str(group)][k] == v for k,v in counts.items())
                tpr.append(counts['tp'] / (counts['tp']+counts['fn']))
                rate.append((counts['tp']+counts['fp']) / int(mask.sum()))
            computed = dict(accuracy=int(np.sum(prediction==y))/len(y), aeod=abs(tpr[0]-tpr[1]), aspd=abs(rate[0]-rate[1]))
            assert all(r['views'][view][k] == v for k,v in computed.items())
off = read(FOLDER / 'offserver_verification.json')
assert off['pass'] and off['different_host_observed'] and off['members_verified'] == 74
old = read(ROOT / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/MECHANISM_VALID_ONE_ROOT_VERIFICATION.json')
assert set(old['accepted_new_ids']).isdisjoint(ids)
proof = dict(status='ROOT_SEVEN_ARCHIVE_AND_INDEPENDENT_ARRAY_CHECKS_PASS',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), archive_sha256=sha(archive), members_verified=74,
    delivery_seal_sha256=sha(FOLDER / 'FILES_SHA256.json'), offserver_verification_sha256=sha(FOLDER / 'offserver_verification.json'),
    accepted_new_ids=ids, cumulative_new_mechanism_three_views=8, metric_checks=63, confusion_count_checks=168,
    prediction_rule_checks=21, native_max_abs_difference=0, original_models_repacked=0, Full_reinference=0,
    single_seed91002_repacked=0, paired_Full_six_three_views_missing=True, test_started=False, goal_complete=False)
out = ROOT / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/MECHANISM_VALID_SEVEN_ROOT_VERIFICATION.json'
with out.open('x', encoding='utf8') as stream:
    json.dump(proof, stream, indent=2); stream.write('\n')
print(json.dumps(proof))
