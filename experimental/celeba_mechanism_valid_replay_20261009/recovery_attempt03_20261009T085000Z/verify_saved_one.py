"""Independent validation counts from saved arrays; no model/image inference."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parents[1] / 'celeba_final_valid_replay_20261009'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    path = HERE / 'saved_one'
    receipt, bridge, acceptance = (read(path / n) for n in ('receipt.json', 'bridge_receipt.json', 'strict_acceptance.json'))
    inventory = read(HERE.parent / 'inventory_actual8_Full100refs.json')
    record = next(r for r in inventory['records'] if r['id'] == 'minus_U_IID_Benign_seed91002')
    assert receipt['id'] == bridge['id'] == acceptance['id'] == record['id']
    assert acceptance['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED'
    assert bridge['source_before'] == bridge['source_after'] and bridge['artifact_before'] == bridge['artifact_after']
    assert len(bridge['source_before']) == 35 and len(bridge['artifact_before']) == 7
    assert receipt['checkpoint_sha256'] == acceptance['checkpoint_sha256'] == record['checkpoint']['sha256']
    assert receipt['weights_before'] == receipt['weights_after'] and not receipt['optimizer_created'] and not receipt['gradients_created']
    assert receipt['runtime']['device'] == 'cpu' and receipt['runtime']['cuda_device_count'] == 0
    assert receipt['runtime']['torch_threads'] == 8 and receipt['runtime']['interop_threads'] == 1 and receipt['runtime']['nice'] == 10
    assert receipt['valid_n'] == 19867 and receipt['root_reconstruction']['root_n'] == 16277
    assert sha(path / 'validation_predictions.npz') == receipt['prediction_arrays_sha256']
    assert sha(path / 'receipt.json') == bridge['scientific_body_receipt_sha256']
    assert sha(path / 'bridge_receipt.json') == acceptance['bridge_receipt_sha256']
    assert bridge['approval_sha256'] == '34fc8399e0cc8271c95001289e9b05a1412bbdcb9ed3894bc43fa38e27969162'
    for resources in (receipt['before_resources'], receipt['after_resources']):
        assert all(cpus == list(range(112, 120)) for cpus in resources['thread_cpu_affinities'].values())
        assert 'RUNNING' in resources['service']['stdout'] and not resources['training_queue_snapshot']['failed']
    cache = BASELINE / 'verification_inputs/original_valid_cache.npz'
    assert sha(cache) == '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
    with np.load(cache, allow_pickle=False) as data:
        y, sensitive = data['valid_y'], data['valid_sensitive']
    differences, checks = {}, 0
    with np.load(path / 'validation_predictions.npz', allow_pickle=False) as data:
        assert len(data['valid_margins']) == len(y) == 19867 and len(data['root_margins']) == 16277
        for array, contract in [('valid', 'evaluation'), ('root', 'root')]:
            assert hashlib.sha256(data[array + '_image_ids'].tobytes()).hexdigest() == record['data_contract'][contract + '_image_ids_sha256']
        for view in ('native', 'raw', 'shared_calibration'):
            prediction, fit = data['prediction_' + view], receipt['fits'][view]
            if fit['rule'] == 'argmax_margin_strictly_positive':
                assert fit['thresholds'] is None
                expected = data['valid_margins'] > 0
            else:
                assert fit['rule'] == 'group_margin_greater_equal' and fit['fit_data'] == 'clean_train_root_only'
                thresholds = fit['thresholds']
                expected = np.where(sensitive == 0, data['valid_margins'] >= thresholds['0'], data['valid_margins'] >= thresholds['1'])
            assert np.array_equal(prediction, expected)
            tpr = [int(((sensitive == g) & (y == 1) & (prediction == 1)).sum()) / int(((sensitive == g) & (y == 1)).sum()) for g in (0, 1)]
            rates = [int(((sensitive == g) & (prediction == 1)).sum()) / int((sensitive == g).sum()) for g in (0, 1)]
            direct = {'accuracy': int((prediction == y).sum()) / len(y), 'aeod': abs(tpr[0] - tpr[1]), 'aspd': abs(rates[0] - rates[1])}
            differences[view] = {k: direct[k] - receipt['views'][view][k] for k in direct}
            assert all(d == 0 for d in differences[view].values()) and receipt['views'][view] == acceptance['views'][view]
            for g in (0, 1):
                counts = {name: int(((sensitive == g) & (y == yy) & (prediction == pp)).sum())
                          for name, yy, pp in [('tp', 1, 1), ('fp', 0, 1), ('tn', 0, 0), ('fn', 1, 0)]}
                assert all(receipt['views'][view]['group_confusion_counts'][str(g)][k] == count for k, count in counts.items())
                checks += 4
    native = max(abs(receipt['views']['native'][k] - record['prior_validation_metrics'][k]) for k in ('accuracy', 'aeod', 'aspd'))
    assert native == receipt['native_comparison']['max_abs_difference'] == 0 and native <= 1e-12
    paired = BASELINE / 'v3/phase3_execution_20261009/remote_receipts/phase3_useful/runs/GuardFed-AD2+_IID_Benign_seed91002/receipt.json'
    assert sha(paired) == '254f05813b19d7489af1cebdc23f1c5908e9251bb54387997feb9e248233692c'
    full = read(paired)
    assert full['checkpoint_sha256'] == record['paired_full']['checkpoint_sha256']
    deltas = {view: {k: receipt['views'][view][k] - full['views'][view][k] for k in ('accuracy', 'aeod', 'aspd')}
              for view in ('native', 'raw', 'shared_calibration')}
    proof = {'status': 'INDEPENDENT_SAVED_ARRAYS_ALL_THREE_VIEWS_PASS', 'id': record['id'],
             'independent_metric_checks': 9, 'independent_confusion_count_checks': checks,
             'prediction_rule_checks': 3, 'differences': differences, 'native_max_abs_difference': native,
             'checkpoint_sha256': receipt['checkpoint_sha256'], 'receipt_sha256': sha(path / 'receipt.json'),
             'prediction_arrays_sha256': receipt['prediction_arrays_sha256'],
             'Full_reference_receipt_sha256': sha(paired), 'minus_U_minus_existing_Full_three_view_differences': deltas,
             'new_Full_inference': 0, 'new_training': 0, 'test_inference': False,
             'root_fit_recomputed_by_strict_server_bridge': True, 'local_recompute_uses_only_common_valid_labels': True,
             'elapsed_seconds': receipt['elapsed_seconds'], 'effective_cpu_cores': receipt['effective_cpu_cores'],
             'torch_imported': 'torch' in sys.modules, 'verifier_sha256': sha(Path(__file__)),
             'claim_limit': 'One validation terminal replay, not a mean or final-test claim; earlier failed inference is not counted.'}
    with (HERE / 'independent_saved_array_verification.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
