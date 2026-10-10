"""Read-only existing Hybrid canary reuse contract. No replay or refit entry point."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ID = 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read_pin(pin):
    path = Path(pin['path'])
    need(path.suffix in {'.json', '.py'} and path.stat().st_size <= 2_000_000, 'Compact inputs only')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == pin['sha256'] and len(raw) == pin['bytes'], 'Input bytes changed')
    return json.loads(raw) if path.suffix == '.json' else raw


def one(rows, rid):
    matches = [r for r in rows if r['id'] == rid]
    need(len(matches) == 1, 'Missing or duplicate record')
    return matches[0]


def check():
    pins = json.loads((HERE / 'INPUT_PINS.json').read_bytes())
    need(pins['root']['sha256'] == '631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456', 'Wrong adopted interface root')
    docs = {key: read_pin(pin) for key, pin in pins.items()}
    spec = importlib.util.spec_from_file_location('_existing_hybrid_identity_bridge', pins['private_bridge']['path'])
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    current = bridge.identity_record(ID)
    root, transport, receipt = [docs[k] for k in ('root', 'transport', 'receipt')]
    adopted = one(root['records'], ID)
    need(root['root_adoption'] is True and root['interface_records_accepted'] == 3, 'Existing interface not adopted')
    need(current['checkpoint']['sha256'] == receipt['checkpoint_sha256'] == adopted['checkpoint_sha256'], 'Wrong checkpoint')
    for field, value in [('id', ID), ('method', 'CosineFairnessHybrid'), ('seed', 91002),
                         ('config_canonical_sha256', current['config_canonical_sha256']),
                         ('original_result_sha256', current['result']['sha256']),
                         ('original_job_sha256', current['raw_job']['sha256'])]:
        need(receipt[field] == value, 'Private bridge/receipt mismatch: ' + field)
    need(receipt['root_reconstruction']['root_image_ids_sha256'] == current['data_contract']['root_image_ids_sha256']
         and receipt['valid_image_ids_sha256'] == current['data_contract']['evaluation_image_ids_sha256']
         and receipt['valid_n'] == 19867, 'Wrong root/valid identity')
    need(receipt['native_comparison']['max_abs_difference'] == adopted['native_max_abs_difference'] == 0
         and receipt['native_comparison']['tolerance'] == 1e-12, 'Native comparison drift')
    need(receipt['runtime']['device'] == 'cpu' and receipt['runtime']['original_config_device'] == 'cuda'
         and all(v['dtype'] == '<f4' for v in receipt['weights_before'].values())
         and receipt['weights_before'] == receipt['weights_after'], 'Original FP32 CPU/checkpoint drift')
    need(not receipt['test_labels_accessed'] and not receipt['test_inference_performed'], 'Not valid-only')
    ev = bridge.science_bindings()
    need(ev.SHARED_CALIBRATION == docs['private_check']['shared_calibration'], 'Shared recipe drift')
    for name in ('native', 'raw'):
        fit = receipt['fits'][name]
        need(fit['rule'] == 'argmax_margin_strictly_positive' and fit['thresholds'] is None
             and fit['fit_data'] == 'none', 'Native/raw >0 rule drift')
    shared = receipt['fits']['shared_calibration']
    need(shared['rule'] == 'group_margin_greater_equal' and shared['fit_data'] == 'clean_train_root_only', 'Shared >=/root-only drift')
    need(shared['fit_diagnostics']['server_calibration_max_acc_drop'] == ev.SHARED_CALIBRATION['ad2_calibration_max_acc_drop']
         and shared['fit_diagnostics']['server_calibration_budget'] == ev.SHARED_CALIBRATION['ad2_calibration_budget'], 'Saved calibration recipe drift')
    linux, arrays = [one(docs[k]['records'], ID) for k in ('linux_whole', 'windows_array_block')]
    need(linux['root_receipt_exact'] and linux['cached_root_fit_exact']
         and linux['saved_predictions_metrics_counts_exact'], 'Missing original Linux whole proof')
    need(arrays['root_fit_parameters_and_diagnostics_exact'] and arrays['all_saved_predictions_and_counts_and_metrics_exact']
         and arrays['local_full_root_receipt_exact'], 'Missing original Windows array/root-receipt proof')
    need(root['Linux_whole_original_saved_check_pass'] is True and root['Windows_whole_saved_check_pass'] is False
         and adopted['windows_whole_root_receipt_exact'] is True, 'Preserve platform proof scope')
    for key in ('LINUX_EXACT_ROOT_CHECK.json', 'OFFSERVER_ARRAY_REFIT_CHECK.json', 'TRANSPORT_VERIFICATION.json'):
        name = {'LINUX_EXACT_ROOT_CHECK.json': 'linux_whole', 'OFFSERVER_ARRAY_REFIT_CHECK.json': 'windows_array_block',
                'TRANSPORT_VERIFICATION.json': 'transport'}[key]
        need(root['proof_files_sha256'][key] == pins[name]['sha256'], 'Root/consumer pin drift')
    members = {}
    for name, rootkey in [('receipt.json', 'receipt_sha256'), ('validation_predictions.npz', 'array_sha256')]:
        member = 'bundle/' + ID + '/' + name
        pin = transport['members'][member]
        path = Path(transport['verified_extract']) / member
        need(path.is_file() and path.stat().st_size == pin['bytes'], 'Existing saved member absent')
        need(pin['sha256'] == adopted[rootkey] == linux[rootkey] == arrays[rootkey], 'Member/whole/array proof mismatch')
        members[name] = dict(pin, path=path.as_posix())
    need(pins['receipt']['sha256'] == members['receipt.json']['sha256']
         and receipt['prediction_arrays_sha256'] == members['validation_predictions.npz']['sha256'], 'Receipt/array link drift')
    # Only the explicit native Benign10 index, not unaccepted future100 jobs.
    need(docs['native_table_root']['root_adopted'] is True
         and docs['native_table_root']['files_sha256']['records.json'] == pins['native_benign10']['sha256'], 'Native10 root/index pin drift')
    old = docs['native_benign10']['records']
    need(len(old) == 10 and [r['seed'] for r in old] == list(range(91001, 91011)), 'Native Benign10 identity scope drift')
    hybrid_accepted = [r for r in root['records'] if r['method'] == 'CosineFairnessHybrid']
    by_checkpoint = {r['checkpoint_sha256']: r for r in hybrid_accepted}
    replayed, missing = [], []
    for record in old:
        existing = by_checkpoint.get(record['checkpoint_sha256'])
        if existing:
            need(existing['id'] == record['id'], 'Checkpoint alias/ID mismatch')
            replayed.append(record)
        else:
            missing.append(record)
    need([r['id'] for r in replayed] == [ID] and len(missing) == 9, 'Unexpected accepted intersection')
    need('torch' not in sys.modules, 'Metadata-only contract imported Torch')
    return dict(status='EXISTING_HYBRID91002_CANARY_REUSE_IDENTITY_CONFIRMED_NO_NEW_REPLAY',
        root_pin=pins['root'], checkpoint_sha256=current['checkpoint']['sha256'], existing_saved_members=members,
        linux_whole_pass_for_record=True, Windows_saved_array_block_pass_for_record=True,
        Windows_whole_root_receipt_exact_for_record=True, Windows_combined_whole_checker_pass=False,
        preserved_Windows_failure_method='FLGMM', shared_calibration=ev.SHARED_CALIBRATION,
        native_raw_rule='margin > 0', shared_rule='margin >= group_threshold', native_max_abs_difference=0,
        already_replayed_Benign_records=replayed, missing_Benign_records=missing,
        remaining_metadata_only=True, future_jobs_created=0, new_forward=0, new_fit=0, new_training=0,
        new_scientific_acceptances=0, duplicate_replay_cancelled=True, dispatch_authorized=False,
        array_bytes_rehashed_by_this_check=False, original_proofs_reused_without_rerunning=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', required=True, action='store_true')
    parser.parse_args()
    print(json.dumps(check(), indent=2, allow_nan=False))
