"""Adopt three measured interfaces; retain the failed Windows audit check."""
from pathlib import Path
import datetime, hashlib, json, subprocess

R = Path(__file__).resolve().parents[1]
O = R / 'tmp/celeba_added_cnn_exact3_root_execution_20261010'
F = Path('F:/YananResearchStorage/GuardFed/added_cnn_exact3_valid_20261010/attempt001')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
pins = {
    'ROOT_SOURCE_REVIEW.json': 'e81d178f0390ca9c5be0939a0c62ff9da762b2b3dc59f51b9b8d82b26f878323',
    'LINUX_EXACT_ROOT_CHECK.json': '928769533c2b3cf15afdfcb4beab31c9c2059441e1bc7837a1693756ee379ecf',
    'OFFSERVER_ARRAY_REFIT_CHECK.json': 'ed94f63646c6178bf941058b8ee83f56ba0b38a83f708fc4ae206d0fedd99f00',
}
for name, h in pins.items():
    assert sha(O / name) == h, name
assert not (O / 'ROOT_SCIENTIFIC_ADOPTION.json').exists()
volume = json.loads(subprocess.check_output([
    'powershell', '-NoProfile', '-Command',
    "Get-Volume -DriveLetter F | Select-Object DriveLetter,FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress"
], text=True))
assert volume['FileSystemLabel'] == 'Yanan 2TB' and volume['HealthStatus'] == 'Healthy'
assert volume['SizeRemaining'] > 1024**3
t = read(O / 'TRANSPORT_VERIFICATION.json')
assert sha(F / 'exact3_saved_arrays_and_metadata.zip') == t['archive_sha256'] == '74ea40f40f49c76356710a359144a9fef1789b1c7eeb9addda65db127d7dbb61'
for name, pin in t['members'].items():
    p = F / 'verified_extract' / name
    assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], name
gate = read(F / 'verified_extract/bundle/GATE_RESULT.json')
linux, local = read(O / 'LINUX_EXACT_ROOT_CHECK.json'), read(O / 'OFFSERVER_ARRAY_REFIT_CHECK.json')
failed, diag = read(O / 'SCIENCE_EXIT.json'), read(O / 'OFFSERVER_ROOT_DIAGNOSTIC.json')
assert gate['status'] == 'EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED'
assert linux['status'] == 'LINUX_ORIGINAL_EXACT3_SAVED_ROOT_AND_ARRAY_CHECK_PASS_NOT_ROOT_ADOPTED'
assert local['status'] == 'OFFSERVER_ORIGINAL_ARRAY_FIT_METRICS_BLOCK_PASS_FULL_ROOT_AUDIT_REMAINS_LINUX_ONLY'
assert read(O / 'LINUX_EXACT_ROOT_EXIT.json')['exit'] == 0 and failed['exit'] == 1
assert sha(O / 'SCIENCE_STDERR.txt') == failed['stderr_sha256']
assert 'Root identity or model weights drift' in (O / 'SCIENCE_STDERR.txt').read_text()
ids = read(O / 'ROOT_SOURCE_REVIEW.json')['exact_ids']
assert [r['id'] for r in gate['receipts']] == ids
assert [r['id'] for r in linux['records']] == ids == [r['id'] for r in local['records']]
assert len(ids) == 3 and len(set(ids)) == 3
assert linux['original_check_saved_sha256'] == local['original_saved_check_file_sha256'] == 'd512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
assert linux['native_tolerance_unchanged'] == local['native_tolerance_unchanged'] == 1e-12
rows = []
for g, a, b, d in zip(gate['receipts'], linux['records'], local['records'], diag['records']):
    assert g['id'] == a['id'] == b['id'] == d['id']
    assert a['root_receipt_exact'] and a['cached_root_fit_exact'] and a['saved_predictions_metrics_counts_exact']
    assert b['root_ID_partition_original_assertions_pass'] and b['root_fit_parameters_and_diagnostics_exact'] and b['all_saved_predictions_and_counts_and_metrics_exact']
    assert g['weights_before'] == g['weights_after']
    assert not any(g[k] for k in ('optimizer_created', 'gradients_created', 'test_labels_accessed', 'test_inference_performed', 'final_dispatch_created'))
    assert g['valid_n'] == 19867 and g['root_reconstruction']['root_n'] == 16277
    assert set(g['views']) == {'native', 'raw', 'shared_calibration'}
    assert g['runtime']['device'] == 'cpu' and g['runtime']['torch_threads'] == 8
    for r in (g, a, b):
        assert r['native_comparison']['accepted'] and r['native_comparison']['max_abs_difference'] == 0.0
    assert a['array_sha256'] == b['array_sha256'] == g['prediction_arrays_sha256']
    assert a['checkpoint_sha256'] == g['checkpoint_sha256']
    assert a['receipt_sha256'] == b['receipt_sha256']
    rows.append(dict(id=g['id'], method=g['method'], distribution=g['distribution'], attack=g['attack'], seed=g['seed'], checkpoint_sha256=g['checkpoint_sha256'], receipt_sha256=a['receipt_sha256'], array_sha256=a['array_sha256'], native_max_abs_difference=0.0, linux_whole_original_check_exact=True, offserver_array_fit_predictions_metrics_exact=True, windows_whole_root_receipt_exact=b['local_full_root_receipt_exact'], preserved_root_audit_differences=b['preserved_root_audit_differences']))
diff = rows[0]['preserved_root_audit_differences']
assert len(diff) == 1 and diff[0]['path'] == '/server_sampling_audit/group_kl'
assert diff[0]['difference'] == -2.168404344971009e-19
assert not rows[0]['windows_whole_root_receipt_exact']
assert all(r['windows_whole_root_receipt_exact'] and not r['preserved_root_audit_differences'] for r in rows[1:])
proof_names = list(pins) + ['AUTHORIZATION.json', 'START_RECEIPT.json', 'SOURCE_DEPLOYMENT.json', 'ROOT_OFFSERVER_SOURCE_REVIEW.json', 'TRANSPORT_VERIFICATION.json', 'SCIENCE_COMMAND.json', 'SCIENCE_EXIT.json', 'SCIENCE_STDERR.txt', 'OFFSERVER_ROOT_DIAGNOSTIC.json', 'LINUX_EXACT_ROOT_EXIT.json']
proof = dict(
    status='ROOT_EXACT3_VALID_INTERFACES_ADOPTED_WITH_PRESERVED_WINDOWS_AUDIT_FAILURE',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), root_adoption=True,
    interface_records_accepted=3, mechanism_three_view_cutoff_unchanged=251,
    final_test=False, cross_platform_audit_failure_preserved=True,
    exact_ids=ids, records=rows, metric_values_checked=27, integer_base_counts_checked=72,
    prediction_rules_checked=9, native_max_abs_difference=0.0,
    Linux_whole_original_saved_check_pass=True, Windows_whole_saved_check_pass=False,
    Windows_original_array_refit_block_pass=True, original_tolerance_unchanged=1e-12,
    original_saved_check_file_sha256=linux['original_check_saved_sha256'],
    exact_saved_array_AST_block_sha256=local['exact_saved_array_AST_block_sha256'],
    F_archive_path=str(F / 'exact3_saved_arrays_and_metadata.zip'), archive_sha256=t['archive_sha256'], archive_members=12,
    storage=volume, proof_files_sha256={name: sha(O / name) for name in proof_names},
    adoption_source_path=Path(__file__).relative_to(R).as_posix(), adoption_source_sha256=sha(__file__),
    actual_interface_inference_records=3, new_training=0, original_cached_root_refits=3,
    verification_cached_root_refits=6, test_labels_accessed=False, test_inference_performed=False,
    full100_complete=False, full17_complete=False, final_primary_endpoint_decided=False,
    limitations=[
        'Three representative accepted checkpoints only; FedNGA is one search candidate, not a chosen recipe.',
        'Full root receipt equality passes on Linux; Windows whole checker failed at FLGMM group_kl before fitting, and is not relabelled as passed.',
        'The measured audit-field last-bit difference is preserved; its precise platform/library cause has not been experimentally isolated.',
        'The unchanged original saved-array block independently reproduces root-only fitting, saved predictions, all metrics and base counts on checked F storage.',
        'No tolerance, method, seed, selection rule, checkpoint, training queue or final-test boundary changed.',
        'This does not adopt all added-method records or complete their multi-seed comparison.'
    ])
with (O / 'ROOT_SCIENTIFIC_ADOPTION.json').open('x', encoding='utf-8') as f:
    json.dump(proof, f, ensure_ascii=False, indent=2)
print(json.dumps(dict(status=proof['status'], accepted=3, sha256=sha(O / 'ROOT_SCIENTIFIC_ADOPTION.json'))))
