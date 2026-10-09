"""Exercise real accepted arrays and fail-closed remaining-cohort/backup gates."""
import copy
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
import bounded_remaining as run
import audit_remaining as audit

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
tag = '' if len(sys.argv) == 1 else '_attempt' + str(int(sys.argv[1]))
m = run.read(HERE / 'manifest.json')
inventory = run.read(BASE / 'inputs/model_inventory.json')
prior = run.read(BASE / 'v4/execution_20261009/cumulative_28_accepted.json')
run.validate_contract(m, inventory, prior)
rejected = []


def refuse(name, action):
    try:
        action()
    except (ValueError, KeyError, AssertionError) as exc:
        rejected.append({'case': name, 'error_type': type(exc).__name__, 'error': str(exc)})
    else:
        raise AssertionError('Unexpected acceptance: ' + name)


mutations = {
    'duplicate_remaining_ID': lambda x: x['remaining_ids'].__setitem__(1, x['remaining_ids'][0]),
    'missing_seed_ID': lambda x: x['remaining_ids'].pop(),
    'repeat_previously_accepted_ID': lambda x: x['remaining_ids'].__setitem__(0, x['accepted28_ids'][0]),
    'repeated_chunk': lambda x: x['chunks'].append(copy.deepcopy(x['chunks'][0])),
    'threads_changed': lambda x: x.__setitem__('threads_per_worker', 7),
    'unmeasured_12workers': lambda x: x.__setitem__('workers', 12),
    'CPU_slot_changed': lambda x: x['cpu_allowed_list_indices'].__setitem__(0, 0),
    'tolerance_relaxed': lambda x: x.__setitem__('native_tolerance', 1e-11),
    'test_target': lambda x: x.__setitem__('target_split', 'test'),
    'historical_output': lambda x: x.__setitem__('output', '/workspace/GuardFed-celeba-expanded/results'),
    'automatic_retry': lambda x: x.__setitem__('automatic_retry', True),
    'map_hash_drift': lambda x: x['scientific_cli'].__setitem__('storage-map-sha256', '0' * 64),
    'map_wrong_path': lambda x: x['scientific_cli'].__setitem__('storage-map', '/workspace/wrong.json'),
    'prior_archive_drift': lambda x: x['prior_provenance'][0].__setitem__('archive_sha256', '0' * 64),
    'outer_nice10_double_adjustment': lambda x: x.__setitem__('outer_coordinator_nice', 10),
}
for name, mutate in mutations.items():
    bad = copy.deepcopy(m)
    mutate(bad)
    refuse(name, lambda: run.validate_contract(bad, inventory, prior))

cache = BASE / 'verification_inputs/original_valid_cache.npz'
assert run.sha(cache) == audit.CACHE_SHA
with np.load(cache, allow_pickle=False) as z:
    y, sensitive = z['valid_y'], z['valid_sensitive']
old = BASE / 'v4/phase5_attempt1_execution_20261009'
actual = old / 'remote_receipts/phase5_attempt1_useful/runs'
records = {r['id']: r for r in inventory['records']}
checked = []
first = None
for path in sorted(actual.glob('*/receipt.json')):
    receipt = run.read(path)
    with np.load(path.parent / 'validation_predictions.npz', allow_pickle=False) as z:
        audit.check_array_ids(records[receipt['id']], z)
        diff, counts = audit.direct_checks(receipt, z, y, sensitive)
        checked.append({'id': receipt['id'], 'actual_original_receipt_sha256': run.sha(path), 'actual_original_arrays_sha256': run.sha(path.parent / 'validation_predictions.npz'), 'independent_metric_checks': 9, 'independent_count_checks': counts, 'all_metric_differences_zero': all(v == 0 for metrics in diff.values() for v in metrics.values())})
        if first is None:
            first = receipt, {k: z[k].copy() for k in z.files}
assert len(checked) == 11 and all(r['all_metric_differences_zero'] for r in checked)
r, arrays = first
bad = copy.deepcopy(arrays)
bad['root_image_ids'][0] += 1
refuse('root_ID_tampered', lambda: audit.check_array_ids(records[r['id']], bad))
bad = copy.deepcopy(arrays)
bad['valid_image_ids'][0] += 1
refuse('valid_ID_tampered', lambda: audit.check_array_ids(records[r['id']], bad))
bad = copy.deepcopy(r)
bad['views']['raw']['group_confusion_counts']['0']['tp'] += 1
refuse('confusion_count_tampered', lambda: audit.direct_checks(bad, arrays, y, sensitive))
bad = copy.deepcopy(r)
bad['fits']['shared_calibration']['thresholds']['0'] = 1e6
refuse('threshold_tampered', lambda: audit.direct_checks(bad, arrays, y, sensitive))
bad = copy.deepcopy(r)
bad['fits']['shared_calibration']['rule'] = 'unknown'
refuse('unknown_prediction_rule', lambda: audit.direct_checks(bad, arrays, y, sensitive))
c = audit.collector()
refuse('wrong_archive_SHA', lambda: c.verified_archive(old / 'offserver_verification.json', {'archive_sha256': '0' * 64}))
refuse('partial28_is_not_complete900_statistics', lambda: audit.summarize(BASE / 'v4/execution_20261009/collection_inputs_28.json', HERE / 'MUST_NOT_EXIST_partial_summary'))
assert not (HERE / 'MUST_NOT_EXIST_partial_summary').exists()

# Exercise the real archiver with exact, already accepted inputs. This is a
# structural roundtrip fixture, not new inference or an additional acceptance.
fixture = HERE / ('selfcheck_inputs/archiver_roundtrip' + tag)
fixture.mkdir(parents=True)
run.save(fixture / 'strict_acceptance.json', run.read(old / 'strict_acceptance.json'))
test_m = copy.deepcopy(m)
test_m['source_archive'] = {'sealed_sources/replay_v4.py': str(BASE / 'v4/replay_v4.py'), 'sealed_sources/replay_v3.py': str(BASE / 'v3/replay_v3.py')}
batch = old / 'remote_receipts/phase5_attempt1_useful'
backup = run.archive_chunk(test_m, HERE / 'manifest.json', fixture, batch, 0, [x['id'] for x in checked])
_, data = c.verified_archive(fixture / 'offserver_verification.json', {'archive_sha256': backup['sha256']})
assert len(data) == backup['member_n'] and not any(name.endswith('/model.pt') for name in data)
report = {'status': 'PASS', 'scope': 'LOCAL_STRUCTURAL_AND_EXISTING_REAL_PHASE5_ARRAY_REGRESSION_ONLY', 'new_inference': 0, 'new_scientific_acceptances': 0, 'exact_remaining872_contract': 'PASS', 'actual_previous_phase5_arrays_verified_n': len(checked), 'independent_metric_checks': sum(r['independent_metric_checks'] for r in checked), 'independent_confusion_count_checks': sum(r['independent_count_checks'] for r in checked), 'actual_inputs': checked, 'refused_n': len(rejected), 'refusals': rejected, 'real_evidence_archiver_roundtrip_members': backup['member_n'], 'real_evidence_archiver_roundtrip_sha256': backup['sha256'], 'Linux_supervisor_orchestration_executed': False, 'all900_native_valid_replayed': False, 'source_sha256': {p: run.sha(HERE / p) for p in ('bounded_remaining.py', 'audit_remaining.py', 'selfcheck.py')}, 'manifest_sha256': run.sha(HERE / 'manifest.json')}
run.save(HERE / ('selfcheck' + tag + '.json'), report)
print(json.dumps({k: report[k] for k in ('status', 'actual_previous_phase5_arrays_verified_n', 'independent_metric_checks', 'independent_confusion_count_checks', 'refused_n', 'real_evidence_archiver_roundtrip_members')}, indent=2))
