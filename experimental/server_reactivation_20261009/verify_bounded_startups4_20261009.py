"""Verify copied startup facts; completion and scientific acceptance stay separate."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
def read(path): return json.loads(path.read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def seal(folder, expected):
    assert sha(folder / 'FILES_SHA256.json') == expected
    members = read(folder / 'FILES_SHA256.json')['members']
    for name, row in members.items():
        path = folder / name
        assert path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    return len(members)

seven = ROOT / 'tmp/celeba_mechanism_valid_replay_20261009/remaining_seven_startup_delivery_20261009'
assert seal(seven, '564d349ea15d4dc6af584ce39c7afcbbd59d2a3dfac870eafb412a040c52adb8') == 20
assert sha(seven / 'APPROVED.json') == '270721872c1a8a021d72f114c3300addcaceab4aa9435c4307e5471994101a05'
assert sha(seven / 'sealed_source/FILES_SHA256.json') == '119418ce0ba509c364ad7f8c82361abd2d806d018ace847f6cd46714ea6b431d'
for name, value in read(seven / 'sealed_source/FILES_SHA256.json')['files'].items():
    assert sha(seven / 'sealed_source' / name) == value
s = read(seven / 'startup_receipt.json'); live = read(seven / 'live_20261009T095644Z.json')
assert s['different_host_observed'] and s['source_host'] != s['offserver_verification_host']
assert s['selected_original_terminals'] == 7 and s['excluded'] == 'minus_U_IID_Benign_seed91002'
assert s['actual_nominal_reservations'] == 114 <= s['actual_quota']
assert s['not_measured_cpu_utilization'] and s['new_training'] == s['new_Full_inference'] == s['new_test_inference'] == 0
workers = [row for row in live['processes'] if row['pid'] == s['compute_worker_pid']]
assert len(workers) == 1 and workers[0]['cpus'] == list(range(112, 120))
assert workers[0]['nice'] == 10 and workers[0]['cuda_visible_devices'] == ''
assert workers[0]['user_seconds'] + workers[0]['system_seconds'] > 0 and live['batch_failure'] is None

hybrid = ROOT / 'tmp/celeba_hybrid_realimage_gate_20261009/startup_delivery_20261009'
assert seal(hybrid, '3dc3ed4cbf8b388d7f262fab48fe9149819df1431b26033d88fe3eae9f22bd29') == 20
h = read(hybrid / 'startup_receipt.json'); hlive = read(hybrid / 'live_repair_20261009T095649Z.json')
assert h['APPROVED_sha256'] == sha(hybrid / 'APPROVED.json') == '59b0c3b4298050ee55d9dd2b0d9e9e28c4b3b90390cd3e0c3b79c4a73c6d3bd1'
assert h['source9_seal_sha256'] == '54c67847b44a26a654b380c6c4d03863279261126a61de29489bfa78572e332b'
assert h['first_round']['round'] == 1 and h['first_round']['pid'] == h['actual_python_pid']
assert h['new_completed_CANARY'] == 0 and h['reused_completed_IID_CANARY'] == 2 and h['scientific_table_records'] == 0
assert h['engineering_failures_preserved'] == 2 and h['repeated_IID_training_or_inference'] == 0
assert hlive['new_repair_failure'] is None and hlive['new_repair_summary'] is None
worker = hlive['actual_python_processes'][0]
assert worker['pid'] == h['actual_python_pid'] and worker['cpus'] == list(range(8, 16))
assert worker['nice'] == 10 and worker['cuda_visible_devices'] == ''
proof = dict(status='ROOT_COPIED_STARTUP_SHA_AND_ACTUAL_WORKER_FACTS_PASS',
             verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             seven_delivery_seal_sha256=sha(seven / 'FILES_SHA256.json'), seven_members_verified=20,
             seven_startup_receipt_sha256=sha(seven / 'startup_receipt.json'),
             seven_service_at_observation=live['service'], seven_observed_utc=live['utc'],
             seven_original_terminals=7, seven_accepted_offserver_in_this_proof=0,
             hybrid_delivery_seal_sha256=sha(hybrid / 'FILES_SHA256.json'), hybrid_members_verified=20,
             hybrid_startup_receipt_sha256=sha(hybrid / 'startup_receipt.json'),
             hybrid_service_at_observation=hlive['service'], hybrid_observed_utc=hlive['utc'],
             hybrid_first_round_observed=1, hybrid_complete_cohort_accepted=False,
             original_failure_and_two_engineering_failures_preserved=True,
             scientific_table_records_from_startups=0, test_started=False, goal_complete=False)
with (OUT / 'BOUNDED_STARTUPS4_ROOT_VERIFICATION.json').open('x', encoding='utf8') as stream:
    json.dump(proof, stream, indent=2); stream.write('\n')
print(json.dumps(proof))
