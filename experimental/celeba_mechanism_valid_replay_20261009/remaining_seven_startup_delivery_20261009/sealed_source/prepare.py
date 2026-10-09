"""Prepare only the seven already accepted original terminals, never dispatch."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREPARED = HERE.parent
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
inventory = read(PREPARED / 'inventory_actual8_Full100refs.json')
assert digest(PREPARED / 'inventory_actual8_Full100refs.json') == 'be4d1d34b21443572a25e6a189710e8d75b108a8fcbdb58896a71f2ba1d88cc8'
selected = ['minus_U_IID_Benign_seed' + str(s) for s in (91001,91003,91004,91005,91006,91007,91008)]
records = {row['id']:row for row in inventory['records']}
remote = '/workspace/guardfed_checks/celeba_mechanism_valid_replay_20261009/remaining_seven_prepared_20261009'
scope = dict(status='PREPARED_NOT_APPROVED', selected_ids=selected,
    original_inventory_sha256=digest(PREPARED / 'inventory_actual8_Full100refs.json'),
    original_bridge_sha256=digest(PREPARED / 'bridge.py'), views=['native','raw','shared_calibration'], native_tolerance=1e-12,
    selected_records=[dict(id=identity, checkpoint_sha256=records[identity]['checkpoint']['sha256'],
        original_result_sha256=records[identity]['result']['sha256'], original_job_sha256=records[identity]['raw_job']['sha256'],
        paired_full_reference=records[identity]['paired_full']) for identity in selected],
    outputs={identity:remote + '/runs/' + identity for identity in selected},
    dependency_paths=read(PREPARED / 'execution_attachments_v2/dispatch_receipt.PREPARED.json')['dependency_paths'],
    already_accepted_excluded='minus_U_IID_Benign_seed91002', original_pending792_unchanged=True,
    new_training=False, test_authorized=False, new_full_inference=0, Full_weights_repacked=0,
    cpu_ids=list(range(112,120)), compute_threads=8, max_processes=1,
    paired_Full_display=dict(seed91001='SOURCE_BOUND_EXISTING_THREE_VIEW_RECEIPT_REFERENCE_ONLY',
        seed91003_to_91008='MISSING_THREE_VIEW_REPLAY_PENDING_900_ACTUAL_ACCEPTANCE',
        original_Full100_native='STRICT_ORIGINAL_TRAINING_RESULTS_REFERENCED',
        missing_paired_display_does_not_invalidate_mechanism_terminal=True),
    final_protocol_status='PREPARED_NOT_FROZEN', inference_runs_performed_by_preparation=0)
for name, value in [('SCOPE.json',scope), ('APPROVAL_TEMPLATE.json',dict(status='PREPARED_NOT_APPROVED',
    scope_sha256=None, execution_seal_sha256=None, selected_ids=selected, outputs=scope['outputs'],
    dependency_paths=scope['dependency_paths'], inventory_sha256=scope['original_inventory_sha256'],
    bridge_sha256=scope['original_bridge_sha256'], allowed_cpus=scope['cpu_ids'], compute_threads=8,max_processes=1,
    target_split='valid',native_tolerance=1e-12,final_test_dispatch=False,new_full_inference=0,automatic_retry_authorized=False))]:
    path=HERE/name; assert not path.exists(); path.write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
print('PREPARED seven existing accepted terminal records; no training or inference')
