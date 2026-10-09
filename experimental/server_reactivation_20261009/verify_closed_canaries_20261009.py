"""Independent off-server archive and delivery verification; no training imports."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('frozen_v4', ROOT / 'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py')
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)
assert sha(Path(spec.origin)) == '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'

p = ROOT / 'tmp/celeba_mechanism_valid_replay_20261009/recovery_attempt03_20261009T085000Z'
seal = read(p / 'FILES_SHA256.json')
assert sha(p / 'FILES_SHA256.json') == '4b3ec4848a7833b1c8b60ac395c93f25fad91c78f983ac9e5a0bcacefcda72ff'
for name, expected in seal['members'].items():
    assert sha(p / name) == expected['sha256'] and (p / name).stat().st_size == expected['bytes']
backup = p / 'accepted_one_backup_20261009'
receipt = read(backup / 'backup_receipt.json')
v4.verify_archive(backup / 'accepted_one_valid_replay.tar.gz', receipt)
independent = read(p / 'independent_saved_array_verification.json')
assert independent['independent_metric_checks'] == 9 and independent['independent_confusion_count_checks'] == 24
assert independent['prediction_rule_checks'] == 3 and independent['native_max_abs_difference'] == 0
assert not independent['new_Full_inference'] and not independent['new_training'] and not independent['test_inference']
proof = dict(status='ROOT_DELIVERY_ARCHIVE_AND_SAVED_ARRAY_PROOFS_VERIFIED',
    observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), sealed_members=14,
    archive_sha256=receipt['archive_sha256'], archive_members_verified=14,
    accepted_new_ids=receipt['accepted_new_ids'], delivery_seal_sha256=sha(p/'FILES_SHA256.json'),
    independent_proof_sha256=sha(p/'independent_saved_array_verification.json'),
    source_host=receipt['source_host'], new_Full_inference=False, new_training=False, test_inference=False)
with (OUT / 'MECHANISM_VALID_ONE_ROOT_VERIFICATION.json').open('x', encoding='utf8') as f:
    json.dump(proof, f, indent=2); f.write('\n')
print(json.dumps({k: proof[k] for k in ['status','sealed_members','archive_members_verified','accepted_new_ids']}))

p = ROOT / 'tmp/celeba_gradient_realimage_gate_20261009/completed_four_backup_20261009'
delivery = read(p/'strict_delivery.json')
assert delivery['status'] == 'FOUR_EXPLORATORY_REAL_IMAGE_CANARIES_STRICT_PASS'
assert delivery['real_image_runs'] == 4 and delivery['actual_rounds'] == 12 and delivery['scientific_table_records'] == 0
for row in delivery['records']:
    assert row['real_same_point_client_gradients'] == 60 and row['attack_sign_oracle_exact']
    assert row['local_optimizer_steps'] == 0
archives = list(p.glob('*.tar.gz'))
assert len(archives) == 1
receipts = [q for q in p.glob('*.json') if 'archive_sha256' in read(q) and 'inventory_sha256' in read(q)]
assert len(receipts) == 1
receipt = read(receipts[0])
assert receipt['archive_sha256'] == '354d430da8c60a116d5392a6361afbeffb4b1764dfb0167cb4aded45fce01a76'
v4.verify_archive(archives[0], receipt)
proof = dict(status='ROOT_GRADIENT_FOUR_CANARIES_ARCHIVE_VERIFIED',
    observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), canaries=4,
    actual_rounds=12, same_point_gradient_checks=240, archive_members_verified=86,
    archive_sha256=receipt['archive_sha256'], delivery_sha256=sha(p/'strict_delivery.json'),
    offserver_proof_sha256=sha(p/'offserver_verification.json'), scientific_table_records=0)
with (OUT / 'GRADIENT_FOUR_ROOT_VERIFICATION.json').open('x', encoding='utf8') as f:
    json.dump(proof, f, indent=2); f.write('\n')
print(json.dumps({k: proof[k] for k in ['status','canaries','archive_members_verified','scientific_table_records']}))
