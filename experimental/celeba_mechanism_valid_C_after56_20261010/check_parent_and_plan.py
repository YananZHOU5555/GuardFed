"""Read-only parent verification for the pending exact4 construction plan."""
from pathlib import Path
import hashlib, json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.with_name('celeba_mechanism_valid_C_after50_20261010')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
SELECTED = [f'minus_C_non-IID_Benign_seed{s}' for s in range(91007, 91011)]
PINS = {
    'FILES_SHA256.json': '6e79e65b34a1ff78628884993aa6089fdc8c5d07951ed97ce016ad9b3ca6aaf0',
    'execution_candidate/EXECUTION_SOURCE_SHA256.json': '073bfde67b2286f6e29fe5f6e2c1f7580f74465b4b0a4e42c583ab0332aa6595',
    'inventory_actual156_Full100refs.json': '2131f2386cb3a851f990d6ed4baa38f5ea90c9acadcd62fad50d24bf97277bcd',
    'execution_candidate/backups/incremental_20261010T002635Z/ROOT_ADOPTION_REVIEW.json': 'a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080',
}
for name, digest in PINS.items():
    assert sha(PARENT/name) == digest, name
for name in ['FILES_SHA256.json', 'execution_candidate/EXECUTION_SOURCE_SHA256.json']:
    folder = (PARENT/name).parent
    for row in read(PARENT/name)['members']:
        p = folder/row['path']
        assert sha(p) == row['sha256'] and p.stat().st_size == row['size'], str(p)
inventory = read(PARENT/'inventory_actual156_Full100refs.json')
ids = [r['id'] for r in inventory['records']]
assert len(ids) == len(set(ids)) == 156 and not set(ids).intersection(SELECTED)
assert len(inventory['full_references']) == 100
adoption = read(PARENT/'execution_candidate/backups/incremental_20261010T002635Z/ROOT_ADOPTION_REVIEW.json')
assert adoption['cumulative_three_view_models'] == 156 and adoption['accepted_new'] == 6
assert adoption['original150_unchanged'] and adoption['all_native_differences_zero']
print(json.dumps(dict(status='PLAN_ONLY_WAITING_FOR_ROOT_ACTUAL_NATIVE_DELTA', current_native_accepted=156,
    current_three_view_accepted=156, expected_selected_ids=SELECTED, expected_scope_n=4,
    actual_next_native_root_proof=None, actual_next_native_ledger=None, actual_new_inventory_created=False,
    source_ready=False, external_approval_created=False, new_CNN=0, SSH=False, STATE_changed=False, Git_changed=False)))
