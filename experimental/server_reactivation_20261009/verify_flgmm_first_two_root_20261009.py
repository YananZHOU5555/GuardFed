"""Root member and frozen single-result acceptance of two completed FLGMM jobs."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import tarfile
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_flgmm_screen_20261009_v2_dispatch'
FOLDER = BASE / 'backups/first_two_20261009'
RELEASE = ROOT / 'tmp/celeba_flgmm_screen_20261009_v2_frozen_release'
def read(path): return json.loads(path.read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(FOLDER / 'OFFSERVER_ACCEPTANCE.json') == 'a53c87cbe62cdcc206dddb18cec0f62426292c05adb28a4a00de03fa64fe3750'
receipt = read(FOLDER / 'BACKUP_SHA256.json')
assert sha(FOLDER / 'accepted_first_two.tar.gz') == receipt['archive_sha256'] == '638d7bba07fcc1f3a9953a8f71fb412516d8b5ffcc134eb3b3794cf82dedcbbf'
assert sha(FOLDER / 'MEMBERS.json') == receipt['inventory_sha256'] == 'e95f2e7cf9a36ebc0e717c48556be05a56c0783ceb597bb8d87491d0a207f5d8'
assert sha(FOLDER / 'PARTIAL_ACCEPTANCE.json') == receipt['acceptance_sha256']
specs = read(FOLDER / 'MEMBERS.json')['members']
with tarfile.open(FOLDER / 'accepted_first_two.tar.gz') as bundle:
    members = bundle.getmembers()
    assert len(members) == len({m.name for m in members}) == 22
    assert set(bundle.getnames()) == set(specs) | {'MEMBERS.json'}
    for member in members:
        assert member.isfile()
        data = bundle.extractfile(member).read()
        spec = specs.get(member.name, dict(sha256=receipt['inventory_sha256'], size=(FOLDER / 'MEMBERS.json').stat().st_size))
        assert len(data) == spec['size'] and hashlib.sha256(data).hexdigest() == spec['sha256']
partial = read(FOLDER / 'PARTIAL_ACCEPTANCE.json')
assert partial['before_source_data'] == partial['after_source_data']
assert not partial['candidate_selection_performed'] and not partial['final_test'] and not partial['formal100']
sys.path.insert(0, str(RELEASE))
from screen_common import local_identity, accepted
protocol, manifest = local_identity()
assert sha(RELEASE / 'PACKAGE_SHA256.json') == receipt['package_sha256'] == 'aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'
records = []
for identity in receipt['accepted_new_ids']:
    item = next(row for row in manifest['jobs'] if row['id'] == identity)
    result = accepted(item, FOLDER / 'restored/runs' / identity)
    assert result is not None and result['rounds'] == 70 and result['seed'] == 91001
    assert result['config']['celeba_evaluation_split'] == 'valid'
    records.append(dict(id=identity, checkpoint_sha256=sha(FOLDER / 'restored/runs' / identity / 'model.pt'), metrics=result['metrics']))
assert len(records) == 2
proof = dict(status='ROOT_ARCHIVE_AND_FROZEN_SINGLE_RESULT_ACCEPTANCE_PASS',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), accepted=2, planned=32,
    archive_sha256=receipt['archive_sha256'], archive_members_verified=22, records=records,
    offserver_acceptance_sha256=sha(FOLDER / 'OFFSERVER_ACCEPTANCE.json'),
    original_single_result_checker_replayed=True, no_new_training_or_image_inference=True,
    candidate_selection_performed=False, formal100=False, test_started=False, goal_complete=False)
out = ROOT / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/FLGMM_FIRST_TWO_ROOT_VERIFICATION.json'
with out.open('x', encoding='utf8') as stream:
    json.dump(proof, stream, indent=2); stream.write('\n')
print(json.dumps(proof))
