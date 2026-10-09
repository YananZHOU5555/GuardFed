"""Root adoption check of the already executed, sealed off-server verifier."""
from pathlib import Path
import datetime
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_valid_gpu_remaining464_evidence_20261009'
OUT = BASE / 'chunk_000'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
assert sha(BASE / 'PACKAGE_SHA256.json') == '52f46820fd3d01ea532af1ec70f7a8c731116662541c85307549730ac20f609b'
for name, row in read(BASE / 'PACKAGE_SHA256.json')['members'].items():
    assert sha(BASE / name) == row['sha256'] and (BASE / name).stat().st_size == row['bytes']
assert sha(ROOT / 'tmp/celeba_valid_gpu_recovery_execution_20261009/cumulative_436_accepted.json') == sha(BASE / 'inputs/base436.json')
prior = read(BASE / 'inputs/base436.json')
collector = read(OUT / 'cumulative_447_accepted.json')
proof = read(OUT / 'ROOT_OFFSERVER_VERIFICATION.json')
inventory = read(OUT / 'remote_archive_inventory.json')
plan = read(ROOT / 'tmp/celeba_valid_gpu_remaining464_prepared_20261009/manifest.json')
assert collector['accepted_ids'] == prior['accepted_ids'] + plan['chunks'][0]['ids']
assert collector['accepted_n'] == len(set(collector['accepted_ids'])) == 447
assert sha(OUT / 'ROOT_OFFSERVER_VERIFICATION.json') == collector['new_proof_sha256'] == '1b4b479a186294538744561d840942eeb1d33907ee10662703067b2032618fec'
assert proof['saved_metrics_verified'] == 99 and proof['saved_confusion_counts_verified'] == 264 and proof['saved_prediction_rules_verified'] == 33
assert proof['native_max_abs_difference'] == 0 and proof['local_root_refit'] is False and proof['root_refit_verified_in_original_remote_strict']
assert proof['new_CNN_inference'] == 0 and proof['final_test'] is False
archive = OUT / 'chunk_evidence.tar.gz'
assert sha(archive) == proof['archive_sha256'] == inventory['sha256']
with tarfile.open(archive) as bundle:
    members = bundle.getmembers()
    assert len(members) == len(set(bundle.getnames())) == proof['archive_members_verified']
    for member in members:
        data = bundle.extractfile(member).read()
        assert hashlib.sha256(data).hexdigest() == inventory['members'][member.name]['sha256']
        assert len(data) == inventory['members'][member.name]['bytes']
report = dict(status='ROOT_CHUNK0_ADOPTION_SOURCE_CHAIN_ARCHIVE_PASS',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), verifier_package_sha256=sha(BASE / 'PACKAGE_SHA256.json'),
    accepted_n=447, new_n=11, CPU_provenance_n=434, GPU_provenance_n=13,
    proof_sha256=sha(OUT / 'ROOT_OFFSERVER_VERIFICATION.json'), collector_sha256=sha(OUT / 'cumulative_447_accepted.json'),
    archive_sha256=sha(archive), member_n=len(members), original436_unchanged=True,
    root_refit_original_remote_strict_only=True, uniform_device_comparison=False, test=False, whole_goal_complete=False)
with (OUT / 'ROOT_ADOPTION_REVIEW.json').open('x', encoding='utf8', newline='\n') as stream:
    json.dump(report, stream, indent=2); stream.write('\n')
print(json.dumps(report))
