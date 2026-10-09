"""Restore twelve byte-identical provenance sources in an isolated subdirectory."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import tarfile
import datetime

CHECKS = Path('/workspace/guardfed_checks/celeba_validation900_restore_20261009/adapter_sources_v1')
PIN = '1c57cbd6901e3ef9c03f16817d522974cf829abb2b74ca380cb6120c55daf05d'
TARGET = Path('/workspace/GuardFed-celeba-expanded/deployment/baseline_adapters_20260928/validation_replay_exact_sources_20261009')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(CHECKS / 'receipt.json') == PIN
receipt = json.loads((CHECKS / 'receipt.json').read_text())
assert receipt['remote_target'] == str(TARGET)
assert receipt['inventory_sha256'] == '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
assert not (CHECKS / 'restored12.json').exists()
archive = CHECKS / receipt['archive']
assert archive.parent == CHECKS and sha(archive) == receipt['archive_sha256']
expected = {r['member']: r for r in receipt['members'].values()}
assert len(expected) == 12
payloads = []
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == 12 and {m.name for m in members} == set(expected)
    for member in members:
        relative = PurePosixPath(member.name)
        assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
        row = expected[member.name]
        assert relative.parts[0] == row['sha256'] and len(relative.parts) == 2
        target = TARGET / member.name
        assert target.resolve().is_relative_to(TARGET.resolve())
        data = stream.extractfile(member).read()
        assert len(data) == row['bytes'] == member.size and hashlib.sha256(data).hexdigest() == row['sha256']
        if target.exists():
            assert target.is_file() and sha(target) == row['sha256']
        payloads.append((target, data, row))
created = 0
for target, data, row in payloads:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open('xb') as handle:
            handle.write(data)
        created += 1
    assert sha(target) == row['sha256']
report = {'status': 'EXACT12_MISSING_ADAPTER_SOURCES_RESTORED',
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'receipt_sha256': PIN, 'archive_sha256': sha(archive), 'installer_sha256': sha(Path(__file__)),
    'created': created, 'identical_skipped': 12 - created, 'verified_sources': 12,
    'target': str(TARGET), 'original_files_modified': 0, 'new_inference': 0, 'new_training': 0,
    'restored_files': {str(p): r['sha256'] for p, _, r in payloads}}
with (CHECKS / 'restored12.json').open('x') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
print(json.dumps(report))
