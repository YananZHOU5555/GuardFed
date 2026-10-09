"""Restore only exact nonFull artifacts inside the dedicated isolated directory."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

BASE = Path('/workspace/guardfed_checks/celeba_validation900_restore_20261009')
REPO = Path('/workspace/GuardFed-celeba-expanded')
INV_SHA = '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def target_for_member(name):
    rel = PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] == 'artifact_store'
    target = BASE / str(rel)
    assert target.resolve().is_relative_to((BASE / 'artifact_store').resolve())
    assert target != BASE / 'artifact_store'
    return target


def run(receipt_sha, inventory_path):
    receipt_path = BASE / 'restore_bundle.json'
    assert digest(receipt_path) == receipt_sha
    receipt = json.loads(receipt_path.read_text())
    assert receipt['inventory_sha256'] == INV_SHA and digest(inventory_path) == INV_SHA
    inventory = json.loads(inventory_path.read_text())
    by_id = {r['id']: r for r in inventory['records']}
    assert len(by_id) == 900
    map_path = BASE / 'storage_map.json'
    assert digest(map_path) == receipt['storage_map_sha256']
    mapping = json.loads(map_path.read_text())
    assert mapping['inventory_sha256'] == INV_SHA and len(mapping['records']) == 2700
    assert len({(r['id'], r['kind']) for r in mapping['records']}) == 2700
    assert receipt['artifact_root'] == str(BASE) and mapping['artifact_root'] == str(BASE)
    assert not (BASE / 'restore_acceptance.json').exists(), 'Keep any previous acceptance'
    full_checked = 0
    targets = {}
    for row in mapping['records']:
        original = by_id[row['id']]
        obj = original[row['kind']]
        assert row['method'] == original['method']
        assert all(row[k] == obj[k] for k in ('archive', 'archive_sha256', 'member', 'sha256', 'bytes'))
        target = Path(row['target'])
        if row['existing_full_reuse']:
            assert original['method'] == 'GuardFed-AD2+'
            expected = REPO / obj['member'] if row['kind'] == 'raw_job' else Path(original['original_remote_output']) / ('model.pt' if row['kind'] == 'checkpoint' else 'result.json')
            assert target == expected and target.is_file()
            assert target.stat().st_size == row['bytes'] and digest(target) == row['sha256']
            full_checked += 1
        else:
            assert original['method'] != 'GuardFed-AD2+'
            relative = target.relative_to(BASE).as_posix()
            assert target_for_member(relative) == target
            assert relative in receipt['members']
            assert receipt['members'][relative] == {'sha256': row['sha256'], 'bytes': row['bytes']}
            assert target not in targets
            targets[target] = row
    assert full_checked == 300 and len(targets) == len(receipt['members']) == 2400
    bundle = BASE / receipt['archive']
    assert bundle.parent == BASE and digest(bundle) == receipt['sha256']
    # Full preflight of the entire archive before creating any artifact file.
    seen, existing = set(), 0
    with tarfile.open(bundle, 'r|gz') as archive:
        for member in archive:
            target = target_for_member(member.name)
            assert member.isfile() and target in targets and target not in seen
            seen.add(target)
            row = targets[target]
            data = archive.extractfile(member).read()
            assert member.size == len(data) == row['bytes']
            assert hashlib.sha256(data).hexdigest() == row['sha256']
            if target.exists():
                assert target.is_file() and target.stat().st_size == row['bytes'] and digest(target) == row['sha256']
                existing += 1
    assert seen == set(targets)
    created = 0
    with tarfile.open(bundle, 'r|gz') as archive:
        for member in archive:
            target = target_for_member(member.name)
            if target.exists():
                continue
            data = archive.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest() == targets[target]['sha256']
            target.parent.mkdir(parents=True, exist_ok=True)
            assert target_for_member(member.name) == target
            with target.open('xb') as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            created += 1
    for row in mapping['records']:
        target = Path(row['target'])
        assert target.stat().st_size == row['bytes'] and digest(target) == row['sha256']
    result = {'status': 'EXACT_900_ARTIFACT_STORAGE_VERIFIED',
              'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'inventory_sha256': INV_SHA, 'receipt_sha256': receipt_sha,
              'storage_map_sha256': digest(map_path), 'bundle_sha256': digest(bundle),
              'installer_sha256': digest(Path(__file__)), 'models': 900,
              'existing_full_files_verified': full_checked, 'isolated_members_created': created,
              'identical_isolated_members_skipped': existing, 'all_artifact_file_hashes_verified': 2700,
              'original_output_files_modified': 0, 'new_training': 0, 'new_inference': 0,
              'scientific_acceptance_scope': 'Preserved archive/member identities only; valid replay still required'}
    with (BASE / 'restore_acceptance.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args()
    run(args.receipt_sha256, args.inventory)
