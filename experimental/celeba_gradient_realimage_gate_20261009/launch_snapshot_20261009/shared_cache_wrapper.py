"""Exact registered shared-cache physical paths; keep the 22-file gate sealed."""
import argparse
from pathlib import Path

import gate

ORIGINAL_HASHES = gate.hashes


def bind_shared_paths():
    binding = gate.read(gate.HERE / 'shared_cache_bindings.json')
    gate.require(binding['status'] == 'APPROVED_EXACT_SHARED_INPUT_TARGETS', 'Shared-cache registration not approved')
    scope = gate.read(gate.HERE / 'scope.json')
    expected_names = {
        'data/celeba/derived/rgb64_v1/manifest.json', 'data/celeba/derived/rgb64_v1/metadata.npz',
        'data/celeba/derived/rgb64_v1/images.npy', 'data/celeba/derived/rgb64_v1/available.npy'}
    if binding.get('official_metadata_shared_paths_authorized') is True:
        expected_names |= {'data/celeba/list_attr_celeba.txt', 'data/celeba/list_eval_partition.txt'}
    gate.require(set(binding['bindings']) == expected_names, 'Unexpected external path registration')
    for name, row in binding['bindings'].items():
        gate.require(row['resolved_absolute'] == '/workspace/GuardFed-revision/' + name and
                     row['expected_sha256'] == scope['protected_source_hashes'][name], 'Target/source SHA registration differs')

    def bound_hashes(root, expected):
        root = Path(root).resolve()
        approved_path = gate.HERE / 'dispatch_receipt.APPROVED.json'
        if approved_path.exists():
            ORIGINAL_HASHES(gate.HERE, gate.read(approved_path)['execution_attachment_sha256'])
        if root != gate.REPO:
            return ORIGINAL_HASHES(root, expected)
        actual = {}
        for name, sha in expected.items():
            if name not in binding['bindings']:
                actual.update(ORIGINAL_HASHES(root, {name: sha})); continue
            row = binding['bindings'][name]
            gate.require(not Path(name).is_absolute() and '..' not in Path(name).parts and sha == row['expected_sha256'],
                         'Shared path cannot change logical name/SHA')
            path = root / name
            gate.require(str(path.resolve()) == row['resolved_absolute'] and path.is_file() and
                         path.stat().st_size == row['size_bytes'], 'Registered physical target/size changed')
            actual[name] = gate.digest(path)
            gate.require(actual[name] == sha, 'Registered shared input SHA changed: ' + name)
        return actual

    gate.hashes = bound_hashes
    return binding


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('entry', choices=['approve', 'run'])
    args = parser.parse_args()
    if args.entry == 'run':
        ORIGINAL_HASHES(gate.HERE, gate.read(gate.HERE / 'dispatch_receipt.APPROVED.json')['execution_attachment_sha256'])
    bind_shared_paths()
    if args.entry == 'approve':
        import approve_dispatch
        approve_dispatch.main()
    else:
        import sys
        sys.argv = [str(gate.HERE / 'gate.py'), 'run', '--dispatch-receipt', str(gate.HERE / 'dispatch_receipt.APPROVED.json')]
        gate.main()


if __name__ == '__main__':
    main()
