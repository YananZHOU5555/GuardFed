"""Prepare an immutable release after parent review; never install or launch it."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from screen_common import HERE, digest, local_identity, read, write_json


def freeze(output, reviewed_sha):
    if digest(HERE / 'PACKAGE_SHA256.json') != reviewed_sha:
        raise ValueError('Parent-reviewed package SHA mismatch')
    local_identity(require_frozen=False)
    seal = read(HERE / 'PACKAGE_SHA256.json')
    if seal['status'] != 'PREPARED_NOT_FROZEN':
        raise ValueError('Freeze only a prepared source package')
    output.mkdir(parents=True, exist_ok=False)
    for name in seal['files']:
        if name.startswith('jobs/'):
            continue
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, target)
    protocol = read(output / 'source/protocol.json')
    protocol['status'] = 'FROZEN'
    write_json(output / 'source/protocol.json', protocol)
    subprocess.run([sys.executable, str(output / 'source/prepare_jobs.py'), '--out', str(output / 'jobs')], check=True)
    write_json(output / 'RELEASE_LINEAGE.json', dict(
        reviewed_prepared_sha256=reviewed_sha,
        scientific_change='Only protocol.status PREPARED_NOT_FROZEN -> FROZEN; regenerate original job hashes',
        execution_authorized=False, next='Parent independently reviews release SHA and live resources; writes EXECUTION_AUTHORIZATION.json'))
    files = {str(p.relative_to(output)).replace('\\', '/'): digest(p) for p in sorted(output.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts}
    write_json(output / 'PACKAGE_SHA256.json', dict(status='FROZEN_PENDING_EXECUTION', files=files))
    print(json.dumps(dict(output=str(output), package_sha256=digest(output / 'PACKAGE_SHA256.json'),
                          execution_authorized=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reviewed-sha', required=True)
    args = parser.parse_args()
    freeze(args.out.resolve(), args.reviewed_sha)
