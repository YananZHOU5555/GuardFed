"""Package this completed two-job canary only; never runs training."""
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


root = Path(__file__).resolve().parent
complete = json.loads((root / 'COMPLETE.json').read_text())
assert complete['status'] == 'PASS' and complete['completed_canaries'] == 2
assert complete['formal_screen_jobs_started'] == 0 and not complete['formal_table_eligible']
assert not list((root / 'runs').rglob('failure.json'))
archive = root / 'completed_canary_backup.tar.gz'
inventory_path = root / 'BACKUP_MEMBERS.json'
receipt_path = root / 'BACKUP_SHA256.json'
assert not archive.exists() and not inventory_path.exists() and not receipt_path.exists()
files = sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
inventory = {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files}
inventory_path.write_text(json.dumps(inventory, indent=2) + '\n')
with tarfile.open(archive, 'w:gz') as tar:
    for name in inventory:
        tar.add(root / name, arcname=name, recursive=False)
    tar.add(inventory_path, arcname=inventory_path.name, recursive=False)
receipt = dict(archive=archive.name, sha256=sha(archive), bytes=archive.stat().st_size,
               member_count=len(inventory), inventory_sha256=sha(inventory_path))
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
