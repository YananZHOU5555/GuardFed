"""One source-only upload; never starts or overwrites an experiment."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tmp'))
from guardfed_local_storage import STORAGE_ROOT, check_bulk_storage
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'tmp/celeba_gradient_screen64_20261010'
REMOTE = '/workspace/guardfed_checks/celeba_gradient_screen64_20261010'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal = SOURCE / 'FILES_SHA256.json'
assert sha(seal) == 'eb389aca69b194ef3287049d6940e2c94b5d27d7e06ba2b3ee3b0317098e738b'
members = json.loads(seal.read_bytes())['files']
assert all(sha(SOURCE / n) == h for n, h in members.items())
storage = check_bulk_storage(sum((SOURCE / n).stat().st_size for n in members) + 1048576)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
archive = STORAGE_ROOT / 'deployments/celeba_gradient_screen64_20261010' / (stamp + '.tar.gz')
archive.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive, 'x:gz') as t:
    for name in list(members) + ['FILES_SHA256.json', 'HANDOFF.json']:
        t.add(SOURCE / name, arcname=name, recursive=False)
    t.add(HERE / 'remote_preflight.py', arcname='root_operations/remote_preflight.py', recursive=False)
remote_archive = '/workspace/guardfed_checks/gradient64_source_' + stamp + '.tar.gz'
r = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-P', '60350', str(archive),
                    'root@89.22.197.55:' + remote_archive], capture_output=True, timeout=90)
(HERE / ('upload_' + stamp + '.stdout')).write_bytes(r.stdout)
(HERE / ('upload_' + stamp + '.stderr')).write_bytes(r.stderr)
r.check_returncode()
code = """from pathlib import Path
import hashlib,json,tarfile
base=Path(REMOTE); archive=Path(ARCHIVE)
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==ARCHIVE_SHA
assert not base.exists(), 'Preserve existing source/output; no overwrite'
with tarfile.open(archive) as t:
 for m in t.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
 base.mkdir()
 t.extractall(base,filter='data')
seal=base/'FILES_SHA256.json'
assert hashlib.sha256(seal.read_bytes()).hexdigest()=='eb389aca69b194ef3287049d6940e2c94b5d27d7e06ba2b3ee3b0317098e738b'
files=json.loads(seal.read_bytes())['files']
assert all(hashlib.sha256((base/n).read_bytes()).hexdigest()==h for n,h in files.items())
print(json.dumps(dict(status='SOURCE_ONLY_85_MEMBERS_UPLOADED_VERIFIED_NO_DISPATCH',path=str(base),members=len(files))))
""".replace('ARCHIVE_SHA', repr(sha(archive))).replace('ARCHIVE)', repr(remote_archive) + ')').replace('REMOTE)', repr(REMOTE) + ')')
r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350', 'root@89.22.197.55',
                    'python -B -'], input=code.encode(), capture_output=True, timeout=45)
(HERE / ('unpack_' + stamp + '.stdout')).write_bytes(r.stdout)
(HERE / ('unpack_' + stamp + '.stderr')).write_bytes(r.stderr)
r.check_returncode()
receipt = dict(utc=stamp, storage_preflight=storage, local_archive=str(archive), archive_sha256=sha(archive),
               remote_archive=remote_archive, remote_source=REMOTE, sealed_members=85,
               source_seal_sha256=sha(seal), actual_dispatch=False, remote_verification=json.loads(r.stdout))
(HERE / 'SOURCE_UPLOAD.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt['remote_verification']))
