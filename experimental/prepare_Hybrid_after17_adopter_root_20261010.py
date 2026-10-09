"""Copy the sealed new Hybrid delivery and rebind the existing strict root adopter."""
from pathlib import Path
import ast, hashlib, json, re, shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'tmp/celeba_hybrid_screen_delta_after17_20261010'
target = ROOT/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after17_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
assert sha(source/'DELIVERY_FILES_SHA256.json') == '5d477b71e186c372fbc81490ca2cb4a4648addf7862861d9d33f094952b5674a'
assert sha(source/'ROOT_READY_CHAIN_LINK.json') == 'e8e9dbcc882b4c3a30ab8f239e17adb862a0f250d95e5070cf6612c05d875b71'
for row in read(source/'DELIVERY_FILES_SHA256.json')['members']:
    path = source/row['path']
    assert path.resolve().is_relative_to(source.resolve()) and sha(path) == row['sha256'] and path.stat().st_size == row['size']
old = (ROOT/'tmp/adopt_Hybrid_after16_root_20261010.py').read_text(encoding='utf8')
changes = {
    'actual sixteen-record chain': 'actual seventeen-record chain',
    "'accepted_delta_after16_20261010', 16, 1": "'accepted_delta_after17_20261010', 17, 1",
    'd1ff1c5baf6212a7812fa9acbc268df18b1b155f7a38f5915309cd4ed619ac89': '5d477b71e186c372fbc81490ca2cb4a4648addf7862861d9d33f094952b5674a',
    '5c62b8160cb08ae3d06e2c9342fab4519f01d8118876588bd53d0be78d243284': 'e8e9dbcc882b4c3a30ab8f239e17adb862a0f250d95e5070cf6612c05d875b71',
    'hybrid_after16_delta.tar.gz': 'hybrid_after17_delta.tar.gz',
    '2f78217c5d0d63778d647f787cc4405c08cc62f0e2bf3aec3713aa5895a6df3e': '273c42bbd4fe93b915612be3ec8fff1254cc03c486efdacdbe18d3586f0a4126',
}
for before in changes:
    assert old.count(before) == 1, before
pattern = '|'.join(re.escape(k) for k in sorted(changes, key=len, reverse=True))
updated = re.sub(pattern, lambda m: changes[m.group()], old)
ast.parse(updated)
assert not target.exists()
files = []
for path in source.rglob('*'):
    assert not path.is_symlink()
    if path.is_file() and '__pycache__' not in path.parts:
        dest = target/path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        assert sha(dest) == sha(path)
        files.append({'path': dest.relative_to(target).as_posix(), 'sha256': sha(dest)})
with (target/'ROOT_DELIVERY_COPY.json').open('x', encoding='utf8') as stream:
    json.dump({'source': source.relative_to(ROOT).as_posix(), 'source_delivery_seal_sha256': sha(source/'DELIVERY_FILES_SHA256.json'), 'files': files, 'old_files_modified': False}, stream, indent=2)
    stream.write('\n')
with (ROOT/'tmp/adopt_Hybrid_after17_root_20261010.py').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(updated)
print(json.dumps({'copied_files': len(files), 'adopter_source_prepared': True, 'LATEST_not_changed': True}))
