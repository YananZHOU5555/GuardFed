"""Bind the prior Hybrid root adoption to the sealed actual eighteen-to-nineteen delta."""
from pathlib import Path
import ast, hashlib, json, re, shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'tmp/celeba_hybrid_screen_delta_after18_20261010'
target = ROOT/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after18_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
assert sha(source/'DELIVERY_FILES_SHA256.json') == '8a54c41b25a743aaab9e26a5bfe387cd55ca37c064974a1f8db39fe46e78c228'
assert sha(source/'ROOT_READY_CHAIN_LINK.json') == 'f1104d7b0b4f75e4bb5a8e9f2623da42bcf9c72ea57eeea7737abe7a96357556'
for row in read(source/'DELIVERY_FILES_SHA256.json')['members']:
    path = source/row['path']
    assert path.resolve().is_relative_to(source.resolve()) and sha(path) == row['sha256'] and path.stat().st_size == row['size']
old = (ROOT/'tmp/adopt_Hybrid_after17_root_20261010.py').read_text(encoding='utf8')
changes = {
    'actual seventeen-record chain': 'actual eighteen-record chain',
    "'accepted_delta_after17_20261010', 17, 1": "'accepted_delta_after18_20261010', 18, 1",
    '5d477b71e186c372fbc81490ca2cb4a4648addf7862861d9d33f094952b5674a': '8a54c41b25a743aaab9e26a5bfe387cd55ca37c064974a1f8db39fe46e78c228',
    'e8e9dbcc882b4c3a30ab8f239e17adb862a0f250d95e5070cf6612c05d875b71': 'f1104d7b0b4f75e4bb5a8e9f2623da42bcf9c72ea57eeea7737abe7a96357556',
    'hybrid_after17_delta.tar.gz': 'hybrid_after18_delta.tar.gz',
    '273c42bbd4fe93b915612be3ec8fff1254cc03c486efdacdbe18d3586f0a4126': '19f35049e2a79445e658b3f2586f353a223fe551a343e159d59f02cda2f5cd79',
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
with (ROOT/'tmp/adopt_Hybrid_after18_root_20261010.py').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(updated)
print(json.dumps({'copied_files': len(files), 'adopter_source_prepared': True, 'LATEST_not_changed': True}))
