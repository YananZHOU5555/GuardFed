"""Reuse the accepted Hybrid adopter with only the new increment's metadata bindings."""
from pathlib import Path
import ast, hashlib, json, re, shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'tmp/celeba_hybrid_delta_after22_20261010'
target = ROOT/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after22_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
assert sha(source/'DELIVERY_FILES_SHA256.json') == '769ed746706e2232b2510d07bc1cbe1c14fe5229d456ceb66dbf9b3f07cf1091'
assert sha(source/'ROOT_READY_CHAIN_LINK.json') == '48a65ff52d82b1f23b892e3be1e067a37bcfd613c034f07025471f621e2a33fd'
assert read(source/'ROOT_READY_CHAIN_LINK.json')['previous_root_adoption_sha256'] == '90b06ac4ee6bfed02bac3037d18115e7c0c66592c0f934c4ed3d6b301b517787'
for row in read(source/'DELIVERY_FILES_SHA256.json')['members']:
    path = source/row['path']
    assert path.resolve().is_relative_to(source.resolve()) and sha(path) == row['sha256'] and path.stat().st_size == row['size']
original = ROOT/'tmp/adopt_Hybrid_after21_root_20261010.py'
assert sha(original) == '37ea55db88b7e344186a4a4bdfbda14bd52d223005c1ca3824ed8d3dc54a7893'
old = original.read_text(encoding='utf8')
bindings = {
    "'accepted_delta_after21_20261010', 21, 1": "'accepted_delta_after22_20261010', 22, 1",
    'ac4e4b0d5368066efe156ae364b9321954c5c6ff15a4d4b5da08f87560f9ff28': '769ed746706e2232b2510d07bc1cbe1c14fe5229d456ceb66dbf9b3f07cf1091',
    '7e1cbecd0127fadfda81124389b3382de9a36d723bca0dcfc712c35232210b60': '48a65ff52d82b1f23b892e3be1e067a37bcfd613c034f07025471f621e2a33fd',
    'hybrid_after21_delta.tar.gz': 'hybrid_after22_delta.tar.gz',
    '6148a33dbf0bcad54fb8d4d5d5d20630ef8e04342e3c7582c5ce606624f9f577': 'dbca388d5384d482bfbe84f13592742df38041d4989579c8091c68e71c59213e',
}
for before in bindings:
    assert old.count(before) == 1
pattern = '|'.join(re.escape(k) for k in sorted(bindings, key=len, reverse=True))
bound = re.sub(pattern, lambda m: bindings[m.group()], old)
ast.parse(bound)
assert not target.exists()
for path in source.rglob('*'):
    assert not path.is_symlink()
    if path.is_file() and '__pycache__' not in path.parts:
        dest = target/path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        assert sha(dest) == sha(path)
(target/'ROOT_DELIVERY_COPY.json').write_text(json.dumps(dict(source=source.relative_to(ROOT).as_posix(), source_seal_sha256=sha(source/'DELIVERY_FILES_SHA256.json'), original_adopter_sha256=sha(original), metadata_bindings=bindings, scientific_body_changed=False), indent=2)+'\n', encoding='utf8')
exec(compile(bound, str(original), 'exec'), {'__file__': str(original), '__name__': '__main__'})
