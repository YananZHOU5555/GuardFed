"""Copy the sealed delivery unchanged, then rebind the original one-record root adopter."""
from pathlib import Path
import ast, hashlib, json, shutil

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'tmp/celeba_hybrid_screen_delta_after16_20261010'
target=ROOT/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after16_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(source/'DELIVERY_FILES_SHA256.json')=='d1ff1c5baf6212a7812fa9acbc268df18b1b155f7a38f5915309cd4ed619ac89'
assert sha(source/'ROOT_READY_CHAIN_LINK.json')=='5c62b8160cb08ae3d06e2c9342fab4519f01d8118876588bd53d0be78d243284'
for row in read(source/'DELIVERY_FILES_SHA256.json')['members']:
    path=source/row['path']
    assert path.resolve().is_relative_to(source.resolve()) and sha(path)==row['sha256'] and path.stat().st_size==row['size']
assert not target.exists()
files=[]
for path in source.rglob('*'):
    assert not path.is_symlink()
    if path.is_file() and '__pycache__' not in path.parts:
        dest=target/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);assert sha(dest)==sha(path)
        files.append({'path':dest.relative_to(target).as_posix(),'sha256':sha(dest)})
with (target/'ROOT_DELIVERY_COPY.json').open('x',encoding='utf8') as stream:
    json.dump({'source':source.relative_to(ROOT).as_posix(),'source_delivery_seal_sha256':sha(source/'DELIVERY_FILES_SHA256.json'),'files':files,'old_files_modified':False},stream,indent=2);stream.write('\n')
old=(ROOT/'tmp/adopt_Hybrid_after15_root_20261009.py').read_text(encoding='utf8')
changes={
    'actual fifteen-record chain':'actual sixteen-record chain',
    "'accepted_delta_after15_20261009', 15, 1":"'accepted_delta_after16_20261010', 16, 1",
    '084845d1fce698bb87d31eaed22bea3eec54b3bb7bf6d1873eecda159e2f9144':'d1ff1c5baf6212a7812fa9acbc268df18b1b155f7a38f5915309cd4ed619ac89',
    'b8f80af856c097a589343afb57022ef1e9db8b2b10a0860f1b6464378cdd2388':'5c62b8160cb08ae3d06e2c9342fab4519f01d8118876588bd53d0be78d243284',
    'hybrid_after15_delta.tar.gz':'hybrid_after16_delta.tar.gz',
    '27fb954f55e113d938f03f289048d84cc16303abfd9c437a40fd09b180370d2c':'2f78217c5d0d63778d647f787cc4405c08cc62f0e2bf3aec3713aa5895a6df3e',
}
for before,after in changes.items():assert before in old;old=old.replace(before,after)
ast.parse(old)
output=ROOT/'tmp/adopt_Hybrid_after16_root_20261010.py'
with output.open('x',encoding='utf8',newline='\n') as stream:stream.write(old)
print(json.dumps({'copied_files':len(files),'adopter':output.relative_to(ROOT).as_posix()}))
