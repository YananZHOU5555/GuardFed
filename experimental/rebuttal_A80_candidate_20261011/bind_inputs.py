"""Bind only an explicitly SHA-identified, root-adopted A80 table; no generation."""
from pathlib import Path
import argparse,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def pin(p):return dict(path=p.resolve().as_posix(),sha256=sha(p),bytes=p.stat().st_size)
p=argparse.ArgumentParser();p.add_argument('--A80-root',type=Path,required=True);p.add_argument('--A80-root-sha256',required=True);a=p.parse_args()
assert __debug__ and not (H/'SOURCE_PINS.json').exists()
assert sha(a.A80_root)==a.A80_root_sha256
root=read(a.A80_root)
assert root['root_adoption'] is True and root['paired_models']==80 and root['preserved_records']==160
assert root['complete_scenes']==8 and root['complete_IID_scenes']==5 and root['complete_nonIID_scenes']==['Benign','F Flip','FedSA']
assert root['seed_panels']==[10,9,6] and root['IID_seed_first_JSON_bytes_exact'] and not root['test'] and not root['primary_endpoint_selected']
fixed=read(H/'FIXED_INPUT_PINS.json')
for key,value in fixed.items():
    path=Path(value['path']);assert sha(path)==value['sha256'] and path.stat().st_size==value['bytes'],key
pins=dict(fixed,A80_root=pin(a.A80_root))
for key,name in [('tables','tables.json'),('iid_seed_first','IID_SEED_FIRST.json'),('summary','SUMMARY.json'),('official_table','TABLES.md'),('saved_verification','SAVED_VERIFICATION.json')]:
    path=a.A80_root.parent/name;assert sha(path)==root['files_sha256'][name]
    pins[key]=pin(path)
assert read(Path(pins['summary']['path']))['native_shared_metrics_and_counts_exact'] is True
with (H/'SOURCE_PINS.json').open('x',encoding='utf-8') as f:json.dump(pins,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':'ACTUAL_A80_ROOT_BOUND_NO_DOCUMENT_GENERATION','A80_root_sha256':a.A80_root_sha256,'pins_sha256':sha(H/'SOURCE_PINS.json')}))
