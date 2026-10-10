"""Bind an explicitly SHA-identified, actually root-adopted A90 table."""
from pathlib import Path
import argparse,hashlib,json
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def pin(p):return dict(path=p.resolve().as_posix(),sha256=sha(p),bytes=p.stat().st_size)
p=argparse.ArgumentParser();p.add_argument('--A90-root',type=Path,required=True);p.add_argument('--A90-root-sha256',required=True);a=p.parse_args()
assert __debug__ and not (H/'SOURCE_PINS.json').exists()
assert sha(a.A90_root)==a.A90_root_sha256
root=read(a.A90_root)
assert root['root_adoption'] is True and root['paired_models']==90 and root['preserved_records']==180
assert root['complete_scenes']==9 and root['complete_IID_scenes']==5 and root['complete_nonIID_scenes']==['Benign','F Flip','FedSA','S-DFA']
assert root['seed_panels']==[10,9,6] and root['IID_seed_first_JSON_bytes_exact'] and not root['test'] and not root['primary_endpoint_selected']
assert root['accepted_partial_SpDFA_seeds_excluded']==list(range(91001,91006))
fixed=read(H/'FIXED_INPUT_PINS.json')
for key,value in fixed.items():
 path=Path(value['path']);assert sha(path)==value['sha256'] and path.stat().st_size==value['bytes'],key
assert fixed['prior_A80_table_root']['sha256']==read(Path(fixed['reader_root']['path']))['A80_table_root_sha256']
pins=dict(fixed,A90_root=pin(a.A90_root))
for key,name in [('tables','tables.json'),('iid_seed_first','IID_SEED_FIRST.json'),('summary','SUMMARY.json'),('official_table','TABLES.md'),('saved_verification','SAVED_VERIFICATION.json')]:
 path=a.A90_root.parent/name;assert sha(path)==root['files_sha256'][name]
 pins[key]=pin(path)
assert read(Path(pins['summary']['path']))['native_shared_metrics_and_counts_exact'] is True
with (H/'SOURCE_PINS.json').open('x',encoding='utf8',newline='\n') as f:json.dump(pins,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status='ACTUAL_A90_ROOT_BOUND_NO_DOCUMENT_GENERATION',A90_root_sha256=a.A90_root_sha256,pins_sha256=sha(H/'SOURCE_PINS.json'))))
