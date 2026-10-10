"""Bind an actual root-adopted A100 table; never generate drafts or scientific values."""
from pathlib import Path
import argparse,hashlib,json
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def pin(p):return dict(path=p.resolve().as_posix(),sha256=sha(p),bytes=p.stat().st_size)
p=argparse.ArgumentParser();p.add_argument('--A100-root',type=Path,required=True);p.add_argument('--A100-root-sha256',required=True);a=p.parse_args()
assert __debug__ and not (H/'SOURCE_PINS.json').exists()
assert sha(a.A100_root)==a.A100_root_sha256
root=read(a.A100_root)
assert root['root_adoption'] is True and (root['paired_models'],root['preserved_records'],root['complete_scenes'])==(100,200,10)
assert root['complete_IID_scenes']==5 and root['complete_nonIID_scenes']==['Benign','F Flip','FedSA','S-DFA','Sp-DFA']
assert root['seed_panels']==[10,9,6] and root['IID_seed_first_JSON_bytes_exact'] and not root['test'] and not root['primary_endpoint_selected']
assert root['old180_record_bytes_order_exact'] and root['old1458_scalars_exact'] and root['old729_cells_exact']
fixed=read(H/'FIXED_INPUT_PINS.json')
for key,value in fixed.items():
 path=Path(value['path']);assert sha(path)==value['sha256'] and path.stat().st_size==value['bytes'],key
reader=read(Path(fixed['reader_root']['path']))
assert reader['A90_incorporated'] and reader['author_review_only'] and not reader['manuscript_applied']
assert fixed['prior_A90_table_root']['sha256']==reader['A90_table_root_sha256']
pins=dict(fixed,A100_root=pin(a.A100_root))
for key,name in [('tables','tables.json'),('iid_seed_first','IID_SEED_FIRST.json'),('cross_scene','CROSS_SCENE_ADDITIONAL.json'),('summary','SUMMARY.json'),('official_table','TABLES.md'),('saved_verification','SAVED_VERIFICATION.json')]:
 path=a.A100_root.parent/name;assert sha(path)==root['files_sha256'][name]
 pins[key]=pin(path)
assert root['canonical_table']==str(Path(pins['official_table']['path']).relative_to(H.parents[1]).as_posix())
assert read(Path(pins['summary']['path']))['native_shared_metrics_and_counts_exact'] is True
assert (root['mean_SD_scalars_recomputed'],root['display_cells'])==(2106,1053)
with (H/'SOURCE_PINS.json').open('x',encoding='utf8',newline='\n') as f:json.dump(pins,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status='ACTUAL_A100_ROOT_BOUND_NO_DOCUMENT_GENERATION',A100_root_sha256=a.A100_root_sha256,pins_sha256=sha(H/'SOURCE_PINS.json'))))
