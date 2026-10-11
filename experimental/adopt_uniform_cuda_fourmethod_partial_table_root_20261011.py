"""Adopt verified descriptive statistics; create no new experiment records."""
from pathlib import Path
import datetime,hashlib,json
R=Path(__file__).resolve().parents[1];H=R/'tmp/celeba_uniform_cuda_fourmethod_partial_20261011';O=H/'actual_partial400'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
review=R/'tmp/celeba_uniform_cuda_fourmethod_partial_independent_review_20261011/SOURCE_REVIEW.json'
assert sha(review)=='154c53a04f6216f0e8a651506d9ff648986f9cf0a422636b8e1432a74ddb09d7'
r=read(review);receipt=O/'BUILD_RECEIPT.json';b=read(receipt)
assert sha(receipt)==r['build_receipt_sha256']=='14ab89c336f540dd1109f1bd8d07b706096dd52db78e7bd0e8acf25885328f83'
assert r['status']=='PASS_INDEPENDENT_ARITHMETIC_AND_FORMAT_VERIFICATION' and not r['blocking_findings'] and not r['unit_findings']
assert r['identity_grid_records']==b['records']==400 and r['numeric_comparisons']==2772 and r['max_abs_difference']<1e-14
assert b['view']=='native' and b['ddof']==1 and not b['test'] and not b['new_primary_endpoint']
for name,pin in b['files'].items():assert sha(O/name)==pin['sha256'] and (O/name).stat().st_size==pin['bytes']
for pin in r['roots'].values():assert sha(pin['path'])==pin['sha256'] and read(pin['path'])['accepted']==100
proof=dict(status='ROOT_PARTIAL_FOUR_METHOD_UNIFORM_CUDA_VALIDATION_TABLE_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_accepted=True,records=400,methods=b['methods'],roots=r['roots'],build_receipt=dict(path=str(receipt),sha256=sha(receipt)),independent_review=dict(path=str(review),sha256=sha(review)),table=dict(path=str(O/'TABLES.md'),sha256=sha(O/'TABLES.md')),outputs_all_SHA_rechecked=True,numeric_comparisons=2772,ddof=1,seed_panels=[10,9,6],seed_first_aggregation=True,constant_predictors=r['constant_predictors_by_method'],new_scientific_adoptions=0,whole_comparison_complete=False,whole_rebuttal_complete=False,test=False,primary_endpoint_selected=False,old_tables_preserved=True)
p=H/'ROOT_TABLE_ADOPTION.json'
with p.open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(p),sha256=sha(p),records=400,new_scientific_adoptions=0)))
