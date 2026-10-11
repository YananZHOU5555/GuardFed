from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
assert not (H/'FILES_SHA256.json').exists()
check=json.loads((H/'SOURCE_CHECK.json').read_bytes());assert check['identity_rows']==400 and check['mean_SD_computed'] is False
for p in H.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
put('HANDOFF.json',dict(status='SOURCE_READY_ACTUAL_FOUR100_ROOTS_BOUND_STATISTICS_NOT_EXECUTED',title='partial four-method uniformly-CUDA validation',roots=json.loads((H/'INPUT_PINS.json').read_bytes())['roots'],source_check_sha256=sha(H/'SOURCE_CHECK.json'),build_sha256=sha(H/'build.py'),statistics_source_sha256=sha(H/'statistics_original.py'),original_math_reused=True,original_formatter_reused=True,view='native',methods=4,records_expected=400,seed_panel_sizes=[10,9,6],ddof=1,seed_first_scene_averaging=True,all_negative_and_constant_results_retained=True,old_metrics_used=False,actual_statistics=None,actual_outputs=None,root_source_review=None,F_access=False,SSH=False,arrays=False,fit=False,CNN=False,new_primary_endpoint=False,final_comparison=False))
files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(H.iterdir()) if p.is_file()}
put('FILES_SHA256.json',dict(status='SOURCE_ONLY_SEAL_NOT_STATISTICAL_AUTHORIZATION',files=files,members=len(files),bytes=sum(v['bytes'] for v in files.values())))
print(json.dumps(dict(seal=sha(H/'FILES_SHA256.json'),handoff=sha(H/'HANDOFF.json'),members=len(files),build=sha(H/'build.py'),source_check=sha(H/'SOURCE_CHECK.json'))))
