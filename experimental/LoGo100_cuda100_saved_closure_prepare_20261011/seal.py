"""Seal prepared local sources only; no runtime execution."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def put(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
assert not (H/'FILES_SHA256.json').exists()
checks=read(H/'SOURCE_CHECK.json');chain=read(H/'METADATA_CHAIN_CHECK.json')
assert checks['status']=='SOURCE_ONLY_PASS_NOT_RUNTIME_OR_SCIENTIFIC_ACCEPTANCE' and chain['positive']==1 and chain['rejected']==9
for p in H.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
scope=read(H/'SCOPE.json');assert len(scope['ids'])==100 and len(scope['remaining_ids'])==98
put('HANDOFF.json',dict(status='SEALED_SOURCE_ONLY_LOGO100_DELTA98_NOT_EXECUTED',parent_source_seal_sha256='eb08186667ef3ca700cd853aa4dfc919181c86f080b89e5b2b1484203051e3a3',prior2_root_sha256='cdf9a578fd26a9a85ae674aff7e55932e9cd6e75b895b8e3b39adc1ad12753bc',cohort_sha256=sha(H/'EXACT100_REFS.json'),source_check_sha256=sha(H/'SOURCE_CHECK.json'),metadata_chain_check_sha256=sha(H/'METADATA_CHAIN_CHECK.json'),source_pins={n:sha(H/n) for n in ['metadata_gate.py','remote_export.py','transport.py','audit_saved.py','prior2_overlay.py']},expected=dict(records=100,new_scores_npz=98,adopted2_scores_referenced=2,retained_payload_members=33,saved_decisions=1986700,metrics=300,primitive_confusion_counts=800),original_scientific_loop_raw_AST_exact=True,original_science_files_raw_exact=3,complete_raw_inverse_count=5,auth_canary_ids_still_original2=True,per_record_seed_rule='SHA-authenticated original job seed equals frozen original post_id seed; same job/result retained',old_CPU_not_new_CUDA=True,all_seeds_kept=True,full_order_retained=True,actual_transport=None,actual_archive=None,actual_audit=None,actual_root=None,SSH=False,F_access=False,arrays=False,CNN=False,fit=False,STATE=False,Git=False,CLI='See README.md: exact transport once, real pins, audit once; root adopts separately.'))
files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(H.iterdir()) if p.is_file()}
put('FILES_SHA256.json',dict(status='SOURCE_ONLY_SEAL_NOT_AUTHORIZATION',files=files,members=len(files),bytes=sum(v['bytes'] for v in files.values())))
print(json.dumps(dict(seal=sha(H/'FILES_SHA256.json'),handoff=sha(H/'HANDOFF.json'),members=len(files),bytes=sum(v['bytes'] for v in files.values()),audit=sha(H/'audit_saved.py'),remote=sha(H/'remote_export.py'),overlay=sha(H/'prior2_overlay.py'))))
