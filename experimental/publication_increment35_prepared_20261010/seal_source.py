"""Seal this source-only delivery once; no Git or evidence collection."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;O=H.with_name('publication_increment34_prepared_v2_20261010')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
old=(O/'publish_increment34.py').read_text();new=(H/'publish_increment35.py').read_text()
marker='    try:\n        for name, source in sources.items():'
assert old[old.index(marker):]==new[new.index(marker):]
assert read(H/'SELF_CHECK.json')['actual_Git_calls']==0
for name in ['publish_increment35.py','verify_increment35.py','prepare_source.py','prepare_metadata.py','check_prepared.py']:ast.parse((H/name).read_text())
with (H/'MINIMAL_SOURCE_DIFF.patch').open('w',encoding='utf-8',newline='\n') as f:
 for a,b in [('publish_increment34.py','publish_increment35.py'),('verify_increment34.py','verify_increment35.py')]:
  f.writelines(difflib.unified_diff((O/a).read_text().splitlines(True),(H/b).read_text().splitlines(True),fromfile='sealed34-v2/'+a,tofile='increment35/'+b))
reuse={'status':'SOURCE_ONLY_INCREMENT35_TRANSPORT_REUSE_NOT_PUBLICATION','parent_publisher_sha256':sha(O/'publish_increment34.py'),'parent_verifier_sha256':sha(O/'verify_increment34.py'),'index_mutation_copy_force_add_renormalize_blob_verify_and_failure_block_byte_exact':True,'publisher_lines':len(new.splitlines()),'verifier_lines':len((H/'verify_increment35.py').read_text().splitlines()),'root_actual_inputs_required':True,'future_adoption_or_table_SHA_fabricated':False,'Git_mutation':False,'SSH':False,'CNN':False}
with (H/'SOURCE_REUSE.json').open('x',encoding='utf-8') as f:json.dump(reuse,f,indent=2);f.write('\n')
files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(H.iterdir()) if p.is_file() and p.name!='FILES_SHA256.json'}
with (H/'FILES_SHA256.json').open('x',encoding='utf-8') as f:json.dump({'status':'SOURCE_PREPARED_ONLY_NOT_STAGED_COMMITTED_OR_PUSHED','files':files},f,indent=2);f.write('\n')
for n,pin in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/n)==pin['sha256']
print(json.dumps({'status':reuse['status'],'source_seal_sha256':sha(H/'FILES_SHA256.json'),'members':len(files),'publisher_sha256':sha(H/'publish_increment35.py'),'verifier_sha256':sha(H/'verify_increment35.py'),'selfcheck_sha256':sha(H/'SELF_CHECK.json'),'Git_calls':0}))
