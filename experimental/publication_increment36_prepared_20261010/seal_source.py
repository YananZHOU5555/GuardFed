"""Seal the source-only increment36 delivery once, without Git or collection."""
from pathlib import Path
import ast, difflib, hashlib, json

H = Path(__file__).resolve().parent
O = H.with_name('publication_increment35_prepared_20261010')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
assert sha(O / 'publish_increment35.py') == '0772f210851449811934614913d2cb40a18cf852eb882d973df2a744b0da012d'
assert sha(O / 'verify_increment35.py') == '8267976b8f2b89f14c9c2440e25bc11c8b626049a56f9183dd13c7207f5230b6'
old = (O / 'publish_increment35.py').read_text()
new = (H / 'publish_increment36.py').read_text()
marker = '    try:\n        for name, source in sources.items():'
assert old[old.index(marker):] == new[new.index(marker):]
def function(text, name):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)
assert function(old, 'verify_blobs') == function(new, 'verify_blobs')
check = read(H / 'SELF_CHECK_FINAL.json')
assert check['refusal_count'] == 26 and check['actual_Git_calls'] == 0
assert not any(check[k] for k in ['SSH', 'network', 'CNN', 'stage', 'commit', 'push'])
template = read(H / 'ROOT_CLOSED_INPUTS_TEMPLATE.json')
assert template['counts'] is None and template['C50_table'] is None
assert all(p['path'] is None and p['sha256'] is None for p in template['closure_pins'].values())
assert read(H / 'AUTHOR_REVIEW_BINDING_CHECK.json')['required_extra_pins_added'] == 15
for p in H.glob('*.py'):
    ast.parse(p.read_text(encoding='utf-8'))
with (H / 'MINIMAL_SOURCE_DIFF.patch').open('x', encoding='utf-8', newline='\n') as f:
    for a, b in [('publish_increment35.py', 'publish_increment36.py'), ('verify_increment35.py', 'verify_increment36.py')]:
        f.writelines(difflib.unified_diff((O/a).read_text().splitlines(True), (H/b).read_text().splitlines(True), fromfile='sealed35/'+a, tofile='increment36/'+b))
reuse = dict(status='SOURCE_ONLY_INCREMENT36_REUSE_NOT_PUBLICATION', parent_publisher_sha256=sha(O/'publish_increment35.py'), parent_verifier_sha256=sha(O/'verify_increment35.py'),
    index_mutation_copy_force_add_renormalize_blob_verify_and_failure_block_byte_exact=True, verify_blobs_function_byte_exact=True,
    publisher_lines=len(new.splitlines()), verifier_lines=len((H/'verify_increment36.py').read_text().splitlines()),
    changes=['150+6 closure and exact non-IID Benign IDs', 'FL9+2=11 closure', 'unchanged published Hybrid19 proof only; old namespace refused', 'C50_table stays null; actual canonical complete C50 author-review extra pins', 'parent35/source pins and receipt names'],
    actual_C6_adoption_fabricated=False, required_actual_root_inputs=True, actual_Git_calls=0, SSH=False, CNN=False)
with (H / 'SOURCE_REUSE.json').open('x', encoding='utf-8', newline='\n') as f:
    json.dump(reuse, f, indent=2); f.write('\n')
handoff = dict(status='SOURCE_PREPARED_ONLY_ROOT_ACTUAL_C6_CLOSURE_REQUIRED', parent_commit=template['parent_commit'], required_closed_counts=template['required_closed_counts'],
    template='ROOT_CLOSED_INPUTS_TEMPLATE.json', prepared_inputs='PREPARED_INPUTS.json', publisher='publish_increment36.py', verifier='verify_increment36.py',
    selfcheck='SELF_CHECK_FINAL.json', selfcheck_sha256=sha(H/'SELF_CHECK_FINAL.json'), refusal_count=26,
    publisher_sha256=sha(H/'publish_increment36.py'), verifier_sha256=sha(H/'verify_increment36.py'),
    actual_C6_adoption_sha256=None, C50_table=None, unchanged_Hybrid19_proof_only=True, old_Hybrid_archives_republished=False,
    source_pins_verified=31, complete_C50_author_review_required_extra_pins=15, actual_stage=False, commit=False, push=False, SSH=False, CNN=False,
    remaining_boundary='Root independently binds actual C6 adoption/state/live and executes staging, commit and remote verification. Running observations are not accepted counts.')
with (H / 'HANDOFF.json').open('x', encoding='utf-8', newline='\n') as f:
    json.dump(handoff, f, indent=2); f.write('\n')
files = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(H.iterdir()) if p.is_file() and p.name != 'FILES_SHA256.json'}
with (H / 'FILES_SHA256.json').open('x', encoding='utf-8', newline='\n') as f:
    json.dump(dict(status='SOURCE_PREPARED_ONLY_NOT_STAGED_COMMITTED_OR_PUSHED', files=files), f, indent=2); f.write('\n')
for n, pin in read(H/'FILES_SHA256.json')['files'].items():
    assert sha(H/n) == pin['sha256']
print(json.dumps(dict(status=reuse['status'], source_seal_sha256=sha(H/'FILES_SHA256.json'), members=len(files), publisher_sha256=sha(H/'publish_increment36.py'), verifier_sha256=sha(H/'verify_increment36.py'), selfcheck_sha256=sha(H/'SELF_CHECK_FINAL.json'), handoff_sha256=sha(H/'HANDOFF.json'), actual_Git_calls=0)))
