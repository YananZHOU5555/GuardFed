"""Preserve the sealed publisher and make the reviewed Git-only repair in a new directory."""
from pathlib import Path
import ast, hashlib, json, shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'tmp/publication_increment34_prepared_20261010'
NEW = ROOT/'tmp/publication_increment34_prepared_v2_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
assert sha(OLD/'FILES_SHA256.json') == 'fd98b16b4bf3f1bec7d86438adea8dd3a5738f9549ef683a6d15d99505daadd0'
for name, row in read(OLD/'FILES_SHA256.json')['files'].items():
    assert sha(OLD/name) == row['sha256'] and (OLD/name).stat().st_size == row['bytes']
review = ROOT/'tmp/publication_increment34_root_review_20261010'
assert sha(review/'ROOT_REVIEW.json') == '8a4156dbc27f09477eb4ee1086615c7ac446a6841b02ee8f0ea88af880fe6667'
original = (OLD/'publish_increment34.py').read_text(encoding='utf8')
changes = {
    "git('add', '--', *names[start:start + 25])": "git('add', '-f', '--', *names[start:start + 25])",
    "for path in (hd / 'ROOT_ADOPTION_REVIEW.json', hybrid / 'LATEST_BACKUP.json', hybrid / 'BACKUP_CHAIN_accepted_delta_after17_20261010.json')": "for path in (hd / 'ROOT_DELIVERY_COPY.json', hd / 'ROOT_ADOPTION_REVIEW.json', hybrid / 'LATEST_BACKUP.json', hybrid / 'BACKUP_CHAIN_accepted_delta_after17_20261010.json')",
    "    seal(HERE, 'FILES_SHA256.json')": "    seal(ROOT / 'tmp/publication_increment34_prepared_20261010', 'FILES_SHA256.json')\n    tree(ROOT / 'tmp/publication_increment34_root_review_20261010')\n    add(ROOT / 'tmp/prepare_publication34_v2_root_20261010.py')\n    seal(HERE, 'FILES_SHA256.json')",
}
updated = original
for before, after in changes.items():
    assert updated.count(before) == 1, before
    updated = updated.replace(before, after)
ast.parse(updated)
assert updated.count("git('add', '-f', '--'") == 1
assert "git('add', '--renormalize', '--'" in updated
NEW.mkdir(exist_ok=False)
with (NEW/'publish_increment34.py').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(updated)
for name in ('verify_increment34.py', 'PREPARED_INPUTS.json'):
    shutil.copyfile(OLD/name, NEW/name)
    assert sha(OLD/name) == sha(NEW/name)
note = dict(status='PRE_EXECUTION_ENGINEERING_REPAIR_NOT_SCIENTIFIC_FAILURE',
    original_source_seal_sha256=sha(OLD/'FILES_SHA256.json'), independent_review_sha256=sha(review/'ROOT_REVIEW.json'),
    original_source_preserved=True, original_publisher_not_executed=True,
    code_changes=['Force-add the exact verified mapping, as the previous publisher did, because five archives are ignored.',
                  'Include the root copy provenance and preserve both source versions plus the independent review.'],
    science_closure_and_count_guards_changed=False, verifier_byte_exact=True, prepared_inputs_byte_exact=True,
    source_AST_pass=True, source_ready_not_staged=True, shared_state_changed=False)
with (NEW/'ROOT_PATCH_NOTE.json').open('x', encoding='utf8') as stream:
    json.dump(note, stream, indent=2)
    stream.write('\n')
with (NEW/'README.md').open('x', encoding='utf8', newline='\n') as stream:
    stream.write('# Increment34 V2 — Git transport repair only\n\n'
                 'Original six-member sealed source and independent review are preserved. '
                 'This source adds force-add for the exact verified mapping and includes provenance. '
                 'All science, actual closure, 147/147, FL7/Hybrid18, parent and remote checks remain unchanged.\n\n'
                 'Run the V2 publisher with the same required actual paths/SHA flags as the original README, '
                 'using an output directly beneath this V2 directory. Then commit, verify committed blobs, '
                 'push, and separately verify the remote branch. Source preparation is not publication.\n')
files = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(NEW.iterdir()) if p.is_file()}
with (NEW/'FILES_SHA256.json').open('x', encoding='utf8') as stream:
    json.dump({'status': 'REVIEWED_GIT_ONLY_REPAIR_SOURCE_READY_NOT_EXECUTED', 'files': files}, stream, indent=2)
    stream.write('\n')
print(json.dumps({'publisher_sha256': sha(NEW/'publish_increment34.py'), 'source_seal_sha256': sha(NEW/'FILES_SHA256.json'), 'original_seal_unchanged': sha(OLD/'FILES_SHA256.json')}))
