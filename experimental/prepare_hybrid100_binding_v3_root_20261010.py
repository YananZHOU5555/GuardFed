"""New namespace for one reviewed metadata-only bind; preserve first failure."""
from pathlib import Path
import ast, datetime, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
SOURCE = ROOT / 'tmp/celeba_hybrid_fullcoverage_implementation_v3_20261010'
REVIEW = ROOT / 'tmp/celeba_hybrid_fullcoverage_v3_delta_review_20261010/REVIEW.json'
OLD_HELPER = ROOT / 'tmp/bind_hybrid100_metadata_root_v3_20261010.py'
NEW_HELPER = ROOT / 'tmp/bind_hybrid100_metadata_root_v4_20261010.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())

def save(path, value):
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2); f.write('\n')

def main():
    assert not HERE.exists() and not NEW_HELPER.exists()
    assert sha(OLD_HELPER) == '0fdfb69f7be05d170be394e7bae4868324381dd1d9e95251e4d8857dafa99160'
    assert sha(SOURCE/'FILES_SHA256.json') == 'bce84aa075a242ec89c2fd016dbac8fb7dea4a1b369074a30fefd5ce6e61d7af'
    review = read(REVIEW)
    assert review['status'] == 'INDEPENDENT_IMPLEMENTATION_V3_RECORD_ARITHMETIC_DELTA_PASS'
    assert review['v3_source_seal_sha256'] == sha(SOURCE/'FILES_SHA256.json')
    assert not review['confirmed_blockers']
    for name, pin in read(SOURCE/'FILES_SHA256.json')['files'].items():
        assert sha(SOURCE/name) == pin['sha256'] and (SOURCE/name).stat().st_size == pin['bytes']
    original = OLD_HELPER.read_text('utf8')
    replacements = {
        "HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_20261010'": "HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'",
        "SOURCE=ROOT/'tmp/celeba_hybrid_fullcoverage_implementation_v2_20261010'": "SOURCE=ROOT/'tmp/celeba_hybrid_fullcoverage_implementation_v3_20261010'",
        "REMOTE='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v2_20261010'": "REMOTE='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010'",
    }
    changed = original
    for before, after in replacements.items():
        assert changed.count(before) == 1
        changed = changed.replace(before, after)
    ast.parse(changed)
    restored = changed
    for before, after in replacements.items(): restored = restored.replace(after, before)
    assert restored == original
    HERE.mkdir()
    with NEW_HELPER.open('x', encoding='utf8', newline='\n') as f: f.write(changed)
    prior = ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_20261010'
    approval = read(prior/'BIND_APPROVAL.json')
    approval.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_seal_sha256=sha(SOURCE/'FILES_SHA256.json'),
        output='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/stage',
        independent_source_review_path=REVIEW.relative_to(ROOT).as_posix(),
        independent_source_review_sha256=sha(REVIEW),
        independent_source_review_seal_sha256=sha(REVIEW.parent/'FILES_SHA256.json'))
    approval['recovery_scope'] = 'One new metadata namespace; preserve v2 failure; exact Python3.10 summary arithmetic across server3.12, no tolerance/ranking/science change.'
    save(HERE/'BIND_APPROVAL.json', approval)
    lineage = dict(status='ROOT_REVIEWED_NAMESPACE_ONLY_HELPER_REBIND',
        original_helper_sha256=sha(OLD_HELPER), helper_sha256=sha(NEW_HELPER),
        exact_changes=replacements, all_other_helper_bytes_unchanged=True,
        inherited_helper_review='tmp/celeba_hybrid100_bind_helper_v3_delta_review_20261010/REVIEW.json',
        implementation_review_sha256=sha(REVIEW),
        prior_failure_sha256=sha(prior/'REMOTE_BIND_FAILURE.json'),
        actual_diagnostic_sha256=sha(prior/'RANK_RUNTIME_DIAGNOSTIC.json'),
        approval_sha256=sha(HERE/'BIND_APPROVAL.json'),
        executed=False, training_authorized=False, final_test=False)
    save(HERE/'ROOT_HELPER_REVIEW.json', lineage)
    print(json.dumps(lineage))

if __name__ == '__main__': main()
