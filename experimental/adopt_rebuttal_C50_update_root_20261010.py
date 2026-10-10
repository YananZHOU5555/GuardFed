"""Copy the source-checked C50 author-review update; does not apply manuscript text."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'tmp/rebuttal_C50_update_prepared_20261010'
TARGET = ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_C50_update_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
assert not sys.flags.optimize and not TARGET.exists()
seal = SOURCE/'FILES_SHA256.json'
assert sha(seal) == '3e8480134c8351e33ab8299d3ee8964b2def72620108e9c983e23ae83e47fa08'
files = read(seal)['files']
assert len(files) == 10
for name, pin in files.items():
    assert Path(name).name == name
    assert sha(SOURCE/name) == pin['sha256'] and (SOURCE/name).stat().st_size == pin['bytes']
handoff = read(SOURCE/'HANDOFF.json')
assert sha(SOURCE/'HANDOFF.json') == 'affbaa511cad5c0ef01d5b9578c6d75ed2a4929860d7f975cf069cd75777c71c'
assert handoff['root_adoption_sha256'] == '811ad551c1398f6e57681f59046c316b05c609ef30d4897158460b8121b1a28d'
assert sha(ROOT/handoff['root_adoption_path']) == handoff['root_adoption_sha256']
assert handoff['source_comments_and_original_complete_response_unchanged'] and handoff['negative_results_retained']
assert not handoff['manuscript_applied'] and not handoff['new_statistics'] and handoff['new_inference'] == handoff['new_training'] == 0
checked = subprocess.run([sys.executable, '-B', str(SOURCE/'check_sources.py')], cwd=ROOT, capture_output=True, text=True, encoding='utf8', check=True)
report = json.loads(checked.stdout)
expected = dict(source_pins=12, quoted_mean_SD_cells=8, quoted_scalar_pointers=16,
    original_comments_exact=2, fact_bindings=19, direction_checks=24, links_checked=9,
    unique_records=100, matched_pairs=50, complete_IID_scenes=5,
    native_shared_equal_metrics_and_counts=100, manuscript_candidate_paragraphs=2)
assert report['status'] == 'PASS_C50_SHORT_WRITING_SOURCES_NUMBERS_COMMENTS_SCOPE_LINKS'
assert all(report[k] == value for k, value in expected.items())
assert report['source_pointers_sha256'] == sha(SOURCE/'SOURCE_POINTERS.json')
assert sha(seal) == '3e8480134c8351e33ab8299d3ee8964b2def72620108e9c983e23ae83e47fa08'
TARGET.mkdir()
for name in [*files, 'FILES_SHA256.json']:
    shutil.copyfile(SOURCE/name, TARGET/name)
    assert sha(SOURCE/name) == sha(TARGET/name)
proof = dict(status='ROOT_C50_AUTHOR_REVIEW_UPDATE_QUOTED_VALUES_SCOPE_AND_SOURCE_PINS_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_seal_sha256=sha(seal),
    root_table_sha256=handoff['root_adoption_sha256'], independent_root_check=report,
    scalar_pointer_checks=16, display_cells_checked=8, scope_fact_checks=19, source_pins_checked=12,
    links_checked=9, verbatim_comment_excerpts=2, complete_C_scenes=5, old_24_comment_draft_unchanged=True,
    all_negative_results_retained=True, addendum_sha256=sha(TARGET/'C50_REVIEWER_ADDENDUM.md'),
    insertions_sha256=sha(TARGET/'C50_MANUSCRIPT_INSERTIONS.md'), author_review_only=True,
    manuscript_applied=False, final_endpoint_selected=False, final_test=False,
    whole_rebuttal_complete=False, new_statistics=False, new_CNN=0, new_training=0)
path = TARGET/'ROOT_REVIEW.json'
with path.open('x', encoding='utf8') as stream:
    json.dump(proof, stream, indent=2); stream.write('\n')
print(json.dumps(dict(status=proof['status'], root_path=str(path), root_sha256=sha(path), counts=expected)))
