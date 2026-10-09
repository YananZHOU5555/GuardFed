"""Independently bind the C40 author-review addendum to actual quoted evidence."""
from pathlib import Path
import argparse, datetime, hashlib, json, re, shutil, sys

if sys.flags.optimize:
    raise RuntimeError('Optimized Python forbidden')
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'tmp/rebuttal_C40_addendum_prepared_20261010'
TARGET = ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_C40_addendum_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())


def pointer(value, path):
    for part in path.lstrip('/').split('/') if path else []:
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-seal-sha256', required=True)
    args = parser.parse_args()
    assert not TARGET.exists(), 'Preserve prior adoption'
    seal = SOURCE/'FILES_SHA256.json'
    assert sha(seal) == args.source_seal_sha256
    files = read(seal)['files']
    for name, pin in files.items():
        assert Path(name).name == name
        assert sha(SOURCE/name) == pin['sha256'] and (SOURCE/name).stat().st_size == pin['bytes']
    handoff = read(SOURCE/'HANDOFF.json')
    pointers = read(SOURCE/'SOURCE_POINTERS.json')
    checked = read(SOURCE/'CHECK_RESULTS.json')
    assert handoff['root_table_sha256'] == pointers['root_adoption_sha256'] == 'eb08dbc328339e8293133bbd35f60e627ca70b7436ec3b20871bc149e6047172'
    assert sha(ROOT/handoff['root_table_path']) == handoff['root_table_sha256']
    assert handoff['author_review_only'] and handoff['DO_NOT_SUBMIT']
    assert not handoff['manuscript_applied'] and not handoff['old_24_comment_response_modified']
    data = {}
    for name, pin in pointers['source_pins'].items():
        path = ROOT/name
        assert sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes']
        if path.suffix == '.json':
            data[name] = read(path)
    documents = {name:(SOURCE/name).read_text('utf8') for name in handoff['new_documents']}
    assert set(documents) == {'C40_REVIEWER_ADDENDUM.md', 'C40_MANUSCRIPT_INSERTIONS.md'}
    for row in pointers['quoted_scalar_bindings']:
        value = pointer(data[row['source_path']], row['pointer'])
        assert value == row['value'] and type(value) == type(row['value'])
        for line in row['document_lines']:
            assert row['paired_cell'] in documents[row['document']].splitlines()[line-1]
    for row in pointers['quoted_mean_SD_cells']:
        value = pointer(data[row['source_path']], row['base_pointer'])
        places = row['decimal_places']
        display = f"{value['mean']:+.{places}f} ± {value['sample_sd_ddof1']:.{places}f}"
        assert display == row['display']
        for line in row['document_lines']:
            assert display in documents[row['document']].splitlines()[line-1]
    for row in pointers['fact_bindings']:
        assert pointer(data[row['source_path']], row['pointer']) == row['value']
        for name, phrase in row['document_phrases'].items():
            assert phrase in documents[name]
    for row in pointers['verbatim_comment_bindings']:
        assert (ROOT/row['source_path']).read_text('utf-8-sig').splitlines()[row['source_line']-1] == row['text']
        assert row['text'] in documents[row['document']]
    links = []
    for name, text in documents.items():
        assert 'AUTHOR_REVIEW / DO_NOT_SUBMIT' in text and 'not applied' in text
        assert ('six C' in text or 'six other C scenes' in text) and 'six other image-control variants' in text
        for target in re.findall(r'\]\(([^)]+)\)', text):
            assert Path(target).is_absolute() and Path(target).is_file()
            links.append((name, target))
    assert checked['status'] == 'PASS_C40_AUTHOR_REVIEW_WRITING_NUMERIC_POINTERS_SCOPE_AND_LINKS'
    assert checked['source_pointers_sha256'] == sha(SOURCE/'SOURCE_POINTERS.json')
    assert checked['verbatim_original_comment_excerpts_exact'] == len(pointers['verbatim_comment_bindings']) == 2
    assert checked['negative_direction_checks_passed'] >= 7
    assert sha(seal) == args.source_seal_sha256
    TARGET.mkdir()
    for name in list(files) + ['FILES_SHA256.json']:
        shutil.copyfile(SOURCE/name, TARGET/name)
        assert sha(SOURCE/name) == sha(TARGET/name)
    proof = dict(status='ROOT_C40_AUTHOR_REVIEW_ADDENDUM_QUOTED_VALUES_SCOPE_AND_SOURCE_PINS_PASS',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_seal_sha256=sha(seal), root_table_sha256=handoff['root_table_sha256'],
        scalar_pointer_checks=len(pointers['quoted_scalar_bindings']), display_cells_checked=len(pointers['quoted_mean_SD_cells']),
        scope_fact_checks=len(pointers['fact_bindings']), source_pins_checked=len(pointers['source_pins']), links_checked=len(links), verbatim_comment_excerpts=2,
        old_24_comment_draft_unchanged=True, complete_C_scenes=4, excluded_partial_C_records=0, all_negative_results_retained=True,
        addendum_sha256=sha(TARGET/'C40_REVIEWER_ADDENDUM.md'), insertions_sha256=sha(TARGET/'C40_MANUSCRIPT_INSERTIONS.md'),
        author_review_only=True, manuscript_applied=False, final_endpoint_selected=False, final_test=False,
        whole_rebuttal_complete=False, new_CNN=0, new_training=0)
    path = TARGET/'ROOT_REVIEW.json'
    with path.open('x', encoding='utf8') as stream:
        json.dump(proof, stream, indent=2)
        stream.write('\n')
    print(json.dumps(proof | dict(root_path=str(path), root_sha256=sha(path))))


if __name__ == '__main__':
    main()
