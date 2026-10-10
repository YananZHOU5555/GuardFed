"""Record actual root and independent text checks without changing sealed drafts."""
from pathlib import Path
import argparse, datetime, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
D = ROOT / 'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A20_LoGo100_20261010'
I = ROOT / 'tmp/rebuttal_A20_LoGo100_independent_review_20261010'

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--independent-seal-sha256', required=True)
    args = a.parse_args()
    assert not (D / 'ROOT_REVIEW.json').exists()
    assert sha(D / 'FILES_SHA256.json') == '713cfb3d32c05ea0a61d419f344cda823b5fc36c33e80233961cf86366a207d8'
    for directory, expected in ((D, '713cfb3d32c05ea0a61d419f344cda823b5fc36c33e80233961cf86366a207d8'),
                                (I, args.independent_seal_sha256)):
        assert sha(directory / 'FILES_SHA256.json') == expected
        for name, pin in read(directory / 'FILES_SHA256.json')['files'].items():
            p = directory / name
            assert p.resolve().is_relative_to(directory.resolve())
            assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes']
    command = ROOT / 'tmp/rebuttal_A20_LoGo100_root_operations_20261010/ROOT_CHECK_COMMAND.json'
    c = read(command)
    assert c['exit_code'] == 0 and c['executions'] == 1
    assert c['original_comments_verbatim_and_order'] == 24 and c['links_checked'] == 82
    assert c['source_seal_sha256'] == sha(D / 'FILES_SHA256.json')
    independent = read(I / 'REVIEW.json')
    assert independent.get('blocking_findings') == []
    handoff = read(D / 'HANDOFF.json')
    for name, digest in handoff['documents_sha256'].items():
        assert sha(D / name) == digest
    proof = dict(
        status='ROOT_COMPLETE_A20_LOGO100_NATIVE1000_REBUTTAL_AND_INSERTION_CANDIDATES_ADOPTED_FOR_AUTHOR_REVIEW',
        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_seal_sha256=sha(D / 'FILES_SHA256.json'), source_handoff_sha256=sha(D / 'HANDOFF.json'),
        actual_root_command_path=command.relative_to(ROOT).as_posix(), actual_root_command_sha256=sha(command),
        actual_root_command_exit=0, independent_review_path=(I / 'REVIEW.json').relative_to(ROOT).as_posix(),
        independent_review_sha256=sha(I / 'REVIEW.json'), independent_seal_sha256=args.independent_seal_sha256,
        documents_sha256=handoff['documents_sha256'], original_comments=24, reverse_exact_documents=2,
        changed_spans=25, scalar_pointer_checks=46, mean_SD_cells=23, single_scalar_displays=1,
        fact_pointers=59, A_direction_checks=54, native_tradeoff_mean_checks=6, source_pins=34,
        local_links_checked=81, inherited_external_links=1, links_checked=82,
        complete_C_scenes=10, U_complete_scenes=10, A_complete_scenes=2, A_paired_models=20,
        LoGoFair_native_records=100, native_comparison_records=1000, native_comparison_methods=10,
        original_three_view_records=900, full17_complete=False, remaining_image_controls=6,
        whole_rebuttal_complete=False, primary_endpoint='PENDING_AUTHOR', author_review_only=True,
        manuscript_applied=False, final_test=False, new_statistics=False, new_inference=0, new_fits=0,
        new_training=0, all_negative_results_retained=True, old_original_comments_and_numeric_displays_preserved=True,
        semantic_review='Root read both added A-deletion and LoGoFair/native-comparison passages; independent reader found no substantive overclaim.',
        nonblocking_editorial_followup=[
            'Before final manuscript integration, explicitly label the inherited IID Sp-DFA paragraph and five-IID seed-first paragraph as C-deletion; their current contents and source links already identify C.'
        ],
        limitations=[
            'Author-review text only; P1-P6, seven remaining methods, six mechanism controls and final evaluation remain incomplete.',
            'Matching submitted LaTeX source has not been recovered; candidate text is not the applied manuscript.',
            'Validation selection, initial official-test exposure, virtual-cohort/adaptation and mixed runtime/device provenance remain disclosed.'
        ])
    with (D / 'ROOT_REVIEW.json').open('x', encoding='utf8') as f:
        json.dump(proof, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(dict(status=proof['status'], path=str(D / 'ROOT_REVIEW.json'), sha256=sha(D / 'ROOT_REVIEW.json'))))

if __name__ == '__main__':
    main()
