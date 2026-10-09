"""Independent delivery checks; writes a receipt and optional file manifest."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(name: str):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def rows(name: str):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert abs(float(a) - float(b)) <= 1e-12, (a, b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true', help='Write receipt and final SHA manifest after checks.')
    args = parser.parse_args()
    a = load('acceptance.json')
    assert a['status'] == 'PARTIAL_PROVENANCE_WITH_VERIFIED_NUMERICAL_RECOVERY'
    assert not a['training_run'] and not a['old_results_edited']
    for entry in load('input_receipts.json'):
        path = Path(entry['path'])
        assert digest(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
    archive = load('archive_search.json')
    assert digest(ROOT / archive['archive']) == archive['archive_sha256']
    restored = 0
    for member in archive['inspected_members']:
        if 'restored_as' in member:
            assert digest(OUT / member['restored_as']) == member['sha256']
            restored += 1
    additional = load('additional_source_search.json')
    for bundle in additional['additional_archives']:
        assert digest(ROOT / bundle['archive']) == bundle['sha256']
        for member in bundle['source_files']:
            if 'restored_as' in member:
                assert digest(OUT / member['restored_as']) == member['sha256']
                restored += 1
    # Original line bytes/line numbers are compared to the pinned full raw file.
    original = (ROOT / 'tmp/table2_rawaudit_20261002/paper_tables_raw.jsonl').read_bytes().splitlines(keepends=True)
    delivered = (OUT / 'server_generation_original_260.jsonl').read_bytes().splitlines(keepends=True)
    table = rows('per_record_260.csv')
    assert len(table) == len(delivered) == 260
    missing_common = 0
    pca_missing_common = 0
    identity = set()
    for row, line in zip(table, delivered):
        assert line == original[int(row['raw_line']) - 1]
        assert hashlib.sha256(line).hexdigest() == row['raw_line_sha256']
        r = json.loads(line)
        assert row['run_id'] == r['run_id'] and r['seed'] == 123
        identity.add((r['dataset'], r['distribution'], r['attack'], r['config']['experiment_tag']))
        values = np.array([[s['metrics'][key] for key in ['accuracy', 'aeod', 'aspd']]
                           for s in r['last10_metrics']], dtype=float)
        assert values.shape == (10, 3)
        assert [s['round'] for s in r['last10_metrics']] == list(range(61, 71))
        selected = np.array([values[:, 0].max(), values[:, 1].min(), values[:, 2].min()])
        for key, value in zip(['ACC', 'AEOD', 'ASPD'], selected):
            close(row[key], value)
        match = np.isclose(values, selected, atol=1e-12, rtol=0).all(axis=1)
        matching_rounds = (np.flatnonzero(match) + 61).tolist()
        assert matching_rounds == json.loads(row['common_attaining_rounds'])
        missing_common += not bool(matching_rounds)
        pca_missing_common += row['synthetic_method'] == 'pca_gaussian' and not bool(matching_rounds)
        joint_idx = int(np.argmax(values[:, 0] - .5 * (values[:, 1] + values[:, 2])))
        assert int(row['joint_round']) == joint_idx + 61
        for key, value in zip(['joint_ACC', 'joint_AEOD', 'joint_ASPD'], values[joint_idx]):
            close(row[key], value)
        for key, value in zip(['terminal_ACC', 'terminal_AEOD', 'terminal_ASPD'], values[-1]):
            close(row[key], value)
    assert len(identity) == 260 and missing_common == 250 and pca_missing_common == 40
    pca = rows('pca_records_40.csv')
    assert len(pca) == 40 and {r['raw_line_sha256'] for r in pca} == {
        r['raw_line_sha256'] for r in table if r['synthetic_method'] == 'pca_gaussian'}
    points = rows('candidate_figure_points_26.csv')
    assert len(points) == 26
    for p in points:
        source = [r for r in table if r['dataset'] == p['dataset'] and r['tag'] == p['tag']]
        assert len(source) == 10 and {r['seed'] for r in source} == {'123'}
        assert len({(r['distribution'], r['attack']) for r in source}) == 10
        close(p['all_scenarios_ACC_pct'], np.mean([float(r['ACC_pct']) for r in source]))
        half_sum = np.mean([.5 * (float(r['AEOD']) + float(r['ASPD'])) for r in source])
        close(p['all_scenarios_1_minus_half_sum'], 1 - half_sum)
        close(p['all_scenarios_1_minus_sum'], 1 - 2 * half_sum)
    # Verify the submitted PDF's actual embedded Fig.3 image identity anew.
    pdf_receipt = load('submission_figure_receipt.json')
    pdf = Path(pdf_receipt['pdf_path'])
    assert digest(pdf) == pdf_receipt['pdf_sha256']
    with fitz.open(pdf) as doc:
        page = doc[10]
        assert any(info[0] == 35 for info in page.get_images(full=True))
        payload = doc.extract_image(35)['image']
        assert hashlib.sha256(payload).hexdigest() == digest(OUT / 'page11_image_35.png')
        text = page.get_text()
        assert 'the results under S-DFA' in text and 'AEOD + ASPD' in text
    checks = rows('raster_candidate_checks_104.csv')
    assert len(checks) == 104
    raster = load('raster_compatibility.json')
    for name, stats in raster['hypotheses'].items():
        group = [r for r in checks if r['hypothesis'] == name]
        assert len(group) == 26
        distances = [float(r['nearest_same_colour_distance_px']) for r in group]
        assert sum(d <= 2 for d in distances) == stats['within_2px']
        assert sum(d <= 22 for d in distances) == stats['within_22px']
        close(max(distances), stats['max_distance_px'])
    source_receipt = load('pca_source_receipt.json')
    core = OUT / 'original_sources/scripts/reproduce_paper_tables.py'
    assert digest(core) == source_receipt['core_sha256']
    text = core.read_text(encoding='utf-8')
    functions = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    for receipt in source_receipt['function_receipts']:
        segment = ast.get_source_segment(text, functions[receipt['function']])
        assert hashlib.sha256(segment.encode()).hexdigest() == receipt['source_sha256']
        assert (OUT / receipt['excerpt']).read_text(encoding='utf-8') == segment + '\n'
    report = {'status': 'PASS_WITH_EXPLICIT_PARTIAL_PROVENANCE',
              'original_raw_lines_checked': 260, 'pca_lines_checked': 40,
              'old_triplets_without_common_checkpoint': 250,
              'pca_old_triplets_without_common_checkpoint': 40,
              'candidate_aggregates_checked': 26, 'raster_hypotheses_checks': 104,
              'restored_original_source_files_checked': restored,
              'archive_hashes_checked': 3, 'pdf_embedded_image_identity_checked': True,
              'finite_source_search': {
                  'archive_text_source_members': archive['text_source_count'] + sum(len(r['source_files']) for r in additional['additional_archives']),
                  'archive_rasters': len(archive['raster_images']) + sum(len(r['raster_images']) for r in additional['additional_archives']),
                  'local_text_sources': len(load('bounded_search.json')['local_text_files']) + len(additional['extra_local_sources']),
                  'local_rasters': len(load('bounded_search.json')['local_raster_images']),
                  'reachable_git_commits': load('bounded_search.json')['git_commit_count'],
                  'git_text_source_blobs': load('bounded_search.json')['git_source_blob_count'] + sum('hits' in r for r in additional['additional_git_blobs']),
                  'git_raster_blobs': sum('size' in r for r in additional['additional_git_blobs']),
                  'exact_submission_asset_found': False,
                  'original_Fig3_script_found': False,
                  'historical_ForestDiffusion_call_implementation_found': False,
                  'scope': 'Named archives and explicit project paths only; inventories retain duplicate copies and do not prove global absence.'},
              'limitations_preserved': a['still_open']}
    if args.seal:
        (OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        files = []
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.name != 'FILES_SHA256.json' and '__pycache__' not in path.parts:
                files.append({'path': str(path.relative_to(OUT)).replace('\\', '/'),
                              'sha256': digest(path), 'bytes': path.stat().st_size})
        (OUT / 'FILES_SHA256.json').write_text(json.dumps({'files': files}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    else:
        existing = load('FILES_SHA256.json')['files']
        actual = {str(p.relative_to(OUT)).replace('\\', '/') for p in OUT.rglob('*')
                  if p.is_file() and p.name != 'FILES_SHA256.json' and '__pycache__' not in p.parts}
        assert actual == {r['path'] for r in existing}
        for item in existing:
            path = OUT / item['path']
            assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        report['sealed_files_checked'] = len(existing)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
