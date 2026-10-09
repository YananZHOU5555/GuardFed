"""Read-only, bounded recovery of the historical synthetic figure/PCA lineage.

Writes only beside this file. Does not train, execute archived source, or change
the submitted figure/results. Raster compatibility is not source identity.
"""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
ARCHIVE = ROOT / 'tmp/git_sync_assets/guardfed-5090-20260923.tar.gz'
RAW = ROOT / 'tmp/table2_rawaudit_20261002/paper_tables_raw.jsonl'
PDF = Path('E:/Edge下载/IEEE_TDSC__二次版___GuardFed____.pdf')
EXPECT_ARCHIVE = 'a2a0384c9e02d1e3c0cca176d2ea53ef99e37016eff3475ccc779765961772e0'
EXPECT_RAW = 'c94441fc33b4a460008bcb6eac409a856ae4ad79ed8baa132866342120f4e541'
EXPECT_PDF = '549a4191b1ac560bdaba79d9dce3b11693ab69c4a8cc0f7f52be31afa1110071'
RAW_MEMBER = './results/paper_tables/raw_results.jsonl'
CORE_MEMBER = './scripts/reproduce_paper_tables.py'
DRIVER_MEMBER = './scripts/run_ad2plus_advisor_experiments.py'
CSV_MEMBER = './results/paper_tables/advisor_experiments/server_generation_ablation.csv'
SUMMARY = ROOT / ('outputs/guardfed_tables/FOUR_CORE_EXPERIMENTS/'
                  '03_synthetic_generation_10pct/goal_revision_v2/'
                  'server_generation_ablation_summary_from_existing.csv')
LOCAL_CSV = ROOT / 'outputs/guardfed_tables/advisor_experiments_final/server_generation_ablation.csv'
LOCAL_GOAL_CSV = SUMMARY.with_name('server_generation_ablation_raw_from_existing.csv')
LOCAL_GOAL_PRODUCER = ROOT / 'outputs/guardfed_tables/FOUR_CORE_EXPERIMENTS/run_goal_revision_v2.py'
SEARCH_ROOTS = ['scripts', '.codex_remote', '.codex_transfer',
                'outputs/guardfed_tables', 'tmp/publish-5090/scripts',
                'tmp/publish-5090/outputs/guardfed_tables']
PATTERNS = {
    'forest': re.compile(r'ForestDiffusion|forest_diffusion', re.I),
    'plot_label': re.compile(r'Fairness score|FairScore'),
    'figure_input': re.compile(r'server_generation_ablation_summary_from_existing'),
    'pca': re.compile(r'pca_gaussian'),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def put_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def put_csv(name: str, rows: list[dict]) -> None:
    assert rows
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def identity(r: dict, raw: bool = False) -> tuple:
    if raw:
        return (r['dataset'], r['distribution'], r['attack'], int(r['seed']),
                r['config']['experiment_tag'])
    return (r['dataset'], r['distribution'], r['attack'], int(r['seed']), r['tag'])


def compatible(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


def search_hits(data: bytes) -> dict:
    s = data.decode('utf-8', errors='replace')
    out = {}
    for label, pat in PATTERNS.items():
        rows = [{'line': n, 'text': line[:500]} for n, line in enumerate(s.splitlines(), 1)
                if pat.search(line)]
        if rows:
            out[label] = rows
    return out


def archive_recovery() -> tuple[dict, dict]:
    assert file_sha(ARCHIVE) == EXPECT_ARCHIVE
    wanted = {RAW_MEMBER, CORE_MEMBER, DRIVER_MEMBER, CSV_MEMBER,
              './scripts/run_goal_revision_v2.py',
              './results/paper_tables/advisor_experiments/forest_diffusion_run.log',
              './results/paper_tables/advisor_experiments/synthetic_run.log'}
    saved = {}
    files, sources, binaries, images = [], [], [], []
    target = Image.open(OUT / 'page11_image_35.png').convert('RGB')
    target_pixel_sha = sha(target.tobytes())
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            suffix = Path(name).suffix.lower()
            inspect = (name in wanted or suffix in {'.py', '.tex', '.ipynb', '.pyc',
                                                   '.png', '.jpg', '.jpeg', '.svg', '.eps'})
            if not inspect:
                continue
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data = tar.extractfile(member).read()
            row = {'member': name, 'bytes': len(data), 'sha256': sha(data)}
            files.append(row)
            if name in wanted:
                saved[name] = data
                if name != RAW_MEMBER:
                    dest = OUT / 'original_sources' / name.lstrip('./')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    row['restored_as'] = str(dest.relative_to(OUT)).replace('\\', '/')
            if suffix in {'.py', '.tex', '.ipynb'}:
                sources.append({**row, 'hits': search_hits(data)})
            if suffix == '.pyc':
                binaries.append({**row, 'forest_byte_tokens': bool(re.search(b'forest', data, re.I))})
            if suffix in {'.png', '.jpg', '.jpeg'}:
                with Image.open(io.BytesIO(data)) as im:
                    im = im.convert('RGB')
                    images.append({**row, 'size': list(im.size), 'pixel_sha256': sha(im.tobytes()),
                                   'exact_submission_pixel_match': im.size == target.size
                                   and sha(im.tobytes()) == target_pixel_sha})
    assert sha(saved[RAW_MEMBER]) == EXPECT_RAW == file_sha(RAW)
    report = {'archive': str(ARCHIVE.relative_to(ROOT)).replace('\\', '/'),
              'archive_sha256': EXPECT_ARCHIVE, 'raw_member': RAW_MEMBER,
              'raw_sha256': EXPECT_RAW, 'inspected_members': files,
              'text_source_count': len(sources), 'text_sources': sources,
              'compiled_bytecode_count': len(binaries), 'compiled_bytecode': binaries,
              'raster_images': images, 'submission_pixel_sha256': target_pixel_sha,
              'scope': 'All .py/.tex/.ipynb/.pyc/.png/.jpg/.jpeg/.svg/.eps members in this one archive; named raw/log/CSV members.'}
    put_json('archive_search.json', report)
    return saved, report


def local_and_git_search() -> dict:
    # rg only indexes explicitly bounded project/source/output paths.
    p = subprocess.run(['rg', '--files', '-uuu', *SEARCH_ROOTS], cwd=ROOT,
                       stdout=subprocess.PIPE, check=True)
    paths = sorted(set(p.stdout.decode('utf-8').splitlines()))
    texts, images, other_plot_assets = [], [], []
    target = Image.open(OUT / 'page11_image_35.png').convert('RGB')
    pixel_sha = sha(target.tobytes())
    for rel in paths:
        path = ROOT / rel
        if path.suffix.lower() in {'.py', '.tex', '.ipynb'}:
            data = path.read_bytes()
            texts.append({'path': rel.replace('\\', '/'), 'sha256': sha(data),
                          'bytes': len(data), 'hits': search_hits(data)})
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            with Image.open(path) as im:
                im = im.convert('RGB')
                images.append({'path': rel.replace('\\', '/'), 'size': list(im.size),
                               'file_sha256': file_sha(path),
                               'exact_submission_pixel_match': im.size == target.size
                               and sha(im.tobytes()) == pixel_sha})
        if path.suffix.lower() in {'.svg', '.eps'}:
            data = path.read_bytes()
            other_plot_assets.append({'path': rel.replace('\\', '/'), 'sha256': sha(data),
                                      'hits': search_hits(data)})
    # Reachable Git objects from the checked-out repository; no network fetching.
    refs = subprocess.check_output(['git', 'rev-list', '--all'], cwd=ROOT).decode().splitlines()
    listed = subprocess.check_output(['git', 'rev-list', '--objects', '--all', '--',
                                      'scripts', '.codex_remote', '.codex_transfer',
                                      'outputs/guardfed_tables'], cwd=ROOT).decode('utf-8')
    objects = {}
    for line in listed.splitlines():
        if ' ' not in line:
            continue
        oid, path = line.split(' ', 1)
        if Path(path).suffix.lower() in {'.py', '.tex', '.ipynb', '.pyc', '.svg', '.eps'}:
            objects[oid] = path
    git_sources = []
    for oid, path in sorted(objects.items()):
        data = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=ROOT)
        row = {'git_blob': oid, 'path': path, 'sha256': sha(data), 'bytes': len(data)}
        row['hits'] = (search_hits(data) if Path(path).suffix.lower() != '.pyc'
                       else {'forest_byte_tokens': bool(re.search(b'forest', data, re.I))})
        git_sources.append(row)
    report = {'local_scope': SEARCH_ROOTS, 'local_text_files': texts,
              'local_raster_images': images, 'local_other_plot_assets': other_plot_assets,
              'git_reachable_commits': refs, 'git_commit_count': len(refs),
              'git_paths_scope': ['scripts', '.codex_remote', '.codex_transfer', 'outputs/guardfed_tables'],
              'git_source_blob_count': len(git_sources), 'git_sources': git_sources,
              'limitations': 'No personal-directory/global-disk scan, no unreachable/dangling Git objects, no remote access. Exact raster comparison covers inventoried PNG/JPEG files, not resized/cropped/re-encoded PDF/SVG derivatives.'}
    put_json('bounded_search.json', report)
    return report


def audit_records(saved: dict) -> tuple[list[dict], list[dict], dict]:
    original_lines = saved[RAW_MEMBER].splitlines(keepends=True)
    records = [(n, line, json.loads(line)) for n, line in enumerate(original_lines, 1)
               if b'server_generation_ablation' in line]
    records = [(n, line, r) for n, line, r in records
               if r.get('config', {}).get('experiment_suite') == 'server_generation_ablation']
    assert len(records) == 260
    assert len(set(identity(r, True) for _, _, r in records)) == 260
    assert set(r['seed'] for _, _, r in records) == {123}
    assert set(r['method'] for _, _, r in records) == {'GuardFed-AD2+'}
    archived_csv = read_csv(saved[CSV_MEMBER])
    table_sets = {'archive_advisor_csv': archived_csv,
                  'local_advisor_csv': read_csv(LOCAL_CSV.read_bytes()),
                  'local_goal_csv': read_csv(LOCAL_GOAL_CSV.read_bytes())}
    maps = {name: {identity(r): r for r in rows} for name, rows in table_sets.items()}
    assert all(len(rows) == len(mapping) == 260 for (name, rows), mapping in zip(table_sets.items(), maps.values()))
    details, raw_rows = [], []
    mismatch_count = 0
    max_err = 0.
    with (OUT / 'server_generation_original_260.jsonl').open('wb') as f:
        for n, line, r in records:
            f.write(line)
            cfg = r['config']
            samples = r['last10_metrics']
            assert len(samples) == 10 and [x['round'] for x in samples] == list(range(61, 71))
            assert r['rounds'] == cfg['rounds'] == 70
            assert r['alpha'] == (5000. if r['distribution'] == 'IID' else 5.)
            last = samples[-1]
            assert all(compatible(last['metrics'][k], r['metrics'][k]) for k in ('accuracy', 'aeod', 'aspd'))
            extrema = {k: (max if k == 'accuracy' else min)(x['metrics'][k] for x in samples)
                       for k in ('accuracy', 'aeod', 'aspd')}
            attaining = {k: [x['round'] for x in samples if compatible(x['metrics'][k], extrema[k])]
                         for k in extrema}
            common = sorted(set.intersection(*(set(v) for v in attaining.values())))
            if not common:
                mismatch_count += 1
            joint = max(samples, key=lambda x: x['metrics']['accuracy'] - .5 *
                        (x['metrics']['aeod'] + x['metrics']['aspd']))
            values = {'ACC': extrema['accuracy'], 'ACC_pct': 100 * extrema['accuracy'],
                      'AEOD': extrema['aeod'], 'ASPD': extrema['aspd'],
                      'fair_avg': .5 * (extrema['aeod'] + extrema['aspd']),
                      'score': extrema['accuracy'] - .5 * (extrema['aeod'] + extrema['aspd'])}
            for label, mapping in maps.items():
                row = mapping[identity(r, True)]
                for k, v in values.items():
                    err = abs(float(row[k]) - v)
                    max_err = max(max_err, err)
                    assert err <= 1e-12, (label, identity(r, True), k, err)
                assert row['synthetic_method'] == cfg['synthetic_method']
                assert compatible(row['synthetic_ratio'], cfg['synthetic_ratio'])
                if 'server_ratio' in row:
                    assert compatible(row['server_ratio'], cfg['server_ratio'])
            dc = r['data_contract']
            detail = {'raw_line': n, 'raw_line_sha256': sha(line), 'run_id': r['run_id'],
                      'dataset': r['dataset'], 'distribution': r['distribution'],
                      'alpha': r['alpha'], 'attack': r['attack'], 'seed': r['seed'],
                      'tag': cfg['experiment_tag'], 'synthetic_method': cfg['synthetic_method'],
                      'server_ratio': cfg['server_ratio'], 'synthetic_ratio': cfg['synthetic_ratio'],
                      'server_sampling': cfg['server_sampling'], 'rounds': r['rounds'],
                      'train_rows': dc['train_rows'], 'test_rows': dc['test_rows'],
                      'root_clean_rows': dc['root_clean_rows'], 'root_synthetic_rows': dc['root_synthetic_rows'],
                      'recorded_synthetic_method': dc.get('synthetic_method'),
                      'config_sha256_canonical': sha(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()),
                      **values,
                      'ACC_attaining_rounds': json.dumps(attaining['accuracy']),
                      'AEOD_attaining_rounds': json.dumps(attaining['aeod']),
                      'ASPD_attaining_rounds': json.dumps(attaining['aspd']),
                      'common_attaining_rounds': json.dumps(common),
                      'joint_round': joint['round'], 'joint_ACC': joint['metrics']['accuracy'],
                      'joint_AEOD': joint['metrics']['aeod'], 'joint_ASPD': joint['metrics']['aspd'],
                      'terminal_ACC': r['metrics']['accuracy'], 'terminal_AEOD': r['metrics']['aeod'],
                      'terminal_ASPD': r['metrics']['aspd']}
            details.append(detail)
            raw_rows.append(r)
    put_csv('per_record_260.csv', details)
    summary_rows = read_csv(SUMMARY.read_bytes())
    grouped = collections.defaultdict(list)
    for r in details:
        grouped[(r['dataset'], r['tag'])].append(r)
    assert len(grouped) == len(summary_rows) == 26
    points = []
    summary_maxerr = 0.
    for row in summary_rows:
        group = grouped[(row['dataset'], row['tag'])]
        assert len(group) == int(row['n']) == 10
        assert len(set((r['distribution'], r['attack']) for r in group)) == 10
        assert len(set(r['seed'] for r in group)) == 1
        for k in ('ACC_pct', 'AEOD', 'ASPD', 'fair_avg', 'score'):
            vals = [r[k] for r in group]
            for op, value in [('mean', sum(vals) / len(vals)), ('min', min(vals)), ('max', max(vals))]:
                err = abs(value - float(row[f'{k}_{op}']))
                summary_maxerr = max(summary_maxerr, err)
                assert err <= 1e-12, (row['tag'], k, op, err)
        sdfa = [r for r in group if r['attack'] == 'S-DFA']
        points.append({'dataset': row['dataset'], 'tag': row['tag'],
                       'synthetic_method': row['synthetic_method'],
                       'real_ratio': group[0]['server_ratio'], 'synthetic_ratio': group[0]['synthetic_ratio'],
                       'seed': 123, 'independent_seeds': 1, 'scenario_count': 10,
                       'all_scenarios_ACC_pct': float(row['ACC_pct_mean']),
                       'all_scenarios_AEOD': float(row['AEOD_mean']),
                       'all_scenarios_ASPD': float(row['ASPD_mean']),
                       'all_scenarios_1_minus_half_sum': 1 - float(row['fair_avg_mean']),
                       'all_scenarios_1_minus_sum': 1 - 2 * float(row['fair_avg_mean']),
                       'sdfa_two_distributions_ACC_pct': sum(r['ACC_pct'] for r in sdfa) / 2,
                       'sdfa_two_distributions_1_minus_half_sum': 1 - sum(r['fair_avg'] for r in sdfa) / 2,
                       'sdfa_two_distributions_1_minus_sum': 1 - sum(r['AEOD'] + r['ASPD'] for r in sdfa) / 2,
                       'raw_lines': json.dumps([r['raw_line'] for r in group]),
                       'records_without_common_checkpoint': sum(not json.loads(r['common_attaining_rounds']) for r in group)})
    put_csv('candidate_figure_points_26.csv', points)
    pca = [r for r in details if r['synthetic_method'] == 'pca_gaussian']
    put_csv('pca_records_40.csv', pca)
    receipt = {'record_count': len(details), 'method_counts': dict(collections.Counter(r['synthetic_method'] for r in details)),
               'attack_counts': dict(collections.Counter(r['attack'] for r in details)),
               'seed_counts': dict(collections.Counter(r['seed'] for r in details)),
               'rounds': 70, 'last10_rounds': list(range(61, 71)),
               'three_raw_csvs_matched': list(table_sets), 'raw_csv_max_abs_error': max_err,
               'records_without_joint_checkpoint_for_old_triplet': mismatch_count,
               'pca_record_count': len(pca),
               'pca_records_without_joint_checkpoint_for_old_triplet': sum(not json.loads(r['common_attaining_rounds']) for r in pca),
               'summary_settings': len(points), 'summary_scenarios_per_setting': 10,
               'summary_independent_seeds_per_setting': 1, 'summary_max_abs_error': summary_maxerr,
               'historical_selection': 'Separately maximize ACC and minimize AEOD and ASPD over test rounds 61..70; not one common checkpoint in the indicated records.',
               'checkpoint_limitation': 'Metric-round identities are present; this task did not recover historical weight/checkpoint binary SHA. Numerical/record lineage is not generator or model byte identity.'}
    return details, points, receipt


def raster_compatibility(points: list[dict]) -> dict:
    im = np.asarray(Image.open(OUT / 'page11_image_35.png').convert('RGB')).astype(float)
    palette = {'none': (0, 114, 178), 'ctgan': (230, 159, 0),
               'forest_diffusion': (0, 158, 115), 'gaussian_copula': (213, 94, 0),
               'pca_gaussian': (204, 121, 167), 'smote': (240, 228, 66), 'tvae': (0, 161, 200)}
    distance = np.stack([np.linalg.norm(im - np.asarray(rgb), axis=2) for rgb in palette.values()])
    nearest = distance.argmin(axis=0)
    dmin = distance.min(axis=0)
    # Manual axis ticks from the native submitted raster. Stated uncertainty is
    # +/-1 px. No curve digitization is claimed. Each marker has radius ~25 px.
    calibration = {
        'adult': {'x_tick0': 76., 'x_px0': 142., 'x_px_per_unit': 88.6,
                  'y_tick0': 1., 'y_px0': 302.5, 'y_px_per_unit': 4200.,
                  'crop': [122, 261, 858, 997]},
        'compas': {'x_tick0': 60., 'x_px0': 1045., 'x_px_per_unit': 97.2,
                   'y_tick0': .95, 'y_px0': 392., 'y_px_per_unit': 2910.,
                   'crop': [1022, 261, 1758, 997]},
    }
    hypotheses = {
        'ten_scenarios_half_sum': ('all_scenarios_ACC_pct', 'all_scenarios_1_minus_half_sum'),
        'ten_scenarios_paper_sum': ('all_scenarios_ACC_pct', 'all_scenarios_1_minus_sum'),
        'sdfa_only_half_sum': ('sdfa_two_distributions_ACC_pct', 'sdfa_two_distributions_1_minus_half_sum'),
        'sdfa_only_paper_sum': ('sdfa_two_distributions_ACC_pct', 'sdfa_two_distributions_1_minus_sum'),
    }
    rows = []
    for p in points:
        c = calibration[p['dataset']]
        idx = list(palette).index(p['synthetic_method'])
        left, top, right, bottom = c['crop']
        mask = (nearest == idx) & (dmin < 65)
        mask[:top] = False
        mask[bottom:] = False
        mask[:, :left] = False
        mask[:, right:] = False
        yy, xx = np.nonzero(mask)
        for hypothesis, (xkey, ykey) in hypotheses.items():
            x = c['x_px0'] + (p[xkey] - c['x_tick0']) * c['x_px_per_unit']
            y = c['y_px0'] + (c['y_tick0'] - p[ykey]) * c['y_px_per_unit']
            d = np.hypot(xx - x, yy - y)
            at = int(d.argmin())
            rows.append({'dataset': p['dataset'], 'tag': p['tag'], 'hypothesis': hypothesis,
                         'candidate_ACC_pct': p[xkey], 'candidate_fairness': p[ykey],
                         'candidate_x_px': x, 'candidate_y_px': y,
                         'nearest_same_colour_x_px': int(xx[at]), 'nearest_same_colour_y_px': int(yy[at]),
                         'nearest_same_colour_distance_px': float(d[at]),
                         'within_2px': bool(d[at] <= 2), 'within_22px': bool(d[at] <= 22),
                         'occluded_centre_under_ten_scenario_candidate':
                         p['tag'] == 'real10_none' and p['dataset'] == 'adult' or
                         p['tag'] == 'real5_forest_diffusion_synth5' and p['dataset'] == 'compas'})
    put_csv('raster_candidate_checks_104.csv', rows)
    result = {}
    for h in hypotheses:
        rs = [r for r in rows if r['hypothesis'] == h]
        result[h] = {'within_2px': sum(r['within_2px'] for r in rs),
                     'within_22px': sum(r['within_22px'] for r in rs),
                     'max_distance_px': max(r['nearest_same_colour_distance_px'] for r in rs)}
    report = {'kind': 'Raster compatibility only, not provenance certification or exact digitization.',
              'image': 'page11_image_35.png', 'image_sha256': file_sha(OUT / 'page11_image_35.png'),
              'calibration': calibration, 'axis_tick_uncertainty_px': 1,
              'method_colour_palette': palette, 'colour_nearest_palette_cutoff_rgb_distance': 65,
              'central_colour_tolerance_px': 2, 'marker_extent_tolerance_px': 22,
              'two_occluded_centres': ['adult/real10_none', 'compas/real5_forest_diffusion_synth5'],
              'hypotheses': result,
              'interpretation': 'Ten-scenario 1-minus-half-sum candidate agrees strongly with the 26 submitted marker locations. Two marker centres are occluded by other methods. Original plotting script/input receipt is still absent; do not turn this compatibility test into a file-SHA provenance claim.'}
    put_json('raster_compatibility.json', report)
    return report


def pca_source_receipt(saved: dict) -> dict:
    data = saved[CORE_MEMBER]
    code = data.decode('utf-8')
    parsed = ast.parse(code)
    names = ['pca_gaussian_augment', 'make_synthetic_root', '_project_synthetic_columns', 'load_bundle']
    funcs = []
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            source = ast.get_source_segment(code, node)
            assert source
            dest = OUT / 'source_excerpts' / f'{node.name}.py.txt'
            dest.parent.mkdir(exist_ok=True)
            dest.write_text(source + '\n', encoding='utf-8')
            funcs.append({'function': node.name, 'first_line': node.lineno, 'last_line': node.end_lineno,
                          'source_sha256': sha(source.encode()), 'ast_sha256': sha(ast.dump(node, include_attributes=False).encode()),
                          'excerpt': str(dest.relative_to(OUT)).replace('\\', '/')})
    assert len(funcs) == 4
    pca = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'pca_gaussian_augment')
    calls = sorted(set(ast.unparse(node.func) for node in ast.walk(pca) if isinstance(node, ast.Call)))
    report = {'archive_core_member': CORE_MEMBER, 'core_sha256': sha(data), 'function_receipts': funcs,
              'pca_calls': calls,
              'pca_implementation_observed': 'Fits root-column mean and full covariance, adds diagonal shrinkage, samples multivariate normal; diagonal fallback on LinAlgError; projects columns. No explicit PCA decomposition or dimension reduction is present.',
              'record_link_limit': 'Archive preserves this implementation and 40 PCA-labelled result records. Those records do not carry immutable per-run source/dependency/model/cache hashes; this source is a historical candidate, not a proved executed-source byte identity.',
              'forest_implementation_in_this_core': False,
              'executed_generator_reproduction_claim': False}
    put_json('pca_source_receipt.json', report)
    return report


def main() -> None:
    assert ROOT.name == 'GuardFed', ROOT
    assert file_sha(PDF) == EXPECT_PDF
    saved, archive_search = archive_recovery()
    search = local_and_git_search()
    details, points, numeric = audit_records(saved)
    raster = raster_compatibility(points)
    pca = pca_source_receipt(saved)
    inputs = [SUMMARY, LOCAL_CSV, LOCAL_GOAL_CSV, LOCAL_GOAL_PRODUCER,
              ROOT / 'tmp/pdfs/tdsc_submission/submission.txt', RAW, PDF]
    receipts = [{'path': str(p), 'sha256': file_sha(p), 'bytes': p.stat().st_size} for p in inputs]
    put_json('input_receipts.json', receipts)
    report = {'status': 'PARTIAL_PROVENANCE_WITH_VERIFIED_NUMERICAL_RECOVERY',
              'training_run': False, 'old_results_edited': False,
              'numeric_audit': numeric,
              'raster_compatibility': raster['hypotheses'],
              'pca_archived_candidate_implementation_sha256': pca['core_sha256'],
              'archive_text_sources_checked': archive_search['text_source_count'],
              'archive_bytecode_checked': archive_search['compiled_bytecode_count'],
              'archive_raster_assets_checked': len(archive_search['raster_images']),
              'local_text_files_checked': len(search['local_text_files']),
              'local_raster_assets_checked': len(search['local_raster_images']),
              'reachable_git_commits_checked': search['git_commit_count'],
              'git_source_blobs_checked': search['git_source_blob_count'],
              'archive_exact_submission_raster_matches': [r for r in archive_search['raster_images'] if r['exact_submission_pixel_match']],
              'local_exact_submission_raster_matches': [r for r in search['local_raster_images'] if r['exact_submission_pixel_match']],
              'closed': ['260 original numerical records with raw line/config/round identity',
                         '40 PCA-labelled records and archived candidate PCA-Gaussian source',
                         '26 candidate input points: 10 scenarios, one seed, exact summary arithmetic'],
              'still_open': ['Original Fig.3 plotting script and exact point-to-input source receipt',
                             'Historical executed ForestDiffusion implementation/dependency/fit/cache identity',
                             'Per-run immutable source/model/cache hashes for historical PCA and other generators',
                             'Historical checkpoint binary SHA; original records retain only metric/round identity in this recovery'],
              'paper_description_conflicts': ['Fig.3 prose says S-DFA; the strong raster-compatible candidate averages both distributions and all five attacks.',
                                             'Eq.(32) says 1-(AEOD+ASPD); the strong raster-compatible candidate is 1-0.5*(AEOD+ASPD).',
                                             'The strong raster-compatible candidate inherits independent extrema, not a common checkpoint for most records.'],
              'claim_boundary': 'Conflicts are evidence-led discrepancies in the recovered candidate chain. Missing original script prevents absolute attribution of the submitted graphic to that chain. Do not claim full figure/generator provenance or repaired historical evaluation.'}
    put_json('acceptance.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
