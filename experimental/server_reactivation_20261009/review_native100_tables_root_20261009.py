"""Independently check accepted native100 tables, then copy the exact sealed packet."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_mechanism_native100_tables_20261009'
TARGET = ROOT / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
assert sha(BASE / 'FILES_SHA256.json') == 'f42edf527aa9bf6ed94933d508c13c351b62c5fe6f25f2ea3399f82f7fc5385b'
seal = read(BASE / 'FILES_SHA256.json')['files']
for name, pin in seal.items():
    assert sha(BASE / name) == pin['sha256'] and (BASE / name).stat().st_size == pin['bytes']
inputs = read(BASE / 'INPUTS.json')['files']
for name, pin in inputs.items():
    assert sha(ROOT / name) == pin['sha256'] and (ROOT / name).stat().st_size == pin['bytes']
inspection_path = next(ROOT / name for name in inputs if name.endswith('/inspection.json'))
inspection = read(inspection_path)
assert inspection['new_count'] == 104 and inspection['reused_count'] == 100 and not inspection['invalid']
rows = inspection['records']
assert len(rows) == 204
indexed = {(r['variant'], r['distribution'], r['attack'], r['seed']): r for r in rows}
assert len(indexed) == len(rows)
tables = read(BASE / 'rendered/tables.json')
assert tables['inspection_sha256'] == sha(inspection_path) and tables['complete_paired_scenes'] == 10
assert set(tables['accepted_new_ids']) == set(inspection['accepted_new_ids'])
expected_scenes = {('IID', a) for a in ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']}
expected_scenes |= {('non-IID', a) for a in ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']}
checks = 0
max_difference = 0.
for panel, seeds in zip(tables['panels'], [range(91001, 91011), range(91002, 91011), range(91005, 91011)]):
    seeds = list(seeds)
    assert panel['seeds'] == seeds and len(panel['rows']) == 30
    assert {(r['distribution'], r['attack']) for r in panel['rows']} == expected_scenes
    assert len({(r['distribution'], r['attack'], r['variant']) for r in panel['rows']}) == 30
    for row in panel['rows']:
        assert row['n'] == row['expected_n'] == len(seeds) and row['complete'] and row['seeds'] == seeds
        assert row['variant'] in ('Full', 'minus_U', 'minus_U minus Full')
        for metric in ('accuracy_pct', 'aeod', 'aspd'):
            points = []
            for seed in seeds:
                key = row['distribution'], row['attack'], seed
                if row['variant'] == 'minus_U minus Full':
                    points.append(indexed[('minus_U', *key)][metric] - indexed[('Full', *key)][metric])
                else:
                    points.append(indexed[(row['variant'], *key)][metric])
            mean = math.fsum(points) / len(points)
            sd = math.sqrt(math.fsum((x - mean) ** 2 for x in points) / (len(points) - 1))
            for name, value in [('mean', mean), ('sample_sd_ddof1', sd)]:
                saved = row[metric][name]
                assert f'{value:.10f}' == f'{saved:.10f}'
                max_difference = max(max_difference, abs(value - saved))
                checks += 1
assert checks == 540
old_table_path = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim92_20261009/TABLES.md'
assert old_table_path.relative_to(ROOT).as_posix() in inputs
old_lines = [x for x in old_table_path.read_text(encoding='utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
new_lines = [x for x in (BASE / 'rendered/TABLES.md').read_text(encoding='utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
assert len(old_lines) == 54 and len(new_lines) == 60 and all(x in new_lines for x in old_lines)
assert not TARGET.exists() and not (BASE / 'ROOT_REVIEW.json').exists()
proof = dict(status='ROOT_NATIVE100_TEN_SCENE_GRID_AND_INDEPENDENT_STATISTICS_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_seal_sha256=sha(BASE / 'FILES_SHA256.json'), inspection_sha256=sha(inspection_path),
    inputs_verified=len(inputs), sealed_members_verified=len(seal), accepted_native100=100, accepted_native_controls=104, minus_C_partial=4,
    Full_references=100, displayed_pairs=100, complete_scenes=10, scalar_checks=checks,
    max_abs_difference=max_difference, precision_checked_decimal_places=10,
    prior_nine_scene_display_lines_exact=54, new_three_view_acceptance=0,
    partial_other_variant_excluded=True, new_inference=0, test=False,
    native_acceptance_tolerance_unchanged=True, whole_rebuttal_complete=False)
with (BASE / 'ROOT_REVIEW.json').open('x', encoding='utf8', newline='\n') as stream:
    json.dump(proof, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
TARGET.mkdir(parents=True)
for name in [*seal, 'FILES_SHA256.json', 'ROOT_REVIEW.json']:
    (TARGET / name).parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(BASE / name, TARGET / name)
    assert sha(BASE / name) == sha(TARGET / name)
print(json.dumps(dict(status=proof['status'], root_proof_sha256=sha(BASE / 'ROOT_REVIEW.json'),
    complete_scenes=10, scalar_checks=checks, canonical=str(TARGET))))
