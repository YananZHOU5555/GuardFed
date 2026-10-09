"""Independently join the delivered table to original accepted receipts; no inference."""
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
import datetime, hashlib, itertools, json, math, shutil, tarfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'tmp/celeba_nine_method_three_view_tables_20261009'
DEST = ROOT / 'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'

def read(p):
    return json.loads(p.read_bytes())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

assert sha(SRC / 'FILES_SHA256.json') == '1bc00ab3e2f5b69f915730dc1b94fb9da49cc8cc69f9756f8b2ea092b3e33c8e'
assert sha(SRC / 'HANDOFF.json') == '80a119c820a0f647ec12f7cd6aee8ff32fb0e22e9d560cd05d091a3ceb64a1b1'
seal = read(SRC / 'FILES_SHA256.json')
for name, row in seal.items():
    p = SRC / name
    assert p.resolve().is_relative_to(SRC.resolve())
    assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], name
inputs = read(SRC / 'input_files_SHA256.json')
for name, row in inputs.items():
    p = ROOT / name
    assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], name

data = read(SRC / 'records_three_views_900.json')
records = data['records']
collector = read(ROOT / data['final_collector_path'])
inventory = read(ROOT / data['inventory_path'])
inv = {r['id']: r for r in inventory['records']}
old = read(ROOT / 'outputs/guardfed_tables/celeba_nine_method_final_20261004/source_acceptance_snapshot.json')
old_records = {r['id']: r for r in old['all_conditions']}
by_id = {r['id']: r for r in records}
assert len(records) == len(by_id) == 900 and set(by_id) == set(collector['accepted_ids']) == set(inv)
methods = sorted({r['method'] for r in records})
assert len(methods) == 9
expected = set(itertools.product(methods, ['IID', 'non-IID'], ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA'], range(91001, 91011)))
assert {tuple(r['scientific_cell']) for r in records} == expected
receipt_checks = count_checks = native_checks = 0
with ExitStack() as stack:
    archives = {}
    for r in records:
        assert r['original_inventory_record'] == inv[r['id']]
        assert r['same_checkpoint_all_views'] and not r['test_evaluation_performed']
        binding = r['source_binding']
        path = binding['archive']['path']
        if path not in archives:
            assert sha(ROOT / path) == binding['archive']['sha256']
            archives[path] = stack.enter_context(tarfile.open(ROOT / path))
        raw = archives[path].extractfile(binding['receipt_member']).read()
        assert hashlib.sha256(raw).hexdigest() == r['receipt_sha256']
        receipt = json.loads(raw)
        for k in ('checkpoint_sha256', 'model_inventory_record_sha256', 'config_canonical_sha256',
                  'runtime', 'root_reconstruction', 'fits', 'views', 'native_comparison', 'prediction_arrays_sha256'):
            record_key = 'prediction_arrays_sha256' if k == 'prediction_arrays_sha256' else k
            assert receipt[k] == r[record_key], (r['id'], k)
        assert receipt['valid_n'] == 19867 and not receipt['test_inference_performed']
        assert receipt['checkpoint_sha256'] == inv[r['id']]['checkpoint']['sha256']
        receipt_checks += 1
        prior = old_records[r['old_table_id']]
        assert prior['checkpoint_sha256'] == r['checkpoint_sha256']
        for k in ('accuracy', 'aeod', 'aspd'):
            assert r['views']['native'][k] == prior[k] == inv[r['id']]['prior_validation_metrics'][k]
            native_checks += 1
        for view, v in r['views'].items():
            g0, g1 = v['group_confusion_counts']['0'], v['group_confusion_counts']['1']
            n = g0['n'] + g1['n']
            assert n == v['prediction_count'] == 19867
            accuracy = sum(g['tp'] + g['tn'] for g in (g0, g1)) / n
            aeod = abs(g0['tp'] / g0['positives'] - g1['tp'] / g1['positives'])
            aspd = abs((g0['tp'] + g0['fp']) / g0['n'] - (g1['tp'] + g1['fp']) / g1['n'])
            for k, value in [('accuracy', accuracy), ('aeod', aeod), ('aspd', aspd)]:
                assert value == v[k], (r['id'], view, k)
                count_checks += 1

summary = read(SRC / 'summary_statistics.json')
panels = {'ten': list(range(91001, 91011)), 'nonselection_nine': list(range(91002, 91011)),
          'matching_six': list(range(91005, 91011))}
stat_checks = 0
max_difference = 0.0
for view, panels_data in summary.items():
    assert set(panels_data) == set(panels)
    for panel, rows in panels_data.items():
        assert len(rows) == 90
        for row in rows:
            assert row['seeds'] == panels[panel] and row['n'] == len(panels[panel])
            selected = [by_id[i] for i in row['IDs']]
            assert len(selected) == len(panels[panel])
            assert sorted(r['seed'] for r in selected) == panels[panel]
            assert all((r['method'], r['distribution'], r['attack']) == (row['method'], row['distribution'], row['attack']) for r in selected)
            for metric in ('accuracy', 'aeod', 'aspd'):
                values = [r['views'][view][metric] for r in selected]
                mean = math.fsum(values) / len(values)
                sd = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / (len(values) - 1))
                for computed, actual in ((mean, row[metric]['mean']), (sd, row[metric]['sample_sd'])):
                    difference = abs(computed - actual)
                    assert difference <= 1e-12
                    max_difference = max(max_difference, difference)
                    stat_checks += 1

devices = Counter('CPU' if r['runtime']['device'] == 'cpu' else 'GPU' for r in records)
training = Counter(r['training_torch'] for r in records)
assert devices == {'CPU': 434, 'GPU': 466}
assert training == {'2.11.0+cu128': 886, '2.11.0+cu130': 14}
proof = dict(status='ROOT_NINE_METHOD900_THREE_VIEW_RECEIPTS_COUNTS_AND_TABLE_STATISTICS_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_seal_sha256=sha(SRC / 'FILES_SHA256.json'),
    handoff_sha256=sha(SRC / 'HANDOFF.json'), input_files_rehashed=len(inputs), sealed_files=len(seal),
    unique_records=len(records), complete_cells=90, saved_source_receipts_rejoined=receipt_checks,
    group_count_metric_recomputations=count_checks, original_native_metrics_exact=native_checks,
    independent_mean_and_sampleSD_scalars=stat_checks, max_statistics_abs_difference=max_difference,
    original_native_displayed_cells_exact=1080, original_snapshot_SD_lastbit_differences=94,
    original_snapshot_SD_max_difference=2.7755575615628914e-17, original_tolerance=1e-12,
    inference_devices=dict(devices), original_training_environments=dict(training), new_inference=0,
    root_refit=False, final_test=False, formal_full17_complete=False, main_endpoint_pending=True)
with (SRC / 'ROOT_REVIEW.json').open('x', encoding='utf8', newline='\n') as f:
    json.dump(proof, f, ensure_ascii=False, indent=2)
    f.write('\n')
assert not DEST.exists()
DEST.mkdir(parents=True)
for name in [*seal, 'FILES_SHA256.json', 'HANDOFF.json', 'ROOT_REVIEW.json']:
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC / name, target)
    assert sha(target) == sha(SRC / name)
print(json.dumps(proof))
