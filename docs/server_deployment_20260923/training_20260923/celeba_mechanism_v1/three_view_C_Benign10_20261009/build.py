"""PREPARED ONLY: join actual adopted C1+C11 receipts, then render one C scene."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
C1 = R / 'tmp/celeba_mechanism_valid_C1_gate_20261009'
C11 = R / 'tmp/celeba_mechanism_valid_C_after1_20261009'
B1 = C1 / 'execution_candidate/backups/incremental_20261009T185829Z'
FULL = R / 'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
EXPECTED = [f'minus_C_IID_Benign_seed{s}' for s in range(91002, 91011)] + [f'minus_C_IID_F Flip_seed{s}' for s in [91001, 91002]]
VIEWS = ('native', 'raw', 'shared_calibration')


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_bytes())
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result
def write(path, value):
    with path.open('x', encoding='utf-8') as f: f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def scope(prior, current):
    old = {r['id']: r for r in prior['records']}; actual = {r['id']: r for r in current['records']}
    need(len(old) == len(prior['records']) == 101 and len(actual) == len(current['records']) == 112, 'Duplicate/missing scientific inventory')
    need(prior['selected_replay_ids'] == ['minus_C_IID_Benign_seed91001'] and current['selected_replay_ids'] == EXPECTED, 'Exact C1/C11 scope required')
    need(set(actual) - set(old) == set(EXPECTED) and set(current['excluded_prior_replay_ids']) == set(old), 'Prior/new ID difference drift')
    need(all(actual[k] == row for k, row in old.items()), 'Original101 record changed')
    need(current['full_references'] == prior['full_references'] and len(current['full_references']) == 100, 'Original Full100 references changed')
    need(current['native_tolerance'] == prior['native_tolerance'] == 1e-12, 'Native tolerance changed')
    controls = {key: row for key, row in actual.items() if row['variant'] == 'minus_C'}
    need(set(controls) == set(EXPECTED) | {'minus_C_IID_Benign_seed91001'}, 'Only exact native C12 allowed')
    need(all(r['original_split'] == 'valid' and r['terminal_round'] == 70 and r['original_n_eval'] == 19867 for r in controls.values()), 'Split/round/count drift')
    need({(r['distribution'], r['attack'], r['seed']) for r in controls.values()} == {('IID', 'Benign', s) for s in range(91001, 91011)} | {('IID', 'F Flip', s) for s in [91001, 91002]}, 'Wrong C scene/seed')
    return controls


def adoption_gate(path, expected_sha):
    need(path.name == 'ROOT_ADOPTION_REVIEW.json' and path.resolve().is_relative_to((C11 / 'execution_candidate/backups').resolve()), 'Only actual C-after1 adoption permitted')
    need(len(expected_sha) == 64 and sha(path) == expected_sha, 'Actual external root review SHA required')
    proof = read(path)
    need(proof['prior_three_view_models'] == 101 and proof['accepted_new'] == 11 and proof['cumulative_three_view_models'] == 112, 'Actual adopted101+11 required')
    need(proof['accepted_new_ids'] == EXPECTED and proof['original101_unchanged'] and proof['source_scope_complete'], 'Adopted source/prior/exact cohort incomplete')
    need(proof['prior101_root_adoption_sha256'] == sha(B1 / 'ROOT_ADOPTION_REVIEW.json'), 'C1 root adoption chain drift')
    need(proof['new_training'] == proof['new_Full_inference'] == 0 and proof['test_inference'] is False, 'Forbidden new science')
    return proof


def source_functions(basis):
    reused = module('accepted92_saved_receipt_join', R / basis['accepted_join'])
    ns = dict(reused.__dict__)
    source = (R / basis['accepted_join']).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'accepted_increment')
    original = ast.get_source_segment(source, node)
    need(original.count("'minus_U'") == 2, 'Unexpected scientific join source')
    mapped = original.replace("'minus_U'", "'minus_C'")
    exec(compile(mapped, str(R / basis['accepted_join']), 'exec'), ns)
    function_proof = dict(accepted_increment_original_sha256=hashlib.sha256(original.encode()).hexdigest(), accepted_increment_C_variant_sha256=hashlib.sha256(mapped.encode()).hexdigest(), only_two_variant_literals_rebound=True)
    scientific_ns = dict(need=need, VIEWS=VIEWS, hashlib=hashlib, json=json)
    function_proof.update(reused.funcs(R / basis['receipt_identity_source'], ['receipt_identity', 'normalized'], scientific_ns))
    function_proof.update(reused.funcs(C11 / 'bridge.py', ['canonical'], scientific_ns))
    return ns['accepted_increment'], scientific_ns, function_proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-review', type=Path, required=True); parser.add_argument('--root-review-sha', required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    need(args.output.resolve().parent == H.resolve() and not args.output.exists(), 'Fresh owned output only')
    basis = read(H / 'INPUTS.json')
    for name, pin in basis['files'].items(): need(sha(R / name) == pin['sha256'] and (R / name).stat().st_size == pin['bytes'], 'Pinned source/input changed ' + name)
    i1 = read(C1 / 'inventory_actual101_Full100refs.json'); i12 = read(C11 / 'inventory_actual112_Full100refs.json'); controls = scope(i1, i12)
    adoption_gate(args.root_review, args.root_review_sha)
    join, scientific, function_proof = source_functions(basis)
    first, chain1 = join(B1, sha(B1 / 'ROOT_ADOPTION_REVIEW.json'), i1, C1 / 'inventory_actual101_Full100refs.json', C1 / 'bridge.py', 100, 101, scientific['receipt_identity'], scientific['normalized'], scientific['canonical'])
    new, chain11 = join(args.root_review.parent, args.root_review_sha, i12, C11 / 'inventory_actual112_Full100refs.json', C11 / 'bridge.py', 101, 112, scientific['receipt_identity'], scientific['normalized'], scientific['canonical'])
    c_records = first + new
    need(len(c_records) == len({r['id'] for r in c_records}) == 12 and {r['id'] for r in c_records} == set(controls), 'Missing/duplicate C receipts')
    baseline = read(FULL / 'records_three_views_900.json'); byid = {r['id']: r for r in baseline['records']}
    refs = {r['id']: r for r in i12['full_references']}; parent = module('accepted_U100_full_reference', R / basis['parent_builder'])
    full_ids = {row['paired_full']['id'] for row in controls.values()}
    full = [parent.full_record(byid[rid], refs[rid]) for rid in sorted(full_ids)]
    need(len(full) == 12, 'Exact corresponding Full12 references required')
    fullcells = {(r['distribution'], r['attack'], r['seed']): r for r in full}
    for record in c_records:
        inv = controls[record['id']]
        need(record['checkpoint_sha256'] == inv['checkpoint']['sha256'] and record['config_sha256'] == inv['config_canonical_sha256'] and record['data_contract'] == inv['data_contract'], 'C identity drift')
        need(record['data_contract'] == fullcells[record['distribution'], record['attack'], record['seed']]['data_contract'], 'Paired root/train/valid/client partition drift')
        need(all(abs(record['views']['native'][m] - inv['prior_validation_metrics'][m]) <= 1e-12 for m in ['accuracy', 'aeod', 'aspd']), 'Original native differs')
    records = full + c_records
    original = module('accepted_evidence_statistics', R / basis['evidence']); pure = module('one_C_scene', H / 'panels.py'); verifier = module('independent_C_table_arithmetic', H / 'verify_numeric.py')
    panels, coverage, paired = pure.panels(records, original); checks = verifier.verify(records, panels)
    native = read(R / basis['native_C_tables'])
    for oldpanel in native['panels']:
        newpanel = next(p for p in panels if p['view'] == 'native' and p['label'] == oldpanel['label'])
        for left, right in zip(oldpanel['rows'], newpanel['rows']):
            need(left['variant'] == right['variant'] and left['seeds'] == right['seeds'], 'Native panel/variant differs')
            for metric in original.METRICS:
                for stat in ['mean', 'sample_sd_ddof1']: need(abs(left[metric][stat] - right[metric][stat]) <= 1e-12, 'Original native C table differs')
    text = ['# CelebA Full–minus_C: IID Benign, three views', '', 'One complete scene, ten shared terminal checkpoints, valid-only (19,867), round70. Mean ± sample SD, ddof=1; paired differences are minus_C − Full. ACC is percent and its paired difference is percentage points. Primary endpoint remains pending.', '']
    for panel in panels:
        text += ['## ' + panel['view'] + ' — ' + panel['label'], '', '| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |', '|---|---:|---:|---:|---:|']
        for row in panel['rows']:
            values = [f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in original.METRICS]
            text.append('| ' + ' | '.join([row['variant'], str(row['n']), *values]) + ' |')
        text.append('')
    text += ['AEOD is absolute TPR gap, not full equalized odds. Native retains each procedure’s original root-only calibration; raw is uncalibrated; shared uses the frozen common root-only rule. All views use the same terminal checkpoint. No new calibration fit or Full inference is performed by this table builder.', 'IID F Flip has only two paired seeds; all four records are retained for identity/coverage but excluded from every mean. Other eight C scenes remain absent. Mixed CPU/GPU replay, historical/current CUDA/driver provenance, seed91001 selection, prior validation/test exposure and author-pending primary endpoint remain limitations. The 9/6 panels apply the same seed rule to both variants and are descriptive, not untouched confirmation sets.', 'No significance test, C necessity/causality claim, final test, or completed mechanism900 claim.', '']
    lines = [line for line in text if line.startswith('| ') and ' ± ' in line]
    flat_rows = [row for panel in panels for row in panel['rows']]
    need(len(lines) == len(flat_rows) == 27, 'Missing displayed row')
    displayed = 0
    for line, row in zip(lines, flat_rows):
        cells = line.strip('| ').split(' | ')
        for index, metric in enumerate(original.METRICS, 2):
            precision = 3 if metric == 'accuracy_pct' else 5
            need(cells[index] == f"{row[metric]['mean']:.{precision}f} ± {row[metric]['sample_sd_ddof1']:.{precision}f}", 'Rendered cell drift')
            displayed += 1
    need(displayed == 81, 'Missing displayed cell')
    checks.update(display_mean_sd_cells=displayed,original_native_table_scalar_checks=54,source_functions=function_proof)
    args.output.mkdir()
    write(args.output / 'records.json', dict(records=records)); write(args.output / 'tables.json', dict(status='C_SINGLE_COMPLETE_SCENE_THREE_VIEWS_PENDING_ROOT_TABLE_REVIEW', complete_scenes=1, table_model_records=20, preserved_records=24, partial_pairs=2, panels=panels, primary_endpoint_selected=False, final_test=False, new_inference=0, new_training=0, replay_devices={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v and r['attack']=='Benign')) for v in ['Full','minus_C']}))
    write(args.output / 'coverage.json', coverage); write(args.output / 'paired_per_seed.json', paired); write(args.output / 'verification.json', checks)
    write(args.output / 'SOURCE_BINDINGS.json', dict(prepared_input_pins=basis['files'],actual_root_review=str(args.root_review),actual_root_review_sha256=args.root_review_sha,archive_chains=[chain1,chain11],source_functions=function_proof))
    (args.output / 'TABLES.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps(dict(status='BUILT_C_SINGLE_SCENE_THREE_VIEWS_PENDING_ROOT_TABLE_REVIEW', scenes=1, displayed_records=20, preserved_records=24, mean_sd_scalars=162)))


if __name__ == '__main__': main()
