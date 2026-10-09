"""Prepared record-only C three-scene tables; actual after28 C8 adoption is required."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
OLD = R / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_two_scenes_20261009'
BENIGN = OLD.with_name('three_view_C_Benign10_20261009')
C8 = R / 'tmp/celeba_mechanism_valid_C_after28_20261009'
C25 = R / 'tmp/celeba_mechanism_valid_C_after25_20261009'
C20 = R / 'tmp/celeba_mechanism_valid_C_after20_20261009'
B20 = C20 / 'execution_candidate/backups/incremental_20261009T210932Z'
B25 = C25 / 'execution_candidate/backups/incremental_20261009T213828Z'
PRIOR = B25 / 'ROOT_ADOPTION_REVIEW.json'
EXPECTED = [f'minus_C_IID_FedSA_seed{s}' for s in [91009,91010]] + [f'minus_C_IID_S-DFA_seed{s}' for s in range(91001,91007)]
FEDSA = [f'minus_C_IID_FedSA_seed{s}' for s in range(91001,91011)]
METRICS = ['accuracy_pct', 'aeod', 'aspd']
VIEWS = ['native', 'raw', 'shared_calibration']


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result
def write(path, value):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def record_spans(text):
    position=text.index('[',text.index('"records"'))+1; decoder=json.JSONDecoder(); spans=[]
    while True:
        while text[position].isspace() or text[position]==',': position+=1
        if text[position]==']': return spans
        _,end=decoder.raw_decode(text,position); spans.append(text[position:end]); position=end


def scope(prior, current):
    a = {r['id']: r for r in prior['records']}; b = {r['id']: r for r in current['records']}
    need(len(a) == len(prior['records']) == 128 and len(b) == len(current['records']) == 136, 'Exact native128/native136 inventories required')
    need(set(b)-set(a) == set(EXPECTED) and all(b[k] == r for k,r in a.items()), 'Prior128 or exact C8 difference changed')
    need(current['selected_replay_ids'] == EXPECTED and set(current['excluded_prior_replay_ids']) == set(a), 'after28 C8 boundary changed')
    need(prior['full_references'] == current['full_references'] and len(current['full_references']) == 100, 'Full100 references changed')
    need(prior['native_tolerance'] == current['native_tolerance'] == 1e-12, 'Native tolerance changed')
    controls = {k:r for k,r in b.items() if r['variant'] == 'minus_C'}
    complete = {('IID',a,s) for a in ['Benign','F Flip','FedSA'] for s in range(91001,91011)}
    partial = {('IID','S-DFA',s) for s in range(91001,91007)}
    need(len(controls) == 36 and {(r['distribution'],r['attack'],r['seed']) for r in controls.values()} == complete | partial, 'Only complete C30 and explicit partial S-DFA6 allowed')
    need(all((r['terminal_round'],r['original_split'],r['original_n_eval']) == (70,'valid',19867) for r in controls.values()), 'Incomplete/non-valid control')
    return controls


def adoption_gate(path, expected_sha):
    need(path.name == 'ROOT_ADOPTION_REVIEW.json' and path.resolve().is_relative_to((C8/'execution_candidate/backups').resolve()), 'Only actual C8 backup ROOT adoption permitted')
    need(len(expected_sha) == 64 and set(expected_sha) <= set('0123456789abcdef') and sha(path) == expected_sha, 'Actual external C8 ROOT SHA required')
    proof = read(path)
    need(proof['status'] == 'ROOT_C_AFTER28_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS', 'C8 ROOT has not adopted strict/offserver outputs')
    need((proof['prior_three_view_models'],proof['accepted_new'],proof['cumulative_three_view_models']) == (128,8,136), 'Actual adopted128+8 required')
    need(proof['accepted_new_ids'] == EXPECTED and proof['original128_unchanged'] and proof['source_scope_complete'] and proof['negative_results_preserved'], 'Incomplete exact C8/prior/source preservation')
    need(proof['prior128_root_adoption_sha256'] == sha(PRIOR), 'Prior after25 adoption changed')
    need(proof['new_training'] == proof['new_Full_inference'] == 0 and proof['test_inference'] is False, 'Forbidden new science')
    return proof


def displayed_cells(text, panels):
    lines = [x for x in text.splitlines() if x.startswith('| ') and ' ± ' in x]
    rows = [r for p in panels for r in p['rows']]
    need(len(lines) == len(rows) == 81, 'Incomplete displayed rows')
    for line,row in zip(lines,rows):
        cells = line.strip('| ').split(' | ')
        for index,metric in enumerate(METRICS,3):
            precision = 3 if metric == 'accuracy_pct' else 5
            need(cells[index] == f"{row[metric]['mean']:.{precision}f} ± {row[metric]['sample_sd_ddof1']:.{precision}f}", 'Display drift')
    return len(rows)*3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--C8-adoption', type=Path, required=True)
    parser.add_argument('--C8-adoption-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(args.output.resolve().parent == H.resolve() and not args.output.exists(), 'Fresh owned output only')
    adoption_gate(args.C8_adoption, args.C8_adoption_sha256)
    inputs = read(H/'INPUTS.json')
    for name,pin in inputs['files'].items():
        need(sha(R/name) == pin['sha256'] and (R/name).stat().st_size == pin['bytes'], 'Input/source changed '+name)
    for name,pin in read(H/'FILES_SHA256.json')['files'].items():
        need(sha(H/name) == pin['sha256'], 'Prepared member changed '+name)
    prior = read(C25/'inventory_actual128_Full100refs.json'); current = read(C8/'inventory_actual136_Full100refs.json')
    controls = scope(prior,current)
    # Original accepted join, all member/strict/offserver guards, unchanged.
    accepted = module('accepted_C_table_source', BENIGN/'build.py')
    accepted.R = R; accepted.C11 = R/'tmp/celeba_mechanism_valid_C_after1_20261009'
    basis = read(BENIGN/'INPUTS.json')
    join,scientific,function_proof = accepted.source_functions(basis)
    added = []; chains = []
    for folder,stage,total_before,total_after in [(B20,C20,120,125),(B25,C25,125,128),(args.C8_adoption.parent,C8,128,136)]:
        inventory_path = stage/f'inventory_actual{total_after}_Full100refs.json'
        rows,chain = join(folder,sha(folder/'ROOT_ADOPTION_REVIEW.json'),read(inventory_path),inventory_path,stage/'bridge.py',total_before,total_after,scientific['receipt_identity'],scientific['normalized'],scientific['canonical'])
        added.extend(rows); chains.append(chain)
    need(len(added)==len({r['id'] for r in added})==16, 'Exact five plus three plus eight source receipts required')
    new_records = [r for r in added if r['attack']=='FedSA']
    partial_records = [r for r in added if r['attack']=='S-DFA']
    need({r['id'] for r in new_records}==set(FEDSA) and len(new_records)==10, 'Missing/duplicate FedSA10')
    need({r['id'] for r in partial_records}==set(EXPECTED[2:]) and len(partial_records)==6, 'Partial S-DFA6 must remain explicit')
    old_records = read(OLD/'snapshot/records.json')['records']
    need(len(old_records)==len({r['id'] for r in old_records})==40, 'Accepted old40 changed')
    full_source = read(R/inputs['full900_path'])
    byid = {r['id']:r for r in full_source['records']}; refs = {r['id']:r for r in current['full_references']}
    full_builder = module('accepted_Full_reference_normalizer',R/basis['parent_builder'])
    full = [full_builder.full_record(byid[controls[key]['paired_full']['id']],refs[controls[key]['paired_full']['id']]) for key in FEDSA]
    records = old_records + full + new_records
    need(len(records)==len({r['id'] for r in records})==60 and records[:40]==old_records, 'Exact60/original40 identity changed')
    old_spans=record_spans((OLD/'snapshot/records.json').read_text('utf8'))
    new_spans=record_spans(json.dumps(dict(records=records),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    need(len(old_spans)==40 and new_spans[:40]==old_spans, 'Original40 JSON record bytes changed')
    pairs = {(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
    for record in records:
        need(record['variant'] in ['Full','minus_C'], 'U/other variant entered C table')
        if record['variant'] == 'minus_C':
            inv = controls[record['id']]; paired = pairs['Full',record['distribution'],record['attack'],record['seed']]
            need(record['checkpoint_sha256'] == inv['checkpoint']['sha256'] and record['config_sha256'] == inv['config_canonical_sha256'], 'C checkpoint/config changed')
            need(record['data_contract'] == inv['data_contract'] == paired['data_contract'], 'Paired root/train/valid/client support changed')
            need(all(abs(record['views']['native'][m]-inv['prior_validation_metrics'][m]) <= 1e-12 for m in ['accuracy','aeod','aspd']), 'Native source mismatch')
    original = module('original_statistics',R/basis['evidence'])
    pure = module('two_C_scenes',H/'panels.py'); verifier = module('independent_two_C_scenes',H/'verify_numeric.py')
    panels,coverage,paired = pure.panels(records,original); checks = verifier.verify(records,panels)
    old_panels = read(OLD/'snapshot/tables.json')['panels']
    prior_panels = [dict(p,rows=[r for r in p['rows'] if r['attack'] in ['Benign','F Flip']]) for p in panels]
    need(prior_panels == old_panels, 'Old two-scene324 statistics must remain exact')
    old_verifier = module('accepted_two_scene_numeric_regression',OLD/'verify_numeric.py')
    regression = old_verifier.verify(old_records,prior_panels)
    old_checks = read(OLD/'snapshot/verification.json')
    need(all(value == old_checks[key] for key,value in regression.items()), 'Old two-scene independent checks changed')
    text = ['# CelebA Full–minus_C: IID Benign, F Flip and FedSA, three views','','Three complete scenes, thirty matched Full–C checkpoint pairs, valid-only (19,867), round70. Mean ± sample SD (ddof=1); differences are minus_C − Full. ACC is percent; ΔACC is percentage points. ACC higher and gaps lower are better. Primary endpoint remains pending.','']
    for panel in panels:
        text += ['## '+panel['view']+' — '+panel['label'],'','| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |','|---|---|---:|---:|---:|---:|']
        for row in panel['rows']:
            values = [f"{row[m]['mean']:.{3 if m == 'accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m == 'accuracy_pct' else 5}f}" for m in METRICS]
            text.append('| '+' | '.join([row['distribution']+' '+row['attack'],row['variant'],str(row['n']),*values])+' |')
        text.append('')
    text += ['AEOD is absolute TPR gap, not full equalized odds. Native retains each original root-only calibration; raw is uncalibrated; shared uses the frozen common root-only rule. All views use each model’s same checkpoint. No model inference or threshold refit occurs in this table builder.','Mixed CPU/GPU replay, training CUDA/driver differences, seed91001 recipe selection, prior validation/test exposure and pending author endpoint remain disclosed. 9/6 panels are descriptive subsets with identical seeds for Full and C, not untouched confirmation sets.','Only C IID Benign, F Flip and FedSA are complete. Six accepted S-DFA C records are retained separately and excluded from all means; seven other C scenes and the full mechanism grid remain incomplete. No C necessity/causality, significance, final-test or whole-rebuttal completion claim.','']
    rendered = '\n'.join(text); cells = displayed_cells(rendered,panels)
    old_lines = [line for line in (OLD/'snapshot/TABLES.md').read_text('utf8').splitlines() if line.startswith('| ') and ' ± ' in line]
    new_prior_lines = [line for line in rendered.splitlines() if line.startswith('| IID Benign |') or line.startswith('| IID F Flip |')]
    need(old_lines == new_prior_lines and len(old_lines)*3 == 162, 'Old two-scene162 displayed cells changed')
    checks.update(display_mean_sd_cells=cells,old_two_scene_regression=regression,old_two_scene_display_cells_exact=162,old_two_scene_statistic_scalars_exact=324,old40_preserved_exact=True,excluded_partial_C_records=6)
    for name,pin in inputs['files'].items(): need(sha(R/name) == pin['sha256'], 'Source changed during build '+name)
    args.output.mkdir()
    write(args.output/'records.json',dict(records=records))
    write(args.output/'tables.json',dict(status='C_THREE_COMPLETE_SCENES_THREE_VIEWS_PENDING_ROOT_TABLE_REVIEW',complete_scenes=3,table_model_records=60,preserved_records=60,excluded_partial_C_records=6,panels=panels,primary_endpoint_selected=False,final_test=False,new_inference=0,new_training=0,replay_devices={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ['Full','minus_C']},training_torch={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ['Full','minus_C']}))
    write(args.output/'excluded_partial_C_records.json',dict(records=partial_records,paired_Full_reference_only=[controls[r['id']]['paired_full'] for r in partial_records],reason='Six of ten S-DFA seeds; excluded from all means'))
    write(args.output/'coverage.json',coverage); write(args.output/'paired_per_seed.json',paired); write(args.output/'verification.json',checks)
    write(args.output/'SOURCE_BINDINGS.json',dict(prepared_input_pins=inputs['files'],actual_C8_adoption=str(args.C8_adoption),actual_C8_adoption_sha256=args.C8_adoption_sha256,new_archive_chains=chains,original_two_scene_root_sha256=sha(OLD/'ROOT_VERIFICATION.json'),old40_records_sha256=sha(OLD/'snapshot/records.json'),source_functions=function_proof,new_inference=0))
    (args.output/'TABLES.md').write_text(rendered,encoding='utf8',newline='\n')
    write(args.output/'FILES_SHA256.json',dict(files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(args.output.iterdir()) if p.is_file()}))
    print(json.dumps(dict(status='BUILT_PENDING_INDEPENDENT_ROOT_REVIEW',unique_records=60,mean_sd_scalars=486,display_cells=243,count_metric_checks=540,excluded_partial_C_records=6)))


if __name__ == '__main__': main()
