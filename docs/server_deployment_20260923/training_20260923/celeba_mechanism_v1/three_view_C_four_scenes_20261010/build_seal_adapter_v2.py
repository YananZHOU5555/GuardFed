"""Prepared record-only C four-scene tables; actual after36 C4 adoption is required."""
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
OLD = R / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_three_scenes_20261009'
BENIGN = OLD.with_name('three_view_C_Benign10_20261009')
C4 = R / 'tmp/celeba_mechanism_valid_C_after36_20261010'
C28 = R / 'tmp/celeba_mechanism_valid_C_after28_20261009'
PRIOR = C28 / 'execution_candidate/backups/incremental_20261009T221150Z/ROOT_ADOPTION_REVIEW.json'
EXPECTED = [f'minus_C_IID_S-DFA_seed{s}' for s in range(91007,91011)]
SDFA = [f'minus_C_IID_S-DFA_seed{s}' for s in range(91001,91011)]
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
    a = {r['id']:r for r in prior['records']}; b = {r['id']:r for r in current['records']}
    need(len(a)==len(prior['records'])==136 and len(b)==len(current['records'])==140, 'Exact native136/native140 inventories required')
    need(set(b)-set(a)==set(EXPECTED) and all(b[k]==r for k,r in a.items()), 'Prior136 or exact C4 difference changed')
    need(current['selected_replay_ids']==EXPECTED and set(current['excluded_prior_replay_ids'])==set(a), 'after36 C4 boundary changed')
    need(prior['full_references']==current['full_references'] and len(current['full_references'])==100, 'Full100 references changed')
    need(prior['native_tolerance']==current['native_tolerance']==1e-12, 'Native tolerance changed')
    controls={k:r for k,r in b.items() if r['variant']=='minus_C'}
    complete={('IID',a,s) for a in ['Benign','F Flip','FedSA','S-DFA'] for s in range(91001,91011)}
    need(len(controls)==40 and {(r['distribution'],r['attack'],r['seed']) for r in controls.values()}==complete, 'Only exact four complete C IID scenes allowed')
    need(all((r['terminal_round'],r['original_split'],r['original_n_eval'])==(70,'valid',19867) for r in controls.values()), 'Incomplete/non-valid control')
    return controls


def adoption_gate(path, expected_sha):
    need(path.name == 'ROOT_ADOPTION_REVIEW.json' and path.resolve().is_relative_to((C4/'execution_candidate/backups').resolve()), 'Only actual C4 backup ROOT adoption permitted')
    need(len(expected_sha) == 64 and set(expected_sha) <= set('0123456789abcdef') and sha(path) == expected_sha, 'Actual external C4 ROOT SHA required')
    proof = read(path)
    need(proof['status'] == 'ROOT_C_AFTER36_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS', 'C4 ROOT has not adopted strict/offserver outputs')
    need((proof['prior_three_view_models'],proof['accepted_new'],proof['cumulative_three_view_models']) == (136,4,140), 'Actual adopted136+4 required')
    need(proof['accepted_new_ids'] == EXPECTED and proof['original136_unchanged'] and proof['source_scope_complete'] and proof['negative_results_preserved'], 'Incomplete exact C4/prior/source preservation')
    need(proof['prior136_root_adoption_sha256'] == sha(PRIOR), 'Prior after28 adoption changed')
    need(proof['new_training'] == proof['new_Full_inference'] == 0 and proof['test_inference'] is False, 'Forbidden new science')
    return proof


def displayed_cells(text, panels):
    lines = [x for x in text.splitlines() if x.startswith('| ') and ' ± ' in x]
    rows = [r for p in panels for r in p['rows']]
    need(len(lines) == len(rows) == 108, 'Incomplete displayed rows')
    for line,row in zip(lines,rows):
        cells = line.strip('| ').split(' | ')
        for index,metric in enumerate(METRICS,3):
            precision = 3 if metric == 'accuracy_pct' else 5
            need(cells[index] == f"{row[metric]['mean']:.{precision}f} ± {row[metric]['sample_sd_ddof1']:.{precision}f}", 'Display drift')
    return len(rows)*3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--C4-adoption', type=Path, required=True)
    parser.add_argument('--C4-adoption-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(args.output.resolve().parent == H.resolve() and not args.output.exists(), 'Fresh owned output only')
    adoption_gate(args.C4_adoption, args.C4_adoption_sha256)
    inputs = read(H/'INPUTS.json')
    for name,pin in inputs['files'].items():
        need(sha(R/name) == pin['sha256'] and (R/name).stat().st_size == pin['bytes'], 'Input/source changed '+name)
    for pin in read(H/'FILES_SHA256.json')['files']:
        name = pin['path']
        need(sha(H/name) == pin['sha256'], 'Prepared member changed '+name)
    prior = read(C28/'inventory_actual136_Full100refs.json'); current = read(C4/'inventory_actual140_Full100refs.json')
    controls = scope(prior,current)
    # Original accepted join, all member/strict/offserver guards, unchanged.
    accepted = module('accepted_C_table_source', BENIGN/'build.py')
    accepted.R = R; accepted.C11 = R/'tmp/celeba_mechanism_valid_C_after1_20261009'
    basis = read(BENIGN/'INPUTS.json')
    join,scientific,function_proof = accepted.source_functions(basis)
    added,chain = join(args.C4_adoption.parent,args.C4_adoption_sha256,current,C4/'inventory_actual140_Full100refs.json',C4/'bridge.py',136,140,scientific['receipt_identity'],scientific['normalized'],scientific['canonical'])
    partial_source = read(OLD/'snapshot/excluded_partial_C_records.json')
    old_six = partial_source['records']
    need(len(old_six)==6 and [r['id'] for r in old_six]==SDFA[:6], 'Adopted previous S-DFA6 identity/order changed')
    need(partial_source['paired_Full_reference_only']==[controls[r['id']]['paired_full'] for r in old_six], 'Prior S-DFA6 paired Full references changed')
    need(len(added)==4 and [r['id'] for r in added]==EXPECTED, 'Exact four source receipts required')
    new_records = old_six + added
    need([r['id'] for r in new_records]==SDFA, 'Missing/duplicate/reordered S-DFA10')
    old_records = read(OLD/'snapshot/records.json')['records']
    need(len(old_records)==len({r['id'] for r in old_records})==60, 'Accepted old60 changed')
    full_source = read(R/inputs['full900_path'])
    byid = {r['id']:r for r in full_source['records']}; refs = {r['id']:r for r in current['full_references']}
    full_builder = module('accepted_Full_reference_normalizer',R/basis['parent_builder'])
    full = [full_builder.full_record(byid[controls[key]['paired_full']['id']],refs[controls[key]['paired_full']['id']]) for key in SDFA]
    records = old_records + full + new_records
    need(len(records)==len({r['id'] for r in records})==80 and records[:60]==old_records, 'Exact80/original60 identity changed')
    old_spans=record_spans((OLD/'snapshot/records.json').read_text('utf8'))
    new_spans=record_spans(json.dumps(dict(records=records),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    need(len(old_spans)==60 and new_spans[:60]==old_spans, 'Original60 JSON record bytes changed')
    previous_six_spans=record_spans((OLD/'snapshot/excluded_partial_C_records.json').read_text('utf8'))
    need(len(previous_six_spans)==6 and new_spans[70:76]==previous_six_spans, 'Previously accepted S-DFA6 JSON record bytes changed')
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
    prior_panels = [dict(p,rows=[r for r in p['rows'] if r['attack'] in ['Benign','F Flip','FedSA']]) for p in panels]
    need(prior_panels == old_panels, 'Old three-scene486 statistics must remain exact')
    old_verifier = module('accepted_three_scene_numeric_regression',OLD/'verify_numeric.py')
    regression = old_verifier.verify(old_records,prior_panels)
    old_checks = read(OLD/'snapshot/verification.json')
    need(all(value == old_checks[key] for key,value in regression.items()), 'Old three-scene independent checks changed')
    text = ['# CelebA Full–minus_C: IID Benign, F Flip, FedSA and S-DFA, three views','','Four complete scenes, forty matched Full–C checkpoint pairs, valid-only (19,867), round70. Mean ± sample SD (ddof=1); differences are minus_C − Full. ACC is percent; ΔACC is percentage points. ACC higher and gaps lower are better. Primary endpoint remains pending.','']
    for panel in panels:
        text += ['## '+panel['view']+' — '+panel['label'],'','| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |','|---|---|---:|---:|---:|---:|']
        for row in panel['rows']:
            values = [f"{row[m]['mean']:.{3 if m == 'accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m == 'accuracy_pct' else 5}f}" for m in METRICS]
            text.append('| '+' | '.join([row['distribution']+' '+row['attack'],row['variant'],str(row['n']),*values])+' |')
        text.append('')
    text += ['AEOD is absolute TPR gap, not full equalized odds. Native retains each original root-only calibration; raw is uncalibrated; shared uses the frozen common root-only rule. All views use each model’s same checkpoint. No model inference or threshold refit occurs in this table builder.','Mixed CPU/GPU replay, training CUDA/driver differences, seed91001 recipe selection, prior validation/test exposure and pending author endpoint remain disclosed. 9/6 panels are descriptive subsets with identical seeds for Full and C, not untouched confirmation sets.','Only C IID Benign, F Flip, FedSA and S-DFA are complete. The six previously excluded S-DFA C records are now paired with four newly accepted records under the same ten-seed rule; six other C scenes and the full mechanism grid remain incomplete. No C necessity/causality, significance, final-test or whole-rebuttal completion claim.','']
    rendered = '\n'.join(text); cells = displayed_cells(rendered,panels)
    old_lines = [line for line in (OLD/'snapshot/TABLES.md').read_text('utf8').splitlines() if line.startswith('| ') and ' ± ' in line]
    new_prior_lines = [line for line in rendered.splitlines() if line.startswith('| IID Benign |') or line.startswith('| IID F Flip |') or line.startswith('| IID FedSA |')]
    need(old_lines == new_prior_lines and len(old_lines)*3 == 243, 'Old three-scene243 displayed cells changed')
    checks.update(display_mean_sd_cells=cells,old_three_scene_regression=regression,old_three_scene_display_cells_exact=243,old_three_scene_statistic_scalars_exact=486,old60_preserved_exact=True,previously_partial_S_DFA6_now_complete=True)
    for name,pin in inputs['files'].items(): need(sha(R/name) == pin['sha256'], 'Source changed during build '+name)
    args.output.mkdir()
    write(args.output/'records.json',dict(records=records))
    write(args.output/'tables.json',dict(status='C_FOUR_COMPLETE_SCENES_THREE_VIEWS_PENDING_ROOT_TABLE_REVIEW',complete_scenes=4,table_model_records=80,preserved_records=80,previously_partial_S_DFA6_now_complete=True,panels=panels,primary_endpoint_selected=False,final_test=False,new_inference=0,new_training=0,replay_devices={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ['Full','minus_C']},training_torch={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ['Full','minus_C']}))
    write(args.output/'coverage.json',coverage); write(args.output/'paired_per_seed.json',paired); write(args.output/'verification.json',checks)
    write(args.output/'SOURCE_BINDINGS.json',dict(prepared_input_pins=inputs['files'],actual_C4_adoption=str(args.C4_adoption),actual_C4_adoption_sha256=args.C4_adoption_sha256,new_archive_chain=chain,previous_S_DFA6_source_sha256=sha(OLD/'snapshot/excluded_partial_C_records.json'),original_three_scene_root_sha256=sha(OLD/'ROOT_VERIFICATION.json'),old60_records_sha256=sha(OLD/'snapshot/records.json'),source_functions=function_proof,new_inference=0))
    (args.output/'TABLES.md').write_text(rendered,encoding='utf8',newline='\n')
    write(args.output/'FILES_SHA256.json',dict(files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(args.output.iterdir()) if p.is_file()}))
    print(json.dumps(dict(status='BUILT_PENDING_INDEPENDENT_ROOT_REVIEW',unique_records=80,mean_sd_scalars=648,display_cells=324,count_metric_checks=720)))


if __name__ == '__main__': main()
