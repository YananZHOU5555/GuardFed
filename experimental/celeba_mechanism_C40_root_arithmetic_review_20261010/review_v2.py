"""Read-only C40 arithmetic/provenance review; never adopt or render a table."""
import argparse, collections, datetime, hashlib, itertools, json, math, sys, traceback
from pathlib import Path
sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Optimized Python is forbidden')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/'tmp/celeba_mechanism_three_view_C_four_scenes_prepared_20261010'
OLD = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_three_scenes_20261009'
VIEWS = ('native', 'raw', 'shared_calibration')
SCENES = ('Benign', 'F Flip', 'FedSA', 'S-DFA')
SEEDS = [list(range(91001, 91011)), list(range(91002, 91011)), list(range(91005, 91011))]
METRICS = ('accuracy_pct', 'aeod', 'aspd')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())

def raw_records(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    pos = text.index('[', text.index('"records"')) + 1
    decoder = json.JSONDecoder(); output = []
    while True:
        while text[pos].isspace() or text[pos] == ',': pos += 1
        if text[pos] == ']': return output
        value, end = decoder.raw_decode(text, pos)
        output.append((value['id'], text[pos:end])); pos = end

def verify(handoff_path, external_sha):
    assert handoff_path.resolve() == BASE/'ACTUAL_HANDOFF.json'
    assert sha(handoff_path) == external_sha
    for name, pin in read(HERE/'PREPARED_INPUTS.json')['files'].items():
        assert sha(ROOT/name) == pin['sha256'] and (ROOT/name).stat().st_size == pin['bytes']
    handoff = read(handoff_path)
    assert handoff['unique_records'] == 80 and handoff['Full'] == handoff['minus_C'] == 40
    assert handoff['prepared_source_seal_sha256'] == 'e8821734d524b47fc485ad1509de54ee1295a05c4e5e7065a2e6db9b20d07f89'
    assert handoff['new_CNN'] == handoff['new_Full_inference'] == handoff['threshold_refits'] == handoff['new_training'] == 0
    assert handoff['test'] is False and handoff['primary_endpoint'] == 'PENDING_AUTHOR'
    source_seal = BASE/'FILES_SHA256.json'
    assert sha(source_seal) == handoff['prepared_source_seal_sha256']
    for pin in read(source_seal)['files']:
        name = pin['path']
        assert sha(BASE/name) == pin['sha256'] and (BASE/name).stat().st_size == pin['size']
    actual_seal = BASE/'ACTUAL_FILES_SHA256.json'; files = read(actual_seal)['files']
    assert files['ACTUAL_HANDOFF.json']['sha256'] == external_sha
    for name, pin in files.items():
        assert sha(BASE/name) == pin['sha256'] and (BASE/name).stat().st_size == pin['bytes']
    for name, pin in handoff['actual_closure_pins'].items():
        assert sha(ROOT/name) == pin['sha256'] and (ROOT/name).stat().st_size == pin['bytes']
    snap = BASE/'snapshot'; bindings = read(snap/'SOURCE_BINDINGS.json')
    for name, pin in bindings['prepared_input_pins'].items():
        assert sha(ROOT/name) == pin['sha256'] and (ROOT/name).stat().st_size == pin['bytes']
    adoption = Path(bindings['actual_C4_adoption']); adopted = read(adoption)
    assert adoption.resolve().parent.parent == ROOT/'tmp/celeba_mechanism_valid_C_after36_20261010/execution_candidate/backups'
    assert sha(adoption) == bindings['actual_C4_adoption_sha256'] == handoff['actual_C4_root_sha256']
    assert adopted['status'] == 'ROOT_C_AFTER36_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
    assert (adopted['prior_three_view_models'], adopted['accepted_new'], adopted['cumulative_three_view_models']) == (136, 4, 140)
    assert adopted['original136_unchanged'] and adopted['source_scope_complete'] and adopted['negative_results_preserved']
    assert adopted['prior136_root_adoption_sha256'] == 'cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0'
    expected4 = [f'minus_C_IID_S-DFA_seed{s}' for s in range(91007, 91011)]
    assert adopted['accepted_new_ids'] == expected4
    assert sha(OLD/'ROOT_VERIFICATION.json') == '2cce0519555efbff75f559875c4c1afe63e07d242cb8e3ebe6af644efc95961d'
    for name, key in [('records.json','records_sha256'), ('tables.json','tables_sha256'), ('TABLES.md','display_sha256')]:
        assert sha(snap/name) == handoff[key]
    assert sha(snap/'FILES_SHA256.json') == handoff['snapshot_seal_sha256']
    for name, pin in read(snap/'FILES_SHA256.json')['files'].items():
        assert sha(snap/name) == pin['sha256'] and (snap/name).stat().st_size == pin['bytes']
    records = read(snap/'records.json')['records']; tables = read(snap/'tables.json')
    by = {(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
    assert len(records) == len({r['id'] for r in records}) == len(by) == 80
    assert set(by) == set(itertools.product(('Full','minus_C'),('IID',),SCENES,range(91001,91011)))
    old_records = read(OLD/'snapshot/records.json')['records']; old_ids = {r['id'] for r in old_records}
    assert len(old_records) == len(old_ids) == 60
    assert [r for r in records if r['id'] in old_ids] == old_records
    assert [(i,t) for i,t in raw_records(snap/'records.json') if i in old_ids] == raw_records(OLD/'snapshot/records.json')
    partial = read(OLD/'snapshot/excluded_partial_C_records.json')['records']
    assert len(partial) == len({r['id'] for r in partial}) == 6
    assert {(r['variant'],r['distribution'],r['attack'],r['seed']) for r in partial} == {('minus_C','IID','S-DFA',s) for s in range(91001,91007)}
    assert [r for r in records if r['id'] in {x['id'] for x in partial}] == partial
    assert [(i,t) for i,t in raw_records(snap/'records.json') if i in {x['id'] for x in partial}] == raw_records(OLD/'snapshot/excluded_partial_C_records.json')
    assert bindings['previous_S_DFA6_source_sha256'] == sha(OLD/'snapshot/excluded_partial_C_records.json')
    metric_checks = count_checks = 0
    for record in records:
        assert record['data_contract']['evaluation_split'] == 'valid' and record['data_contract']['actual_evaluation_rows'] == 19867
        assert record['views']['native'] == record['views']['shared_calibration']
        assert len(record['checkpoint_sha256']) == len(record['config_sha256']) == 64
        for view in record['views'].values():
            groups = view['group_confusion_counts']; a,b = groups['0'],groups['1']; n = a['n']+b['n']
            assert n == view['prediction_count'] == 19867
            for group in (a,b):
                assert all(type(group[k]) is int and group[k] >= 0 for k in ('tp','fp','tn','fn'))
                assert group['tp']+group['fn'] == group['positives'] and group['fp']+group['tn'] == group['negatives']
                assert group['positives']+group['negatives'] == group['n']; count_checks += 4
            calc = dict(accuracy=(a['tp']+a['tn']+b['tp']+b['tn'])/n,
                aeod=abs(a['tp']/(a['tp']+a['fn'])-b['tp']/(b['tp']+b['fn'])),
                aspd=abs((a['tp']+a['fp'])/a['n']-(b['tp']+b['fp'])/b['n']))
            for key,value in calc.items(): assert abs(view[key]-value) <= 1e-12; metric_checks += 1
    panels = tables['panels']; assert len(panels) == 9
    assert {(p['view'],tuple(p['seeds'])) for p in panels} == set(itertools.product(VIEWS,map(tuple,SEEDS)))
    errors = []
    for panel in panels:
        assert len(panel['rows']) == 12
        assert {(x['distribution'],x['attack'],x['variant']) for x in panel['rows']} == set(itertools.product(('IID',),SCENES,('Full','minus_C','minus_C minus Full')))
        for row in panel['rows']:
            seeds = panel['seeds']; assert row['complete'] and row['seeds'] == seeds and row['n'] == row['expected_n'] == len(seeds)
            for metric in METRICS:
                def value(v,s): return by[v,row['distribution'],row['attack'],s]['views'][panel['view']]['accuracy' if metric == 'accuracy_pct' else metric]*(100 if metric == 'accuracy_pct' else 1)
                xs = [value('minus_C',s)-value('Full',s) if row['variant'] == 'minus_C minus Full' else value(row['variant'],s) for s in seeds]
                mu = math.fsum(xs)/len(xs); sd = math.sqrt(math.fsum((x-mu)**2 for x in xs)/(len(xs)-1))
                errors.extend((abs(mu-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])))
    assert len(errors) == 648 and max(errors) <= 1e-12 and metric_checks == 720 and count_checks == 1920
    lines = [x for x in (snap/'TABLES.md').read_text('utf-8').splitlines() if x.startswith('| ') and ' ± ' in x]
    rows = [r for p in panels for r in p['rows']]; assert len(lines) == len(rows) == 108
    for line,row in zip(lines,rows):
        cells = line.strip('| ').split(' | ')
        for index,metric in enumerate(METRICS,3):
            precision = 3 if metric == 'accuracy_pct' else 5
            assert cells[index] == f"{row[metric]['mean']:.{precision}f} ± {row[metric]['sample_sd_ddof1']:.{precision}f}"
    old_panels = read(OLD/'snapshot/tables.json')['panels']; new_panel = {(p['view'],tuple(p['seeds'])):p for p in panels}
    for panel in old_panels: assert panel['rows'] == [r for r in new_panel[panel['view'],tuple(panel['seeds'])]['rows'] if r['attack'] in SCENES[:3]]
    old_lines = [x for x in (OLD/'snapshot/TABLES.md').read_text('utf-8').splitlines() if x.startswith('| ') and ' ± ' in x]
    assert old_lines == [x for x in lines if x.startswith('| IID Benign |') or x.startswith('| IID F Flip |') or x.startswith('| IID FedSA |')]
    pairs = read(snap/'paired_per_seed.json'); paired_checks = 0
    for view in VIEWS:
        assert len(pairs[view]) == 40 and {(x['distribution'],x['attack'],x['seed']) for x in pairs[view]} == set(itertools.product(('IID',),SCENES,range(91001,91011)))
        for pair in pairs[view]:
            assert pair['variant'] == 'minus_C'
            for metric in METRICS:
                key = 'accuracy' if metric == 'accuracy_pct' else metric; scale = 100 if metric == 'accuracy_pct' else 1
                expected = (by['minus_C',pair['distribution'],pair['attack'],pair['seed']]['views'][view][key]*scale - by['Full',pair['distribution'],pair['attack'],pair['seed']]['views'][view][key]*scale)
                assert abs(pair[metric]-expected) <= 1e-12; paired_checks += 1
    from source_connections import verify_connections
    source_connections = verify_connections(records, bindings, adopted, adoption)
    assert sha(handoff_path) == external_sha
    return dict(source_connections=source_connections, status='INDEPENDENT_C40_FOUR_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_handoff_sha256=external_sha,
        actual_delivery_seal_sha256=sha(actual_seal),actual_members_verified=len(files),actual_closure_pins=handoff['actual_closure_pins'],
        actual_C4_root_adoption_sha256=sha(adoption),source_prepared_seal_sha256=sha(source_seal),
        prior_three_scene_root_sha256=sha(OLD/'ROOT_VERIFICATION.json'),records_sha256=sha(snap/'records.json'),tables_sha256=sha(snap/'tables.json'),display_sha256=sha(snap/'TABLES.md'),
        unique_records=80,paired_models=40,complete_scenes=4,mean_SD_scalars_recomputed=648,display_cells=324,
        count_metrics_recomputed=720,confusion_count_checks=1920,paired_seed_metric_checks=paired_checks,max_abs_difference=max(errors),
        old60_records_exact=True,old60_record_JSON_bytes_and_order_exact=True,old486_scalars_exact=True,old243_display_cells_exact=True,
        S_DFA_six_original_record_bytes_preserved=True,native_shared_identical_records=80,
        replay_devices={v:dict(collections.Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ('Full','minus_C')},
        training_torch={v:dict(collections.Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ('Full','minus_C')},
        review_source_sha256=sha(Path(__file__)),canonical_modified=False,adoption_performed=False,new_CNN=0,new_training=0,new_Full_inference=0,test=False,
        primary_endpoint='PENDING_AUTHOR',other_C_scenes_complete=False,whole_rebuttal_complete=False,
        ten_seed_paired_deltas={p['view']:{r['attack']:{m:r[m]['mean'] for m in METRICS} for r in p['rows'] if r['variant']=='minus_C minus Full'} for p in panels if len(p['seeds'])==10})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--handoff-sha256', required=True)
    args = parser.parse_args()
    output = HERE/'ROOT_ARITHMETIC_REVIEW.json'; assert not output.exists()
    proof = verify(args.handoff, args.handoff_sha256)
    with output.open('x', encoding='utf-8') as stream: json.dump(proof,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(path=str(output),sha256=sha(output),status=proof['status'],max_abs_difference=proof['max_abs_difference'])))

if __name__ == '__main__':
    try: main()
    except SystemExit: raise
    except BaseException as error:
        failure = HERE/'ROOT_ARITHMETIC_FAILURE.json'
        if not failure.exists():
            with failure.open('x',encoding='utf-8') as stream: json.dump(dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,canonical_modified=False),stream,indent=2)
        raise
