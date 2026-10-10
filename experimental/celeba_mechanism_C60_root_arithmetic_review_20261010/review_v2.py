"""Independent stdlib C60 arithmetic/source reader; no builder, adoption or inference."""
import argparse, collections, datetime, hashlib, itertools, json, math, sys, traceback
from pathlib import Path
sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Optimized Python is forbidden')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/'tmp/celeba_mechanism_C_six_scenes_prepared_20261010'
OLD = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010'
SOURCE_SEAL = '4a1a2f9d71bd432e58229f98cf7f98b1fdba9e8b60dad4c9338c8801b240067c'
OLD_ROOT = '811ad551c1398f6e57681f59046c316b05c609ef30d4897158460b8121b1a28d'
VIEWS = ('native', 'raw', 'shared_calibration')
SCENES = ('Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA')
CELLS = [('IID',a) for a in SCENES]+[('non-IID','Benign')]
VARIANTS = ('Full', 'minus_C', 'minus_C minus Full')
SEEDS = [list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
METRICS = ('accuracy_pct','aeod','aspd')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())

def raw_records(path):
    text = Path(path).read_text(encoding='utf-8')
    position = text.index('[',text.index('"records"'))+1
    decoder = json.JSONDecoder(); output = []
    while True:
        while text[position].isspace() or text[position] == ',': position += 1
        if text[position] == ']': return output
        value,end = decoder.raw_decode(text,position)
        output.append((value['id'],text[position:end])); position=end

def verify_files(base, files):
    for name,pin in files.items():
        path = base/name
        assert path.resolve().is_relative_to(base.resolve())
        assert sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes'], name

def mean_sd(values):
    mean = math.fsum(values)/len(values)
    return mean,math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1))

def verify(handoff_path, handoff_sha, delivery_sha):
    assert handoff_path.resolve() == (BASE/'ACTUAL_HANDOFF.json').resolve()
    assert sha(handoff_path) == handoff_sha
    assert sha(BASE/'FILES_SHA256.json') == SOURCE_SEAL
    source_files = read(BASE/'FILES_SHA256.json')['members']
    assert len(source_files) == 13
    verify_files(BASE,{r['path']:dict(sha256=r['sha256'],bytes=r['size']) for r in source_files})
    handoff = read(handoff_path)
    assert handoff['unique_records'] == 120 and handoff['Full'] == handoff['minus_C'] == 60
    assert handoff['source_prepared_seal_sha256'] == SOURCE_SEAL
    assert handoff['inference'] == handoff['training'] == 0
    assert handoff['test'] is False and handoff['canonical_modified'] is False
    assert handoff['negative_results_preserved'] is True
    actual_seal = BASE/'ACTUAL_FILES_SHA256.json'
    assert sha(actual_seal) == delivery_sha
    actual_files = read(actual_seal)['files']
    assert actual_files['ACTUAL_HANDOFF.json']['sha256'] == handoff_sha
    verify_files(BASE,actual_files)
    snap = BASE/'snapshot'
    for name,pin in handoff['snapshot_files'].items(): assert sha(snap/name) == pin
    verify_files(snap,read(snap/'FILES_SHA256.json')['files'])
    bindings = read(snap/'SOURCE_BINDINGS.json')
    assert bindings['prepared_input_pins'] == read(BASE/'INPUTS.json')['files']
    verify_files(ROOT,bindings['prepared_input_pins'])
    assert sha(OLD/'ROOT_VERIFICATION.json') == OLD_ROOT == bindings['original_C50_root_sha256']
    binding = bindings['actual_C4_binding']
    stage = (ROOT/binding['stage']).resolve()
    assert stage == (ROOT/'tmp/celeba_mechanism_valid_C_after56_20261010').resolve()
    adoption = (ROOT/binding['adoption']).resolve(); adopted = read(adoption)
    assert adoption.name == 'ROOT_ADOPTION_REVIEW.json' and adoption.parent.parent == stage/'execution_candidate/backups'
    assert sha(adoption) == binding['adoption_sha256'] == handoff['actual_C4_adoption_sha256']
    assert sha(stage/'FILES_SHA256.json') == binding['science_sha256'] == '2095ab384a7844355fc92453dbfa5d2922f88f378838e7b7eda2524578f3bbd6'
    assert sha(stage/'execution_candidate/EXECUTION_SOURCE_SHA256.json') == binding['execution_sha256'] == 'd1679c0bbd53bc66e4ea7ae792000d398efcafe5192bc5164ffd80e7a2eeb236'
    assert sha(stage/'inventory_actual160_Full100refs.json') == binding['inventory_sha256'] == '302dd45e9f05c646671d31d26775607af7a4fe70876fa1e60643939f972742f4'
    for folder,seal in [(stage,stage/'FILES_SHA256.json'),(stage/'execution_candidate',stage/'execution_candidate/EXECUTION_SOURCE_SHA256.json')]:
        for row in read(seal)['members']:
            path=folder/row['path'];assert sha(path)==row['sha256'] and path.stat().st_size==row['size']
    assert adopted['status'] == 'ROOT_C_AFTER56_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
    assert (adopted['prior_three_view_models'],adopted['accepted_new'],adopted['cumulative_three_view_models']) == (156,4,160)
    assert adopted['accepted_new_ids'] == [f'minus_C_non-IID_Benign_seed{s}' for s in range(91007,91011)]
    assert adopted['original156_unchanged'] and adopted['source_scope_complete'] and adopted['negative_results_preserved']
    assert adopted['all_native_differences_zero'] and adopted['server_strict_bound_in_saved_receipts']
    assert adopted['science_seal_sha256']==binding['science_sha256'] and adopted['execution_seal_sha256']==binding['execution_sha256']
    assert adopted['new_training'] == adopted['new_Full_inference'] == 0 and adopted['test_inference'] is False
    records = read(snap/'records.json')['records']; tables = read(snap/'tables.json')
    assert tables['primary_endpoint_selected'] is False and tables['final_test'] is False
    assert tables['new_inference'] == tables['new_training'] == bindings['new_inference'] == 0
    by = {(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
    assert len(records) == len({r['id'] for r in records}) == len(by) == 120
    assert set(by) == {(v,d,a,seed) for v in VARIANTS[:2] for d,a in CELLS for seed in range(91001,91011)}
    old_records = read(OLD/'snapshot/records.json')['records']
    assert len(old_records) == 100 and records[:100] == old_records
    assert raw_records(snap/'records.json')[:100] == raw_records(OLD/'snapshot/records.json')
    count_checks = metric_checks = 0; count_errors = []
    for record in records:
        assert record['data_contract']['evaluation_split'] == 'valid' and record['data_contract']['actual_evaluation_rows'] == 19867
        assert set(record['views']) == set(VIEWS)
        assert len(record['checkpoint_sha256']) == len(record['config_sha256']) == 64
        for view in record['views'].values():
            groups = view['group_confusion_counts']; a,b = groups['0'],groups['1']; n=a['n']+b['n']
            assert n == view['prediction_count'] == 19867
            for group in (a,b):
                assert all(type(group[k]) is int and group[k] >= 0 for k in ('tp','fp','tn','fn'))
                assert group['tp']+group['fn'] == group['positives']
                assert group['fp']+group['tn'] == group['negatives']
                assert group['positives']+group['negatives'] == group['n']; count_checks += 4
            calculated = dict(accuracy=(a['tp']+a['tn']+b['tp']+b['tn'])/n,
                aeod=abs(a['tp']/(a['tp']+a['fn'])-b['tp']/(b['tp']+b['fn'])),
                aspd=abs((a['tp']+a['fp'])/a['n']-(b['tp']+b['fp'])/b['n']))
            for key,value in calculated.items():
                count_errors.append(abs(view[key]-value)); metric_checks += 1
    assert metric_checks == 1080 and count_checks == 2880 and max(count_errors) <= 1e-12
    def value(variant,distribution,attack,seed,view,metric):
        return by[variant,distribution,attack,seed]['views'][view]['accuracy' if metric=='accuracy_pct' else metric]*(100 if metric=='accuracy_pct' else 1)
    panels = tables['panels']; expected_panels = set(itertools.product(VIEWS,map(tuple,SEEDS)))
    assert len(panels) == 9 and {(p['view'],tuple(p['seeds'])) for p in panels} == expected_panels
    errors = []; recomputed = []; direction_rows = []
    for panel in panels:
        assert len(panel['rows']) == 18
        assert {(r['distribution'],r['attack'],r['variant']) for r in panel['rows']} == {(d,a,v) for d,a in CELLS for v in VARIANTS}
        for row in panel['rows']:
            seeds = panel['seeds']; assert row['complete'] and row['seeds'] == seeds and row['n'] == row['expected_n'] == len(seeds)
            numbers = {}
            for metric in METRICS:
                xs = [value('minus_C',row['distribution'],row['attack'],s,panel['view'],metric)-value('Full',row['distribution'],row['attack'],s,panel['view'],metric) if row['variant']==VARIANTS[2] else value(row['variant'],row['distribution'],row['attack'],s,panel['view'],metric) for s in seeds]
                mu,sd = mean_sd(xs); numbers[metric] = (mu,sd)
                errors.extend((abs(mu-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])))
            recomputed.append(numbers)
            if row['variant'] == VARIANTS[2]:
                direction_rows.append(dict(view=panel['view'],n=len(seeds),distribution=row['distribution'],attack=row['attack'],minus_C_minus_Full_mean={m:numbers[m][0] for m in METRICS}))
    assert len(errors) == 972 and max(errors) <= 1e-12
    lines = [x for x in (snap/'TABLES.md').read_text('utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
    assert len(lines) == len(recomputed) == 162
    for line,numbers in zip(lines,recomputed):
        cells=line.strip('| ').split(' | ')
        for index,metric in enumerate(METRICS,3):
            precision=3 if metric=='accuracy_pct' else 5; mu,sd=numbers[metric]
            assert cells[index] == f'{mu:.{precision}f} ± {sd:.{precision}f}'
    old_panels = read(OLD/'snapshot/tables.json')['panels']; new_panels={(p['view'],tuple(p['seeds'])):p for p in panels}
    for panel in old_panels:
        assert panel['rows'] == [r for r in new_panels[panel['view'],tuple(panel['seeds'])]['rows'] if r['distribution']=='IID']
    old_lines=[x for x in (OLD/'snapshot/TABLES.md').read_text('utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
    assert old_lines == [x for x in lines if x.startswith('| IID ')]
    assert len(old_lines)*3 == 405
    paired=read(snap/'paired_per_seed.json'); paired_checks=0
    for view in VIEWS:
        assert len(paired[view]) == 60 and {(r['distribution'],r['attack'],r['seed']) for r in paired[view]} == {(d,a,seed) for d,a in CELLS for seed in range(91001,91011)}
        for pair in paired[view]:
            assert pair['variant'] == 'minus_C'
            for metric in METRICS:
                delta=value('minus_C',pair['distribution'],pair['attack'],pair['seed'],view,metric)-value('Full',pair['distribution'],pair['attack'],pair['seed'],view,metric)
                assert abs(pair[metric]-delta) <= 1e-12; paired_checks += 1
    assert (snap/'cross_scene_seed_first.json').read_bytes()==(OLD/'snapshot/cross_scene_seed_first.json').read_bytes()
    cross=read(snap/'cross_scene_seed_first.json'); aggregates=cross['panels']; aggregate_errors=[]; aggregate_directions=[]
    assert cross['new_endpoint_selected'] is False
    assert len(aggregates)==9 and {(p['view'],tuple(p['seeds'])) for p in aggregates}==expected_panels
    for panel in aggregates:
        assert len(panel['rows'])==3 and {r['variant'] for r in panel['rows']}==set(VARIANTS)
        for row in panel['rows']:
            seeds=panel['seeds']; assert row['seeds']==seeds and row['n']==row['expected_n']==len(seeds) and row['complete']
            assert row['distribution']=='IID' and row['attack']=='Five-scene equal mean within seed'
            means={}
            for metric in METRICS:
                def within(variant,seed): return math.fsum(value(variant,'IID',a,seed,panel['view'],metric) for a in SCENES)/5
                xs=[within('minus_C',s)-within('Full',s) if row['variant']==VARIANTS[2] else within(row['variant'],s) for s in seeds]
                mu,sd=mean_sd(xs); means[metric]=mu
                aggregate_errors.extend((abs(mu-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])))
            if row['variant']==VARIANTS[2]: aggregate_directions.append(dict(view=panel['view'],n=len(seeds),minus_C_minus_Full_mean=means))
    assert len(aggregate_errors)==162 and max(aggregate_errors)<=1e-12
    from source_connections import verify_connections
    connections=verify_connections(records,bindings,adopted,adoption)
    assert sha(handoff_path)==handoff_sha and sha(actual_seal)==delivery_sha
    return dict(status='INDEPENDENT_C60_SIX_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_handoff_sha256=handoff_sha,
        actual_delivery_seal_sha256=delivery_sha,actual_members_verified=len(actual_files),source_prepared_seal_sha256=SOURCE_SEAL,
        actual_C4_root_adoption_sha256=sha(adoption),source_connections=connections,
        records_sha256=sha(snap/'records.json'),tables_sha256=sha(snap/'tables.json'),display_sha256=sha(snap/'TABLES.md'),
        cross_scene_seed_first_sha256=sha(snap/'cross_scene_seed_first.json'),unique_records=120,paired_models=60,complete_scenes=6,
        mean_SD_scalars_recomputed=972,display_cells=486,count_metrics_recomputed=1080,confusion_count_checks=2880,
        paired_seed_metric_checks=paired_checks,cross_scene_mean_SD_scalars_recomputed=162,
        max_abs_difference=max(errors),cross_scene_max_abs_difference=max(aggregate_errors),count_metric_max_abs_difference=max(count_errors),
        old100_record_JSON_bytes_and_order_exact=True,old810_scalars_exact=True,old405_display_cells_exact=True,old162_IID_aggregate_bytes_exact=True,
        native_shared_identical_records=sum(r['views']['native']==r['views']['shared_calibration'] for r in records),
        replay_devices={v:dict(collections.Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in VARIANTS[:2]},
        training_torch={v:dict(collections.Counter(r['training_torch'] for r in records if r['variant']==v)) for v in VARIANTS[:2]},
        all_scene_paired_directions=direction_rows,all_seed_first_paired_directions=aggregate_directions,
        AEOD_definition='absolute TPR gap; not full equalized odds',n_is_seed_count=True,seed_first_scenes_per_seed=5,
        primary_endpoint='PENDING_AUTHOR',other_nonIID_C_scenes_complete=False,whole_mechanism_complete=False,whole_rebuttal_complete=False,
        review_source_sha256=sha(Path(__file__)),canonical_modified=False,adoption_performed=False,new_CNN=0,new_training=0,new_Full_inference=0,test=False,
        limitations=['Frozen descriptive 10/9/6 subsets retain selection and prior valid/test exposure history.',
            'Mixed replay devices and historical training environment remain; no causal necessity, significance or final-test claim.',
            'Six-scene rows remain separate; the preserved seed-first aggregate covers only five IID scenes.',
            'Saved count metrics recomputed; no checkpoint inference, threshold refit or prediction-array recomputation.'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff',type=Path,required=True)
    parser.add_argument('--handoff-sha256',required=True)
    parser.add_argument('--seal-sha256',required=True)
    args=parser.parse_args(); output=HERE/'ROOT_ARITHMETIC_REVIEW.json'
    assert not output.exists()
    proof=verify(args.handoff,args.handoff_sha256,args.seal_sha256)
    with output.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(proof,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(path=str(output),sha256=sha(output),status=proof['status'],max_abs_difference=proof['max_abs_difference'])))

if __name__=='__main__':
    try: main()
    except SystemExit: raise
    except BaseException as error:
        failure=HERE/'ROOT_ARITHMETIC_FAILURE.json'
        if not failure.exists():
            with failure.open('x',encoding='utf8') as stream:
                json.dump(dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,canonical_modified=False),stream,indent=2)
        raise
