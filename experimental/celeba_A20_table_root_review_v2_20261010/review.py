"""Independent A20 arithmetic/provenance review; no table adoption or inference."""
from pathlib import Path
from collections import Counter
import argparse, datetime, hashlib, itertools, json, math, sys, traceback

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[1]
OLD = ROOT / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_Benign10_20261010'
FULL = ROOT / 'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/records_three_views_900.json'
REFS = ROOT / 'tmp/celeba_mechanism_valid_C_after1_20261009/inventory_actual112_Full100refs.json'
PINS = {
    'tmp/celeba_A20_table_root_review_20261010/review.py': '56f98589d50cdebd71e2bc2c44cab2b7d904980f4af4dac3f92a36b808cbb737',
    'tmp/celeba_A20_table_root_review_20261010/REVIEW_FAILURE.json': '51314f65a18108e7d090f4903661d49a411649a9fe674d8a500a0c5342711ca0',
    'tmp/verify_A_Benign10_root_20261010.py': '2e7407b315f599baea9453e7ebe602a65249f684b579afd5d764bf8a87acca2a',
    'tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010/ROOT_ADOPTION.json': '1221482d564a2c735b0de0680fe8a42512c9fe5774a9d157bd3bfda5dd9c858b',
    'tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010/MECHANISM212_INDEX.json': '9a90f3d74a27d9ca4225b850797faec3c3aac2b6e2f83f87d3afe49c21912496',
    'tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010/ROOT_ADOPTION.json': 'e088871fbd98cbc9415cc79a44626667a532389ce0b6dddbaaf2ab25f72a4979',
    'tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010/MECHANISM220_INDEX.json': 'af414a0ac6705230c53d324cd1f51e7a76b893dabd7416bb6914a6690b6fdddc',
    FULL.relative_to(ROOT).as_posix(): '983bca43dff7e94dd79a312f273c65ed3ea1d146fff3ebfbf3bd2a854e124529',
    REFS.relative_to(ROOT).as_posix(): '5e46f03d99be908bf19cf376909b5b7ae11dfcc098be09f1d08f8ad9c5688f32',
    (OLD/'FILES_SHA256.json').relative_to(ROOT).as_posix(): 'c96af519a0fc8ef3eff780f2990c89d40022b6fc1eba25cc73675e3079fcdbb5',
    (OLD/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(): 'c3d65134f0e6eeda36fd37c64b6ac0df3794828af054d920d46775670a54df9d',
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
canonical = lambda d: json.dumps(d, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def checked(pin):
    assert sha(pin['path']) == pin['sha256'], pin['path']
    return read(pin['path'])


def sealed(directory, expected):
    assert sha(directory/'FILES_SHA256.json') == expected, 'external delivery seal drift'
    files = read(directory/'FILES_SHA256.json')['files']
    assert {'records.json','tables.json','TABLES.md','SOURCE_BINDINGS.json'} <= set(files)
    for name, pin in files.items():
        p = (directory/name).resolve()
        assert p.is_relative_to(directory.resolve()), 'delivery path escape'
        assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], name


def review(directory, delivery_seal):
    sealed(directory, delivery_seal)
    for name, digest in PINS.items():
        assert sha(ROOT/name) == digest, name
    sealed(OLD, PINS[(OLD/'FILES_SHA256.json').relative_to(ROOT).as_posix()])
    roots = [ROOT/f'tmp/celeba_mechanism_remaining620_A{n}_root_adoption_20261010/ROOT_ADOPTION.json' for n in (12,20)]
    adoptions = [read(p) for p in roots]
    indexes = [read(ROOT/a['records_index_path']) for a in adoptions]
    for a in adoptions:
        assert sha(ROOT/a['records_index_path']) == a['records_index_sha256']
    prior, new = indexes
    assert new['prior_index_path'] == adoptions[0]['records_index_path']
    assert new['prior_index_sha256'] == adoptions[0]['records_index_sha256']
    assert ROOT/new['prior_adoption_path'] == roots[0] and new['prior_adoption_sha256'] == sha(roots[0])
    assert new['all_ids'][:212] == prior['all_ids'] and len(new['all_ids']) == len(set(new['all_ids'])) == 220
    expected_new = {f'minus_A_IID_F Flip_seed{s}' for s in range(91003,91011)}
    assert set(new['new_ids']) == set(adoptions[1]['accepted_new_ids']) == expected_new
    assert (adoptions[1]['prior_accepted'],adoptions[1]['cumulative_accepted'],adoptions[1]['new_accepted']) == (212,220,8)
    originals = {r['id']:(r,index,root) for index,root in zip(indexes,roots) for r in index['new_records']}
    assert len(originals) == 20 and not set(prior['new_ids']) & expected_new
    full = {r['id']:r for r in read(FULL)['records']}
    refs = {r['id']:r for r in read(REFS)['full_references']}
    assert len(refs) == 100
    records = read(directory/'records.json')['records']; table = read(directory/'tables.json')
    assert len(records) == len({r['id'] for r in records}) == 40
    cells = {(r['variant'],r['attack'],r['seed']):r for r in records}
    attacks = ('Benign','F Flip'); variants = ('Full','minus_A')
    assert set(cells) == set(itertools.product(variants,attacks,range(91001,91011)))
    assert all(r['distribution']=='IID' for r in records)
    assert (table['complete_scenes'],table['paired_models'],table['preserved_records']) == (2,20,40)
    assert table['final_test'] is False and table['primary_endpoint_selected'] is False
    assert table['new_threshold_fits'] == table['new_inference'] == table['new_training'] == 0
    old_records = read(OLD/'records.json')['records']; old_ids = {r['id'] for r in old_records}
    assert [canonical(r) for r in records if r['id'] in old_ids] == [canonical(r) for r in old_records]
    metric_checks = count_checks = 0
    for r in records:
        assert set(r['views']) == {'native','raw','shared_calibration'}
        if r['variant'] == 'Full':
            original = full[r['id']]; ref = refs[r['id']]
            assert r['checkpoint_sha256'] == original['checkpoint_sha256'] == ref['checkpoint_sha256']
            assert original['model_inventory_record_sha256'] == ref['baseline_record_canonical_sha256']
            assert original['same_checkpoint_all_views'] and not original['test_evaluation_performed']
            assert (r['method'],r['distribution'],r['attack'],r['seed']) == tuple(original[k] for k in ('method','distribution','attack','seed'))
            assert r['views'] == original['views'] and r['fits'] == original['fits']
            assert r['config_sha256'] == original['config_canonical_sha256']
            assert r['data_contract'] == original['original_inventory_record']['data_contract']
            assert r['training_torch'] == original['training_torch'] and r['replay_runtime'] == original['runtime']
            assert r['provenance']['receipt_sha256'] == original['receipt_sha256']
            assert r['provenance']['prediction_arrays_sha256'] == original['prediction_arrays_sha256']
            assert r['provenance']['source_binding'] == original['source_binding']
            assert Path(r['provenance']['accepted900_path']) == FULL and r['provenance']['accepted900_sha256'] == PINS[FULL.relative_to(ROOT).as_posix()]
        else:
            original,index,root = originals[r['id']]; binding = index['new_bindings'][r['id']]
            b = binding['record']; provenance = r['provenance']
            assert checked(index['new_binding_files'][r['id']]) == binding
            assert provenance['binding'] == index['new_binding_files'][r['id']]
            assert provenance['artifacts'] == index['new_artifacts'][r['id']]
            assert ROOT/provenance['root_adoption'] == root and provenance['root_adoption_sha256'] == sha(root)
            assert provenance['accepted_index'] == read(root)['records_index_path']
            receipts = {k:checked(pin) for k,pin in provenance['artifacts'].items()}
            receipt = receipts['scientific_receipt']
            assert r['views'] == original['views'] == receipt['views'] and r['fits'] == receipt['fits']
            assert r['checkpoint_sha256'] == original['checkpoint_sha256'] == binding['checkpoint_sha256'] == b['checkpoint']['sha256'] == receipt['checkpoint_sha256']
            assert r['config_sha256'] == b['config_canonical_sha256'] == receipt['config_canonical_sha256']
            assert r['data_contract'] == b['data_contract']
            assert r['method'] == b['method'] == receipt['method'] and receipt['id'] == r['id']
            assert (receipt['distribution'],receipt['attack'],receipt['seed']) == ('IID',r['attack'],r['seed'])
            assert (b['variant'],b['distribution'],b['attack'],b['seed'],b['terminal_round'],b['original_split'],b['original_n_eval']) == ('minus_A','IID',r['attack'],r['seed'],70,'valid',19867)
            assert receipt['original_result_sha256'] == b['result']['sha256'] and receipt['original_job_sha256'] == b['raw_job']['sha256']
            assert r['training_torch'] == b['training_torch'] == receipt['original_training_torch']
            assert r['replay_runtime'] == receipt['runtime'] and receipt['weights_before'] == receipt['weights_after']
            assert not receipt['optimizer_created'] and not receipt['gradients_created'] and not receipt['test_inference_performed']
            assert receipt['native_comparison']['tolerance'] == 1e-12 and receipt['native_comparison']['accepted']
            assert receipt['native_comparison']['max_abs_difference'] <= 1e-12
            assert provenance['prediction_arrays_sha256'] == original['prediction_arrays_sha256'] == receipt['prediction_arrays_sha256']
            assert b['paired_full'] == refs[cells['Full',r['attack'],r['seed']]['id']]
        paired = cells['Full',r['attack'],r['seed']]
        assert r['data_contract'] == paired['data_contract']
        for view in r['views'].values():
            left,right = (view['group_confusion_counts'][str(g)] for g in (0,1))
            for g in (left,right):
                assert all(type(g[k]) is int and g[k]>=0 for k in ('tp','fp','tn','fn'))
                assert g['tp']+g['fn']==g['positives'] and g['fp']+g['tn']==g['negatives']
                assert g['positives']+g['negatives']==g['n']; count_checks+=4
            n=left['n']+right['n']; assert n==view['prediction_count']==19867
            calculated=dict(accuracy=(left['tp']+left['tn']+right['tp']+right['tn'])/n,
                aeod=abs(left['tp']/left['positives']-right['tp']/right['positives']),
                aspd=abs((left['tp']+left['fp'])/left['n']-(right['tp']+right['fp'])/right['n']))
            for key,value in calculated.items():
                assert abs(value-view[key])<=1e-12; metric_checks+=1
    views=('native','raw','shared_calibration')
    seed_sets=[list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
    panels=table['panels']; old_panels=read(OLD/'tables.json')['panels']
    assert len(panels)==9 and {(p['view'],tuple(p['seeds'])) for p in panels}==set(itertools.product(views,map(tuple,seed_sets)))
    errors=[]; all_rows=[]; benign_rows=[]
    for panel in panels:
        seeds=panel['seeds']; view=panel['view']; rows=panel['rows']
        assert len(rows)==6 and {(r['attack'],r['variant']) for r in rows}==set(itertools.product(attacks,(*variants,'minus_A minus Full')))
        old_panel=next(p for p in old_panels if (p['view'],p['seeds'])==(view,seeds))
        assert [r for r in rows if r['attack']=='Benign']==old_panel['rows']
        for row in rows:
            assert row['n']==row['expected_n']==len(seeds) and row['seeds']==seeds and row['complete']
            assert row['distribution']=='IID' and row['attack'] in attacks
            for key in ('accuracy_pct','aeod','aspd'):
                def value(v,s):
                    metrics=cells[v,row['attack'],s]['views'][view]
                    return 100*metrics['accuracy'] if key=='accuracy_pct' else metrics[key]
                xs=[value('minus_A',s)-value('Full',s) if row['variant']=='minus_A minus Full' else value(row['variant'],s) for s in seeds]
                mean=math.fsum(xs)/len(xs); sd=math.sqrt(math.fsum((x-mean)**2 for x in xs)/(len(xs)-1))
                errors.extend((abs(mean-row[key]['mean']),abs(sd-row[key]['sample_sd_ddof1'])))
            all_rows.append(row)
            if row['attack']=='Benign': benign_rows.append(row)
    lines=[x for x in (directory/'TABLES.md').read_text(encoding='utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
    old_lines=[x for x in (OLD/'TABLES.md').read_text(encoding='utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
    assert len(lines)==len(all_rows)==54
    display=0; benign_lines=[]
    for line,row in zip(lines,all_rows):
        columns=line.strip('| ').split(' | ')
        assert columns[0]==f"{row['distribution']} / {row['attack']} / {row['variant']}" and int(columns[1])==row['n']
        for offset,key in enumerate(('accuracy_pct','aeod','aspd'),2):
            digits=3 if key=='accuracy_pct' else 5
            assert columns[offset]==f"{row[key]['mean']:.{digits}f} ± {row[key]['sample_sd_ddof1']:.{digits}f}"; display+=1
        if row['attack']=='Benign': benign_lines.append(line)
    assert [line.strip('| ').split(' | ')[1:] for line in benign_lines]==[line.strip('| ').split(' | ')[1:] for line in old_lines] and len(benign_rows)==27
    assert [line.strip('| ').split(' | ')[0] for line in old_lines]==[row['variant'] for row in benign_rows]
    assert len(errors)==324 and max(errors)<=1e-12 and display==162 and metric_checks==360 and count_checks==960
    return dict(status='INDEPENDENT_A20_TWO_IID_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),paired_models=20,complete_scenes=2,
        preserved_records=40,mean_SD_scalars_recomputed=324,display_cells=162,metrics_from_group_counts=360,
        base_integer_confusion_counts_checked=960,max_abs_difference=max(errors),old24_normalized_JSON_bytes_and_order_exact=True,
        old_Benign162_scalars_exact=True,old_Benign81_display_cells_exact=True,original_Full20_records_exact=True,
        original_A20_saved_views_fits_and_bindings_exact=True,source_acceptance_sha256=sha(roots[1]),
        source_prior_acceptance_sha256=sha(roots[0]),delivery_seal_sha256=delivery_seal,table_directory=str(directory),
        files_sha256={n:sha(directory/n) for n in ('records.json','tables.json','TABLES.md','SOURCE_BINDINGS.json')},
        replay_devices={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in variants},
        training_torch={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in variants},
        root_adoption=False,canonical_written=False,new_CNN=0,new_fits=0,new_training=0,test=False,
        claim_limit='Two IID scenes only; AEOD is absolute TPR gap. Mixed replay devices and validation/selection history remain; no necessity, significance or final-test claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table-dir',required=True,type=Path); p.add_argument('--delivery-seal',required=True)
    a=p.parse_args()
    if sys.flags.optimize: raise RuntimeError('Assertions require unoptimized Python')
    assert not (OWN/'ROOT_ARITHMETIC_REVIEW.json').exists() and not (OWN/'REVIEW_FAILURE.json').exists(), 'one-shot output exists'
    try:
        proof=review(a.table_dir.resolve(),a.delivery_seal)
        with (OWN/'ROOT_ARITHMETIC_REVIEW.json').open('x',encoding='utf8') as f: json.dump(proof,f,indent=2); f.write('\n')
        print(json.dumps({'status':proof['status'],'proof_sha256':sha(OWN/'ROOT_ARITHMETIC_REVIEW.json'),'stats':324,'cells':162}))
    except Exception as e:
        with (OWN/'REVIEW_FAILURE.json').open('x',encoding='utf8') as f:
            json.dump({'status':'REVIEW_FAILED_NO_ADOPTION','command':sys.argv,'error':repr(e),'traceback':traceback.format_exc()},f,indent=2); f.write('\n')
        raise


if __name__=='__main__': main()
