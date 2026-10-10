"""Independent A40 arithmetic/provenance review; no table adoption or inference."""
from pathlib import Path
from collections import Counter
import argparse, datetime, hashlib, itertools, json, math, sys, traceback

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[1]
OLD = ROOT / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_two_scenes20_20261010'
FULL = ROOT / 'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/records_three_views_900.json'
REFS = ROOT / 'tmp/celeba_mechanism_valid_C_after1_20261009/inventory_actual112_Full100refs.json'
PINS = {
    'tmp/celeba_A20_table_root_review_v2_20261010/review.py': 'cc0f6890856cf6de63ab552414959e7972081aeebbeb1be56bf4636f07e1df6e',
    'tmp/celeba_A20_table_root_review_v2_20261010/ROOT_ARITHMETIC_REVIEW.json': 'e6cee89fd0406b006c008200d6fff5a4966967b3dea3271b7a5b5a682bfa8051',
    'tmp/celeba_mechanism_remaining620_A28_root_adoption_20261010/ROOT_ADOPTION.json': 'f6761dd1c2b844724aee91d235d71f6474511ac8aec63f3dc4fe0308272e0966',
    'tmp/celeba_mechanism_remaining620_A28_root_adoption_20261010/MECHANISM228_INDEX.json': '765ea715defea1e54aebbee0115f38c926ee022f47f520d151d1417c6c8592a9',
    'tmp/celeba_mechanism_remaining620_A36_root_adoption_20261010/ROOT_ADOPTION.json': '1513441c17a3d6439df4a2944c6fcf6a0e6bb6aa2280d2176d7bc1527a78c36b',
    'tmp/celeba_mechanism_remaining620_A36_root_adoption_20261010/MECHANISM236_INDEX.json': 'f378bab97b5a2fba436370f4508f122198559d05920dac2ac9075d83ef7f6e59',
    'tmp/celeba_A20_table_root_review_20261010/review.py': '56f98589d50cdebd71e2bc2c44cab2b7d904980f4af4dac3f92a36b808cbb737',
    'tmp/celeba_A20_table_root_review_20261010/REVIEW_FAILURE.json': '51314f65a18108e7d090f4903661d49a411649a9fe674d8a500a0c5342711ca0',
    'tmp/verify_A_Benign10_root_20261010.py': '2e7407b315f599baea9453e7ebe602a65249f684b579afd5d764bf8a87acca2a',
    'tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010/ROOT_ADOPTION.json': '1221482d564a2c735b0de0680fe8a42512c9fe5774a9d157bd3bfda5dd9c858b',
    'tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010/MECHANISM212_INDEX.json': '9a90f3d74a27d9ca4225b850797faec3c3aac2b6e2f83f87d3afe49c21912496',
    'tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010/ROOT_ADOPTION.json': 'e088871fbd98cbc9415cc79a44626667a532389ce0b6dddbaaf2ab25f72a4979',
    'tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010/MECHANISM220_INDEX.json': 'af414a0ac6705230c53d324cd1f51e7a76b893dabd7416bb6914a6690b6fdddc',
    FULL.relative_to(ROOT).as_posix(): '983bca43dff7e94dd79a312f273c65ed3ea1d146fff3ebfbf3bd2a854e124529',
    REFS.relative_to(ROOT).as_posix(): '5e46f03d99be908bf19cf376909b5b7ae11dfcc098be09f1d08f8ad9c5688f32',
    (OLD/'FILES_SHA256.json').relative_to(ROOT).as_posix(): '3841a7498dad4a3dc6a0bf3b8262510082cd042671850b5f56320a079c012ef8',
    (OLD/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(): 'dbc2f8fe481c2a049f4e556a500c29d3122fc8392d69e4ab2bc131c0e3f4b3c0',
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


def record_fragments(text):
    pos=text.index('[',text.index('"records"'))+1; decoder=json.JSONDecoder(); fragments=[]
    while True:
        while text[pos] in ' \t\r\n,': pos+=1
        if text[pos]==']': return fragments
        record,end=decoder.raw_decode(text,pos)
        fragments.append((record['id'],text[pos:end].encode('utf8'))); pos=end


def review(directory, delivery_seal, adoption, adoption_sha256):
    sealed(directory, delivery_seal)
    for name, digest in PINS.items():
        assert sha(ROOT/name) == digest, name
    sealed(OLD, PINS[(OLD/'FILES_SHA256.json').relative_to(ROOT).as_posix()])
    assert sha(adoption)==adoption_sha256, 'external A40 root adoption SHA differs'
    roots = [ROOT/f'tmp/celeba_mechanism_remaining620_A{n}_root_adoption_20261010/ROOT_ADOPTION.json' for n in (12,20,28,36)] + [adoption]
    adoptions = [read(p) for p in roots]
    indexes = [read(ROOT/a['records_index_path']) for a in adoptions]
    assert adoptions[-1]['status']=='ROOT_A40_SAVED_ARRAYS_AND_NATIVE243_RESTORE_CHAIN_ADOPTED'
    assert adoptions[-1]['original236_unchanged']
    assert [len(i['new_ids']) for i in indexes]==[12,8,8,8,4]
    assert [len(i['all_ids']) for i in indexes]==[212,220,228,236,240]
    wanted=[f'minus_A_IID_{attack}_seed{s}' for attack in ('Benign','F Flip','FedSA','S-DFA') for s in range(91001,91011)]
    assert [rid for i in indexes for rid in i['new_ids']]==wanted
    native_maps={}
    for pos,(a,index,root) in enumerate(zip(adoptions,indexes,roots)):
        assert sha(ROOT/a['records_index_path'])==a['records_index_sha256']
        assert a['accepted_new_ids']==index['new_ids'] and a['new_accepted']==len(index['new_ids'])
        assert a['cumulative_accepted']==len(index['all_ids'])==len(set(index['all_ids']))
        assert a['native_max_abs_difference']==0 and a['Full_inference']==a['new_CNN']==a['new_training']==a.get('new_fit',0)==0 and a['test'] is False
        assert a['status'].startswith('ROOT_A') and a['status'].endswith('_ADOPTED')
        assert sha(ROOT/index['native_inspection_path'])==index['native_inspection_sha256']==a['native_inspection_sha256']
        native_rows=read(ROOT/index['native_inspection_path'])['records']
        native_maps[a['records_index_path']]={r['id']:r for r in native_rows}
        assert len(native_maps[a['records_index_path']])==len(native_rows)
        assert set(index['new_bindings'])==set(index['new_binding_files'])==set(index['new_artifacts'])==set(index['new_ids'])
        if pos:
            prior=indexes[pos-1]
            assert index['prior_index_path']==adoptions[pos-1]['records_index_path']
            assert index['prior_index_sha256']==adoptions[pos-1]['records_index_sha256']
            assert ROOT/index['prior_adoption_path']==roots[pos-1] and index['prior_adoption_sha256']==sha(roots[pos-1])
            assert index['all_ids']==prior['all_ids']+index['new_ids'] and a['prior_accepted']==len(prior['all_ids'])
    originals = {r['id']:(r,index,root) for index,root in zip(indexes,roots) for r in index['new_records']}
    assert len(originals)==40 and list(originals)==wanted
    source_bindings=read(directory/'SOURCE_BINDINGS.json')
    assert ROOT/source_bindings['actual_A40_root_adoption']==adoption and source_bindings['actual_A40_root_adoption_sha256']==adoption_sha256
    assert source_bindings['accepted_index_sha256']==adoptions[-1]['records_index_sha256'] and source_bindings['accepted_index']==adoptions[-1]['records_index_path']
    full = {r['id']:r for r in read(FULL)['records']}
    refs = {r['id']:r for r in read(REFS)['full_references']}
    assert len(refs) == 100
    records = read(directory/'records.json')['records']; table = read(directory/'tables.json')
    assert len(records) == len({r['id'] for r in records}) == 80
    cells = {(r['variant'],r['attack'],r['seed']):r for r in records}
    attacks = ('Benign','F Flip','FedSA','S-DFA'); variants = ('Full','minus_A')
    assert set(cells) == set(itertools.product(variants,attacks,range(91001,91011)))
    assert all(r['distribution']=='IID' for r in records)
    assert (table['complete_scenes'],table['paired_models'],table['preserved_records']) == (4,40,80)
    assert table['final_test'] is False and table['primary_endpoint_selected'] is False
    assert table['new_threshold_fits'] == table['new_inference'] == table['new_training'] == 0
    old_records = read(OLD/'records.json')['records']; old_ids = {r['id'] for r in old_records}
    assert [canonical(r) for r in records if r['id'] in old_ids] == [canonical(r) for r in old_records]
    assert len(old_records)==40
    assert [x for x in record_fragments((directory/'records.json').read_text(encoding='utf8')) if x[0] in old_ids]==record_fragments((OLD/'records.json').read_text(encoding='utf8'))
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
            assert b['accepted_v4_row']==native_maps[read(root)['records_index_path']][r['id']]
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
        assert len(rows)==12 and {(r['attack'],r['variant']) for r in rows}==set(itertools.product(attacks,(*variants,'minus_A minus Full')))
        old_panel=next(p for p in old_panels if (p['view'],p['seeds'])==(view,seeds))
        assert [r for r in rows if r['attack'] in ('Benign','F Flip')]==old_panel['rows']
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
            if row['attack'] in ('Benign','F Flip'): benign_rows.append(row)
    lines=[x for x in (directory/'TABLES.md').read_text(encoding='utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
    old_lines=[x for x in (OLD/'TABLES.md').read_text(encoding='utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
    assert len(lines)==len(all_rows)==108
    display=0; benign_lines=[]
    for line,row in zip(lines,all_rows):
        columns=line.strip('| ').split(' | ')
        assert columns[0]==f"{row['distribution']} / {row['attack']} / {row['variant']}" and int(columns[1])==row['n']
        for offset,key in enumerate(('accuracy_pct','aeod','aspd'),2):
            digits=3 if key=='accuracy_pct' else 5
            assert columns[offset]==f"{row[key]['mean']:.{digits}f} ± {row[key]['sample_sd_ddof1']:.{digits}f}"; display+=1
        if row['attack'] in ('Benign','F Flip'): benign_lines.append(line)
    assert [line.strip('| ').split(' | ')[1:] for line in benign_lines]==[line.strip('| ').split(' | ')[1:] for line in old_lines] and len(benign_rows)==54
    assert [line.strip('| ').split(' | ')[0] for line in old_lines]==[f"{row['distribution']} / {row['attack']} / {row['variant']}" for row in benign_rows]
    assert len(errors)==648 and max(errors)<=1e-12 and display==324 and metric_checks==720 and count_checks==1920
    return dict(status='INDEPENDENT_A40_FOUR_IID_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),paired_models=40,complete_scenes=4,
        preserved_records=80,mean_SD_scalars_recomputed=648,display_cells=324,metrics_from_group_counts=720,
        base_integer_confusion_counts_checked=1920,max_abs_difference=max(errors),old40_record_JSON_bytes_and_order_exact=True,
        old_A20_324_scalars_exact=True,old_A20_162_display_cells_exact=True,original_Full40_records_exact=True,
        original_A40_saved_views_fits_and_bindings_exact=True,source_acceptance_sha256=adoption_sha256,
        source_prior_acceptance_sha256=sha(roots[-2]),five_batch_root_sha256=[sha(p) for p in roots],delivery_seal_sha256=delivery_seal,table_directory=str(directory),
        files_sha256={n:sha(directory/n) for n in ('records.json','tables.json','TABLES.md','SOURCE_BINDINGS.json')},
        replay_devices={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in variants},
        training_torch={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in variants},
        root_adoption=False,canonical_written=False,new_CNN=0,new_fits=0,new_training=0,test=False,
        reviewer_role='Prepared old A20 builder and reviewed A40 builder; did not author A40 builder. Numeric/count loops retained from accepted independent A20 reviewer.',
        claim_limit='Four IID scenes only; AEOD is absolute TPR gap. Mixed replay devices and validation/selection history remain; no necessity, significance or final-test claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table-dir',required=True,type=Path); p.add_argument('--delivery-seal',required=True)
    p.add_argument('--adoption',required=True,type=Path); p.add_argument('--adoption-sha256',required=True)
    a=p.parse_args()
    if sys.flags.optimize: raise RuntimeError('Assertions require unoptimized Python')
    assert not (OWN/'ROOT_ARITHMETIC_REVIEW.json').exists() and not (OWN/'REVIEW_FAILURE.json').exists(), 'one-shot output exists'
    try:
        proof=review(a.table_dir.resolve(),a.delivery_seal,a.adoption.resolve(),a.adoption_sha256)
        with (OWN/'ROOT_ARITHMETIC_REVIEW.json').open('x',encoding='utf8') as f: json.dump(proof,f,indent=2); f.write('\n')
        print(json.dumps({'status':proof['status'],'proof_sha256':sha(OWN/'ROOT_ARITHMETIC_REVIEW.json'),'stats':648,'cells':324}))
    except Exception as e:
        with (OWN/'REVIEW_FAILURE.json').open('x',encoding='utf8') as f:
            json.dump({'status':'REVIEW_FAILED_NO_ADOPTION','command':sys.argv,'error':repr(e),'traceback':traceback.format_exc()},f,indent=2); f.write('\n')
        raise


if __name__=='__main__': main()
