"""ROOT-only independent identity/count/statistic/display review and exact promotion."""
from pathlib import Path
import argparse, datetime, hashlib, json, math, shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'tmp/celeba_mechanism_three_view100_tables_20261009'
DEST = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim100_20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seal', required=True)
parser.add_argument('--seal-sha256', required=True)
parser.add_argument('--inputs', required=True)
args = parser.parse_args()
seal_path = BASE/args.seal; inputs_path = BASE/args.inputs
assert seal_path.resolve().is_relative_to(BASE.resolve()) and inputs_path.resolve().is_relative_to(BASE.resolve())
assert sha(seal_path) == args.seal_sha256 and not DEST.exists()
seal = read(seal_path)['files']
for name,row in seal.items():
    p = BASE/name
    assert p.resolve().is_relative_to(BASE.resolve()) and sha(p) == row['sha256'] and p.stat().st_size == row['bytes']
assert args.inputs == 'FINAL_SOURCE_BINDINGS.json'
bindings = read(inputs_path)
assert bindings['prepared_seal_sha256'] == sha(BASE/'FILES_SHA256.json') == '8d085d2a1e39f0c37dcaaea962a593614c055f05591d05691de7cbea0795fe40'
for source_files in [read(BASE/'INPUTS.json')['files'], bindings['actual_new8_files']]:
    for name,row in source_files.items():
        p = ROOT/name
        assert p.resolve().is_relative_to(ROOT.resolve()) and sha(p) == row['sha256'] and p.stat().st_size == row['bytes']
adopt = ROOT/'tmp/celeba_mechanism_valid_incremental_after92_20261009/execution_candidate/backups/incremental_20261009T183102Z/ROOT_ADOPTION_REVIEW.json'
assert sha(adopt) == '9050eb059a797c70f0ca977294989b5ae5757286dbc85b36d529012cb5ab72ee'
assert read(adopt)['cumulative_three_view_models'] == 100
records = read(BASE/'snapshot100/records.json')['records']; tables = read(BASE/'snapshot100/tables.json')
assert len(records) == len({r['id'] for r in records}) == 200
assert (tables['complete_scenes'],tables['paired_checkpoints'],tables['table_model_records'],tables['incomplete_pairs']) == (10,100,200,0)
assert not tables['primary_endpoint_selected'] and not tables['final_test'] and tables['new_inference'] == tables['new_training'] == 0
bycell = {(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
scenes = {(d,a) for d in ['IID','non-IID'] for a in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA']}
assert set(bycell) == {(v,d,a,s) for v in ['Full','minus_U'] for d,a in scenes for s in range(91001,91011)}
full = {(r['distribution'],r['attack'],r['seed']):r for r in read(ROOT/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/records_three_views_900.json')['records'] if r['method'] == 'GuardFed-AD2+'}
original = {r['id']:r for r in read(ROOT/'tmp/celeba_mechanism_valid_incremental_after92_20261009/inventory_actual100_Full100refs.json')['records']}
assert {r['id'] for r in records if r['variant'] == 'minus_U'} == set(original)
metric_count = 0
for r in records:
    assert r['views']['native'] == r['views']['shared_calibration']
    if r['variant'] == 'Full':
        old = full[(r['distribution'],r['attack'],r['seed'])]
        assert r['checkpoint_sha256'] == old['checkpoint_sha256'] and r['config_sha256'] == old['config_canonical_sha256']
        assert r['views'] == old['views'] and r['fits'] == old['fits']
    else:
        old = original[r['id']]
        assert r['checkpoint_sha256'] == old['checkpoint']['sha256'] and r['config_sha256'] == old['config_canonical_sha256']
        for k in ['accuracy','aeod','aspd']:
            assert abs(r['views']['native'][k]-old['prior_validation_metrics'][k]) <= 1e-12
    for view in r['views'].values():
        a,b = view['group_confusion_counts']['0'],view['group_confusion_counts']['1']
        for g in [a,b]:
            assert all(isinstance(g[k],int) and g[k] >= 0 for k in ['tp','tn','fp','fn','n'])
            assert g['tp']+g['tn']+g['fp']+g['fn'] == g['n']
        n = a['n']+b['n']; assert n == view['prediction_count'] == 19867
        metrics = dict(accuracy=(a['tp']+a['tn']+b['tp']+b['tn'])/n,
            aeod=abs(a['tp']/(a['tp']+a['fn'])-b['tp']/(b['tp']+b['fn'])),
            aspd=abs((a['tp']+a['fp'])/a['n']-(b['tp']+b['fp'])/b['n']))
        for k,value in metrics.items():
            assert abs(view[k]-value) <= 1e-12
            metric_count += 1
panels = tables['panels']; assert len(panels) == 9
assert {(p['view'],tuple(p['seeds'])) for p in panels} == {(v,tuple(s)) for v in ['native','raw','shared_calibration'] for s in [range(91001,91011),range(91002,91011),range(91005,91011)]}
scalar_count = 0; maximum = 0.; expected_lines = []
for panel in panels:
    seeds = panel['seeds']; assert len(panel['rows']) == 30
    assert {(r['distribution'],r['attack'],r['variant']) for r in panel['rows']} == {(d,a,v) for d,a in scenes for v in ['Full','minus_U','minus_U minus Full']}
    for row in panel['rows']:
        d,a,v = row['distribution'],row['attack'],row['variant']
        assert row['seeds'] == seeds and row['n'] == row['expected_n'] == len(seeds) and row['complete']
        cells = []
        for key,metric,scale,precision in [('accuracy_pct','accuracy',100,3),('aeod','aeod',1,5),('aspd','aspd',1,5)]:
            values = []
            for seed in seeds:
                if v == 'minus_U minus Full':
                    x = bycell[('minus_U',d,a,seed)]['views'][panel['view']][metric]-bycell[('Full',d,a,seed)]['views'][panel['view']][metric]
                else:
                    x = bycell[(v,d,a,seed)]['views'][panel['view']][metric]
                values.append(x*scale)
            mean = math.fsum(values)/len(values)
            sd = math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1))
            for expected,actual in [(mean,row[key]['mean']),(sd,row[key]['sample_sd_ddof1'])]:
                difference = abs(expected-actual); assert difference <= 1e-12
                maximum = max(maximum,difference); scalar_count += 1
            cells.append(f'{mean:.{precision}f} ± {sd:.{precision}f}')
        expected_lines.append(f'| {d} | {a} | {v} | {len(seeds)} | '+ ' | '.join(cells)+' |')
display_lines = [s for s in (BASE/'snapshot100/TABLES.md').read_text(encoding='utf8').splitlines() if s.startswith('| ') and ' ± ' in s]
assert display_lines == expected_lines and len(display_lines)*3 == 810
old_base = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim92_20261009/snapshot92'
old_records = read(old_base/'records.json')['records']; index_id = {r['id']:r for r in records}
assert len(old_records) == 184 and all(index_id[r['id']] == r for r in old_records)
old_rows = 0
for before,after in zip(read(old_base/'tables.json')['panels'],panels):
    assert before['view'] == after['view'] and before['seeds'] == after['seeds']
    lookup = {(r['distribution'],r['attack'],r['variant']):r for r in after['rows']}
    for row in before['rows']:
        assert lookup[(row['distribution'],row['attack'],row['variant'])] == row
        old_rows += 1
assert scalar_count == 1620 and metric_count == 1800 and old_rows == 243
for name,row in seal.items():
    target = DEST/name; target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(BASE/name,target); assert sha(target) == row['sha256']
shutil.copyfile(seal_path,DEST/args.seal)
proof = dict(status='ROOT_THREE_VIEW100_TEN_SCENE_COUNTS_PAIRED_STATISTICS_DISPLAY_AND_PRIOR_ROWS_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_seal_sha256=sha(seal_path),source_seal_name=args.seal,
    source_entry=BASE.relative_to(ROOT).as_posix(), sealed_members_verified=len(seal), accepted_three_view100=100,
    paired_checkpoints=100,preserved_records=200,complete_scenes=10,incomplete_paired_checkpoints=0,display_cells_verified=810,
    independently_reconstructed_metrics=metric_count,independent_mean_sd_scalars=scalar_count,max_abs_statistic_difference=maximum,
    old_nine_scene_statistic_rows_exact=old_rows,old_records_exact=184,original_Full900_view_fit_checkpoint_config_exact=True,
    original_native100_control_identity_exact=True,native_shared_identical_records=200,new8_root_adoption_sha256=sha(adopt),
    tables_sha256=sha(DEST/'snapshot100/tables.json'),records_sha256=sha(DEST/'snapshot100/records.json'),
    verification_sha256=sha(DEST/'snapshot100/verification.json'),source_helper_sha256=sha(Path(__file__)),
    new_inference=0,new_training=0,primary_endpoint_selected=False,test=False,negative_results_preserved=True)
with (DEST/'ROOT_REVIEW.json').open('x',encoding='utf8',newline='\n') as stream:
    json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=proof['status'],root_proof_sha256=sha(DEST/'ROOT_REVIEW.json'),scalars=scalar_count,
    reconstructed_metrics=metric_count,display_cells=810,canonical=str(DEST))))
