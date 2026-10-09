"""Recompute and adopt the actual C40 validation table after independent review."""
from pathlib import Path, PurePosixPath
import argparse, collections, datetime, hashlib, itertools, json, math, shutil, sys

if sys.flags.optimize:
    raise RuntimeError('Optimized Python is forbidden')
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'tmp/celeba_mechanism_three_view_C_four_scenes_prepared_20261010'
OLD = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_three_scenes_20261009'
DEST = OLD.parent/'three_view_C_four_scenes_20261010'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff-sha256', required=True)
    parser.add_argument('--independent-review', required=True, type=Path)
    parser.add_argument('--independent-sha256', required=True)
    args = parser.parse_args()
    assert not DEST.exists(), 'Preserve any prior adoption'
    handoff_path = BASE/'ACTUAL_HANDOFF.json'
    assert sha(handoff_path) == args.handoff_sha256
    handoff = read(handoff_path)
    independent_path = args.independent_review.resolve()
    assert independent_path.is_relative_to(ROOT/'tmp/celeba_mechanism_C40_root_arithmetic_review_20261010')
    assert sha(independent_path) == args.independent_sha256
    independent = read(independent_path)
    assert independent['status'] == 'INDEPENDENT_C40_FOUR_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION'
    assert independent['actual_handoff_sha256'] == sha(handoff_path)
    assert (independent['unique_records'], independent['paired_models'], independent['complete_scenes'],
            independent['mean_SD_scalars_recomputed'], independent['display_cells'], independent['count_metrics_recomputed']) == (80, 40, 4, 648, 324, 720)
    for key in ('old60_record_JSON_bytes_and_order_exact', 'old486_scalars_exact',
                'old243_display_cells_exact', 'S_DFA_six_original_record_bytes_preserved'):
        assert independent[key]
    actual_seal = BASE/'ACTUAL_FILES_SHA256.json'
    assert independent['actual_delivery_seal_sha256'] == sha(actual_seal)
    files = read(actual_seal)['files']
    assert isinstance(files, dict) and files['ACTUAL_HANDOFF.json']['sha256'] == sha(handoff_path)
    for name, pin in files.items():
        rel = PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts and rel.suffix != '.pt'
        assert sha(BASE/name) == pin['sha256'] and (BASE/name).stat().st_size == pin['bytes']
    bindings = read(BASE/'snapshot/SOURCE_BINDINGS.json')
    for name, pin in bindings['prepared_input_pins'].items():
        assert sha(ROOT/name) == pin['sha256'] and (ROOT/name).stat().st_size == pin['bytes']
    assert handoff['actual_C4_adoption_sha256'] == bindings['actual_C4_adoption_sha256'] == independent['actual_C4_root_adoption_sha256'] == 'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a'
    assert sha(ROOT/bindings['actual_C4_adoption']) == bindings['actual_C4_adoption_sha256']
    assert sha(ROOT/bindings['new_archive_chain']['archive']) == bindings['new_archive_chain']['archive_sha256'] == '9e7206998d61dafbccc99a4b3f35088f72f887ddf0e6fea3d6a7349d5135b3f1'
    assert handoff['inference'] == handoff['training'] == bindings['new_inference'] == 0 and handoff['test'] is False
    assert sha(OLD/'ROOT_VERIFICATION.json') == '2cce0519555efbff75f559875c4c1afe63e07d242cb8e3ebe6af644efc95961d'
    records = read(BASE/'snapshot/records.json')['records']
    tables = read(BASE/'snapshot/tables.json')
    by = {(r['variant'], r['distribution'], r['attack'], r['seed']): r for r in records}
    scenes = ('Benign', 'F Flip', 'FedSA', 'S-DFA')
    assert len(records) == len({r['id'] for r in records}) == len(by) == 80
    assert set(by) == set(itertools.product(('Full', 'minus_C'), ('IID',), scenes, range(91001, 91011)))
    old_records = read(OLD/'snapshot/records.json')['records']
    old_ids = {r['id'] for r in old_records}
    assert len(old_ids) == 60 and [r for r in records if r['id'] in old_ids] == old_records
    errors = []
    count_metrics = 0
    for record in records:
        assert record['data_contract']['evaluation_split'] == 'valid'
        assert record['data_contract']['actual_evaluation_rows'] == 19867
        assert record['views']['native'] == record['views']['shared_calibration']
        for view in record['views'].values():
            a, b = view['group_confusion_counts']['0'], view['group_confusion_counts']['1']
            assert a['n'] + b['n'] == 19867
            calc = dict(accuracy=(a['tp']+a['tn']+b['tp']+b['tn'])/19867,
                aeod=abs(a['tp']/(a['tp']+a['fn'])-b['tp']/(b['tp']+b['fn'])),
                aspd=abs((a['tp']+a['fp'])/a['n']-(b['tp']+b['fp'])/b['n']))
            for metric, value in calc.items():
                assert abs(view[metric]-value) <= 1e-12
                count_metrics += 1
    for panel in tables['panels']:
        assert len(panel['rows']) == 12
        for row in panel['rows']:
            for metric in ('accuracy_pct', 'aeod', 'aspd'):
                def value(variant, seed):
                    key = 'accuracy' if metric == 'accuracy_pct' else metric
                    return by[(variant, row['distribution'], row['attack'], seed)]['views'][panel['view']][key]*(100 if metric == 'accuracy_pct' else 1)
                xs = [value('minus_C', s)-value('Full', s) if row['variant'] == 'minus_C minus Full' else value(row['variant'], s) for s in panel['seeds']]
                mean = math.fsum(xs)/len(xs)
                sd = math.sqrt(math.fsum((x-mean)**2 for x in xs)/(len(xs)-1))
                errors.extend((abs(mean-row[metric]['mean']), abs(sd-row[metric]['sample_sd_ddof1'])))
    assert len(errors) == 648 and max(errors) <= 1e-12 and count_metrics == 720
    for old_panel, new_panel in zip(read(OLD/'snapshot/tables.json')['panels'], tables['panels']):
        assert old_panel['rows'] == [r for r in new_panel['rows'] if r['attack'] in scenes[:3]]
    assert sha(handoff_path) == args.handoff_sha256 and sha(independent_path) == args.independent_sha256
    DEST.mkdir()
    for name in list(files) + ['ACTUAL_FILES_SHA256.json']:
        target = DEST/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE/name, target)
        assert sha(target) == sha(BASE/name)
    proof = dict(status='ROOT_C40_FOUR_SCENE_THREE_VIEW_TABLES_ADOPTED',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        independent_review_path=independent_path.relative_to(ROOT).as_posix(), independent_review_sha256=sha(independent_path),
        actual_delivery_seal_sha256=sha(actual_seal), actual_members_verified=len(files), C4_root_adoption_sha256=handoff['actual_C4_adoption_sha256'],
        records_sha256=sha(BASE/'snapshot/records.json'), tables_sha256=sha(BASE/'snapshot/tables.json'), display_sha256=sha(BASE/'snapshot/TABLES.md'),
        unique_records=80, paired_models=40, complete_scenes=4, mean_SD_scalars_recomputed=648, display_cells=324,
        count_metrics_recomputed=720, max_abs_difference=max(errors), original_three_scene60_records_exact=True,
        original_three_scene486_statistics_exact=True, original_three_scene243_cells_preserved=True,
        prior_S_DFA_six_original_record_bytes_preserved=True, native_shared_identical_records=80,
        replay_devices={v:dict(collections.Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ('Full','minus_C')},
        training_torch={v:dict(collections.Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ('Full','minus_C')},
        seed_panels=[10,9,6], all_negative_results_retained=True, new_CNN=0, new_training=0, new_Full_inference=0, test=False,
        primary_endpoint='PENDING_AUTHOR', other_C_scenes_complete=False, whole_rebuttal_complete=False,
        canonical_table=(DEST/'snapshot/TABLES.md').relative_to(ROOT).as_posix())
    target = DEST/'ROOT_VERIFICATION.json'
    with target.open('x', encoding='utf8') as stream:
        json.dump(proof, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(root_path=str(target), root_sha256=sha(target), table=proof['canonical_table'])))


if __name__ == '__main__':
    main()
