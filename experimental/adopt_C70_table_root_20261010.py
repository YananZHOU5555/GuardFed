"""Adopt the actual C70 table after its independent arithmetic review; no inference."""
from pathlib import Path, PurePosixPath
from collections import Counter
import datetime, hashlib, json, shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'tmp/celeba_mechanism_C70_table_preparation_20261010'
REVIEW = ROOT/'tmp/celeba_mechanism_C70_independent_review_20261010'
BINDING = ROOT/'tmp/celeba_mechanism_C70_root_operations_20261010/C10_BINDING.json'
OLD = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010'
DEST = OLD.parent/'three_view_C_seven_scenes_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())


def files_in_seal(base, name, expected):
    assert sha(base/name) == expected
    seal = read(base/name)
    rows = seal.get('members')
    if rows is None:
        rows = [dict(path=k, sha256=v['sha256'], size=v['bytes']) for k, v in seal['files'].items()]
    for row in rows:
        rel = PurePosixPath(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts and ':' not in row['path'] and '\\' not in row['path']
        path = base/row['path']
        assert not path.is_symlink() and path.resolve().is_relative_to(base.resolve())
        assert sha(path) == row['sha256'] and path.stat().st_size == row['size']
    return [r['path'] for r in rows] + [name]


def main():
    assert not DEST.exists()
    src = files_in_seal(SOURCE, 'FILES_SHA256.json', '758d49ac894a423f151f64cb0619f630bb6fdb697d9e8d16949b163ba09f336e')
    snap = files_in_seal(SOURCE/'snapshot', 'FILES_SHA256.json', 'cf81e43fd957b098ce8f72821b2530c4db6854cca8726c03d0f8e4c1a8084d2e')
    rev = files_in_seal(REVIEW, 'FINAL_FILES_SHA256.json', '9b1e8624c7ebb47d53b6fb20112ba5ee957fa3d99cf2392684e8a0d7d99b6ed2')
    assert (len(src)-1, len(snap)-1, len(rev)-1) == (12, 8, 10)
    proof = read(REVIEW/'ROOT_ARITHMETIC_REVIEW.json')
    assert sha(REVIEW/'ROOT_ARITHMETIC_REVIEW.json') == 'dcb2f2cbb0e19490f07f7f9c41556feb0995074f7067c42f637dc37ea8c7371d'
    assert proof['status'] == 'INDEPENDENT_C70_SEVEN_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION'
    assert proof['snapshot_seal_sha256'] == sha(SOURCE/'snapshot/FILES_SHA256.json')
    assert proof['source_prepared_seal_sha256'] == sha(SOURCE/'FILES_SHA256.json')
    fields = ('unique_records', 'paired_models', 'complete_scenes', 'mean_SD_scalars_recomputed', 'display_cells', 'count_metrics_recomputed', 'confusion_count_checks', 'paired_seed_metric_checks', 'cross_scene_mean_SD_scalars_recomputed')
    assert tuple(proof[k] for k in fields) == (140, 70, 7, 1134, 567, 1260, 3360, 630, 162)
    for key in ('old120_record_JSON_bytes_and_order_exact', 'old972_scalars_exact', 'old486_display_cells_exact', 'old162_IID_aggregate_bytes_exact', 'n_is_seed_count'):
        assert proof[key] is True
    assert proof['max_abs_difference'] <= 1e-12 and proof['count_metric_max_abs_difference'] == 0
    assert proof['primary_endpoint'] == 'PENDING_AUTHOR' and proof['seed_first_scenes_per_seed'] == 5
    for key in ('adoption_performed', 'canonical_modified', 'test', 'whole_mechanism_complete', 'whole_rebuttal_complete', 'other_three_nonIID_C_scenes_complete', 'six_other_image_controls_complete'):
        assert proof[key] is False
    assert proof['new_CNN'] == proof['new_training'] == proof['new_Full_inference'] == 0
    assert sha(BINDING) == 'a50061f0f70102babe08ccac3151da5c308c20a04e79762110bf42dcb60ed9e1'
    binding = read(BINDING)
    c10 = ROOT/binding['adoption']
    assert sha(c10) == binding['adoption_sha256'] == proof['actual_C10_root_adoption_sha256'] == '7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8'
    c = read(c10)
    assert (c['prior_three_view_models'], c['accepted_new'], c['cumulative_three_view_models']) == (160, 10, 170)
    assert c['accepted_new_ids'] == [f'minus_C_non-IID_F Flip_seed{s}' for s in range(91001, 91011)]
    assert c['all_native_differences_zero'] and c['original160_unchanged'] and c['source_scope_complete']
    assert c['science_seal_sha256'] == binding['science_sha256'] and c['execution_seal_sha256'] == binding['execution_sha256']
    assert c['new_training'] == c['new_Full_inference'] == 0 and not c['test_inference']
    bindings = read(SOURCE/'snapshot/SOURCE_BINDINGS.json')
    assert bindings['actual_C10_binding'] == binding and bindings['external_binding_sha256'] == sha(BINDING)
    assert bindings['original_C60_root_sha256'] == sha(OLD/'ROOT_VERIFICATION.json') == 'f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742'
    assert bindings['new_inference'] == 0
    for name, key in [('records.json', 'records_sha256'), ('tables.json', 'tables_sha256'), ('TABLES.md', 'display_sha256'), ('cross_scene_seed_first.json', 'cross_scene_seed_first_sha256')]:
        assert sha(SOURCE/'snapshot'/name) == proof[key]
    records = read(SOURCE/'snapshot/records.json')['records']
    old_records = read(OLD/'snapshot/records.json')['records']
    assert records[:120] == old_records and len(records) == len({r['id'] for r in records}) == 140
    scenes = [('IID', a) for a in ('Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA')] + [('non-IID', a) for a in ('Benign', 'F Flip')]
    assert {(r['variant'], r['distribution'], r['attack'], r['seed']) for r in records} == {(v, d, a, s) for v in ('Full', 'minus_C') for d, a in scenes for s in range(91001, 91011)}
    tables = read(SOURCE/'snapshot/tables.json')
    assert (tables['unique_records'], tables['paired_models'], tables['complete_scenes']) == (140, 70, 7)
    assert tables['nonIID_complete_scenes'] == ['Benign', 'F Flip'] and not tables['full_nonIID_coverage']
    assert not tables['primary_endpoint_selected'] and not tables['final_test'] and tables['new_inference'] == 0
    assert {(p['view'], tuple(p['seeds'])) for p in tables['panels']} == {(v, tuple(ss)) for v in ('raw', 'native', 'shared_calibration') for ss in (range(91001, 91011), range(91002, 91011), range(91005, 91011))}
    assert (SOURCE/'snapshot/cross_scene_seed_first.json').read_bytes() == (OLD/'snapshot/cross_scene_seed_first.json').read_bytes()
    # Source, result snapshot and independent review stay separate and byte exact.
    DEST.mkdir()
    for base, prefix, names in [(SOURCE, 'source', src), (SOURCE/'snapshot', 'snapshot', snap), (REVIEW, 'independent_review', rev)]:
        for name in names:
            target = DEST/prefix/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base/name, target)
            assert sha(target) == sha(base/name)
    shutil.copyfile(BINDING, DEST/'C10_BINDING.json')
    adopted = dict(status='ROOT_C70_SEVEN_SCENE_THREE_VIEW_TABLES_ADOPTED', checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        independent_review_path=(DEST/'independent_review/ROOT_ARITHMETIC_REVIEW.json').relative_to(ROOT).as_posix(), independent_review_sha256=sha(REVIEW/'ROOT_ARITHMETIC_REVIEW.json'),
        independent_review_seal_sha256=sha(REVIEW/'FINAL_FILES_SHA256.json'), prepared_source_seal_sha256=sha(SOURCE/'FILES_SHA256.json'), snapshot_seal_sha256=sha(SOURCE/'snapshot/FILES_SHA256.json'),
        C10_root_adoption_path=c10.relative_to(ROOT).as_posix(), C10_root_adoption_sha256=sha(c10), **{k:proof[k] for k in fields},
        records_sha256=sha(SOURCE/'snapshot/records.json'), tables_sha256=sha(SOURCE/'snapshot/tables.json'), display_sha256=sha(SOURCE/'snapshot/TABLES.md'), aggregate_sha256=sha(SOURCE/'snapshot/cross_scene_seed_first.json'),
        old120_records_preserved=True, old972_scalars_preserved=True, old486_cells_preserved=True, old162_IID_aggregate_bytes_exact=True, all_negative_results_retained=True,
        replay_devices={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ('Full', 'minus_C')},
        training_torch={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ('Full', 'minus_C')},
        seed_panels=[10, 9, 6], IID_complete_scenes=5, nonIID_complete_scenes=['Benign', 'F Flip'], full_nonIID_coverage=False,
        aggregate_scope='Original five IID scenes only; no imbalanced seven-scene mean', primary_endpoint='PENDING_AUTHOR',
        reviewer_authored_C10_evaluator_source=True, reviewer_authored_C70_table_builder=False, reviewer_authored_C70_numeric_verifier=False,
        auxiliary_fixture_target_correction_sha256=proof['auxiliary_fixture_target_correction_sha256'],
        new_CNN=0, new_training=0, new_Full_inference=0, test=False, other_C_scenes_complete=False, whole_rebuttal_complete=False, incorporated_into_full_rebuttal=False,
        canonical_table=(DEST/'snapshot/TABLES.md').relative_to(ROOT).as_posix())
    (DEST/'ROOT_VERIFICATION.json').write_text(json.dumps(adopted, indent=2)+'\n', encoding='utf8')
    sealed = {p.relative_to(DEST).as_posix():dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'ACTUAL_FILES_SHA256.json').write_text(json.dumps(dict(files=sealed), indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(root_sha256=sha(DEST/'ROOT_VERIFICATION.json'), actual_seal_sha256=sha(DEST/'ACTUAL_FILES_SHA256.json'), members=len(sealed), table=adopted['canonical_table'])))


if __name__ == '__main__':
    main()
