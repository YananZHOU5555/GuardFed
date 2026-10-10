"""Reuse the pinned C70 adopter with the actual C80 metadata; no inference."""
from pathlib import Path
import difflib, hashlib

ROOT=Path(__file__).resolve().parents[1]
parent=ROOT/'tmp/adopt_C70_table_root_20261010.py'
parent_sha='8cdc30c0b5e6b501da59016448f68d032eeb324b6b419ddca2fb09de4cb76834'
assert hashlib.sha256(parent.read_bytes()).hexdigest()==parent_sha
original=parent.read_text(encoding='utf8')
source=original
changes={
    'actual C70 table':'actual C80 table',
    "SOURCE = ROOT/'tmp/celeba_mechanism_C70_table_preparation_20261010'":"SOURCE = ROOT/'tmp/celeba_mechanism_C_eight_scenes_20261010'",
    "REVIEW = ROOT/'tmp/celeba_mechanism_C70_independent_review_20261010'":"REVIEW = ROOT/'tmp/celeba_mechanism_C_eight_scenes_review_20261010'",
    "BINDING = ROOT/'tmp/celeba_mechanism_C70_root_operations_20261010/C10_BINDING.json'":"BINDING = SOURCE/'C10_BINDING.json'",
    "OLD = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010'":"OLD = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_seven_scenes_20261010'",
    "DEST = OLD.parent/'three_view_C_seven_scenes_20261010'":"DEST = OLD.parent/'three_view_C_eight_scenes_20261010'",
    '758d49ac894a423f151f64cb0619f630bb6fdb697d9e8d16949b163ba09f336e':'54d6f53696587eeb0d29a195e73ca86dd57c341d46ef210a3500c5be4f8a7fbc',
    'cf81e43fd957b098ce8f72821b2530c4db6854cca8726c03d0f8e4c1a8084d2e':'b22d017bff4e13f1acc21a97205914b454d80b2c495b2b1798ede8bcc24185bc',
    'FINAL_FILES_SHA256.json':'FILES_SHA256.json',
    '9b1e8624c7ebb47d53b6fb20112ba5ee957fa3d99cf2392684e8a0d7d99b6ed2':'53e30067b019e302152e4bbf841de1be46ab50203c688bf869e8f0eaedb15e0c',
    '(12, 8, 10)':'(13, 8, 5)',
    'dcb2f2cbb0e19490f07f7f9c41556feb0995074f7067c42f637dc37ea8c7371d':'e4eefac8876a732b05828f6c1b4a8e4b96fe75b84bc11a692882e2cedbd4746b',
    'INDEPENDENT_C70_SEVEN_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION':'INDEPENDENT_C80_EIGHT_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION',
    '(140, 70, 7, 1134, 567, 1260, 3360, 630, 162)':'(160, 80, 8, 1296, 648, 1440, 3840, 720, 162)',
    'old120':'old140',
    'old972':'old1134',
    'old486':'old567',
    'other_three_nonIID_C_scenes_complete':'other_two_nonIID_C_scenes_complete',
    'a50061f0f70102babe08ccac3151da5c308c20a04e79762110bf42dcb60ed9e1':'6bb73000047cc2659e63ab54cb0733e00e739cbf5afaed78011452724b263f93',
    '7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8':'3fc1e49e927a971a577d648dd9a7ff44ec7ac81552ea250026349d4f2e06d615',
    '(160, 10, 170)':'(170, 10, 180)',
    "f'minus_C_non-IID_F Flip_seed{s}'":"f'minus_C_non-IID_FedSA_seed{s}'",
    "c['original160_unchanged']":"c['original170_unchanged']",
    "bindings['original_C60_root_sha256']":"bindings['original_C70_root_sha256']",
    'f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742':'abab0188adfa2d857316b23a282fd104c50dd282ca00ffed628b78b2accaea70',
    'records[:120]':'records[:140]',
    'len({r[\'id\'] for r in records}) == 140':'len({r[\'id\'] for r in records}) == 160',
    "('Benign', 'F Flip')":"('Benign', 'F Flip', 'FedSA')",
    "['Benign', 'F Flip']":"['Benign', 'F Flip', 'FedSA']",
    '(140, 70, 7)':'(160, 80, 8)',
    'ROOT_C70_SEVEN_SCENE_THREE_VIEW_TABLES_ADOPTED':'ROOT_C80_EIGHT_SCENE_THREE_VIEW_TABLES_ADOPTED',
    'no imbalanced seven-scene mean':'no imbalanced eight-scene mean',
    'reviewer_authored_C10_evaluator_source=True':'reviewer_authored_C10_evaluator_source=False',
    'reviewer_authored_C70_table_builder':'reviewer_authored_C80_table_builder',
    'reviewer_authored_C70_numeric_verifier':'reviewer_authored_C80_numeric_verifier',
    "auxiliary_fixture_target_correction_sha256=proof['auxiliary_fixture_target_correction_sha256'],":"auxiliary_review_failure_preserved=proof['auxiliary_review_failure_preserved'], reviewer_reviewed_after70_evaluator_source=True, reviewer_authored_reused_after60_parent_source=True, parent_adopter_sha256='"+parent_sha+"', actual_delivery_seal_sha256=proof['actual_delivery_seal_sha256'],",
}
for old,new in changes.items():
    assert old in source,old
    source=source.replace(old,new)
assert "assert records[:140] == old_records and len(records) == len({r['id'] for r in records}) == 160" in source
assert "(170, 10, 180)" in source and "proof['count_metric_max_abs_difference'] == 0" in source
assert "seed_first_scenes_per_seed'] == 5" in source

if __name__=='__main__':
    out=ROOT/'tmp/celeba_mechanism_C80_root_adoption_20261010'
    assert not out.exists()
    out.mkdir()
    (out/'SOURCE_DIFF.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile=str(parent),tofile='effective_C80_adopter')),encoding='utf8')
    namespace={'__file__':str(Path(__file__).resolve()),'__name__':'pinned_C80_adopter'}
    exec(compile(source,str(parent)+'[C80 metadata]','exec'),namespace)
    namespace['main']()
