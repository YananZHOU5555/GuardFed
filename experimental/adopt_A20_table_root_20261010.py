"""Record root adoption of the actually reviewed two-scene A table."""
from pathlib import Path
import datetime, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
review_path = ROOT/'tmp/celeba_A20_table_root_review_v2_20261010/ROOT_ARITHMETIC_REVIEW.json'
assert sha(review_path) == 'e6cee89fd0406b006c008200d6fff5a4966967b3dea3271b7a5b5a682bfa8051'
proof = read(review_path)
assert proof['status'] == 'INDEPENDENT_A20_TWO_IID_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION'
assert (proof['paired_models'], proof['complete_scenes'], proof['preserved_records']) == (20, 2, 40)
assert (proof['mean_SD_scalars_recomputed'], proof['display_cells'], proof['metrics_from_group_counts'], proof['base_integer_confusion_counts_checked']) == (324, 162, 360, 960)
assert proof['max_abs_difference'] <= 1e-12 and not proof['test']
assert all(proof[k] for k in ('old24_normalized_JSON_bytes_and_order_exact', 'old_Benign162_scalars_exact', 'old_Benign81_display_cells_exact', 'original_Full20_records_exact', 'original_A20_saved_views_fits_and_bindings_exact'))
directory = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_two_scenes20_20261010'
assert Path(proof['table_directory']) == directory
assert sha(directory/'FILES_SHA256.json') == proof['delivery_seal_sha256'] == '3841a7498dad4a3dc6a0bf3b8262510082cd042671850b5f56320a079c012ef8'
for name, digest in proof['files_sha256'].items():
    assert sha(directory/name) == digest
failure = ROOT/'tmp/celeba_A20_table_root_review_20261010/REVIEW_FAILURE.json'
assert read(failure)['status'] == 'REVIEW_FAILED_NO_ADOPTION'
proof.update(status='ROOT_A20_TWO_COMPLETE_IID_SCENE_THREE_VIEW_TABLE_ADOPTED',
    adopted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), root_adoption=True,
    independent_review_path=review_path.relative_to(ROOT).as_posix(), independent_review_sha256=sha(review_path),
    preserved_first_review_failure_path=failure.relative_to(ROOT).as_posix(), preserved_first_review_failure_sha256=sha(failure),
    failure_scope='Display-label schema only; original failed review preserved, source v2 fixes exact scene labels and retains old numerical checks.',
    canonical_table=(directory/'TABLES.md').relative_to(ROOT).as_posix(),
    incorporated_into_full_rebuttal=False, manuscript_applied=False, whole_rebuttal_complete=False)
target = directory/'ROOT_VERIFICATION.json'
with target.open('x', encoding='utf8') as f:
    json.dump(proof, f, ensure_ascii=False, indent=2); f.write('\n')
print(json.dumps(dict(path=str(target), sha256=sha(target), status=proof['status'])))
