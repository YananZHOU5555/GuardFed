"""Record-only publication gate. No scientific computation or future-result discovery."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2]
T=Path('docs/server_deployment_20260923/training_20260923')
C=Path('tmp/celeba_mechanism_valid_C_after70_20261010')
D=C/'execution_candidate/backups/incremental_20261010T025719Z'
TABLE=T/'celeba_mechanism_v1/three_view_C_eight_scenes_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def check(c,d,roots,pinned):
    r=d['C10'];t=d['table'];s=d['state'];m=s['celeba_mechanism_v1']
    assert roots['C10']==(R/D/'ROOT_ADOPTION_REVIEW.json').resolve() and sha(roots['C10'])=='3fc1e49e927a971a577d648dd9a7ff44ec7ac81552ea250026349d4f2e06d615'
    assert r['status']=='ROOT_C_AFTER70_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
    ids=[f'minus_C_non-IID_FedSA_seed{seed}' for seed in range(91001,91011)]
    assert r['accepted_new_ids']==ids and (r['prior_three_view_models'],r['accepted_new'],r['cumulative_three_view_models'])==(170,10,180)
    assert (r['archive_members_verified'],r['content_members_verified'])==(103,102)
    assert r['original170_unchanged'] and r['all_native_differences_zero'] and r['negative_results_preserved'] and r['source_scope_complete']
    assert r['new_training']==r['new_Full_inference']==r['new_CNN_inference_for_root_review']==0 and not r['test_inference']
    for name,key in [('incremental_valid_three_views.tar.gz','archive_sha256'),('backup_receipt.json','backup_receipt_sha256'),('OFFSERVER_VERIFICATION.json','offserver_verification_sha256')]:assert sha(R/D/name)==r[key]
    off=read(R/D/'OFFSERVER_VERIFICATION.json')
    assert off['accepted_new_ids']==ids and (off['independent_metric_checks'],off['independent_confusion_count_checks'],off['prediction_rule_checks'])==(90,240,30)
    assert roots['table']==(R/TABLE/'ROOT_VERIFICATION.json').resolve()
    # Actual root schema is checked once adoption exists; no provisional table counts are accepted.
    assert t['status']=='ROOT_C80_EIGHT_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert (t['unique_records'],t['paired_models'],t['complete_scenes'],t['mean_SD_scalars_recomputed'],t['display_cells'])==(160,80,8,1296,648)
    assert t['C10_root_adoption_sha256']==sha(roots['C10']) and t['seed_panels']==[10,9,6]
    assert t['nonIID_complete_scenes']==['Benign','F Flip','FedSA'] and t['IID_complete_scenes']==5
    assert t['all_negative_results_retained'] and t['old162_IID_aggregate_bytes_exact']
    assert not t['full_nonIID_coverage'] and not t['whole_rebuttal_complete'] and not t['incorporated_into_full_rebuttal'] and not t['test']
    review=pinned(c['table_review']);assert sha(review)==t['independent_review_sha256']
    for row in c['artifact_bindings']:pinned(row)
    scope=read(R/C/'SCOPE.json');assert scope['selected_ids']==ids and len(scope['excluded_prior_ids'])==170
    assert set(m['three_view_accepted_ids'])==set(scope['excluded_prior_ids'])|set(ids)
    assert m['scientific_results_offserver_verified']==m['three_view_new_models_offserver_verified']==180 and not m['test_started']
    assert s['flgmm_fullcoverage_v2_20261009']['new_accepted']==18 and s['hybrid_screen32_20261009']['offserver_accepted70round_jobs']==23
    assert s['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted']==900
    assert s['latest_rebuttal_draft']['complete_C_scenes']==6 and s['latest_rebuttal_draft']['root_proof_sha256']=='b2e3958ed4000fe0365929c94408c411cd47c70e637c5151bb40603ad097098a'
    assert c['frozen_snapshot_destinations'][c['roots']['state']['path']]==(T/'TRAINING_STATE.json').as_posix()
