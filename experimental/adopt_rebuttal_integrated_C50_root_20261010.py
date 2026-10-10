"""Adopt the complete source-checked C50 author-review response, without applying the manuscript."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/rebuttal_integrated_C50_prepared_20261010'
REVIEW=ROOT/'tmp/rebuttal_integrated_C50_root_review_20261010'
TARGET=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert not sys.flags.optimize and not TARGET.exists() and not REVIEW.exists()
seal=SOURCE/'FILES_SHA256.json'
assert sha(seal)=='427fb0015ef84793d4ced9c49d576563fbfc8c5b0314459a4a7a568fcbe57389'
files=read(seal)['files'];assert len(files)==13
for name,pin in files.items():
    assert Path(name).name==name and sha(SOURCE/name)==pin['sha256'] and (SOURCE/name).stat().st_size==pin['bytes']
handoff=read(SOURCE/'HANDOFF.json')
assert handoff['status']=='SEALED_COMPLETE_24_COMMENT_C50_AUTHOR_REVIEW_NOT_APPLIED'
assert handoff['accepted_C50_update_root_sha256']=='cb57e51f7ba426583f7975a78078c79b59149e475e63901ba4079c218e1354ff'
assert handoff['accepted_C50_table_root_sha256']=='811ad551c1398f6e57681f59046c316b05c609ef30d4897158460b8121b1a28d'
assert handoff['negative_results_retained'] and handoff['all_unaffected_passages_and_numbers_exact'] and handoff['prior_documents_recoverable_byte_exact']
assert not handoff['manuscript_applied'] and not handoff['whole_rebuttal_complete'] and handoff['new_inference']==handoff['new_training']==0
REVIEW.mkdir()
result=subprocess.run([sys.executable,'-B',str(SOURCE/'check_sources.py'),'--report',str(REVIEW/'CHECK_RESULTS.json')],cwd=ROOT,capture_output=True)
(REVIEW/'STDOUT.json').write_bytes(result.stdout);(REVIEW/'STDERR.log').write_bytes(result.stderr)
assert result.returncode==0, 'Preserve source-check failure; do not adopt'
checked=read(REVIEW/'CHECK_RESULTS.json')
assert checked['status']=='PASS_COMPLETE_24_COMMENT_C50_AUTHOR_REVIEW_TEXT_AND_SOURCES_ONLY'
expected=dict(full_author_review_documents=2,original_comments_verbatim=24,prior_documents_reverse_diff_exact=2,
    changed_spans=11,new_and_retained_C_display_pointer_bindings=19,C_scalar_pointer_checks=38,
    C_scope_fact_pointer_checks=19,direction_checks=24,links_checked=44,C50_unique_records=100,
    C50_matched_pairs=50,C50_complete_IID_scenes=5,C_nonIID_scenes_incomplete=5,other_image_controls_incomplete=6)
assert all(checked[k]==v for k,v in expected.items())
assert checked['original_comments_order_exact'] and checked['old_COMPAS_U100_900_calibration_numbers_preserved']
assert checked['Sp_tradeoff_and_preselected_9_to_6_reversal_retained'] and checked['FedSA_raw9_all_three_mean_counterexample_retained']
assert checked['P1_P6_still_pending'] and checked['AUTHOR_REVIEW_DO_NOT_SUBMIT'] and not checked['manuscript_applied']
assert sha(seal)=='427fb0015ef84793d4ced9c49d576563fbfc8c5b0314459a4a7a568fcbe57389'
TARGET.mkdir()
for name in [*files,'FILES_SHA256.json']:
    shutil.copyfile(SOURCE/name,TARGET/name);assert sha(SOURCE/name)==sha(TARGET/name)
shutil.copyfile(REVIEW/'CHECK_RESULTS.json',TARGET/'ROOT_SOURCE_CHECK.json')
proof=dict(status='ROOT_C50_COMPLETE_AUTHOR_REVIEW_TEXT_REVERSIBLE_DELTA_AND_SOURCE_POINTERS_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal_sha256=sha(seal),
    source_checker_sha256=sha(SOURCE/'check_sources.py'),root_source_check_sha256=sha(TARGET/'ROOT_SOURCE_CHECK.json'),
    rebuttal_sha256=sha(TARGET/'rebuttal_integrated_20261009.md'),insertions_sha256=sha(TARGET/'manuscript_insertions_integrated_20261009.md'),
    root_C50_table_sha256=handoff['accepted_C50_table_root_sha256'],root_C50_short_update_sha256=handoff['accepted_C50_update_root_sha256'],
    **expected,complete_C_scenes=5,all_unaffected_text_and_numbers_exact=True,old_complete_documents_preserved=True,
    all_negative_results_retained=True,author_review_only=True,manuscript_applied=False,final_test=False,
    primary_endpoint='PENDING_AUTHOR',whole_rebuttal_complete=False,new_statistics=False,new_inference=0,new_training=0)
path=TARGET/'ROOT_REVIEW.json'
with path.open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],root_path=path.relative_to(ROOT).as_posix(),root_sha256=sha(path),counts=expected)))
