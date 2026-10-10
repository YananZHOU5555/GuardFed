"""Root reread/recheck and adopt the complete C100 author-review drafts."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, shutil, sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/rebuttal_integrated_C100_prepared_20261010'
REVIEW=ROOT/'tmp/rebuttal_C100_rootreview_20261010'
DEST=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C100_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())

def main():
    assert not sys.flags.optimize and not REVIEW.exists() and not DEST.exists()
    assert sha(SOURCE/'FILES_SHA256.json')=='cc80624a16ea2f3045b6629baf05f30d72d405437dc522114356202fc805b315'
    seal=read(SOURCE/'FILES_SHA256.json')['files'];assert len(seal)==11
    for name,info in seal.items():
        p=SOURCE/name;assert p.resolve().is_relative_to(SOURCE.resolve()) and p.is_file()
        assert sha(p)==info['sha256'] and p.stat().st_size==info['bytes']
    assert sha(SOURCE/'CHECK_RESULTS.json')=='58694f5c6b64e35641171daae35ef498ec59d1bf00db0e9f1a4be5fd742c309c'
    handoff=read(SOURCE/'HANDOFF.json')
    table=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_full100_20261010/ROOT_VERIFICATION.json'
    assert sha(table)==handoff['root_C100_sha256']=='0bed6d372c63a60a978f8faa1353ec3257dcba66cc236b01906abd7097a0d0e7'
    REVIEW.mkdir()
    for name in seal:
        if name!='CHECK_RESULTS.json':shutil.copyfile(SOURCE/name,REVIEW/name)
    # Execute the unchanged checker once against fresh review output; do not rerun producer.
    spec=importlib.util.spec_from_file_location('C100_writing_original_checker',SOURCE/'check_sources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.D=REVIEW;module.R=ROOT;module.main()
    check=read(REVIEW/'CHECK_RESULTS.json');assert check==read(SOURCE/'CHECK_RESULTS.json')
    assert check['original_comments_verbatim_and_order']==24 and check['reverse_reconstruction_exact']==2
    assert check['new_scalar_pointer_checks']==90 and check['direction_checks']==54 and check['links_checked']==60
    assert check['remaining_other_image_controls']==6 and check['P1_P6_pending']
    assert not check['test'] and not check['manuscript_applied'] and not check['whole_rebuttal_complete']
    DEST.mkdir()
    for name in list(seal)+['FILES_SHA256.json']:
        shutil.copyfile(SOURCE/name,DEST/name);assert sha(DEST/name)==sha(SOURCE/name)
    proof=dict(status='ROOT_COMPLETE_C100_REBUTTAL_AND_INSERTION_CANDIDATES_ADOPTED_FOR_AUTHOR_REVIEW',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_seal_sha256=sha(SOURCE/'FILES_SHA256.json'),source_handoff_sha256=sha(SOURCE/'HANDOFF.json'),
        actual_root_recheck_path=(REVIEW/'CHECK_RESULTS.json').relative_to(ROOT).as_posix(),actual_root_recheck_sha256=sha(REVIEW/'CHECK_RESULTS.json'),
        root_table_sha256=sha(table),documents_sha256=handoff['documents_sha256'],
        original_comments=24,reverse_exact_documents=2,changed_spans=14,scalar_pointer_checks=90,mean_SD_cells=45,direction_checks=54,links_checked=60,
        complete_C_scenes=10,U_complete_scenes=10,remaining_image_controls=6,whole_rebuttal_complete=False,
        primary_endpoint='PENDING_AUTHOR',author_review_only=True,manuscript_applied=False,final_test=False,new_inference=0,new_training=0,
        all_negative_results_retained=True,old_original_comments_and_numeric_displays_preserved=True,
        limitations=['P1–P6 remain open; full17-method cohort and six mechanism controls unfinished.','Matching submitted-version LaTeX is not recovered; candidate text is not applied manuscript.','Validation selection, prior official-test exposure, adaptation and mixed device/runtime provenance remain disclosed.'])
    (DEST/'ROOT_REVIEW.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=proof['status'],root_sha256=sha(DEST/'ROOT_REVIEW.json'),rebuttal=str(DEST/'rebuttal_integrated_20261009.md'))))

if __name__=='__main__':main()
