"""Adopt bounded A50 integration into both complete author-review drafts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1]
S=R/'tmp/rebuttal_A50_reader_integration_20261010'
D=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A50_reader_20261010'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())

def main():
    assert not sys.flags.optimize and not D.exists()
    assert H(S/'FILES_SHA256.json')=='e28fa1a5c51a269d7fad96af5183aab0125d153b9da4b4e2b6dffef62cae0ee3'
    assert H(S/'HANDOFF.json')=='c8467fc79478372b7ccfdb086f5275416017222047e7a3237cf10a1b8f2d06a6'
    sealed=read(S/'FILES_SHA256.json')['files'];assert len(sealed)==9
    for n,pin in sealed.items():
        p=S/n;assert p.resolve().is_relative_to(S.resolve()) and not p.is_symlink()
        assert H(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],n
    spec=importlib.util.spec_from_file_location('root_A50_editorial_actual',S/'build_and_check.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    actual=checker.check();assert actual==read(S/'SELF_CHECK.json')
    assert actual['status']=='PASS_A50_READER_INTEGRATION_FOR_AUTHOR_REVIEW'
    assert actual['documents'][0]['original_comment_count']==24
    assert sum(d['all_old_number_string_occurrences_preserved'] for d in actual['documents'])==2292
    assert sum(d['all_old_links_preserved'] for d in actual['documents'])==88
    assert sum(d['edit_operations'] for d in actual['documents'])==16
    assert actual['new_mean_sd_pairs_bound_to_adopted_JSON_pointers']==12
    assert all(d['forward_and_inverse_bytes_exact'] and d['all_old_scientific_tables_exact'] for d in actual['documents'])
    assert all(v['positive']==4 and v['negative']==6 for v in actual['SpDFA_accuracy_seed_signs'].values())
    assert not actual['final_test'] and not actual['primary_endpoint_selected'] and not actual['manuscript_applied']
    D.mkdir(parents=True)
    mapping={'rebuttal_integrated_A50_reader_20261010.md':'rebuttal_integrated_20261010.md','manuscript_insertions_integrated_A50_reader_20261010.md':'manuscript_insertions_integrated_20261010.md'}
    for old,new in mapping.items():
        (D/new).write_bytes((S/old).read_bytes());assert H(D/new)==H(S/old)
    (D/'ROOT_ACTUAL_CHECK.json').write_text(json.dumps(actual,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    proof=dict(status='ROOT_COMPLETE24_A50_READER_REBUTTAL_CANDIDATES_ADOPTED_FOR_AUTHOR_REVIEW',utc=datetime.now(timezone.utc).isoformat(),
        source_seal_sha256=H(S/'FILES_SHA256.json'),source_directory=S.relative_to(R).as_posix(),
        prior_root_review_sha256='0b46e6ba9a8f30f0b2523cbe7c6947ef47a402e0af55a3ce578e18036c94a511',
        A50_table_root_sha256='2ab91a854691624c79124a5eccbb43efe30541474cb7ef542b76db300554f7c4',
        documents_sha256={n:H(D/n) for n in mapping.values()},entry=(D/'rebuttal_integrated_20261010.md').relative_to(R).as_posix(),
        manuscript_candidate=(D/'manuscript_insertions_integrated_20261010.md').relative_to(R).as_posix(),
        actual_root_check_path=(D/'ROOT_ACTUAL_CHECK.json').relative_to(R).as_posix(),actual_root_check_sha256=H(D/'ROOT_ACTUAL_CHECK.json'),
        actual_root_check='Original A50 check() and its original writing checker run, exact SELF_CHECK reproduction',
        original_comments=24,number_strings_preserved=2292,links_preserved=88,whole_scientific_tables_preserved=11,
        reversible_edits=16,paired_mean_SD_values_bound_to_JSON=12,only_pending_row_P2_updated=True,
        A50_incorporated=True,A_complete_scenes=5,A_paired_models=50,A_nonIID_complete_scenes=0,
        partial_nonIID_excluded='minus_A_non-IID_Benign_seed91001',C_complete_scenes=10,LoGoFair_native_records=100,native_comparison_records=1000,
        root_editorial_review=True,author_review_only=True,manuscript_applied=False,whole_rebuttal_complete=False,
        primary_endpoint_selected=False,final_test=False,new_scientific_results=0,
        limitations='Five IID A scenes only; all five non-IID A scenes and five other image controls incomplete. Native/shared coincide, not independent confirmation. Negative and sensitivity-panel reversals retained. Submitted source and final endpoint/test boundaries unresolved; preserved full drafts are not submission-length compressed.')
    (D/'ROOT_REVIEW.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=proof['status'],proof_sha256=H(D/'ROOT_REVIEW.json'),entry=proof['entry'],original_comments=24,A_paired_models=50)))

if __name__=='__main__':main()
