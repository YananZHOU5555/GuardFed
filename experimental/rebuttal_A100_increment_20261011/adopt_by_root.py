"""Adopt the root-checked A100 editorial increment; no scientific reruns."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys
R=Path(__file__).resolve().parents[2]
S=R/'tmp/rebuttal_A100_increment_20261011'
D=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A100_reader_20261011'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())

assert not sys.flags.optimize and not D.exists()
p=argparse.ArgumentParser();p.add_argument('--delivery-seal-sha256',required=True);p.add_argument('--root-command-sha256',required=True);p.add_argument('--root-stdout-sha256',required=True);args=p.parse_args()
assert H(S/'FILES_SHA256.json')==args.delivery_seal_sha256
sealed=read(S/'FILES_SHA256.json')['files'];assert set(sealed)==set(read(S/'DELIVERY_MEMBERS.json')['files'])
for name,pin in sealed.items():
    p=S/name
    assert p.resolve().is_relative_to(S.resolve()) and not p.is_symlink()
    assert H(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],name
assert H(S/'ROOT_CHECK_COMMAND.json')==args.root_command_sha256
assert H(S/'ROOT_CHECK.stdout')==args.root_stdout_sha256
assert read(S/'ROOT_CHECK_COMMAND.json')['exit_code']==0 and not (S/'ROOT_CHECK.stderr').read_bytes()
actual=read(S/'ROOT_CHECK.stdout');assert actual==read(S/'SELF_CHECK.json')
assert read(S/'ROOT_CHECK_COMMAND.json')['command'][-2:]==['tmp/rebuttal_A100_increment_20261011/build_and_check.py','--check'] or [str(x).replace('\\','/') for x in read(S/'ROOT_CHECK_COMMAND.json')['command'][-2:]]==[str(S/'build_and_check.py').replace('\\','/'),'--check']
assert actual['checker_sha256']==H(S/'build_and_check.py') and actual['manifest_sha256']==H(S/'EDIT_MANIFEST.json') and actual['fact_bindings_sha256']==H(S/'FACT_BINDINGS.json')
assert actual['status']=='PASS_A100_DETAILED_READER_INTEGRATION_AUTHOR_REVIEW_ONLY'
assert actual['documents'][0]['original_comment_count']==24
assert sum(d['all_old_number_string_occurrences_preserved'] for d in actual['documents'])==3012
assert sum(d['all_old_links_preserved'] for d in actual['documents'])==128
assert sum(d['edit_operations'] for d in actual['documents'])==13
assert actual['new_mean_sd_pairs_bound_to_JSON_pointers']==39
assert actual['fixed_10_9_6_direction_panels_bound']==9
assert actual['explicit_SpDFA_direction_panels_sign_guarded']==9
assert all(d['forward_and_inverse_bytes_exact'] and d['all_old_scientific_tables_exact'] for d in actual['documents'])
assert not any(actual[k] for k in ['source_statistics_recomputed','final_test','primary_endpoint_selected','manuscript_applied'])
A=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/ROOT_VERIFICATION.json'
assert H(A)==actual['A100_root_sha256']=='adbc7f81c5952e85ee1a38ff4b5515d98a8706e4e60acd2e5a259425adcf1237'
mapping={'rebuttal_integrated_A100_reader_20261011.md':'rebuttal_integrated_20261011.md',
         'manuscript_insertions_integrated_A100_reader_20261011.md':'manuscript_insertions_integrated_20261011.md'}
assert {d['candidate'] for d in actual['documents']}==set(mapping)
for d in actual['documents']:assert H(S/d['candidate'])==d['sha256']
D.mkdir(parents=True)
for old,new in mapping.items():
    (D/new).write_bytes((S/old).read_bytes());assert H(D/new)==H(S/old)
(D/'ROOT_ACTUAL_CHECK.json').write_text(json.dumps(actual,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
proof=dict(status='ROOT_COMPLETE24_A100_READER_REBUTTAL_CANDIDATES_ADOPTED_FOR_AUTHOR_REVIEW',utc=datetime.now(timezone.utc).isoformat(),
    source_seal_sha256=H(S/'FILES_SHA256.json'),source_directory=S.relative_to(R).as_posix(),
    prior_root_review_sha256=actual['root_reader_sha256'],A100_table_root_sha256=H(A),
    documents_sha256={n:H(D/n) for n in mapping.values()},entry=(D/'rebuttal_integrated_20261011.md').relative_to(R).as_posix(),
    manuscript_candidate=(D/'manuscript_insertions_integrated_20261011.md').relative_to(R).as_posix(),
    actual_root_check_path=(D/'ROOT_ACTUAL_CHECK.json').relative_to(R).as_posix(),actual_root_check_sha256=H(D/'ROOT_ACTUAL_CHECK.json'),
    root_command_path=(S/'ROOT_CHECK_COMMAND.json').relative_to(R).as_posix(),root_command_sha256=H(S/'ROOT_CHECK_COMMAND.json'),
    original_comments=24,number_strings_preserved=3012,links_preserved=128,whole_scientific_tables_preserved=True,
    reversible_edits=13,paired_mean_SD_values_bound_to_JSON=39,new_mean_SD_scalar_pointers=78,fixed_direction_panels=9,interpretation_sign_bindings=9,
    A100_incorporated=True,A_complete_scenes=10,A_paired_models=100,A_IID_complete_scenes=5,A_nonIID_complete_scenes=5,
    A_nonIID_scenes=['Benign','F Flip','FedSA','S-DFA','Sp-DFA'],A_remaining_nonIID_scenes=[],partial_SpDFA_seeds_excluded=[],
    prior_A90_values_tables_and_IID_seed_first_bytes_preserved=True,
    root_editorial_review=True,author_review_only=True,manuscript_applied=False,whole_rebuttal_complete=False,
    independent_review_scope='Original editorial checker executed by root: reversible edits,78 JSON scalar pointers and nine exact expected-sign panels; no scientific-statistics rerun. Human semantic review is separate from this adopter.',
    primary_endpoint_selected=False,final_test=False,new_scientific_results=0,
    limitations='A100 ten-scene A comparison only; five other image controls and seven baseline coverages remain incomplete. Sp-DFA native/shared three-direction six-seed reversal and raw accuracy-disparity tradeoff retained, with seed-first nonIID/balanced aggregates. All A90 text/numbers/tables and prior counterexamples preserved. P1/P3-P6, endpoint/final evaluation and manuscript remain pending. No universal necessity, significance or final-test claim.')
(D/'ROOT_REVIEW.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=proof['status'],proof_sha256=H(D/'ROOT_REVIEW.json'),entry=proof['entry'])))
