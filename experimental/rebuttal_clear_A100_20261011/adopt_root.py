"""Root-only adoption of the saved clear A100 reader; no rebuild or editorial rerun."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json

S=Path(__file__).resolve().parent;R=S.parents[1]
D=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_clear_A100_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--actual-inputs-sha256',required=True)
args=parser.parse_args()
assert __debug__ and not D.exists()
assert sha(S/'ROOT_ADOPTION_INPUTS.json')==args.actual_inputs_sha256
actual=read(S/'ROOT_ADOPTION_INPUTS.json')
assert sha(S/'FILES_SHA256.json')==actual['delivery_seal_sha256']
sealed=read(S/'FILES_SHA256.json')['files']
assert len(sealed)==actual['delivery_members']
for name,pin in sealed.items():
    p=S/name
    assert p.resolve().is_relative_to(S.resolve()) and not p.is_symlink()
    assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],name
assert sha(S/'HANDOFF.json')==actual['handoff_sha256']
assert sha(S/'build_and_check.py')==actual['editorial_source_sha256']
# Reuse the sealed metadata identity guard only; never call edited() or check().
from build_and_check import inputs
binding=inputs()
facts=read(S/'FACT_BINDINGS.json')
assert facts['root_A100_sha256']=='adbc7f81c5952e85ee1a38ff4b5515d98a8706e4e60acd2e5a259425adcf1237'
assert sha(S/'ROOT_CHECK_COMMAND.json')==actual['root_command_sha256']
command=read(S/'ROOT_CHECK_COMMAND.json')
assert command['exit_code']==0 and command['source_sha256']==sha(S/'build_and_check.py')
assert '--check' in command['command'] and '--build' not in command['command']
assert command['stdout_sha256']==sha(S/'ROOT_CHECK.stdout')==actual['root_stdout_sha256']
assert command['stderr_sha256']==sha(S/'ROOT_CHECK.stderr')
check=read(S/'ROOT_CHECK.stdout');build=read(S/'BUILD_RECEIPT.json')
assert check['status']=='CLEAR_A100_EDITORIAL_CHECK_PASS'
assert check['draft_sha256']==sha(S/'rebuttal_clear_A100_20261011.md')==actual['draft_sha256']==build['draft_sha256']
assert check['source_sha256']==build['source_sha256']=='c2041b79aa3ceedc3fe62f0c76b18ffa7656d7e9fb9b994ef7571a4344d89e46'
assert check['original_comments']==check['response_sections']==24
assert check['quotation_lines_exact_and_ordered'] and check['headings_exact_and_ordered'] and check['forward_and_inverse_text_exact']
assert check['edit_groups']==build['edit_groups']==11 and check['local_link_occurrences_checked']==12
assert check['A100_table_root_sha256']==facts['root_A100_sha256'] and check['detailed_A100_binding']==binding
assert build['editorial_checker_executed'] is False
assert not any(check[k] for k in ('science_recomputed','final_test','submitted_manuscript_applied','whole_rebuttal_complete'))
copies={'rebuttal_clear_20261011.md':'rebuttal_clear_A100_20261011.md','EDITORIAL_CHECK.json':'ROOT_CHECK.stdout','ROOT_ACTUAL_CHECK.json':'ROOT_CHECK.stdout'}
D.mkdir(parents=True)
for dest,source in copies.items():
    (D/dest).write_bytes((S/source).read_bytes());assert sha(D/dest)==sha(S/source)
(D/'README.md').write_text('清晰版A100回复已由root采用为作者审阅稿：24条原话与顺序保留，A100十场景与Sp-DFA固定面板反转/汇总取舍已纳入。入口rebuttal_clear_20261011.md；P1–P6中的其余工作未完成，正文插入尚未应用，未进行最终test，不代表返修或投稿材料完成。\n',encoding='utf8')
proof=dict(status='ROOT_CLEAR_A100_COMPLETE24_EDITORIAL_CANDIDATE_ADOPTED_FOR_AUTHOR_REVIEW',utc=datetime.now(timezone.utc).isoformat(),root_adoption=True,source_directory=S.relative_to(R).as_posix(),source_seal_sha256=sha(S/'FILES_SHA256.json'),source_handoff_sha256=sha(S/'HANDOFF.json'),entry=(D/'rebuttal_clear_20261011.md').relative_to(R).as_posix(),draft_sha256=sha(D/'rebuttal_clear_20261011.md'),prior_clear_sha256=check['source_sha256'],A100_table_root_sha256=check['A100_table_root_sha256'],detailed_A100_root_sha256=binding['root_proof_sha256'],actual_root_command_path=(S/'ROOT_CHECK_COMMAND.json').relative_to(R).as_posix(),actual_root_command_sha256=sha(S/'ROOT_CHECK_COMMAND.json'),actual_root_command_exit=0,actual_root_check_sha256=sha(D/'ROOT_ACTUAL_CHECK.json'),root_check_JSON_matches_bound_draft_and_build=True,author_editorial_check_executions=0,root_editorial_check_executions=1,original_comments=24,response_sections=24,original_quotes_and_order_exact=True,reversible_edit_groups=11,local_link_occurrences_checked=12,source_words=check['source_words'],draft_words=check['draft_words'],A100_incorporated=True,A_complete_scenes=10,A_paired_models=100,partial_nonIID_SpDFA_accepted_seeds_excluded=0,negative_results_preserved=True,files_sha256={name:sha(D/name) for name in list(copies)+['README.md']},science_recomputed=False,new_fit=0,new_CNN=0,new_training=0,test=False,submitted_manuscript_applied=False,whole_rebuttal_complete=False)
with (D/'ROOT_REVIEW.json').open('x',encoding='utf8') as f:json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],root_sha256=sha(D/'ROOT_REVIEW.json'),entry=proof['entry'],draft_sha256=proof['draft_sha256'])))
