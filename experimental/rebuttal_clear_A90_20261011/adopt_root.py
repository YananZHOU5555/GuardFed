"""Root-only adoption of the saved, independently checked clear A90 draft; no rebuild."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

S = Path(__file__).resolve().parent
R = S.parents[1]
D = R/'docs/server_deployment_20260923/revision_20260923/rebuttal_clear_A90_20261011'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
assert __debug__ and not D.exists()
assert sha(S/'FILES_SHA256.json') == 'd0ed58e816e31380a7f46cf2c099784b0a5971da13901c3d0a7a4b509936f48d'
sealed = read(S/'FILES_SHA256.json')['files']; assert len(sealed) == 17
for name, pin in sealed.items():
    p = S/name
    assert p.resolve().is_relative_to(S.resolve()) and not p.is_symlink()
    assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], name
assert sha(S/'HANDOFF.json') == 'f335722650fd02b38a129871b106100e3eb925784ce3654ea0e971e765d12b98'
for name, pin in read(S/'SOURCE_PINS.json').items():
    p = R/name
    assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], name
binding = read(S/'DETAIL_BINDING.json')
for key in ['root_proof','rebuttal','insertions']:
    assert sha(R/binding[key]) == binding[key+'_sha256']
assert binding['root_adopted'] and read(R/binding['root_proof'])['status'] == binding['root_status']
assert sha(S/'ROOT_CHECK_COMMAND.json') == 'd22a57a8eba7e9e7f6df27d326861e26940ba3c095dfaf5fb0dff3a50a11b457'
command = read(S/'ROOT_CHECK_COMMAND.json')
assert command['exit_code'] == 0 and command['source_sha256'] == sha(S/'check_candidate.py')
assert command['stdout_sha256'] == sha(S/'ROOT_CHECK.stdout') == '4b4144e674eb1e6b77e69d2ed7856b22559ee2af0cebc0ef866237e153994cb4'
assert command['stderr_sha256'] == sha(S/'ROOT_CHECK.stderr')
check = read(S/'ROOT_CHECK.stdout')
assert check == read(S/'SELF_CHECK.json')  # JSON equality; original platform newline bytes retained.
assert check['status'] == 'CLEAR_A90_EDITORIAL_CHECK_PASS'
assert check['draft_sha256'] == sha(S/'rebuttal_clear_A90_20261011.md') == 'c2041b79aa3ceedc3fe62f0c76b18ffa7656d7e9fb9b994ef7571a4344d89e46'
assert check['original_comments'] == check['response_sections'] == 24
assert check['quotation_lines_exact_and_ordered'] and check['headings_exact_and_ordered'] and check['forward_and_inverse_text_exact']
assert check['edit_groups'] == 10 and check['local_link_occurrences_checked'] == 12
assert not check['science_recomputed'] and not check['final_test'] and not check['submitted_manuscript_applied'] and not check['whole_rebuttal_complete']
copies = {'rebuttal_clear_20261011.md':'rebuttal_clear_A90_20261011.md','EDITORIAL_CHECK.json':'SELF_CHECK.json','ROOT_ACTUAL_CHECK.json':'ROOT_CHECK.stdout'}
D.mkdir(parents=True)
for dest, source in copies.items():
    (D/dest).write_bytes((S/source).read_bytes()); assert sha(D/dest) == sha(S/source)
(D/'README.md').write_text('清晰版A90回复已由root采用为作者审阅稿：24条原话与顺序保留，A90九场景及S-DFA实际取舍已更新。入口rebuttal_clear_20261011.md；仍有P1–P6待完成，正文插入尚未应用，未进行最终test，不代表返修或投稿材料已完成。\n',encoding='utf8')
proof = dict(status='ROOT_CLEAR_A90_COMPLETE24_EDITORIAL_CANDIDATE_ADOPTED_FOR_AUTHOR_REVIEW',utc=datetime.now(timezone.utc).isoformat(),root_adoption=True,source_directory=S.relative_to(R).as_posix(),source_seal_sha256=sha(S/'FILES_SHA256.json'),source_handoff_sha256=sha(S/'HANDOFF.json'),entry=(D/'rebuttal_clear_20261011.md').relative_to(R).as_posix(),draft_sha256=sha(D/'rebuttal_clear_20261011.md'),prior_clear_sha256=check['source_sha256'],A90_table_root_sha256=check['A90_table_root_sha256'],detailed_A90_root_sha256=binding['root_proof_sha256'],actual_root_command_path=(S/'ROOT_CHECK_COMMAND.json').relative_to(R).as_posix(),actual_root_command_sha256=sha(S/'ROOT_CHECK_COMMAND.json'),actual_root_command_exit=0,actual_root_check_sha256=sha(D/'ROOT_ACTUAL_CHECK.json'),root_result_JSON_equals_selfcheck=True,original_comments=24,response_sections=24,original_quotes_and_order_exact=True,reversible_edit_groups=10,local_link_occurrences_checked=12,source_words=6918,draft_words=7016,A90_incorporated=True,A_complete_scenes=9,A_paired_models=90,partial_nonIID_SpDFA_accepted_seeds_excluded=5,negative_results_preserved=True,files_sha256={name:sha(D/name) for name in list(copies)+['README.md']},science_recomputed=False,new_fit=0,new_CNN=0,new_training=0,test=False,submitted_manuscript_applied=False,whole_rebuttal_complete=False)
with (D/'ROOT_REVIEW.json').open('x',encoding='utf8') as f:
    json.dump(proof,f,ensure_ascii=False,indent=2); f.write('\n')
print(json.dumps(dict(status=proof['status'],root_sha256=sha(D/'ROOT_REVIEW.json'),entry=proof['entry'],draft_sha256=proof['draft_sha256'])))
