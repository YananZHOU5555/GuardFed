"""Root-only metadata adoption of the already verified A90 tables; no science execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

S = Path(__file__).resolve().parent
R = S.parents[1]
D = R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_nine_scenes90_20261011'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
assert __debug__ and not D.exists()
assert sha(S/'DELIVERY_FILES_SHA256.json') == '811065967b0515f1a8efee376091477378dcaa061387e5b6a9cb7b8adb14f8d5'
sealed = read(S/'DELIVERY_FILES_SHA256.json')['files']; assert len(sealed) == 45
for name, pin in sealed.items():
    p = S/name
    assert p.resolve().is_relative_to(S.resolve()) and not p.is_symlink()
    assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], name
assert sha(S/'DELIVERY_HANDOFF.json') == 'f3aa1a5470d65ea45764434cac2f8b08f643eff9b6114cf43ba55af2033be242'
delivery = read(S/'DELIVERY_HANDOFF.json'); h = delivery['summary']; b = delivery['input_root']
assert sha(R/b['adoption']) == b['adoption_sha256'] == '70689e63467d8866caa3beb06d2be5f911defd0a4d5f7f061ab005d20bd1c1bb'
assert sha(R/b['index']) == b['index_sha256'] == '7c46fcb20c15c377b1df17378f14b6282ee394bdcaacecd8d4f92be344777bef'
root295 = read(R/b['adoption']); index295 = read(R/b['index'])
assert root295['status'] == b['root_status'] and root295['cumulative_accepted'] == len(index295['all_ids']) == 295
assert root295['records_index_sha256'] == b['index_sha256'] and root295['original288_unchanged']
assert delivery['actual_root_verifier_exit0'] and not delivery['agent_repeated_root_verifier']
assert sha(S/'ROOT_VERIFY_COMMAND.json') == delivery['root_saved_verifier_command_sha256']
command = read(S/'ROOT_VERIFY_COMMAND.json')
assert command['exit_code'] == 0 and command['source_sha256'] == sha(S/'verify_saved.py')
assert command['stdout_sha256'] == sha(S/'ROOT_VERIFY.stdout.txt') == '2bfce95aae203672af7645b2bd60747f33f976514a9700ee5a268040a6dfc569'
assert command['stderr_sha256'] == sha(S/'ROOT_VERIFY.stderr.txt')
assert (S/'SAVED_VERIFICATION.json').read_bytes() == (S/'ROOT_VERIFY.stdout.txt').read_bytes()
numeric = read(S/'SAVED_VERIFICATION.json')
assert numeric == delivery['root_saved_verifier'] and numeric['status'] == 'PASS'
assert (numeric['all_scalars'], numeric['all_display_cells'], numeric['per_scene']['receipt_metrics_from_group_counts'], numeric['per_scene']['base_confusion_counts_structurally_checked']) == (1620,810,1620,4320)
assert numeric['old160_object_bytes_order_exact'] and numeric['old1296_scalars_and648_cells_exact'] and numeric['old_IID_seed_first_JSON_bytes_exact']
assert sha(S/'IID_SEED_FIRST.json') == 'ce6088b23dd0e2181e83393510075a919dc52719d4b654f70a29205d398b5baf'
assert sha(S/'ADDED10_FOCUSED_CHECK.json') == delivery['focused_added10_check_sha256']
assert read(S/'ADDED10_FOCUSED_CHECK.json')['status'] == 'ADDED20_RECORDS10_PAIRS_AND162_SCALARS_AUTHOR_FOCUSED_CHECK_PASS'
assert delivery['focused_added_statistics'] == 162 and delivery['focused_check_by_author']
records = read(S/'records.json')['records']
scenes = [('IID',a) for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')] + [('non-IID',a) for a in ('Benign','F Flip','FedSA','S-DFA')]
expected = {(v,d,a,s) for v in ('Full','minus_A') for d,a in scenes for s in range(91001,91011)}
assert len(records) == 180 and {(r['variant'],r['distribution'],r['attack'],r['seed']) for r in records} == expected
assert (h['records'],h['pairs'],h['complete_scenes'],h['complete_nonIID_scenes']) == (180,90,9,['Benign','F Flip','FedSA','S-DFA'])
assert h['training_torch'] == {'Full':{'2.11.0+cu128':88,'2.11.0+cu130':2},'minus_A':{'2.11.0+cu128':90}}
assert h['replay_devices'] == {'Full':{'cpu':5,'cuda:0':85},'minus_A':{'cpu':90}}
assert not h['mixed_distribution_aggregate'] and not h['independent_confirmation'] and not h['primary_endpoint_selected'] and not h['final_test']
assert h['new_fit'] == h['new_CNN'] == h['new_training'] == 0
assert h['accepted_partial_SpDFA_seeds_excluded'] == list(range(91001,91006))
selected = ['records.json','tables.json','IID_SEED_FIRST.json','TABLES.md','TABLES.tex','SOURCE_BINDINGS.json','SAVED_VERIFICATION.json','DIRECTIONS.json','ADDED10_FOCUSED_CHECK.json','ROOT_READY_REPORT.md','ROOT_READY_SUMMARY.json']
D.mkdir(parents=True)
for name in selected:
    (D/name).write_bytes((S/name).read_bytes()); assert sha(D/name) == sha(S/name)
status = 'ROOT_A90_NINE_COMPLETE_SCENE_THREE_VIEW_TABLE_ADOPTED'
summary = dict(h,status=status,root_adopted=True,native_shared_metrics_and_counts_exact=h['native_shared_metrics_counts_exact'])
(D/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(D/'REPORT.md').write_text('当前：A90九场景表已由root采用；下列交接报告的pending描述是采用前历史。当前范围以ROOT_VERIFICATION.json为准。\n\n'+(S/'ROOT_READY_REPORT.md').read_text('utf8'),encoding='utf8')
(D/'README_CURRENT.md').write_text('A90九场景三视图表已采用：180条记录、90对终轮模型，固定10/9/6种子面板。入口TABLES.md；验收ROOT_VERIFICATION.json。仅验证集；non-IID Sp-DFA部分5seed排除，其他变体与最终test未完成。\n',encoding='utf8')
selected += ['SUMMARY.json','REPORT.md','README_CURRENT.md']
proof = dict(status=status,utc=datetime.now(timezone.utc).isoformat(),root_adoption=True,canonical_table=(D/'TABLES.md').relative_to(R).as_posix(),source_candidate=S.relative_to(R).as_posix(),source_seal_sha256=sha(S/'DELIVERY_FILES_SHA256.json'),source_handoff_sha256=sha(S/'DELIVERY_HANDOFF.json'),source_acceptance_path=b['adoption'],source_acceptance_sha256=b['adoption_sha256'],source_index_sha256=b['index_sha256'],preserved_records=180,paired_models=90,complete_scenes=9,complete_IID_scenes=5,complete_nonIID_scenes=['Benign','F Flip','FedSA','S-DFA'],mean_SD_scalars_recomputed=1620,per_scene_scalars=1458,seed_first_scalars=162,display_cells=810,metrics_from_group_counts=1620,base_integer_confusion_counts_checked=4320,old160_record_bytes_order_exact=True,old1296_scalars_exact=True,old648_cells_exact=True,IID_seed_first_JSON_bytes_exact=True,actual_root_command=command['command'],actual_root_command_exit=0,actual_root_verifier_sha256=sha(S/'verify_saved.py'),actual_root_output_matches_saved=True,author_focused_check_sha256=sha(S/'ADDED10_FOCUSED_CHECK.json'),author_focused_check_is_independent_reviewer=False,replay_devices=h['replay_devices'],training_torch=h['training_torch'],files_sha256={n:sha(D/n) for n in selected},seed_panels=[10,9,6],aggregate_scope='Five IID scenes only; mean within seed first; four non-IID scenes reported separately',accepted_partial_SpDFA_seeds_excluded=list(range(91001,91006)),new_CNN=0,new_fit=0,new_training=0,test=False,primary_endpoint_selected=False,tex_compiled=False,incorporated_into_full_rebuttal=False,limitations=['Validation and selection-seed/history exposure retained.','Native/shared equality is not independent confirmation.','Actual mixed CPU/CUDA and cu128/cu130 provenance; no trajectory-equivalence claim.','Sp-DFA partial5 excluded; A100, other controls, full17 coverage, final endpoint/test and submitted-manuscript integration remain incomplete.','New S-DFA negative/mixed mechanism findings and original Windows baseline failures preserved.'])
with (D/'ROOT_VERIFICATION.json').open('x',encoding='utf8') as f:
    json.dump(proof,f,ensure_ascii=False,indent=2); f.write('\n')
print(json.dumps(dict(status=status,proof_sha256=sha(D/'ROOT_VERIFICATION.json'),table=proof['canonical_table'],scalars=1620,cells=810)))
