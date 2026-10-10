"""Root-only metadata adoption of actually verified A100 tables; no science execution."""
import argparse
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

parser = argparse.ArgumentParser(description=__doc__)
for key in ('delivery-seal-sha256','delivery-handoff-sha256','root-command-sha256','saved-verification-sha256'):
    parser.add_argument('--'+key, required=True)
parser.add_argument('--delivery-members', type=int, required=True)
args = parser.parse_args()
assert args.delivery_members > 0
assert all(len(getattr(args,k))==64 for k in ('delivery_seal_sha256','delivery_handoff_sha256','root_command_sha256','saved_verification_sha256'))
S = Path(__file__).resolve().parent.parent
R = S.parents[1]
D = R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
assert __debug__ and not D.exists()
assert sha(S/'DELIVERY_FILES_SHA256.json') == args.delivery_seal_sha256
sealed = read(S/'DELIVERY_FILES_SHA256.json')['files']; assert len(sealed) == args.delivery_members
for name, pin in sealed.items():
    p = S/name
    assert p.resolve().is_relative_to(S.resolve()) and not p.is_symlink()
    assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], name
assert sha(S/'DELIVERY_HANDOFF.json') == args.delivery_handoff_sha256

delivery = read(S/'DELIVERY_HANDOFF.json'); h = delivery['summary']; b = delivery['input_root']
assert b == read(S/'ROOT_BINDING.json')
assert b['status']=='ACTUAL_NATIVE300_AND_REPLAY300_BOUND_FOR_A100_TABLE' and b['root_adopted'] is True
assert sha(R/b['adoption']) == b['adoption_sha256'] == 'e44122480c85e08461687bd58332aa9f30cc0ed807157966c3fd1769deb74a1e'
assert sha(R/b['index']) == b['index_sha256'] == 'dfa3b4745d5cd2a681e0a5e9a38fdff215a7ab9bb6e5994cfb62e2786cf396c8'
assert sha(R/b['native_root']) == b['native_root_sha256'] == 'b52484c71ea0e602e13f38f6505bc446d2ca2a80d95a859cf0236b078e087d1d'
root300=read(R/b['adoption']); index300=read(R/b['index']); native300=read(R/b['native_root'])
exact5=[f'minus_A_non-IID_Sp-DFA_seed{s}' for s in range(91006,91011)]
assert root300['status']==b['root_status']=='ROOT_AFTER295_EXACT5_SAVED_ARRAYS_REPLAY300_ADOPTED'
assert (root300['prior_accepted'],root300['new_accepted'],root300['cumulative_accepted'])==(295,5,300) and root300['original295_unchanged']
assert root300['accepted_new_ids']==index300['new_ids']==b['expected_new_ids']==exact5
assert root300['records_index_sha256']==b['index_sha256'] and len(index300['all_ids'])==len(set(index300['all_ids']))==300
assert index300['prior_adoption_sha256']=='70689e63467d8866caa3beb06d2be5f911defd0a4d5f7f061ab005d20bd1c1bb'
assert index300['prior_index_sha256']=='7c46fcb20c15c377b1df17378f14b6282ee394bdcaacecd8d4f92be344777bef'
assert sha(R/index300['prior_index_path'])==index300['prior_index_sha256']
assert index300['all_ids']==read(R/index300['prior_index_path'])['all_ids']+exact5
assert root300['native_root_sha256']==b['native_root_sha256'] and (R/root300['native_root_path']).resolve()==(R/b['native_root']).resolve()
assert native300['root_adopted'] is True and native300['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS'
assert native300['total_new_strict_and_offserver']==300 and native300['inspection_sha256']==root300['native_inspection_sha256']==index300['native_inspection_sha256']
assert root300['native_max_abs_difference']==0 and root300['Full_inference']==root300['new_CNN']==root300['new_training']==0
assert root300.get('new_fit',0)==0 and root300['test'] is native300['test'] is False

assert delivery['actual_root_verifier_exit0'] and not delivery['agent_repeated_root_verifier']
assert sha(S/'ROOT_VERIFY_COMMAND.json') == delivery['root_saved_verifier_command_sha256'] == args.root_command_sha256
command = read(S/'ROOT_VERIFY_COMMAND.json')
assert command['exit_code'] == 0 and command['source_sha256'] == sha(S/'verify_saved.py')
assert len(command['command'])==3 and Path(command['command'][0]).name.lower() in ('python','python.exe') and command['command'][1]=='-B'
verifier_path=Path(command['command'][2])
assert (verifier_path if verifier_path.is_absolute() else R/verifier_path).resolve()==(S/'verify_saved.py').resolve()
assert command['stdout_sha256'] == sha(S/'ROOT_VERIFY.stdout.txt') == args.saved_verification_sha256
assert command['stderr_sha256'] == sha(S/'ROOT_VERIFY.stderr.txt')
assert (S/'SAVED_VERIFICATION.json').read_bytes() == (S/'ROOT_VERIFY.stdout.txt').read_bytes()
numeric = read(S/'SAVED_VERIFICATION.json')
assert numeric == delivery['root_saved_verifier'] and numeric['status'] == 'PASS'
assert (numeric['all_scalars'], numeric['all_display_cells'], numeric['per_scene']['receipt_metrics_from_group_counts'], numeric['per_scene']['base_confusion_counts_structurally_checked']) == (2106,1053,1800,4800)
assert numeric['per_scene']['mean_sd_scalars']==1620
for key,scene_count in [('seed_first',5),('nonIID_seed_first',5),('balanced_seed_first',10)]:
    assert numeric[key]['mean_sd_scalars']==162 and numeric[key]['n_is_seed_count'] is True and numeric[key]['scene_count_per_seed']==scene_count
for key in ('per_scene','seed_first','nonIID_seed_first','balanced_seed_first'):
    assert numeric[key]['max_abs_difference']<=1e-12
assert numeric['old180_object_bytes_order_exact'] and numeric['old1458_scalars_and729_cells_exact'] and numeric['old_IID_seed_first_JSON_bytes_exact']
assert sha(S/'IID_SEED_FIRST.json') == 'ce6088b23dd0e2181e83393510075a919dc52719d4b654f70a29205d398b5baf'
assert sha(S/'FOCUSED_CHECKS.json') == delivery['focused_check_sha256']
focused=read(S/'FOCUSED_CHECKS.json')
assert focused['status']=='PASS' and not focused['independent_reviewer']
assert focused['per_scene']==read(S/'checks.json') and focused['seed_first']==numeric['seed_first']
assert focused['nonIID_seed_first']==numeric['nonIID_seed_first'] and focused['balanced_seed_first']==numeric['balanced_seed_first']
assert focused['total_statistic_scalars']==numeric['all_scalars']==2106
records = read(S/'records.json')['records']
scenes = [(d,a) for d in ('IID','non-IID') for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')]
expected = {(v,d,a,s) for v in ('Full','minus_A') for d,a in scenes for s in range(91001,91011)}
assert len(records) == 200 and len({r['id'] for r in records})==200 and {(r['variant'],r['distribution'],r['attack'],r['seed']) for r in records} == expected
assert (h['records'],h['pairs'],h['complete_scenes'],h['complete_nonIID_scenes']) == (200,100,10,['Benign','F Flip','FedSA','S-DFA','Sp-DFA'])
assert h['training_torch']=={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ('Full','minus_A')}
assert h['replay_devices']=={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ('Full','minus_A')}
assert h['native_shared_metrics_counts_exact']==all(r['views']['native']==r['views']['shared_calibration'] for r in records)
assert h['mixed_distribution_aggregate'] is True and not h['independent_confirmation'] and not h['primary_endpoint_selected'] and not h['final_test']
assert h['new_fit'] == h['new_CNN'] == h['new_training'] == 0
assert h['accepted_partial_SpDFA_seeds_excluded']==[]
assert (h['per_scene_scalars'],h['seed_first_scalars'],h['nonIID_seed_first_scalars'],h['balanced_seed_first_scalars'],h['total_scalars'],h['display_cells'])==(1620,162,162,162,2106,1053)
header=read(S/'TABLES_HEADER_EDIT.json')
assert header['replacement_count']==1 and header['inverse_original_bytes_exact'] and header['numeric_table_body_bytes_unchanged']
assert header['root_verifier_command_sha256']==args.root_command_sha256 and header['root_verifier_stdout_sha256']==args.saved_verification_sha256
markdown=(S/'TABLES.md').read_bytes()
assert sha(S/'TABLES.md')==header['after_sha256'] and markdown.count(header['after_title'].encode())==1
assert hashlib.sha256(markdown.replace(header['after_title'].encode(),header['before_title'].encode(),1)).hexdigest()==header['before_sha256']
selected = ['records.json','tables.json','paired_per_seed.json','coverage.json','IID_SEED_FIRST.json','CROSS_SCENE_ADDITIONAL.json','TABLES.md','TABLES.tex','TABLES_HEADER_EDIT.json','TABLES_HEADER_EDIT.diff','SOURCE_BINDINGS.json','SAVED_VERIFICATION.json','DIRECTIONS.json','FOCUSED_CHECKS.json','ROOT_READY_REPORT.md','ROOT_READY_SUMMARY.json']
D.mkdir(parents=True)
for name in selected:
    (D/name).write_bytes((S/name).read_bytes()); assert sha(D/name) == sha(S/name)
status = 'ROOT_A100_TEN_COMPLETE_SCENE_THREE_VIEW_TABLE_ADOPTED'
summary = dict(h,status=status,root_adopted=True,native_shared_metrics_and_counts_exact=h['native_shared_metrics_counts_exact'])
(D/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(D/'REPORT.md').write_text('当前：A100十场景表已由root采用；下列交接报告的pending描述是采用前历史。当前范围以ROOT_VERIFICATION.json为准。\n\n'+(S/'ROOT_READY_REPORT.md').read_text('utf8'),encoding='utf8')
(D/'README_CURRENT.md').write_text('A100十场景三视图表已采用：200条记录、100对终轮模型，固定10/9/6种子面板。入口TABLES.md；验收ROOT_VERIFICATION.json。仅验证集；其他五control变体与最终test未完成。\n',encoding='utf8')
selected += ['SUMMARY.json','REPORT.md','README_CURRENT.md']
proof = dict(status=status,utc=datetime.now(timezone.utc).isoformat(),root_adoption=True,
    canonical_table=(D/'TABLES.md').relative_to(R).as_posix(),source_candidate=S.relative_to(R).as_posix(),
    source_seal_sha256=sha(S/'DELIVERY_FILES_SHA256.json'),source_handoff_sha256=sha(S/'DELIVERY_HANDOFF.json'),
    source_acceptance_path=b['adoption'],source_acceptance_sha256=b['adoption_sha256'],source_index_sha256=b['index_sha256'],native_root_sha256=b['native_root_sha256'],
    preserved_records=200,paired_models=100,complete_scenes=10,complete_IID_scenes=5,complete_nonIID_scenes=['Benign','F Flip','FedSA','S-DFA','Sp-DFA'],
    mean_SD_scalars_recomputed=numeric['all_scalars'],per_scene_scalars=numeric['per_scene']['mean_sd_scalars'],seed_first_scalars=numeric['seed_first']['mean_sd_scalars'],
    nonIID_seed_first_scalars=numeric['nonIID_seed_first']['mean_sd_scalars'],balanced_seed_first_scalars=numeric['balanced_seed_first']['mean_sd_scalars'],
    display_cells=numeric['all_display_cells'],metrics_from_group_counts=numeric['per_scene']['receipt_metrics_from_group_counts'],base_integer_confusion_counts_checked=numeric['per_scene']['base_confusion_counts_structurally_checked'],
    old180_record_bytes_order_exact=True,old1458_scalars_exact=True,old729_cells_exact=True,IID_seed_first_JSON_bytes_exact=True,
    actual_root_command=command['command'],actual_root_command_exit=0,actual_root_verifier_sha256=sha(S/'verify_saved.py'),actual_root_output_matches_saved=True,
    root_verification_precedes_title_only_edit=True,title_only_edit_sha256=sha(S/'TABLES_HEADER_EDIT.json'),numeric_table_body_bytes_unchanged_by_title_edit=True,
    author_focused_check_sha256=sha(S/'FOCUSED_CHECKS.json'),author_focused_check_is_independent_reviewer=False,replay_devices=h['replay_devices'],training_torch=h['training_torch'],
    files_sha256={n:sha(D/n) for n in selected},seed_panels=[10,9,6],aggregate_scope='Separate five IID, five non-IID and balanced ten-scene summaries; equal scene mean within seed before seed statistics',
    accepted_partial_SpDFA_seeds_excluded=[],new_CNN=0,new_fit=0,new_training=0,test=False,primary_endpoint_selected=False,tex_compiled=False,incorporated_into_full_rebuttal=False,
    limitations=['Validation and selection-seed/history exposure retained.','Native/shared equality is not independent confirmation.',
    'Actual mixed CPU/CUDA and cu128/cu130 provenance; no trajectory-equivalence claim.',
    'A100 only is complete; other five controls, full17 coverage, final endpoint/test and submitted-manuscript integration remain incomplete.',
    'All mixed, negative and direction-reversal findings and original Windows baseline failures preserved.'])
with (D/'ROOT_VERIFICATION.json').open('x',encoding='utf8') as f:
    json.dump(proof,f,ensure_ascii=False,indent=2); f.write('\n')
print(json.dumps(dict(status=status,proof_sha256=sha(D/'ROOT_VERIFICATION.json'),table=proof['canonical_table'],scalars=numeric['all_scalars'],cells=numeric['all_display_cells'])))
