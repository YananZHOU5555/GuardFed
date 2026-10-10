"""Publish the root-verified eight-scene table without new scientific execution."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path(__file__).resolve().parents[1]
S=R/'tmp/celeba_mechanism_A80_candidate_20261011'
D=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_eight_scenes80_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert not D.exists()
assert sha(S/'DELIVERY_FILES_SHA256.json')=='eecc5fc844a3d61b3d8081aa81d65b582fb6c887f5c9d28a1f9f76be4b60c418'
sealed=read(S/'DELIVERY_FILES_SHA256.json')['files'];assert len(sealed)==44
for name,pin in sealed.items():
    p=S/name
    assert p.resolve().is_relative_to(S.resolve()) and not p.is_symlink()
    assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
assert sha(S/'DELIVERY_HANDOFF.json')=='841642445e2806691408f5c9e99f101a0c4b2d183f9e9e3c1fe680b7e4826931'
delivery=read(S/'DELIVERY_HANDOFF.json');h=dict(delivery['summary'])
h.update(input_root_path=delivery['input_root']['adoption'],input_root_sha256=delivery['input_root']['adoption_sha256'],input_index_path=delivery['input_root']['index'],input_index_sha256=delivery['input_root']['index_sha256'],actual_original_verifier=delivery['root_saved_verifier'],new_CNN_fit_training_SSH_bulk_download=0,final_test=False)
assert delivery['actual_root_verifier_exit0'] and not delivery['agent_repeated_root_verifier']
assert sha(S/'ROOT_VERIFY_COMMAND.json')==delivery['root_saved_verifier_command_sha256']
assert read(S/'ROOT_VERIFY_COMMAND.json')['exit_code']==0
assert (S/'SAVED_VERIFICATION.json').read_bytes()==(S/'ROOT_VERIFY.stdout').read_bytes()
assert sha(S/'ADDED20_FOCUSED_REVIEW.json')==delivery['independent_added20_review_sha256']=='239a27b872ef8ee92107569ced2724a606ac1295b8e221b97b14a22c2d647755'
assert read(S/'ADDED20_FOCUSED_REVIEW.json')['status']=='INDEPENDENT_ADDED40_RECORDS20_PAIRS_AND324_SCALARS_PASS'
assert delivery['independent_added_statistics']==324
assert sha(R/h['input_root_path'])==h['input_root_sha256']=='3da6be148e4ba649a35f2b6b130022051019d7446f519472efdf3bd4a851473a'
assert sha(R/h['input_index_path'])==h['input_index_sha256']=='ed61c7424a510a70a1f695579341a41cea125b3024067cad874f66af4e1e53b2'
assert (h['records'],h['pairs'],h['complete_IID_scenes'],h['complete_nonIID_scenes'])==(160,80,5,['Benign','F Flip','FedSA'])
assert not h['mixed_distribution_aggregate'] and not h['independent_confirmation']
assert h['new_CNN_fit_training_SSH_bulk_download']==0 and not h['final_test'] and not h['primary_endpoint_selected']
records=read(S/'records.json')['records']
scenes=[('IID',a) for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')]+[('non-IID',a) for a in ('Benign','F Flip','FedSA')]
expected={(v,d,a,s) for v in ('Full','minus_A') for d,a in scenes for s in range(91001,91011)}
assert len(records)==160 and {(r['variant'],r['distribution'],r['attack'],r['seed']) for r in records}==expected
numeric=read(S/'SAVED_VERIFICATION.json')
assert numeric==h['actual_original_verifier'] and numeric['status']=='PASS'
assert (numeric['all_scalars'],numeric['all_display_cells'],numeric['per_scene']['receipt_metrics_from_group_counts'],numeric['per_scene']['base_confusion_counts_structurally_checked'])==(1458,729,1440,3840)
assert numeric['old120_object_bytes_order_exact'] and numeric['old972_scalars_and486_cells_exact'] and numeric['old_IID_seed_first_JSON_bytes_exact']
assert sha(S/'IID_SEED_FIRST.json')=='ce6088b23dd0e2181e83393510075a919dc52719d4b654f70a29205d398b5baf'
assert h['training_torch']=={'Full':{'2.11.0+cu128':79,'2.11.0+cu130':1},'minus_A':{'2.11.0+cu128':80}}
selected=['records.json','tables.json','IID_SEED_FIRST.json','TABLES.md','TABLES.tex','SUMMARY.json','SOURCE_BINDINGS.json','SAVED_VERIFICATION.json','DIRECTIONS.json','ADDED20_FOCUSED_REVIEW.json','ROOT_READY_REPORT.md','ROOT_READY_SUMMARY.json','README_CURRENT.md']
D.mkdir(parents=True)
for name in selected:
    (D/name).write_bytes((S/name).read_bytes());assert sha(D/name)==sha(S/name)
# Promote actual ready summaries with an explicit current root status; source files remain immutable.
summary=dict(read(S/'ROOT_READY_SUMMARY.json'),status='ROOT_A80_EIGHT_COMPLETE_SCENE_THREE_VIEW_TABLE_ADOPTED',root_adopted=True)
summary['native_shared_metrics_and_counts_exact']=summary['native_shared_metrics_counts_exact']
(D/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(D/'REPORT.md').write_text('当前状态：A80八场景表已由ROOT实际采用。下列交接报告中的pending描述保留为采用前历史，当前身份和范围以ROOT_VERIFICATION.json为准。\n\n'+(S/'ROOT_READY_REPORT.md').read_text('utf8'),encoding='utf8')
(D/'README_CURRENT.md').write_text('A80八场景三视图表已采用：160条记录、80对终轮模型；固定10/9/6种子面板。入口TABLES.md，身份与统计核验ROOT_VERIFICATION.json。仅验证集；non-IID S-DFA/Sp-DFA及其他变体未齐，最终test未运行。\n',encoding='utf8')
selected += ['SUMMARY.json','REPORT.md']
proof=dict(status='ROOT_A80_EIGHT_COMPLETE_SCENE_THREE_VIEW_TABLE_ADOPTED',utc=datetime.now(timezone.utc).isoformat(),root_adoption=True,canonical_table=(D/'TABLES.md').relative_to(R).as_posix(),source_candidate=S.relative_to(R).as_posix(),source_seal_sha256=sha(S/'DELIVERY_FILES_SHA256.json'),source_handoff_sha256=sha(S/'DELIVERY_HANDOFF.json'),source_acceptance_path=h['input_root_path'],source_acceptance_sha256=h['input_root_sha256'],source_index_sha256=h['input_index_sha256'],preserved_records=160,paired_models=80,complete_scenes=8,complete_IID_scenes=5,complete_nonIID_scenes=['Benign','F Flip','FedSA'],mean_SD_scalars_recomputed=1458,per_scene_scalars=1296,seed_first_scalars=162,display_cells=729,metrics_from_group_counts=1440,base_integer_confusion_counts_checked=3840,old120_record_bytes_order_exact=True,old972_scalars_exact=True,old486_cells_exact=True,IID_seed_first_JSON_bytes_exact=True,actual_root_command='python -B tmp/celeba_mechanism_A80_candidate_20261011/verify_saved.py',actual_root_command_exit=0,actual_root_verifier_sha256=sha(S/'verify_saved.py'),actual_root_output_matches_saved=True,independent_scope_review_agent='/root/baseline_fidelity_v2',independent_scope_review_received_in_current_turn=True,independent_scope_review_time_note='Exact delivery time was not independently measured; review is recorded in the current chat turn.',independent_scope_review_status='PASS',independent_scope_review_limit='Review received in this chat: new40 records,20 paired identities and324 independently recomputed new-scene scalars; preserved old120 records/972 scalars/IID aggregate bytes and distribution labels. Original full checker was replayed by root, not by this reviewer.',replay_devices=h['replay_devices'],training_torch=h['training_torch'],files_sha256={name:sha(D/name) for name in selected},seed_panels=[10,9,6],aggregate_scope='Original five IID scenes only; mean within seed first; non-IID Benign/F Flip/FedSA reported separately',new_CNN=0,new_fit=0,new_training=0,test=False,primary_endpoint_selected=False,tex_compiled=False,incorporated_into_full_rebuttal=False,limitations=['Validation-only; selection seed91001 and historical official-test exposure disclosed.','Native/shared equality is not independent confirmation or added calibration gain.','Mixed CPU/GPU and cu128/cu130 provenance; no full-trajectory equivalence claim.','Two non-IID A scenes, other controls, full17 baseline coverage, final endpoint and submitted-manuscript integration remain incomplete.'])
with (D/'ROOT_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],proof_sha256=sha(D/'ROOT_VERIFICATION.json'),table=proof['canonical_table'],scalars=1458,cells=729)))
