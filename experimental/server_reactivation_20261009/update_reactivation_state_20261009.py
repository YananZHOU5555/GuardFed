"""Update live project entry from measured receipts, leaving frozen protocols intact."""
from pathlib import Path
import datetime
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'docs/server_deployment_20260923/training_20260923'
CHECKS = TRAIN / 'server_reactivation_20261009'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
state_path = TRAIN / 'TRAINING_STATE.json'
state = read(state_path)
live = read(CHECKS / 'latest_preflight_live.json')
formal = (CHECKS / 'formal_launch.json').exists()
if formal:
    live = read(CHECKS / 'latest_formal_live.json')
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
phase = 'FORMAL800_RUNNING' if formal else 'CU128_REAL_IMAGE_GATES_RUNNING'
state['updated_unix'] = time.time()
state['server_reactivation_20261009'].update(status=phase, new_formal_training=len(live['active'])+live['queue_completed'] if formal else 0,
    new_formal_training_started=formal, exact_restore_acceptance='server_reactivation_20261009/restore_acceptance.json',
    cu130_gates_accepted=20, cu130_offserver_members_verified=149,
    cu128_environment='server_reactivation_20261009/cu128_environment.json',
    latest_live=live, environment_difference='New jobs use isolated torch2.11.0+cu128 and driver595.84; Full controls98cu128+2cu130 were mostly produced on driver570.211.01. Three-round checks cannot establish70-round equivalence.')
state['latest_connection_check'].update(recorded_utc=live['checked_utc'],
    current_internal_runtime=live['service'], torch='2.11.0+cu128',
    new_training_started=formal, next='Accept all20cu128 image gates, verify100Full full science-input identities and freeze before800 formal jobs' if not formal else 'Continue frozen800 missing mechanism jobs; strictly accept and incrementally back up new IDs only')
state['celeba_mechanism_v1'].update(status=phase,
    real_image_cu130_gates_accepted=20, current_runtime='torch2.11.0+cu128 driver595.84',
    queue_dispatched=formal, new_started=(len(live['active'])+live['queue_completed'] if formal else 0),
    preflight_queue_completed=live['queue_completed'] if not formal else 20,
    queue_completed_observed=live['queue_completed'] if formal else 0,
    scientific_results_strictly_accepted=state['celeba_mechanism_v1'].get('scientific_results_strictly_accepted',0))
if formal:
    state['celeba_mechanism_v1'].update(real_image_gates_passed=True,
        real_image_cu128_gates_accepted=20, cross_runtime_three_round_exact=20,
        full_reuse_strictly_accepted=100, new_total=800,
        dispatch_receipt_sha256=sha(TRAIN/'celeba_mechanism_v1/dispatch_receipt.json'),
        validation_only=True, test_started=False)
state['active_services'] = ['guardfed_celeba_mechanism_formal' if formal else 'guardfed_celeba_mechanism_preflight']
if formal:
    state['current_stage'] = 'celeba_mechanism_v1'
state['last_health_check'] = live
state['native_monitor_20261009'] = {'id':'guardfed-training-health', 'observed_local_status':'PAUSED',
    'scheduler_status':'not_observable_with_available_tools', 'native_update_tool_available':False,
    'local_prompt_stale_endpoint':True, 'no_scheduler_changed':True,
    'handoff':'server_reactivation_20261009/MONITOR_HANDOFF.md'}
state['synthetic_figure_recovery_20261009'] = {'status':'PASS_WITH_EXPLICIT_PARTIAL_PROVENANCE',
    'original_records':260, 'pca_records':40, 'legacy_triplets_without_common_checkpoint':250,
    'exact_fig3_plot_script_recovered':False, 'executed_forest_diffusion_identity_recovered':False,
    'verification_sha256':sha(TRAIN / 'synthetic_figure_recovery_20261009/verification.json')}
state['final_evaluator_runtime_20261009'] = {'status':'PREPARED_NOT_FROZEN', 'local_acceptance_groups':11,
    'historical_valid_cached_jobs':8, 'all900_native_realimage_valid_replayed':False, 'test_started':False,
    'local_entry':'tmp/celeba_final_evaluator_20261009/README.md',
    'published_entry':'experimental/celeba_final_evaluator_20261009/README.md'}
restore_dir = TRAIN / 'validation900_restore_20261009'
if (restore_dir / 'restore_acceptance.json').exists():
    restored = read(restore_dir / 'restore_acceptance.json')
    independent = read(restore_dir / 'root_canary_independent_verification.json')
    assert restored['status'] == 'EXACT_900_ARTIFACT_STORAGE_VERIFIED'
    assert restored['models'] == 900 and restored['all_artifact_file_hashes_verified'] == 2700
    assert independent['status'] == 'PASS' and len(independent['canaries']) == 2
    state['validation900_restore_20261009'] = {
        'status': restored['status'], 'models': 900, 'artifact_hashes_verified': 2700,
        'existing_full_artifacts_reused': 300, 'new_isolated_artifacts': 2400,
        'original_output_files_modified': 0,
        'restore_acceptance_sha256': sha(restore_dir / 'restore_acceptance.json'),
        'storage_map_sha256': restored['storage_map_sha256'],
        'entry': 'validation900_restore_20261009/README.md'}
    state['final_evaluator_runtime_20261009'].update(
        actual_native_valid_image_replays_accepted=2,
        actual_canary_native_max_abs_difference=0,
        root_independent_verification='validation900_restore_20261009/root_canary_independent_verification.json',
        new_runtime_entry='tmp/celeba_final_valid_replay_20261009/README.md',
        exact_900_artifact_storage_ready=True,
        full900_replay_started=False,
        final_protocol_frozen=False)
reply = ROOT / 'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009'
state['rebuttal_draft_20261009'].update(sha256=sha(reply/'rebuttal_20261009.md'),verification_sha256=sha(reply/'verification.json'))
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
running = TRAIN / 'RUNNING.md'
old = running.read_text(encoding='utf-8')
boundary = '# HISTORICAL: Nine-method coverage COMPLETE'
assert boundary in old
history = old[old.index(boundary):]
top = f'''# CURRENT: GuardFed mechanism {'formal800' if formal else 'cu128 preflight'} — measured {live['checked_utc']}

用户重新提供89.22.197.55:60350并明确授权停止sglang，现使用实例52183675开展缺失返修实验。sglang已停止；两张5090实际CUDA张量检查通过。先读/etc/vast-agents-guide.md（SHA42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa）。旧完成GuardFed队列仍禁止重启；213.224.31.105:26712内部当前状态未知，不自动切回。

源码五文件、全部RGB64缓存与官方标签/划分身份已核，100已验收Full对照及缺失历史依赖准确恢复（307成员验收，303新建+4原有相同）。20项cu130真实图像三轮门检全部严格接受，两套Full与原worker同horizon模型张量/全部指标/诊断精确一致；149备份成员及archive SHA在本机通过。原cu130门检完整保留在preflight_history/cu130_20261009，仅原job字节复制回空输出路径。

当前使用独立/workspace/guardfed_envs/celeba-cu128-20261009/bin/python，torch2.11.0+cu128、driver595.84，未改原环境/驱动/方法/seed/指标。实测服务{live['service']}；完成{live['queue_completed']}，活动{len(live['active'])}，等待{live['pending']}，失败{len(live['failed'])}；资源与轮次见server_reactivation_20261009/latest_{live['phase']}_live.json。{'800新机制正式队列已启动，70round valid-only，8并发；100Full显式复用；检查dispatch receipt与full identity acceptance。' if formal else '目前仅20项cu128三轮门检，未启动800项科学训练/最终test。所有20项严格接受、实际runtime逐条核验、100Full全科学输入/数据契约验收后才冻结启动800新任务。'}

正式机制设计800新+100复用，IID/non-IID×5场景×10共享seed，按原协议固定recipe。Full98cu128+2cu130且多数原产出driver570.211.01，环境混合须披露；短程门检不能证明70round等价，提供排除两条cu130的配对敏感性口径。PROTOCOL.md为不可改动的准备快照，当前执行事实优先读EXECUTION.md、dispatch receipt、本入口及状态JSON。

原三小时任务guardfed-training-health本地TOML为PAUSED且提示仍指旧阶段；当前没有原生automation_update工具，未修改调度器、未建立替代监督机制。服务器supervisor只管理已启动队列，不等于三小时聊天巡检已恢复。可审阅提示与限制见server_reactivation_20261009/MONITOR_HANDOFF.md。

2454个历史新增完整训练及九方法900验证记录/备份保持原值。返修回复已逐字核24块原意见、40本地引用、209项SHA声明；新增260条生成器/PCA数值追溯与20份CelebA联合分组重建已核，投稿Fig3原脚本/FD执行身份仍缺。其余8基线与正式最终评价仍未完成；准备代码/门检不当作论文科学结果。主入口REBUTTAL_COMPLETION_20261009.md。

'''
if (restore_dir / 'restore_acceptance.json').exists():
    top += '''九方法900终轮模型/result/raw-job已全部精确接入当前服务器：100Full复用现存路径，其他800恢复至独立artifact_store，共2700文件逐SHA核验，原历史output修改0。两条完整valid19867/root16277原图CPU重放已接受，native三指标误差0，raw/native/shared三个视图的18指标与48混淆计数经主代理独立复核；52封存文件及27归档成员离机通过。900全批尚未启动，不称最终评价完成；详见validation900_restore_20261009/README.md。Hybrid与FLGMM完整真实图像CPU门检继续运行，首轮证据不等于三轮PASS。

'''
running.write_text(top+history,encoding='utf-8')
execution = TRAIN / 'celeba_mechanism_v1/EXECUTION.md'
text = execution.read_text(encoding='utf-8')
marker = '# HISTORICAL PREPARATION SNAPSHOT — no execution at time of preparation\n\n'
if marker in text:
    text = text.split(marker,1)[1]
execution.write_text(top.replace('# CURRENT:', '# CURRENT EXECUTION:')+marker+text,encoding='utf-8')
print(json.dumps({'status':phase,'measured_utc':live['checked_utc'],'completed':live['queue_completed'],'active':len(live['active']),'formal':formal}))
