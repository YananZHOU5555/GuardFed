"""Update compact current entries from adopted proofs; preserve historical bytes."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, time

R = Path(__file__).resolve().parents[1]
T = R / 'docs/server_deployment_20260923/training_20260923'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
p = argparse.ArgumentParser()
p.add_argument('--F20-root', type=Path, required=True)
p.add_argument('--F20-root-sha256', required=True)
a = p.parse_args()
fpath = (R / a.F20_root).resolve()
assert fpath.is_relative_to(T.resolve()) and sha(fpath) == a.F20_root_sha256
f = read(fpath)
assert f['root_adoption'] and (f['preserved_records'], f['paired_models'], f['complete_scenes']) == (40, 20, 2)
assert not f['test'] and not f['whole_rebuttal_complete']
np = T / 'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T194755Z/ROOT_DELTA_VERIFICATION.json'
vp = R / 'tmp/celeba_mechanism_remaining_F_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json'
fp = R / 'tmp/fl_native_after63_20261011/ROOT_ADOPTION_REVIEW.json'
op = T / 'server_reactivation_20261009/root_five_queue_20261010T195821Z.raw.json'
gp = T / 'server_reactivation_20261009/ROOT_FIVE_QUEUE_GROWTH_20261010T1958.json'
rp = R / 'tmp/fl_three_view_FFlip10_cpu136_20261011/runtime/LAUNCH_REFUSAL.json'
assert sha(np) == '749577f73d151c958f6dc646a1e15714db7c72a8b099a75f22298b58b8c728bb'
assert sha(vp) == 'b427a751a127242d75a6f47426f60067b15345ef47b8704ba000b28925185317'
assert sha(fp) == 'a11cd9ed94ab136b49dd41975d45f684d88236b6918c295461089a98adbc7db0'
assert sha(op) == '78a5ef2b40184bcd652678cb22620dd10811b98c5db53833427b2795373969a5'
assert sha(gp) == '1ff95525cce03031a5a1d155dcc34319b2a5f601e5d0655331c0ecd8bc686929'
assert sha(rp) == 'a974849e5b617dbdf708a1ebc60ccc9b6328b7d5e99ed6fbcdde6fcac5ec3f0a'
refusal = read(rp)
assert refusal['status'] == 'CPU136_LAUNCH_REFUSED_ACTIVE_WIDE_THREAD_NO_RETRY'
assert refusal['no_new_CNN_fit_training'] and not refusal['retry_authorized']
assert not any(refusal['gate_status_files_present'].values())
n, v, fl, obs = map(read, (np, vp, fp, op))
assert n['root_adopted'] and n['total_new_strict_and_offserver'] == 320
assert v['cumulative_accepted'] == 320 and v['original310_unchanged']
assert fl['accepted_total'] == 67 and fl['old63_ordered_prefix_exact']
assert f['source_acceptance_sha256'] == sha(vp) and f['source_native_root_sha256'] == sha(np)
sp = T / 'TRAINING_STATE.json'
s = read(sp); before_state = sha(sp)
m = s['celeba_mechanism_v1']
assert m['scientific_results_strictly_accepted'] == 312 and m['three_view_new_models_offserver_verified'] == 310
assert len(m['incremental_science_backups']) == 45
for k in ('new_completed', 'scientific_results_completed', 'scientific_results_strictly_accepted', 'scientific_results_offserver_verified'):
    m[k] = 320
m['science_acceptance_inspection'] = (np.parent / 'inspection/inspection.json').relative_to(T).as_posix()
m['latest_science_backup_sha256'] = n['archive_sha256']
m['latest_science_backup_members_verified'] = n['root_checked_archive_members']
m['incremental_science_backups'].append(dict(archive_sha256=n['archive_sha256'], new_ids=n['new_ids'], members_verified=n['root_checked_archive_members']))
idx = read(R / v['records_index_path'])
assert idx['all_ids'][:310] == m['three_view_accepted_ids']
m.update(three_view_new_models_accepted=320, three_view_new_models_offserver_verified=320,
    three_view_accepted_ids=idx['all_ids'], three_view_root_proof_sha256=sha(vp),
    latest_three_view_index=v['records_index_path'], latest_three_view_index_sha256=v['records_index_sha256'],
    three_view_counts_by_variant=dict(minus_U=100, minus_C=100, minus_A=100, minus_F=20),
    F_three_view_two_scene_table=dict(root_adoption=True, table_path=f['canonical_table'], root_proof_path=fpath.relative_to(R).as_posix(),
        root_proof_sha256=sha(fpath), complete_scenes=2, paired_models=20, preserved_records=40,
        seed_panels=[10,9,6], mean_SD_scalars=324, display_cells=162, final_test=False, incorporated_into_full_rebuttal=False),
    three_view_scope_limit='U/C/A each100 pairs across ten scenes. F20 covers IID Benign/F Flip only, fixed10/9/6 panels and all negative paired differences. No F cross-scene aggregate, necessity, significance, final-test or complete-rebuttal claim.')
q = obs['main_mechanism']['queue']
m.update(queue_completed_observed=len(q['completed']), queue_active_observed=obs['main_mechanism']['worker_count'],
    queue_pending_observed=q['pending'], queue_failures_observed=len(q['failed']), queue_observation_utc=obs['utc'],
    queue_observation_path=op.relative_to(R).as_posix(), new_started=len(q['completed'])+obs['main_mechanism']['worker_count'], new_started_observation_utc=obs['utc'])
r = s['mechanism_remaining620_valid_20261010']
assert len(r['accepted_ids']) == 130
r.update(new_offserver_accepted=140, accepted_ids=r['accepted_ids']+v['accepted_new_ids'], latest_accepted_new_ids=v['accepted_new_ids'],
    root_adoption_path=vp.relative_to(R).as_posix(), root_adoption_sha256=sha(vp), F_complete_scenes=v['complete_F_scenes'], F_IID_two_scene_table_adopted=True)
s['flgmm_fullcoverage_v2_20261009'].update(new_accepted=67, accepted_new_ids=fl['accepted_job_ids'],
    latest_full70_root_adoption_path=fp.relative_to(R).as_posix(), latest_full70_root_adoption_sha256=sha(fp))
s['latest_five_queue_readonly_observation'] = dict(path=op.relative_to(R).as_posix(), sha256=sha(op), utc=obs['utc'],
    growth_proof_path=gp.relative_to(R).as_posix(), growth_proof_sha256=sha(gp), counts_are_observation_only=True,
    main_terminal=len(q['completed']), main_active=obs['main_mechanism']['worker_count'], main_pending=q['pending'], main_failures=len(q['failed']),
    FLGMM_terminal=len(obs['FLGMM']['terminal_ids']), gradient_terminal=len(obs['gradient64']['terminal_ids']),
    remaining620_closed=obs['remaining620']['remote_closed_n'], Hybrid_terminal=len(obs['Hybrid96']['terminal_records']), new_acceptance=0)
s['FLGMM_FFlip10_resource_rebind_20261011'] = dict(status=refusal['status'],
    original_attempts_preserved=['tmp/fl_three_view_FFlip10_20261011/runtime_v2/LAUNCH_REFUSAL.json','tmp/fl_three_view_FFlip10_20261011/runtime_after97285_exit/LAUNCH_REFUSAL.json'],
    topology_review='tmp/fl_three_view_FFlip10_20261011/resource_rebind_plan/TOPOLOGY_REVIEW.json',
    topology_review_sha256='39810a4c79591a4a730e266008c5d033e6346f059eb9c2d376e8435d9ec24c95',
    resource_decision='Use8 target logical CPUs136–143 with original guards; different socket/NUMA, not exclusive physical cores. Exact10 science/threads/FP32 unchanged; launch requires fresh gate.',
    actual_launch_refusal_path=rp.relative_to(R).as_posix(), actual_launch_refusal_sha256=sha(rp),
    actual_launch_refusal_utc=refusal['utc'], no_new_CNN_fit_training=True, retry_authorized=False,
    new_three_views_accepted=0, test=False)
s['current_entry_writer'] = dict(path=Path(__file__).relative_to(R).as_posix(), sha256=sha(__file__),
    previous_generators='Historical generation sources retain their original cutoffs; do not rerun them to regenerate current entries.')
s['updated_unix'] = time.time()
draft = s['latest_rebuttal_draft'].get('clear_reader_entry', s['latest_rebuttal_draft']['entry'])
pub = s['latest_publication_verification']
text = f'''# CURRENT: GuardFed返修实验

本段更新时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。sglang已按授权停止，文件保留。

## 已接纳结果与实时观测

| 阶段 | 原strict、离机SHA及root验收 | 最近实测（{obs['utc']}，不代替验收） |
|---|---:|---|
| 机制训练 | 320/800新增；100 Full另复用 | 完成320、活动8、等待472、失败0，活动第17–46轮 |
| 机制三视图 | 320终轮checkpoint：U100/C100/A100/F20 | remaining620远端闭合140，与离机范围单列 |
| FLGMM完整覆盖 | 67/96新增；4screen另复用 | 终轮69；三视图仍61，六场景表60，未把新native计作已评价 |
| Fed-NGA/Huber搜索 | 46/64 | 终轮53；未完成全部搜索或选择recipe |
| 组合基线覆盖 | 12/96新增；4screen另复用 | 终轮15；IID Benign十seed native表已接纳 |

该实测双GPU均100%，温度68/64°C，RecoveryAction None；cgroup内存78.71/519.17GB、OOM0，磁盘余1.060TB。源码/数据身份未变，五队列真实终轮集合增长通过：{gp.relative_to(R).as_posix()}。这是一次采样，不能称连续占用或完成全部实验；固定方法、seed、并发、FP32和统计口径不变。

## 新补齐的F消融表

{f['canonical_table']}

IID Benign/F Flip两完整场景，每场景10共享seed；raw/native/shared三视图、固定10/9/6面板。40记录/20配对，独立核算324均值SD标量、162展示格、360计数派生指标、960基础计数；旧Benign20对象、162标量及81展示值原样保留。没有新增推理、拟合或训练。

F Flip十seed native：Full为ACC88.391±0.626%、AEOD0.01068±0.00958、ASPD0.06068±0.01026；去F为88.325±1.212%、0.01439±0.00651、0.06600±0.01819。9/6seed的准确率差值反向，AEOD/ASPD差值仍支持Full；Benign存在另一组取舍。全部方向和负结果保留，不称F在所有指标不可或缺、显著或因果隔离。仅两个IID场景，其他八个F场景和其余控制未齐。native/shared相同不作独立确认；混合设备与选择历史仍披露。

FL新十项评价原CPU120–127两次被活跃训练线程冲突门拒绝，失败保留。一次全系统资源诊断支持候选136–143逻辑掩码当时无冲突，8核/SMT/cache结构同类，但跨socket/NUMA且一SMT兄弟存在轻微活动。新资源版本已完成源码审查和实际部署；{refusal['utc']}的fresh guard又检测到训练线程TID97820在CPU138两秒增加1 tick，已拒绝启动并停止，无重试。三次均未启动评价、CNN或拟合，不接纳新评价结果；证据为{rp.relative_to(R).as_posix()}。待训练结束或资源条件实质改变再接续，不放松检查或重启健康训练。

## 返修边界与接续

十方法native验证表1000格，IID/non-IID各五场景已交付；九方法三视图900记录与2052项共享校准归因已接纳。U/C/A各100配对、十场景表已接纳。尚缺七方法完整覆盖、剩余机制控制、冻结最终评价和提交版正文/rebuttal收尾。FedWA/SmartFL/FedDNA忠实规格、最终主终点/测试边界、提交版LaTeX源及Fig3执行来源仍待解决；已有作者问题不重复询问。

24条原意见的清晰英文作者审阅稿：{draft}。A100已纳入，F20尚未并入完整回复稿。旧TableII480原值已追溯重复数；缺证据的SD不补造。Fig3核260终轮记录/78均值，ForestDiffusion执行及checkpoint来源缺口仍在。

Huber采用已同意的恒等投影，明确CNN项目适配且不继承原理论。当前已验收14个Huber候选均恒负预测，ACC0.5166859616449389、AEOD/ASPD0，保留为退化负结果，不据此称公平性获胜。LoGoFair采用固定图像ID20虚拟cohort，仅人口适配，不称真实client公平性；其100个native结果已接纳。

最终候选partition2的19962个image_id元数据已核，未读标签/像素或模型、未作最终推理；validation选择史、旧test暴露、cu128/cu130及driver差异保持披露。新CPU位置不证明跨平台或跨socket数值等价。下一步继续冻结队列；按完整场景增量接纳，完成FL exact10评价后再补其七场景表；NGA/Huber须64全验收后按冻结规则选recipe及真实门检，不能以部分结果启动192覆盖。

## 保存、Git与巡检

大文件只写F:/YananResearchStorage/GuardFed，写前核F为Yanan 2TB/Healthy且容量足；E仅代码、配置、索引与精简报告。模型/原始数组/归档不进Git，服务器大文件优先原地保留。

最近已验证推送{pub['commit']}，截止以{pub['proof_path']}为准；本次新结果尚未算已推送。三小时聊天任务guardfed-training-health仍PAUSED，本会话无automation_update接口；未建替代cron/Windows任务。supervisor训练与聊天巡检分开。

入口写入器：{Path(__file__).relative_to(R).as_posix()}。旧生成器只保留历史截止，不再用于当前入口。下方历史原字节保留。

'''
files = [(T/'RUNNING.md', b'# HISTORICAL:'), (T/'REBUTTAL_COMPLETION_20261009.md', b'## Historical accepted increment'),
    (R/'docs/返修实验总览.md', '以下为 2026-10-04'.encode()), (T/'server_reactivation_20261009/MONITOR_HANDOFF.md', b'# Historical handoff snapshots'),
    (T/'celeba_mechanism_v1/EXECUTION.md', b'# HISTORICAL PREPARATION SNAPSHOT')]
report = dict(status='CURRENT_ENTRIES_UPDATED_FROM_ACTUAL320_F20_FL67_PROOFS', before_STATE_sha256=before_state,
    source_pins={key:dict(path=p.relative_to(R).as_posix(),sha256=sha(p)) for key,p in [('native',np),('replay',vp),('F20',fpath),('FL67',fp),('observation',op),('growth',gp),('FL_launch_refusal',rp)]}, entries={})
out = R/'tmp/entry_increment59_actual_20261011'; out.mkdir(exist_ok=False)
for path, marker in files:
    old = path.read_bytes(); at = old.index(marker); history = old[at:]
    (out/(path.name+'.prior_current.txt')).write_bytes(old[:at])
    path.write_bytes(text.encode()+history)
    assert path.read_bytes().endswith(history)
    report['entries'][path.relative_to(R).as_posix()] = dict(sha256=sha(path),history_sha256=hashlib.sha256(history).hexdigest(),history_exact=True)
temp=sp.with_name('TRAINING_STATE.increment59.tmp');assert not temp.exists()
temp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(temp,sp)
report.update(STATE_sha256=sha(sp),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (out/'UPDATE_PROOF.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(status=report['status'],STATE_sha256=sha(sp),entry_count=len(files),proof_sha256=sha(out/'UPDATE_PROOF.json'))))
