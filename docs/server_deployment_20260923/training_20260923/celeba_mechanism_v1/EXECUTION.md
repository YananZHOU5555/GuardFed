# CURRENT: GuardFed返修实验

本段更新时间：2026-10-10T20:09:59.058661+00:00。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。sglang已按授权停止，文件保留。

## 已接纳结果与实时观测

| 阶段 | 原strict、离机SHA及root验收 | 最近实测（2026-10-10T19:58:26.072343+00:00，不代替验收） |
|---|---:|---|
| 机制训练 | 320/800新增；100 Full另复用 | 完成320、活动8、等待472、失败0，活动第17–46轮 |
| 机制三视图 | 320终轮checkpoint：U100/C100/A100/F20 | remaining620远端闭合140，与离机范围单列 |
| FLGMM完整覆盖 | 67/96新增；4screen另复用 | 终轮69；三视图仍61，六场景表60，未把新native计作已评价 |
| Fed-NGA/Huber搜索 | 46/64 | 终轮53；未完成全部搜索或选择recipe |
| 组合基线覆盖 | 12/96新增；4screen另复用 | 终轮15；IID Benign十seed native表已接纳 |

该实测双GPU均100%，温度68/64°C，RecoveryAction None；cgroup内存78.71/519.17GB、OOM0，磁盘余1.060TB。源码/数据身份未变，五队列真实终轮集合增长通过：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/ROOT_FIVE_QUEUE_GROWTH_20261010T1958.json。这是一次采样，不能称连续占用或完成全部实验；固定方法、seed、并发、FP32和统计口径不变。

## 新补齐的F消融表

docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/TABLES.md

IID Benign/F Flip两完整场景，每场景10共享seed；raw/native/shared三视图、固定10/9/6面板。40记录/20配对，独立核算324均值SD标量、162展示格、360计数派生指标、960基础计数；旧Benign20对象、162标量及81展示值原样保留。没有新增推理、拟合或训练。

F Flip十seed native：Full为ACC88.391±0.626%、AEOD0.01068±0.00958、ASPD0.06068±0.01026；去F为88.325±1.212%、0.01439±0.00651、0.06600±0.01819。9/6seed的准确率差值反向，AEOD/ASPD差值仍支持Full；Benign存在另一组取舍。全部方向和负结果保留，不称F在所有指标不可或缺、显著或因果隔离。仅两个IID场景，其他八个F场景和其余控制未齐。native/shared相同不作独立确认；混合设备与选择历史仍披露。

FL新十项评价原CPU120–127两次被活跃训练线程冲突门拒绝，失败保留。一次全系统资源诊断支持候选136–143逻辑掩码当时无冲突，8核/SMT/cache结构同类，但跨socket/NUMA且一SMT兄弟存在轻微活动。新资源版本已完成源码审查和实际部署；2026-10-10T20:05:40.550574+00:00的fresh guard又检测到训练线程TID97820在CPU138两秒增加1 tick，已拒绝启动并停止，无重试。三次均未启动评价、CNN或拟合，不接纳新评价结果；证据为tmp/fl_three_view_FFlip10_cpu136_20261011/runtime/LAUNCH_REFUSAL.json。待训练结束或资源条件实质改变再接续，不放松检查或重启健康训练。

## 返修边界与接续

十方法native验证表1000格，IID/non-IID各五场景已交付；九方法三视图900记录与2052项共享校准归因已接纳。U/C/A各100配对、十场景表已接纳。尚缺七方法完整覆盖、剩余机制控制、冻结最终评价和提交版正文/rebuttal收尾。FedWA/SmartFL/FedDNA忠实规格、最终主终点/测试边界、提交版LaTeX源及Fig3执行来源仍待解决；已有作者问题不重复询问。

24条原意见的清晰英文作者审阅稿：docs/server_deployment_20260923/revision_20260923/rebuttal_clear_A100_20261011/rebuttal_clear_20261011.md。A100已纳入，F20尚未并入完整回复稿。旧TableII480原值已追溯重复数；缺证据的SD不补造。Fig3核260终轮记录/78均值，ForestDiffusion执行及checkpoint来源缺口仍在。

Huber采用已同意的恒等投影，明确CNN项目适配且不继承原理论。当前已验收14个Huber候选均恒负预测，ACC0.5166859616449389、AEOD/ASPD0，保留为退化负结果，不据此称公平性获胜。LoGoFair采用固定图像ID20虚拟cohort，仅人口适配，不称真实client公平性；其100个native结果已接纳。

最终候选partition2的19962个image_id元数据已核，未读标签/像素或模型、未作最终推理；validation选择史、旧test暴露、cu128/cu130及driver差异保持披露。新CPU位置不证明跨平台或跨socket数值等价。下一步继续冻结队列；按完整场景增量接纳，完成FL exact10评价后再补其七场景表；NGA/Huber须64全验收后按冻结规则选recipe及真实门检，不能以部分结果启动192覆盖。

## 保存、Git与巡检

大文件只写F:/YananResearchStorage/GuardFed，写前核F为Yanan 2TB/Healthy且容量足；E仅代码、配置、索引与精简报告。模型/原始数组/归档不进Git，服务器大文件优先原地保留。

最近已验证推送31b55b6e925af02db5fdb7d8deec4e25ac54307d，截止以publication_closed_increment58_verified_20261011.json为准；本次新结果尚未算已推送。三小时聊天任务guardfed-training-health仍PAUSED，本会话无automation_update接口；未建替代cron/Windows任务。supervisor训练与聊天巡检分开。

入口写入器：tmp/update_increment59_entries_20261011.py。旧生成器只保留历史截止，不再用于当前入口。下方历史原字节保留。

# HISTORICAL PREPARATION SNAPSHOT — no execution at time of preparation

# Mechanism stage: handoff and execution

This stage is prepared and locally checked. No new real-image or GPU run has started. `prepared_acceptance.json` verifies the complete 800-new/100-reused grid, 18 image gates and two unchanged-worker references, the five frozen core/loader source files, and six CPU/synthetic component checks. It does not verify current server data, real images, GPU numerics or scientific outcomes.

Runtime files are `adapter.py`, `worker.py`, `runner.py`, `prepare.py` under the project-local `tmp/celeba_mechanism_20261009/`. A byte-identical deployment snapshot and setup archive accompany this entry. Preserve file bytes during transfer; remote paths in the manifest are POSIX. A draft Windows-path error was intercepted before launch, corrected and rechecked; see `predispatch_path_failure.json`.

Authorized target: `ssh -p 26712 root@213.224.31.105`, instance52514165. Before remote actions read `/etc/vast-agents-guide.md`, inspect actual GPU/process/resource state, and verify the repository and data. Never start the old server or previously completed queues.

Install runtime files at `/workspace/GuardFed-celeba-expanded/deployment/celeba_mechanism_20261009/` and the prepared stage contents at `/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1/`. Do not replace the frozen core. The three commands below are sequential, with inspection and acceptance at each boundary. They are instructions, not records of execution.

```bash
cd /workspace/GuardFed-celeba-expanded
.venv/bin/python deployment/celeba_mechanism_20261009/runner.py preflight --repo /workspace/GuardFed-celeba-expanded --stage /workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1
.venv/bin/python deployment/celeba_mechanism_20261009/runner.py freeze --repo /workspace/GuardFed-celeba-expanded --stage /workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1
.venv/bin/python deployment/celeba_mechanism_20261009/runner.py run --repo /workspace/GuardFed-celeba-expanded --stage /workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1
```

The preflight queue executes only 20 three-round pipeline jobs: 18 intervention/distribution gates and two unchanged Full references. Formal dispatch needs all gates, exact Full same-horizon regression, the identities of all100 reused Full models/results, source/data/config consistency, no existing training worker and a separate frozen receipt. Eight independent processes use the established two-GPU/one-CPU-thread schedule; no tuning of ablation parameters occurs. Existing accepted outputs are checked before skipping. A failed or partial directory stops recovery; preserve and inspect it. There is no new midround-resume promise.

The runner and its source checks compile locally; the actual queue path remains untested on this currently unreachable server. In particular, live guide/lineage/real-image gates must pass before describing this stage as ready or running. If a legitimate identity or gate mismatch appears, preserve it and fix the deployment/control problem; do not weaken acceptance.

After training, still required: strict900-record merge, per-condition paired ten-seed tables, raw/native/shared-root-only calibration controls, incremental off-server model/result/log backups with member SHA and restore chain. No automatic test evaluation is included. The separate missing-baseline matrix and final evaluation remain outside this stage.
