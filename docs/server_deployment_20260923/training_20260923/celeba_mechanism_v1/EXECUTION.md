# CURRENT: GuardFed返修实验

本段更新时间：2026-10-10T22:00:44.639533+00:00。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。sglang已按授权停止，文件保留。

## 已接纳结果与实时观测

| 阶段 | 原strict、离机SHA及root验收 | 分批实测（UTC见格内，不代替验收） |
|---|---:|---|
| 机制训练 | 333/800新增；100 Full另复用 | 21:44:05：完成336、活动8、等待456、失败0；活动第9–27轮 |
| 机制三视图 | 330终轮checkpoint：U100/C100/A100/F30 | 21:44:05：remaining620远端闭合156，与离机接受150单列 |
| FLGMM完整覆盖 | 67/96新增；4screen另复用 | 21:44:05：新增终轮73，另复用4screen；三视图接受71，七完整场景表70，另保留一个未齐场景记录 |
| Fed-NGA/Huber搜索 | 46/64 | 21:44:05：终轮59/64；未完成全部搜索或选择recipe |
| 组合基线覆盖 | 12/96新增；4screen另复用 | 21:44:05：新增终轮17，另复用4screen；IID Benign三视图十seed表已接纳 |

21:44:05资源实测：双GPU利用率100/100%，温度64/62°C，RecoveryAction None；cgroup内存77.30/519.17GB、OOM0，磁盘余1.060TB。五队列服务、worker身份与相对增长通过；small source/data hash未变，大图像只核stat及历史全量SHA。增长记录：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/ROOT_FIVE_QUEUE_GROWTH_20261010T2144.json。两采样间整个cgroup平均使用13.69/122.88 CPU核，不能称GuardFed专属效率或持续占用。这是一次采样，不能称连续占用或完成全部实验；固定方法、seed、并发、FP32和统计口径不变。

机制native已另接纳13项：去F的IID FedSA完整10seed，加S-DFA的91002/91003/91004三项部分结果；总333/800。134归档成员SHA及终轮/配置/源码/数据/checkpoint身份通过，旧420记录不变；其中FedSA10三视图已增量接纳，累计330；另外三项S-DFA native仍不计作三视图完成。native333及views330均在Git61截止后形成，下一增量发布待纳入；F盘与服务器原件已保留。

## 新补齐的三个场景F消融表

docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_three_scenes30_20261011/TABLES.md

IID Benign/F Flip/FedSA三个完整场景，每场景10共享seed；raw/native/shared三视图、固定10/9/6面板。60记录/30配对，独立核算486均值SD标量、243展示格、540计数派生指标、1440基础计数；旧40对象、324标量及162展示值原样保留。表格制作没有新增推理、拟合或训练。

F Flip十seed native：Full为ACC88.391±0.626%、AEOD0.01068±0.00958、ASPD0.06068±0.01026；去F为88.325±1.212%、0.01439±0.00651、0.06600±0.01819。9/6seed的准确率差值反向，AEOD/ASPD差值仍支持Full；Benign存在另一组取舍。全部方向和负结果保留，不称F在所有指标不可或缺、显著或因果隔离。现为三个IID场景，其他七个F场景和四类其他控制未齐。FedSA十seed去F较Full的配对均值为ACC+0.136pp、AEOD+0.00322、ASPD+0.00393；六seed native/shared的ASPD差值反转为−0.00560。全部结果保留。native/shared相同不作独立确认；混合设备与选择历史仍披露。

FL新增10记录已通过原生指标复核、Linux原校准检查、F盘归档及成员SHA、Windows零拟合计数审计，并由root采用；旧61记录原样保留。七完整场景表按同一10/9/6种子面板展示，另保留一个未齐场景记录。non-IID F Flip十seed native为ACC89.28±0.98%、AEOD0.0525±0.0077、ASPD0.1185±0.0068；共享校准为88.90±1.04%、0.0091±0.0073、0.0713±0.0111，存在准确率代价。旧Windows重拟合失败、五次资源拒绝和采样审计浮点差异均保留，不宣称跨平台逐位重校准一致。表：outputs/guardfed_tables/celeba_flgmm_seven_scenes70_20261011/TABLES.md。

Hybrid IID Benign三视图现已补齐十共享seed：八个既有覆盖checkpoint、一个原screen seed91001及一个已验收seed91002显式复用。原screen身份及选择史保持，不改名为正式训练；同一checkpoint全部指标。新九项通过Linux原校准检查、F盘归档及Windows零拟合审计，原Windows首次Linux resource导入失败及采样审计微小差异保留。仅这一场景齐备，其他场景和完整100覆盖仍在继续。三视图表：outputs/guardfed_tables/celeba_hybrid_three_view_IID_Benign10_20261011/TABLES.md。十seed native ACC88.027%、AEOD0.02920、ASPD0.09251；shared为87.708%、0.02321、0.03383，存在准确率代价。Hybrid为项目控制，不称外部原方法。

## 返修边界与接续

十方法native验证表1000格，IID/non-IID各五场景已交付；九方法三视图900记录与2052项共享校准归因已接纳。U/C/A各100配对、十场景表已接纳。尚缺七方法完整覆盖、剩余机制控制、冻结最终评价和提交版正文/rebuttal收尾。FedWA/SmartFL/FedDNA忠实规格、最终主终点/测试边界、提交版LaTeX源及Fig3执行来源仍待解决；已有作者问题不重复询问。

24条原意见的最新清晰英文作者审阅稿：docs/server_deployment_20260923/revision_20260923/rebuttal_clear_F20_20261011/rebuttal_clear_20261011.md；详细回复及正文插入候选：docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_F20_20261011/rebuttal_integrated_20261011.md、docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_F20_20261011/manuscript_insertions_integrated_20261011.md。三份作者候选已对齐A100及F20，原话与顺序保留；单独新增F30/Hybrid证据补充稿：docs/server_deployment_20260923/revision_20260923/rebuttal_F30_addendum_20261011.md。未把原24条旧截止稿暗改称F30，未应用提交版正文，不代表返修完成。清晰候选、详细稿及正文插入候选均已在Git798ce6；本次FL71/七场景表70及Hybrid10证据已在Gita3fd1aa。旧TableII480原值已追溯重复数；缺证据的SD不补造。Fig3核260终轮记录/78均值，ForestDiffusion执行及checkpoint来源缺口仍在。

Huber采用已同意的恒等投影，明确CNN项目适配且不继承原理论。当前已验收14条Huber搜索结果均恒负预测，ACC0.5166859616449389、AEOD/ASPD0，保留为退化负结果，不据此称公平性获胜。LoGoFair采用固定图像ID20虚拟cohort，仅人口适配，不称真实client公平性；其100个native结果已接纳。

最终候选partition2的19962个image_id元数据已核，未读标签/像素或模型、未作最终推理；validation选择史、旧test暴露、cu128/cu130及driver差异保持披露。新CPU位置不证明跨平台或跨socket数值等价。下一步继续冻结队列；按完整场景增量接纳，FL exact10及七场景表已闭合，继续其剩余覆盖；NGA/Huber须64全验收后按冻结规则选recipe及真实门检，不能以部分结果启动192覆盖。

## 保存、Git与巡检

大文件只写F:/YananResearchStorage/GuardFed，写前核F为Yanan 2TB/Healthy且容量足；E仅代码、配置、索引与精简报告。模型/原始数组/归档不进Git，服务器大文件优先原地保留。

最近已验证推送a3fd1aa9667cf1e177860872192001f8c4111f40，截止以publication_closed_increment61_verified_20261011.json为准（240提交文件哈希及远端分支通过）。本地本段的推送指针刷新发生在该提交之后，不能称已包含在其内。三小时聊天任务guardfed-training-health仍PAUSED，本会话无automation_update接口；未建替代cron/Windows任务。supervisor训练与聊天巡检分开。

入口写入器：tmp/update_increment62_entries_20261011.py。旧生成器只保留历史截止，不再用于当前入口。下方历史原字节保留。

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
