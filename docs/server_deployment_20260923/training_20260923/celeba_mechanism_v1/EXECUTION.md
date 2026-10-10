# CURRENT: GuardFed返修实验

本段更新时间：2026-10-10T22:42:03.307776+00:00。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。sglang已按授权停止，文件保留。

## 已接纳结果与实时观测

| 阶段 | 原strict、离机SHA及root验收 | 22:39:26 UTC实测（不代替验收） |
|---|---:|---|
| 机制训练 | 341/800新增；100 Full另复用 | 完成344、活动8、等待448、失败0；活动第5–23轮 |
| 机制三视图 | 340终轮checkpoint：U100/C100/A100/F40 | remaining620远端闭合164，离机接受160另列 |
| FLGMM完整覆盖 | 67/96新增；4screen另复用 | 新增终轮77；三视图接受71，七完整场景表70，另保留一个未齐场景记录 |
| Fed-NGA/Huber搜索 | 46/64 | 终轮62/64；未完成全部搜索或选择recipe |
| 组合基线覆盖 | 12/96新增；4screen另复用 | 新增终轮19；IID Benign三视图十seed表已接纳 |

同次资源实测：双GPU利用率100/100%，温度66/62°C，显存6040/6020MiB，RecoveryAction None；cgroup内存76.45/519.17GB、OOM0，磁盘余1.060TB。五队列服务与worker身份通过，small source/data hash未变，大图像只核stat及历史全量SHA。记录：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261010T223921Z.raw.json。瞬时采样不证明持续占用或完成全部实验，冻结方法、seed、并发、FP32和统计口径保持。

本批机制native新接纳8项：补齐去F的IID S-DFA十seed，并接纳Sp-DFA seed91001，总341/800。94归档成员SHA及终轮/配置/源码/数据/checkpoint身份通过，旧433记录和47条恢复链前缀不变，当前48条链已索引。S-DFA10保存数组随后核验并接纳，三视图330→340；Sp-DFA一项native尚不计为三视图完成。原始大文件留服务器及F盘，增量证据待Git63发布。

## 新补齐的四个IID场景F消融表

docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_four_scenes40_20261011/TABLES.md

IID Benign/F Flip/FedSA/S-DFA各10共享seed；raw/native/shared三视图及固定10/9/6面板。80记录/40配对，root独立核648均值SD标量、324展示格、720计数派生指标和1920基础计数；旧60记录、486标量、243展示值原样保留。表格制作没有新增拟合、CNN推理或训练。

S-DFA十seed native：Full为ACC88.689±0.809%、AEOD0.00769±0.00463、ASPD0.06377±0.00697；去F为88.412±1.202%、0.01054±0.00697、0.06941±0.01477。去F减Full配对均值为ACC−0.277pp、AEOD+0.00284、ASPD+0.00564；raw/native/shared的10/9/6九个面板三指标均值方向均支持Full，不作显著性、普遍必要性或单独因果主张。

其他场景的取舍保留：F Flip的9/6seed准确率差值反向；FedSA十seed去F准确率均值+0.136pp，六seed native/shared的ASPD差值−0.00560；Benign也有取舍。native/shared相同不作独立确认，混合设备、环境和选择史均披露。F仍缺六场景60项，另四类机制控制400项：共460三视图未接纳；native未接纳459项，不能称这些任务全部尚未训练。

FL七完整场景表70及单列部分记录已接纳，表：outputs/guardfed_tables/celeba_flgmm_seven_scenes70_20261011/TABLES.md。non-IID F Flip十seed native ACC89.28±0.98%、AEOD0.0525±0.0077、ASPD0.1185±0.0068；shared为88.90±1.04%、0.0091±0.0073、0.0713±0.0111，存在准确率代价。旧Windows全量重拟合失败、五次资源拒绝、Linux原校准通过及Windows零拟合计数审计均保留，不称跨平台逐位一致。

Hybrid IID Benign三视图十seed表已接纳，表：outputs/guardfed_tables/celeba_hybrid_three_view_IID_Benign10_20261011/TABLES.md。原screen seed91001身份与选择史、已验收seed91002及八个覆盖checkpoint显式保留；其余场景继续。native ACC88.027%、AEOD0.02920、ASPD0.09251；shared87.708%、0.02321、0.03383，有准确率代价。Hybrid为项目控制，不称外部原方法；原Windows resource导入失败及采样审计微小差异保留。

## 返修边界与接续

十方法native验证表1000格，IID/non-IID各五场景已交付；九方法三视图900记录和2052项归因已接纳。U/C/A各100配对、十场景表已接纳。尚缺七方法完整覆盖、其余机制控制、冻结最终评价及提交版正文/rebuttal收尾。FedWA/SmartFL/FedDNA忠实规格、最终主终点/测试边界、提交版LaTeX源及Fig3执行来源仍待解决；已有作者问题不重复询问。

24条原意见统一英文工作稿：docs/server_deployment_20260923/revision_20260923/rebuttal_working_20261011.md；已root核原话与顺序、F30/Hybrid数值及链接，供作者审阅。该稿内容截止native333/views330/F30；本次F40为单独新证据表，尚未写进该稿。F20旧稿、正文插入候选及F30补充稿保持各自截止；未应用提交版正文，不能称整份返修完成。

Huber按用户同意采用恒等投影，标为CNN项目适配且不继承原约束域理论。已验收14条Huber搜索结果均恒负预测，ACC0.5166859616449389、AEOD/ASPD0，作为退化负结果保留，不称公平性获胜。LoGoFair采用固定图像ID20虚拟cohort，标为人口适配，不称真实client公平性；其100个native结果已接纳。

最终候选partition2的19962个image_id元数据已核，未读标签/像素或作最终模型推理；validation选择史、旧test暴露、cu128/cu130及driver差异披露。NGA/Huber须64项完整验收后按冻结四条件评分规则选recipe，再过真实图像门检，不能以62项观测启动192覆盖。FL下一批non-IID FedSA10已终轮，先按原strict和离机核验接纳，不把77终轮当77验收。

CPU有界诊断：tmp/cpu_throughput_readonly_20261011/ROOT_DIAGNOSIS.json。15.18秒四采样观察约12个活动单线程worker及12.09/122.88核的cgroup平均用量，双GPU均100%；未采到磁盘读取或新增配额节流，符合冻结worker/thread结构。尚不能据此证明具体算子瓶颈或最优吞吐；未改数值、并发或实例设置，利用空闲CPU处理已闭合证据。

## 保存、Git与巡检

大文件只写F:/YananResearchStorage/GuardFed，写前核F为Yanan 2TB/Healthy且容量足；E仅代码、配置、索引与精简报告。模型/原始数组/归档不进Git，服务器大文件优先原地保留。

最近已验证推送fb1f31031aad89206f252969a6c26af6cde9efa4，截止以publication_closed_increment62_verified_20261011.json为准（245提交文件哈希及远端分支通过）。本次341/340/F40、统一工作稿和入口刷新在该提交之后，待Git63实际发布，不能提前称已上传。三小时聊天任务guardfed-training-health仍PAUSED，本会话无automation_update接口；未建替代cron/Windows任务。supervisor训练与聊天巡检分开。

入口写入器：tmp/update_increment63_F40_entries_20261011.py。旧生成器只保留历史截止，下方历史原字节保留。

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
