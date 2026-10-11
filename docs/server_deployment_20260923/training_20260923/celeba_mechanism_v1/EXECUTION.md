# CURRENT: GuardFed返修实验

入口更新：2026-10-11T07:59:14.454238+00:00。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。实际结果采用与运行观测分开。

| 阶段 | 严格验收、F盘SHA与root采用 | 最新实测及边界 |
|---|---:|---|
| 机制训练 | 新400/800，U/C/A/F各100；Full100复用 | 2026-10-11T07:54:58.907976+00:00：427终轮、8活动、365等待、训练失败0；其余4类继续 |
| 统一CUDA首阶段500 | 本地采用7/500；服务器保存验收58/500 | 2026-10-11T07:55:23.702665+00:00：原57逐SHA/stat复用，剩余443有限恢复，已新验收1；当前失败0 |
| 统一CUDA原八方法800 | 本地采用3/800；服务器保存验收23/800 | 2026-10-11T07:54:56.496822+00:00：固定3闭合后其余797已启动，新通过20、777等待，当前失败0；不含Full/FL/LoGo |
| 历史CPU三视图 | 机制380（U/C/A100、F80）、FL100、Hybrid20 | 原620数值失败与旧表保留；新CUDA表独立产生 |
| 主比较表 | 11方法native1100，三页PDF | IID/non-IID各5场景×10seed，另9/6seed；历史评价协议，非最终17方法或1100独立训练 |
| 组合控制训练 | native29/96新；4screen复用另计 | 最新34终轮，不把观察计入接受；三视图20 |
| Fed-NGA/Huber | NGA新9/96采用，screen4另计 | 最新20终轮；9条已采用恒定负类预测保留；Huber96尚未启动 |

作者批准完整比较范围统一CUDA，包括Full。不重训、不改参数/容差、不访问test。统一原CNN clean-root及valid确定性FP32 CUDA前向；原冻结后处理实现与设备保留。协议及追加运行记录：docs/server_deployment_20260923/training_20260923/celeba_uniform_cuda_evaluation_v1/PROTOCOL.md。目前两批实际冻结1300条互不重复记录，本地CUDA采用10=7+3。Full100仅在首500中，与方法表共享一次。完整目标2500仍待六方法/四控制及适配，不能冒充齐备或已完成。

首7原Linux whole验收native三指标最大差0，离机零拟合63指标/168计数一致，F归档110成员逐SHA；root tmp/uniform_cuda7_saved_audit_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json。原八方法固定3（FedAvg、FedAA、LASA）亦均native差0，离机27指标/72计数/9预测规则一致，F归档85成员逐SHA；root tmp/old800_cuda3_saved_audit_prepare_20261011/mixed_producer_v2/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json。其余尚只计服务器保存验收。旧CPU阈值差造成30张预测变化及原1e-12失败保留，不称所有环境等价。

500队列resume1在累计57项后遇/proc/117589/cwd资源读取错误；47为该次新增而非累计。失败Full/non-IID/Benign/91002有receipt/binding但无ACCEPTED，旧部分未采用。独审后的resume2只对明确非Python进程免读cwd，所有PID/task affinity及名义预算仍检查，未知/不可读Python仍拒绝；科学worker原文不变。旧57全核后复用，失败项新目录有限重评已通过、native差0。service guardfed_uniform_cuda500_resume2_valid，BASE /workspace/guardfed_checks/uniform_cuda_formal_20261011_resume2；原失败证据不覆盖，无自动重试，历史进程类别/状态未知。

800批原启动模块命名冲突、第二次资源检查失败均保留。第三目录复用已有FedAvg单例、新做FedAA/LASA两例，固定3本地采用后才实际启动797。service guardfed_uniform_cuda800_remaining_v3_valid，BASE /workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt003。FL100调用器及LoGo100缓存适配仍是源准备/审查，不称实际运行；LoGo须等FedAvg CUDA100完整采用缓存，原冻结netcal后处理不重新择优/拟合。

Hybrid29、NGA9、机制native400、历史CPU三视图380/FL100/Hybrid20严格采用量不因运行观测增加。主表 outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md；FL表 outputs/guardfed_tables/celeba_flgmm_ten_scenes100_20261011/TABLES.md。完整24条英文回复为 docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md，固定380/F80、FL100、Hybrid20/native11截止。保留准确率代价、LoGo较低ASPD、FLTrust较高准确率，不称每个评分项不可或缺或全面领先。

Git70提交4fccc8b8d1d56492ac9cffd0cf27ca390e78bd20已核197文件blob及remote，父0a276220；含上述固定截止稿、NGA9、作者CUDA批准、CUDA首7与Hybrid29等代码/精简证据。发布回执 docs/server_deployment_20260923/training_20260923/publication_closed_increment70_verified_20261011.json。07:55实时观察、800固定3及其后恢复源尚未随此旧提交发布，不改旧回执。无模型/NPZ/数据/大归档入Git。

2026-10-11T07:54:58.907976+00:00 双5090 99/100%、65/62°C，Recovery None；CPU实测12.02核/122.87999配额，OOM0，内存70.15/519.17GB，磁盘1.059TB余。GPU计算限制CPU有效占用，独立可用任务继续。大文件留服务器，最小离机证据只写核验Yanan2TB后的F:/YananResearchStorage/GuardFed，不回退内置盘。三小时聊天automation仍PAUSED，无可调用更新工具，未新建替代监督；supervisor实际运行不等于聊天持续巡检。

仍待六基线、四控制、完整统一CUDA表、主端点/最终评价、FedWA/SmartFL/FedDNA忠实资料、匹配投稿LaTeX整合及最终发布。旧TableII480格有来源，294格n1的SD不可计算；FairGuard/IID/FedSA投稿59.13/历史54.13/round70候选53.94区分保留。当前valid，官方test历史暴露，不称未触碰test；Huber恒等投影仅CNN适配，LoGo20虚拟cohort仅人口适配。整个返修尚未完成。

本次原STATE/入口/reader/protocol备份在tmp/root_CUDA10_two_queues_Git70_20261011；五处历史尾逐SHA保持。

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
