# CURRENT: GuardFed返修实验

入口更新：2026-10-11T09:27:12.686305+00:00。当前服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。以下将实际采用与运行观测分开。

| 阶段 | 严格验收、F盘SHA与root采用 | 最新实测及边界 |
|---|---:|---|
| 机制训练 | 新400/800，U/C/A/F各100；Full100共享复用 | 2026-10-11T09:19:26.732679+00:00：440终轮、8活动、352等待、失败0；其余4类继续 |
| 统一CUDA首阶段500 | 本地采用7/500；服务器保存验收95/500 | 2026-10-11T09:19:12.925665+00:00：旧57复用、新38通过、405等待；失败0 |
| 统一CUDA原八方法800 | 本地去重采用102/800；服务器保存验收220/800 | 2026-10-11T09:19:18.548282+00:00：原固定3加剩余217通过，580等待；失败0；不含Full/FL/LoGo |
| 统一CUDA FLGMM100 | 全部100已采用；新97+原3，非103 | 原Linux完整验收；离机900指标/2400计数/300规则，native最大差0；959新归档成员SHA与28旧F成员核验 |
| 统一CUDA LoGo100 | 固定2已采用，剩98实际启动 | 09:23 UTC原队列启动；2条39734预测/6指标/16计数重算误差0；CPU后处理保持冻结，不重新fit |
| 历史CPU三视图 | 机制380、FL100、Hybrid20 | 旧620数值失败与旧表完整保留，新CUDA表独立产生 |
| 主比较表 | 11方法native1100，三页PDF | IID/non-IID各5场景×10seed，另9/6seed；非最终17方法或1100独立训练 |
| 组合控制训练 | native29/96新；screen4另计 | 最新37终轮，仅观察未增采用；三视图20 |
| Fed-NGA/Huber | NGA新19/96采用；screen4另计 | 最新25终轮；已采用19条恒定负类预测保留；Huber96尚未启动 |

作者批准完整比较范围统一CUDA，包括Full。不重训、不改参数/容差、不访问test。统一原CNN clean-root及valid确定性FP32 CUDA前向；原冻结后处理实现与设备保留。协议：celeba_uniform_cuda_evaluation_v1/PROTOCOL.md。目前实际冻结1500条互不重复记录（500+800+FL100+LoGo100），本地CUDA去重采用211=7+102+100+2。Full100仅在首500中，与方法表共享一次。完整目标2500仍待六方法/四控制及适配，不能称全部冻结或完成。

FedAvg100根采用：tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json。900指标/2400计数/300规则和1058归档成员SHA通过；与旧800固定3重叠一个FedAvg，唯一采用102。FL100根采用：tmp/FL100_cuda100_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json；只运输97新NPZ，旧3原F引用，不重复下载。LoGo固定2根采用：tmp/LoGo_cuda_fixed2_saved_audit_prepare_v2_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json。原Linux完整预测与离机零拟合算术检查均通过；不是离机重做整个netcal预测。Benign恒负类结果保留，不能把零差异单独解释为公平成功。

LoGo712份原产物已逐SHA恢复；FedAvg100 Linux索引actual002已实际通过。独立CPU环境首次安装错误引入torch2.14.1，已保留失败并以原包约束修复；实际Python3.10.20与原3.10.4差别披露，netcal1.3.6/torch2.8CPU保持原直接包版本，两例实际检查通过但不声称全部环境等价。LoGo部署原科学代码未变；fixed2采用后才启动原序98，service guardfed_uniform_cuda_LoGo100_remaining_valid，BASE /workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001。既有712与FedAvg缓存不重复运输，不重新选择/拟合原后处理。

500旧两次/proc资源读取失败保留；resume2仅对明确非Python进程免读cwd，所有PID/task affinity及名义预算仍核，未知不可读Python仍拒绝。旧57逐SHA/stat复用，失效项新目录通过native差0。service guardfed_uniform_cuda500_resume2_valid，BASE /workspace/guardfed_checks/uniform_cuda_formal_20261011_resume2。800旧模块冲突与资源检查失败保留；attempt003原科学worker不变，service guardfed_uniform_cuda800_remaining_v3_valid。无自动重试或失败证据覆盖。

Git72已提交51a27b50c6c1880d3fb3c6b5c45b750d0cc68d12，258个blob、remote分支和clean核验通过，父ba302df5；含FedAvg100、FL3、NGA19与08:38截止入口等。72发布前清单漏SUFFIX_FIX的拒绝发生于Git写入之前，补精确封条成员后发布成功，失败和修复证据保留。后续FL100完整采用、LoGo实际闭环及本次入口待Git73；不追改旧回执。模型、NPZ、数据和大归档不进Git。

2026-10-11T09:19:26.732679+00:00 双5090 98/100%、66/64°C，Recovery None；CPU实测12.02核/122.87999配额，OOM0，内存75.02/519.17GB，服务器磁盘1.0445TB余。活动机制轮次持续增长。CPU低有效占用已定位GPU计算限制，继续独立有效任务，不改协议或重复作业提高占用。大文件优先留服务器；最小离机证据只写核验Yanan2TB后的F:/YananResearchStorage/GuardFed，不回退内置盘。三小时聊天automation仍PAUSED，无可调用更新工具，未创建替代监督；supervisor持续运行不等于聊天持续巡检。

完整24条英文回复：docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md，固定380/F80、FL100、Hybrid20/native11截止，原SHA910b263保留，不把后续部分结果偷偷写入该截止稿。IID/non-IID十一方法表：outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md。保留准确率代价、LoGo较低ASPD和FLTrust较高准确率，不称每评分项不可或缺或全面领先。

仍待六基线、四控制、完整统一CUDA表、主端点/最终评价、FedWA/SmartFL/FedDNA忠实来源、匹配投稿LaTeX整合和最终发布。旧TableII480格有来源，294格n1的SD不可计算；FairGuard/IID/FedSA投稿59.13/历史54.13/round70候选53.94区分保留。当前valid，官方test历史暴露，不称未触碰test；Huber恒等投影仅CNN适配，LoGo20虚拟cohort仅人口适配。整个返修尚未完成。

本次STATE/入口/reader/protocol旧快照在tmp/root_CUDA211_FL100_LoGo2_Git72_20261011；五处历史尾逐SHA保持。

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
