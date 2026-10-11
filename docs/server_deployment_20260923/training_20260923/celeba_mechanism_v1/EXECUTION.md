# CURRENT: GuardFed返修实验

入口更新：2026-10-11T10:01:52.359879+00:00。当前服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。以下将实际采用与带时间的运行观测分开。

| 阶段 | 严格验收、F盘SHA与root采用 | 最新实测及边界 |
|---|---:|---|
| 机制训练 | 新400/800，U/C/A/F各100；Full100共享复用 | 2026-10-11T09:43:16.913428+00:00：447终轮、8活动、345等待、失败0；其余4类继续，观测不增采用 |
| 统一CUDA首阶段500 | 本地去重采用105/500=Full100+旧5条控制 | 2026-10-11T09:48:54.194745+00:00：服务器保存验收108/500（旧57+新51），392等待；失败0，不等同离机采用 |
| 统一CUDA原八方法800 | 本地去重采用102/800；不含Full/FL/LoGo | 2026-10-11T09:43:14.530458+00:00：服务器277/800（原固定3+后续274），523等待；失败0，不等同离机采用 |
| 统一CUDA FLGMM100 | 全部100已采用；新97+原3，非103 | 原Linux完整验收；离机900指标/2400计数/300规则，native最大差0；旧F引用不重复运输 |
| 统一CUDA LoGo100 | 全部100已采用；新98+原2，非102 | 原冻结CPU后处理，1986700预测/300指标/800计数误差0；无新CNN/fit/refit；1条恒负结果保留 |
| 历史三视图 | 机制CPU380、FL CPU100、Hybrid CPU20 | 既有CPU或混合路径基线表和原620数值失败保留；新CUDA表独立产生 |
| 主比较表 | 历史11方法native1100，三页PDF；新四方法CUDA native400阶段表 | 新阶段表已独立算术检查并根采用；两表均非最终17方法表，本次表采用不新增科学记录 |
| 组合控制训练 | native29/96新；screen4另计 | 2026-10-11T09:43:16.913428+00:00：37终轮，仅观察；历史三视图20 |
| Fed-NGA/Huber | NGA新19/96采用；screen4另计 | 2026-10-11T09:43:16.913428+00:00：27终轮；已采用19条恒负预测保留；Huber96尚未启动 |

作者批准完整比较范围统一CUDA，包括Full。不重训、不改参数/容差、不访问test。原CNN clean-root及valid使用确定性FP32 CUDA前向；原冻结后处理实现与设备保留。协议：celeba_uniform_cuda_evaluation_v1/PROTOCOL.md。实际冻结1500条互不重复记录（500+800+FL100+LoGo100）；本地CUDA去重采用407=105+102+100+100，不是411。首500内原7与Full100重叠2条，因此首阶段为105，不能将7+100直接相加。Full100与方法表共享一次。完整目标2500仍待六方法/四控制及适配，不能称全部冻结或完成。

Full100根采用：{'path': 'tmp/Full100_cuda_saved_closure_prepare_v2_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json', 'sha256': '0d33ecb624323f0f334debcb12f888886f85410cd9b0d5607c687dbb6575cc08'}。新98+旧2、原Linux whole saved gate与离机900指标/2400计数/300规则、609新归档成员SHA通过，native最大差0；不是完整500采用。FedAvg100与旧800固定3重叠一个FedAvg，因此该800范围唯一采用102。FL100根采用：{'path': 'tmp/FL100_cuda100_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json', 'sha256': 'ea935ead7da50ba7eff0dd7e4e4b289ca9365427096ad8431808ae4d76acb751'}，只运输97新NPZ并保留旧3引用。LoGo100根采用：{'path': 'tmp/LoGo100_cuda100_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json', 'sha256': 'c926f7ee3cd6dea31f178d3a6eab3661826f9ecd4e7c90e54d3027e6b94c1317'}，原Linux冻结后处理与离机零拟合保存预测算术检查通过；不是离机重新执行整个netcal预测。LoGoFair-DP使用20个固定image-ID虚拟人口、fitseed1719及原CPU netcal1.3.6，DP指demographic parity，不能称真实client公平性或差分隐私。IID Benign seed91001恒定负预测保留，零差异不能单独解释为公平成功。

LoGo712份原产物、FedAvg100 Linux index actual002和独立CPU环境的既有恢复链保留。首次环境安装错误引入torch2.14.1，失败未覆盖；修复后保持冻结直接包版本，实际Python3.10.20相对原3.10.4的差别仍披露，不声称全部环境等价。原fixed2采用后才启动98；本次100完整闭合不重新选择/拟合原后处理、不重复CNN。500旧/proc读取失败及800模块冲突/资源读取失败原件保留；原科学worker未改，无自动重试或追认诊断。

四方法部分表：tmp/celeba_uniform_cuda_fourmethod_partial_20261011/actual_partial400/TABLES.md，build receipt 14ab89c336f540dd1109f1bd8d07b706096dd52db78e7bd0e8acf25885328f83，TABLES SHA09157fd0dad7047fd05e7fb0e54ff7c852532a5cdcf3743e26ef91a6ccd2b1b8，独立检查154c53a04f6216f0e8a651506d9ff648986f9cf0a422636b8e1432a74ddb09d7及根表采用515724881bcad4903b0e2dd011d5c9646916c7e8c3eed7141aa879891f2e697e。它只包含FedAvg、GuardFed-AD2+、FLGMM、LoGoFair-DP各100条CUDA native记录，已生成并独立验收；不是三视图全比较或最终17方法表，不设新主端点，不新增科学采用。FedAvg与LoGo各1条恒定预测保留。

Git73已实际提交f1148225d99e5b5f469e18bcd3d44f0a1c817875，246个blob及remote分支核验通过，回执SHAdd7145de8cc8b9ec311e545836204e7dfd83b2e96e018195a086bcf7f3efcdd6，截止仍211采用。Git72 pre-Git漏SUFFIX_FIX的拒绝和精确清单修复保留；旧回执不追改。本次Full100/LoGo100完整采用及407入口未发布Git74。模型、NPZ、数据和大归档不进Git。

2026-10-11T09:43:16.913428+00:00 资源快照：双5090利用率[100, 100]%，温度[67, 62]°C，Recovery None；CPU实测12.04/122.87999核，cgroup OOM0，RAM 78.19/519.17GB，服务器磁盘1.0444TB余。它与500/800快照时间不同，不作持续巡检。CPU低有效占用保留GPU瓶颈判断，不造重复负载。大文件仅留服务器或核验Yanan2TB后的F:/YananResearchStorage/GuardFed，不回退内置盘。三小时聊天automation仍PAUSED，无可调用更新工具，未创建替代监督。

完整24条英文回复：docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md，固定380/F80、FL100、Hybrid20/native11截止，原SHA910b263不变；后续部分结果未偷偷写入旧稿。十一方法native1100历史表仍保留准确率代价、LoGo较低ASPD和FLTrust较高准确率，不称每评分项不可或缺或全面领先。

仍待六基线、四控制、完整统一CUDA比较、主端点/最终评价、FedWA/SmartFL/FedDNA忠实来源、匹配投稿LaTeX整合和最终发布。旧TableII480格有来源，294格n1的SD不可计算；FairGuard/IID/FedSA投稿59.13/历史54.13/同终轮候选53.94区分保留。当前valid，官方test有历史暴露，不能称未触碰test；Huber恒等投影仅CNN适配，不继承原理论保证。整个返修尚未完成。

本次STATE/五入口/reader/protocol旧bytes将在tmp/root_CUDA407_Full100_LoGo100_Git73_20261011备份；五历史尾逐SHA保持。

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
