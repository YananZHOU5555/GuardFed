# CURRENT: GuardFed返修实验

入口更新：2026-10-11T06:46:49.839641+00:00。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。结果采用与实时观测分开。

| 阶段 | 严格验收、F盘SHA与root采用 | 最新实测及边界 |
|---|---:|---|
| 机制训练 | 新400/800，U/C/A/F各100；Full100复用 | 2026-10-11T06:43:59.720224+00:00：416终轮、8活动、376等待、训练失败0；其余4类继续 |
| 统一CUDA首阶段 | 已采用7/500；服务器严格保存验收11/500 | 2026-10-11T06:44:28.351509+00:00：新目录已恢复Full/IID/Benign/91005，diff0；当前91006，488等待，当前失败0 |
| 历史CPU三视图 | 机制380（U/C/A100、F80）、FL100、Hybrid20 | 原620数值失败与旧表保留，不重启原CPU队列；新CUDA表独立产生 |
| 主比较表 | 11方法native1100，三页PDF | IID/non-IID各5场景×10seed，另9/6seed；历史评价协议，非最终17方法或1100独立训练 |
| 组合控制训练 | native29/96新；4screen复用另计 | IID FedSA10增量严格离机闭合；最新32终轮，不将观察计入接受 |
| Fed-NGA/Huber | NGA新9/96采用，screen4另计 | 最新15终轮，已有9条恒定负类预测保留；Huber96尚未启动 |

作者实际批准完整比较范围统一CUDA，包括Full。不重训、不改参数/容差、不访问test。统一原CNN clean-root及valid确定性FP32 CUDA前向；原后处理实现与设备保留，包括GuardFed NumPy阈值和LoGo netcal CPU。协议：docs/server_deployment_20260923/training_20260923/celeba_uniform_cuda_evaluation_v1/PROTOCOL.md。首阶段严格冻结500=400控制+Full100；Full与方法表共享同一CUDA结果。完整目标2500逻辑记录仍待六方法/四控制及适配，不能冒充已冻结完整清单或已完成。

首固定7原Linux whole验收native三指标最大差均0；离机零拟合63指标/168计数一致，F归档110成员逐SHA，root tmp/uniform_cuda7_saved_audit_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json。此前minus_F/non-IID/Sp-DFA/91004已正式通过，不是把诊断回填。旧CPU模型阈值差导致的30张预测变化及原1e-12失败完整保留，不证明所有模型环境等价。

第一次CUDA队列10项完成后，Full/IID/Benign/91005后置资源检查读取/proc/113052/cwd发生PermissionError，receipt原生差0但缺binding/ACCEPTED，旧项未采用。修复仅当场确认进程消失/Z/X才跳过，存活不可读仍拒绝；未证明历史zombie根因。独审+16边界检查/9元数据检查通过，原科学worker文本不变。新service guardfed_uniform_cuda500_resume1_valid，BASE /workspace/guardfed_checks/uniform_cuda_formal_20261011_resume1；原10CUDA源/模型/数组/binding/stat全核后复用，失败项仅一次有限恢复已通过，剩余原顺序490不自动重试。源/实际启动与观察：tmp/uniform_cuda500_root_dispatch_20261011/RESUME1_INSTALL_COMMAND.json，tmp/uniform_cuda500_root_dispatch_20261011/RESUME1_OBSERVATION_20261011T064429Z.json。runner的root_offserver_accepted=0只是runner不管理本地采用；root真实已接受7，分开记录。

Hybrid原19+新增IID FedSA10=29已采用，root tmp/celeba_hybrid_native29_IIDFedSA10_root_adoption_20261011/ROOT_ADOPTION.json；304归档成员/10模型80张量核验，CPU108释放；三视图仍20，不能称全100完成。FLGMM完整native100与CPU三视图100已闭合，Windows零拟合的一项root group_kl诊断差-2.1684e-19保留，不称Windows whole/refit通过。主表 outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md；FL表 outputs/guardfed_tables/celeba_flgmm_ten_scenes100_20261011/TABLES.md。

完整24条英文回复为 docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md，固定380/F80、FL100、Hybrid20/native11截止，未用首7局部结果替代完整多seed证据。仍保留准确率代价、LoGo较低ASPD、FLTrust较高准确率；不称每个评分项不可或缺或全面领先。旧TableII全部480格有来源但294格n1的SD不可计算；FairGuard/IID/FedSA投稿59.13/历史54.13/round70候选53.94须区分。数值追溯不等于忠实17方法/统一十seed。

Git69提交0a276220b447645020e8dca8ee092ce411367bb7已核251blob；固定native400/views380/FLnative100/views90/NGA1/Hybrid19。其后NGA9、GPU诊断、FL100、新回复、作者CUDA批准、首7与Hybrid29尚未发布，不改旧回执。

2026-10-11T06:43:59.720224+00:00 双5090 100/100%、67/62°C，Recovery None；CPU实测11.03核/122.87999配额，OOM0，内存70.20/519.17GB，磁盘1.059TB余。GPU计算限制CPU有效占用，独立可用任务继续；不以无意义负载填占用。大文件留服务器，最小离机证据仅写前核Yanan2TB的F:/YananResearchStorage/GuardFed，禁止内置盘批量回退。三小时聊天automation仍PAUSED，无可调用更新工具，未新建替代监督；现存supervisor队列正常执行不等于聊天持续巡检。

尚待六基线、四控制、完整统一CUDA表、主端点/最终评价、FedWA/SmartFL/FedDNA忠实资料、匹配投稿LaTeX整合及最终发布。当前valid，官方test历史暴露，不能称未触碰test。Huber恒等投影为CNN适配；LoGo20虚拟cohort仅人口适配；DP指demographic parity。整个返修仍未完成。

本次原STATE/入口/reader备份在tmp/root_CUDA7_resume1_Hybrid29_20261011；历史尾逐SHA保持。

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
