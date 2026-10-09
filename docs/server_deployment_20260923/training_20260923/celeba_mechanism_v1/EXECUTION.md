# CURRENT EXECUTION: GuardFed返修实验 — 实测 2026-10-09T13:01:13.498063+00:00

当前服务器：ssh -p60350 root@89.22.197.55，实例52183675；repo /workspace/GuardFed-celeba-expanded。用户明确授权停止sglang，模型/文件保留。213.224.31.105:26712当前内部状态未知，不自动切换。先遵守/etc/vast-agents-guide.md，既有SHA为42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa。

## 当前执行与验收

| 阶段 | 实际状态与分母 | 接续入口 |
|---|---|---|
| CelebA机制消融 | 当前观测完成56、活动8、等待736、失败0；已独立严格验收并离机54/800新增，另100 Full显式复用 | server_reactivation_20261009/latest_formal_live.json；celeba_mechanism_v1/EXECUTION.md及dispatch receipt |
| FLGMM验证搜索 | 6/32已严格验收并离机；13:03实测11终轮、2活动、0失败，新增5项待验收，不能计为接受 | tmp/celeba_flgmm_screen_20261009_v2_dispatch/LATEST_BACKUP.json及observations/bounded_review_20261009T130206Z/STATUS.json |
| 组合基线验证搜索 | 原32项队列已启动；最新离机快照尚无完整70轮接受，不把首轮或文件存在计作完成 | tmp/celeba_hybrid_screen_execution_20261009/results_incremental_20261009T1133Z/STATUS.json |
| 九方法旧checkpoint三视图评价 | 460/900已严格验收并离机；原CPU872服务因native偏差failstop EXITED，不重启 | tmp/celeba_valid_gpu_remaining464_execution_20261009/partial2_explicit_import/cumulative_460_accepted.json |
| 机制三视图评价 | 23份minus_U已严格验收并离机，另行统计；不是800份均已完成三视图 | server_reactivation_20261009/MECHANISM_VALID_INCREMENTAL_20261009T104749Z_ROOT_VERIFICATION.json |

主机制服务guardfed_celeba_mechanism_formal，固定70round/valid-only/8并发，IID(alpha5000)/non-IID(alpha5)×5场景×10共享seed；100 Full身份已复核，旧权重不重训/重复打包。FLGMM服务guardfed_celeba_flgmm_screen，两张GPU各1任务；组合基线服务guardfed_celeba_hybrid_screen32，GPU0/CPU104单线程。两套32搜索均固定8候选×四条件、seed91001，尚未完整选recipe或启动100项多seed确认，不运行test。

主机制最近实测CPU 11.09/122.88核，RAM 74.64GB，磁盘余1.062TB；GPU/温度/RecoveryAction与近期错误读同一实时JSON。只在真实轮次/日志、进程身份和资源证据支持时判断健康，低瞬时占用不重启。服务标签与完成文件不代替验收。

## 当前恢复与研究选择

原CPU失败为FairGuard/IID/F Flip/seed91009：native超原1e-12，65成员失败现场完整保留，原chunk036的10份strict partial当时未登记，后来通过显式审阅导入派生436账本。独立单模型GPU诊断已复现原三指标，差值全0；当前CPU/GPU native/raw只有image172599一处翻转，共享校准预测无翻转。三份归档与保存数组已独立验收；缺历史GPU逐图数组，不声称唯一历史根因，原424账本保持不变。凭据NATIVE_GPU_DIAGNOSTIC_ROOT_VERIFICATION.json。

旧436来源链及新GPU队列的前2批已严格验收、离机登记，累计460/900（CPU434、GPU26）；import11新增CNN推理0，旧424/425/436账本及原CPU失效现场不改。原464范围按43批次、每批至多11条、单GPU/单线程派发；实际CPU106协调、CPU105 worker/nice10已核。已登记与远端闭合分开统计，保存预测按原规则重建9指标、24混淆计数；root重拟合核验引用原远端strict，未声称本机重新拟合。第三批在worker资源预检、CNN之前触发Protected main800 health failed而自动停下。当前主训练正常，但失败瞬间的三个健康条件未保存原始截图，不能唯一归因为任务交接。原服务未重启，两条成功partial随后通过原strict部分验收与保存数组复核显式登记，新增CNN推理0；仍缺440条，修复需独立版本和不重复已完成项的补集。独立V2工程修复已通过root源码差异审阅、60个健康组合、11个资源拒收边界及两份实际快照schema核验，仅资源预检与输入保全变化，原科学body/strict及24个成员字节不变；尚未上机，新440补集需独立命名空间。 native1e-12、原model/source/data/map/root/valid、同checkpoint全部视图与失败保留规则不变；混合CPU/GPU来源不能冒充统一设备的最终公平比较。入口tmp/celeba_valid_gpu_recovery_implementation_20261009/README.md。监控不自动启动准备包。

LoGoFair虚拟人口映射提案已独立核验：四条件共用固定image-ID哈希20组，root/valid顺序相同，80个root(label,Male)格最小116人；未解码valid标签/score、未拟合或评价。虚拟群体不是原训练client，人口定义仍待用户裁定；原32草案的mapping SHA仍null。入口tmp/celeba_logofair_population_proposal_20261009/REPORT.md。

## 已完成证据与剩余交付

2454项历史新增训练、九方法900原始验证结果及旧备份保持原值；900原始训练结果完整，不等于900统一评价重放完整或最终test已完成。旧TableII的480个原值已追溯实际重复数，缺乏依据的SD不补造。完整入口REBUTTAL_COMPLETION_20261009.md；英文rebuttal已对齐24块原意见、40处本地引用及209项SHA声明，尚不能把待补实验写成完成。

原20项cu130与另20项cu128真实图像三轮门检已严格接受、离机，两套Full与原worker短程张量/指标/诊断精确。Fed-NGA/Huber四条真实图像三轮门检含240梯度/攻击oracle已接受；FLGMM的CPU2/GPU4及组合基线的CPU4/CUDA4门检已接受。所有恒定负类和工程/数值失败保留。三轮门检不证明70轮跨环境等价或科学性能优势；门检服务已EXITED，不重启。

完整17行比较仍缺8方法的完整多seed结果：LoGoFair、Fed-NGA、FedWA、Huber、FLGMM、SmartFL、FedDNA及组合控制。梯度方法正式协议、LoGoFair人口和最终评价主终点/测试边界仍待裁定；FedWA/SmartFL/FedDNA忠实规格仍缺，不能用简化旧分支冒充。主机制800、完整机制三视图、冻结最终评价、正文及最终回复仍未完成。Fig3原脚本/ForestDiffusion执行身份仍缺；已核数值与缺失来源明确区分。

已接受场景的10/9/6种子中期论文表：celeba_mechanism_v1/interim_tables_20261009T124918Z/TABLES.md。仅展示5个齐备的Full–minus_U配对场景，保留所有指标及取舍，不补造未完成场景，不以Full最佳seed对比消融均值。新增Sp-DFA场景Full准确率较高、去U的两个公平性差距更低，不能声称每项不可或缺。AEOD为绝对TPR差，不是完整equalized odds；Full98cu128+2cu130、多数旧driver570.211.01和当前driver595.84差异、seed91001选择历史均披露。native含各方法原校准，不能据此单独证明聚合机制。

## Git与巡检

最近已验证推送：e464f773a628dc49515d2433ca835f94356cc55d，分支codex/revision-evidence-baselines-20260928，76份committed blob逐SHA及远端分支核验；后续本机变化未自动算作已推送。记录publication_closed_increment12_verified_20261009.json。

三小时聊天任务guardfed-training-health仍PAUSED；本会话没有原生automation_update工具，未编辑调度器或建立替代cron/Windows任务。supervisor持续运行训练不等于聊天巡检恢复。待原生接口可用时按server_reactivation_20261009/MONITOR_HANDOFF.md恢复同一任务；不从历史计划自动派发新队列。

下方为历史记录；当前事实以本入口、TRAINING_STATE.json及对应实际凭据为准。源/数据/冻结配置保持一致、无重复worker且已验收项严格跳过时才有限恢复外部中断；数值/逻辑错误保留证据，不循环重试、不改统计/seed/并发或driver/实例。

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
