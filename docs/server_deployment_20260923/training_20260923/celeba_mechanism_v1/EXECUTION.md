# CURRENT — GuardFed revision evidence (2026-10-11T03:03:10.744004+00:00)

用户已授权补全返修实验。Huber 采用恒等投影，明确 CNN 项目适配且不沿用原约束域理论保证；LoGoFair 使用固定图像 ID 的20虚拟 cohort，仅报告人口适配。重大主评价端点和 final-test 边界仍待作者决定；不能保证方法全面获胜。

当前严格采用：机制 native380，三视图370（U100/C100/A100/F70，Full100显式复用）；F70七场景表；FLGMM native86新增＋4旧screen引用、三视图90/九场景论文表；Hybrid native19新增、三视图20/两场景表；梯度调参64。native380新 FedSA10已严格核验和离机备份，但其三视图/F80在此截止尚未采用。保留所有负结果、恒定预测、失败和准确率代价。10/9/6固定种子面板，sampleSD(ddof1)，不做最佳单次对基线均值。

最新实测 2026-10-11T03:00:37.483572+00:00：主队列380/800终轮、8活动、412待运行、当前失败0；FL90/96新增终轮，Hybrid26/96新增终轮，远端remaining620已闭合200但不自动等于离机采用。双GPU100%、65/63°C、约6GB显存各卡，cgroup OOM0，磁盘余量约1.059TB。以上运行计数仅观察，科学采用以对应root proof为准。观察：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261011T030032Z.raw.json。

14个 CPU三轮部署门已严格通过及离机核验，457成员/14完整state/112张量；只证明该短程实现与身份检查，科学性能表记录0。Fed-NGA96已按冻结runner启动，02:50首任务第3轮、parent108479/child108499；70轮科学采用0。Huber96保持随后串行，尚未启动。NGA首次启动的输出父目录接线错误及一次有限修复证据保留，训练数值逻辑未改。旧服务器213实例不启队列，当前SSH60350 root@89.22.197.55。

完整24点回复工作稿：docs/server_deployment_20260923/revision_20260923/rebuttal_working_F70_FL90_Hybrid20_release_20261011.md（0b300bd7b72001f8804d34da4aab824e89ed1285fc78209a2609b9ef03fc1b80），24原话块/38引文行顺序exact。该稿固定native/views370、F70、FL90、Hybrid20及gradient64，故与较新的native380有明确截止差异；尚未应用提交LaTeX。FL新增S-DFA10 native ACC89.17±1.15%、AEOD0.0423±0.0048、ASPD0.1083±0.0105；共享校准ACC88.78±1.17%、AEOD0.0045±0.0039、ASPD0.0659±0.0080。一次group_KL诊断差−2.1684e−19保留，不称跨平台逐位校准相等。原Windows47 exact-refit失败仍保留为失败。

当前已验证Git66：181969f241d938a93e1e72ac0245a85ae39b3e73，455个committed blobs逐个SHA及remote分支通过。其固定截止native/views370、F70、FL81views、Hybrid20；新native380/FL90/whole14/NGA运行接线等待下一次独立发布，不改写Git66事实。

仍需：完成剩余机制和新基线多seed覆盖、后续三视图验收与表；FedWA/SmartFL/FedDNA忠实定义待补齐；最终主评价端点与冻结final-test决策；匹配提交版LaTeX及rebuttal整合。TableII旧480格追溯中的单次结果不能伪造SD。三小时监控最后实测为暂停，当前编辑工具不可用，不能声称已启用；服务器Supervisor独立继续。大型原始数据/权重/归档只留服务器或经检查的F:/YananResearchStorage/GuardFed，不回退内置盘。

本段写入器：tmp/refresh_native380_FL90_gradient14_NGA_current_root_20261011.py。历史尾部逐字保留；本次只更新已有证据入口，无训练/统计/科研验收。

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
