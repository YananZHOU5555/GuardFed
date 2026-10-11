# CURRENT — GuardFed revision evidence (2026-10-11T03:20:15.858260+00:00)

用户已授权补全返修实验。Huber采用恒等投影，标为CNN项目适配且不沿用原约束域理论保证；LoGoFair采用固定图像ID的20虚拟cohort，仅称人口适配。主评价端点和final-test边界仍待作者决定，不能保证方法全面获胜。

严格采用：机制native/views380（U100/C100/A100/F80，Full100显式复用）；F80八场景/160记录表，10/9/6种子及sampleSD(ddof1)、paired差异完整，1296统计标量/648显示格；旧F70的140对象、1134统计标量、567单元原样保留。FLGMM native86新增＋4screen引用、三视图90及九场景表；Hybrid native19新增、三视图20/两场景表；gradient搜索64。控制400和F20尚未闭合，不称整个机制完成。

最新五队列实测2026-10-11T03:10:16.129291+00:00：主队列383/800终轮、8活动、409待运行、失败0；FL90/96新增终轮，Hybrid26/96新增终轮，三视图远端203闭合但本机仅380总采用，观察不等于验收。记录：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261011T031011Z.raw.json。两张5090正常，资源以该实时快照为准，低瞬时利用率不重启健康任务。

14个CPU三轮部署门已严格离机核验（457成员、14完整state、112张量），性能表0记录。NGA96冻结runner运行，2026-10-11T03:08:56.993139+00:00首任务70轮已落result，下一任务第9轮，当前failure0；科学接受仍0，不能把终轮文件存在当验收。Huber96未启动，按原定随后串行。首次输出父目录接线失败及一次有限修复保留，训练源码/数学/参数/seed未改。当前SSH60350 root@89.22.197.55；213旧实例不启队列。

完整24点工作稿：docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL90_Hybrid20_release_20261011.md（a4fd92d065eb78e4cf325dd87f6fa33168380cb4453ce73a62a9015a4d3f795f）。24原话块/38引文行顺序exact，稿固定native/views380/F80、FL90、Hybrid20、gradient64；从保存JSON复制新FedSA六行统计，不重算。non-IID FedSA十seed Full native ACC88.41±0.92%、AEOD0.0078±0.0056、ASPD0.0616±0.0141；去F−Full ΔACC−0.239±1.297pp、ΔAEOD+0.00359±0.01201、ΔASPD+0.00047±0.01507。方向随面板变化和旧F70负结果全部保留，不称普遍必要或显著。FL新S-DFA校准的准确率代价及−2.1684e−19 group_KL诊断差、旧Windows47 exact-refit失败保留；不称跨平台逐位一致。

Git67已验证远端：3ffad7675c82c4ec7a8e29db541c437e93969590，196个committed blobs逐个SHA和remote通过。其固定截止native380/views370/F70、FL90、Hybrid20、14部署及NGA启动；F80/new views10为下一次发布，不改写旧receipt。当前本机证据更近，Git状态只依真实publication receipt。

仍待：剩余机制与新基线完整多seed、三视图验收和IID/non-IID表；FedWA/SmartFL/FedDNA忠实定义；最终主评价端点与冻结final-test决策；匹配提交版LaTeX和正文/rebuttal整合。TableII旧480格追溯中的单次值不能伪造SD。原1390及已完成队列不重跑。当前稿仅作者审阅，未修改提交LaTeX或启用最终测试。

三小时桌面监控最后实际为暂停，当前automation_update不可用，不能声称启用；服务器Supervisor独立继续，无新cron/Windows任务或其他监督机制。大数据、模型、归档留服务器或已检查的F:/YananResearchStorage/GuardFed（Yanan 2TB），不回退内置盘；本机仅代码/配置/必要索引与精简报告。

写入器：tmp/refresh_saved380_F80_editorial_root_20261011.py。本次只更新已验收证据入口，无训练/fit/新科学检查。五入口历史尾部逐字保留。

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
