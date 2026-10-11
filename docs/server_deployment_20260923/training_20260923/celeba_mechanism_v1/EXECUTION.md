# CURRENT: GuardFed返修实验

入口更新：2026-10-11T00:45:53.508890+00:00；最近合并实测：2026-10-11T00:41:52.396973+00:00。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded。先遵守/etc/vast-agents-guide.md。sglang已按授权停止，其他项目不动。

| 阶段 | strict＋离机SHA＋root已采用 | 最近实测（不能代替验收） |
|---|---:|---|
| 机制训练 | 新360/800，Full100另复用 | 360终轮、8活动、432等待、失败0；活动20–54轮 |
| 机制三视图 | U100/C100/A100/F60，共360 | remaining620远端闭合180，离机采用180 |
| FLGMM覆盖 | 新77/96，screen4另复用 | 新83终轮、2活动；三视图81、八完整场景表80，部分1另存 |
| Fed-NGA/Huber搜索 | 全64已验收并按冻结规则选参 | 服务自然退出；扩展192尚未启动 |
| 组合基线覆盖 | 新12/96，screen4另复用 | 新22终轮、1活动；IID Benign三视图10已采用 |

机制native360根证明：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T002530Z/ROOT_DELTA_VERIFICATION.json；新10为non-IID Benign去F，110原归档成员已核验，旧450原记录和49账本前缀不变，现50条链。三视图360根证明：tmp/celeba_mechanism_remaining_F_nonIID_Benign10_root_adoption_20261011/ROOT_ADOPTION.json；旧350对象/顺序不变，新10的原保存数组核90指标/240计数/30规则、native差0，92归档成员和离机SHA通过；零新增训练/拟合/CNN。

F60六场景论文表：docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_six_scenes60_20261011/TABLES.md。五IID场景(alpha5000)加non-IID Benign(alpha5)，每场景10配对；raw/native/shared和固定10/9/6面板、sampleSD(ddof1)。120记录、972均值/SD标量、486显示单元、1080指标和2880计数结构检查通过；旧F50的100记录、810标量、405单元精确保持。新non-IID Benign的native/shared去F−Full在10/9seed是准确率略升且两公平差距略增，在6seed准确率方向反转；raw也保留此反转。IID Sp-DFA/S-DFA和其他场景原取舍保留，不作普遍必要性、显著性或独立因果主张。尚缺四non-IID F场景40和四类控制400，共440条三视图，不能称全部返修完成。

FLGMM表：outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/TABLES.md。non-IID FedSA10 native ACC89.15±0.88%、AEOD0.0428±0.0049、ASPD0.1088±0.0055；shared88.77±0.81%、0.0087±0.0079、0.0689±0.0061，准确率代价保留。原Windows47重拟合失败、资源/传输拒绝和单个groupKL差−2.168404344971009e−19保留；后续Windows审计保存输出零拟合，不称跨平台逐位等价。

梯度搜索64：冻结四条件score均值/精确并列字典序选NGA eta0.001与Huber eta0.3/T0=0.01/M0。Huber27/32、NGA12/32恒负预测，全部负结果、准确率冠军与Pareto保留，n=1无SD/显著性。192新＋8复用覆盖源码已冻结；原真实图像CPU3门检本入口截止7/14单项PASS，整体验收/离机保存/资源复核仍待完成。短门检不是70轮性能或GPU等价证明。FFlip梯度桥接为元数据空效应，初次index0/index4资源拒绝保留，不重试科学失败。

当前24条原话统一英文工作稿：docs/server_deployment_20260923/revision_20260923/rebuttal_working_F60_FL81_gradient64_20261011.md。固定native/views360、F60、FL77新/81三视图/80整场景、gradient64、Hybrid表10；24原话块38行与顺序精确保持，仅供作者审阅，未应用提交版正文。最新Git验证仍为3edbd954d91bbc5e4993546950202645d82ccbb0（529 blob/远端通过），其截止native/views350/F50；本轮F60增量与新稿尚未发布，不能冒称已上传。

Huber恒等投影依用户决定标CNN项目适配，不沿用原理论保证。LoGoFair固定image_id的20虚拟cohort＋官方人口统计平等后处理，标人口适配，不称真实训练client公平性。十方法native1000和九方法三视图900/2052归因已采用；17方法目标仍未齐。FedWA/SmartFL/FedDNA忠实规格、主终点/最终test边界、匹配LaTeX、Fig3来源仍待解决，已有问题不重复问。partition2仅核19962 image_id元数据；未读最终标签/像素、未最终推理。validation选择历史、旧test暴露、cu128/cu130与driver/重放设备差异继续披露。

最近实测双GPU100/100%，67/61°C，6036/5039MiB，RecoveryAction None；cgroup内存76.32/519.17GB，OOM0，服务器磁盘余1.060TB。CPU有界诊断见tmp/cpu_throughput_readonly_20261011/ROOT_DIAGNOSIS.json；未证明最优吞吐，未改并发/数值/实例。

大型证据优先留服务器；本机只按差集取必要产物到F:/YananResearchStorage/GuardFed，写前核Yanan 2TB/Healthy/容量，无内置盘回退。E仅代码/配置/索引/精简报告，模型/数组/数据/归档不进Git。三小时聊天任务guardfed-training-health仍PAUSED，现无automation_update接口；未另建cron/Windows监督，supervisor训练独立继续。

当前写入器：tmp/update_F60_closed_current_entries_20261011.py。旧current前缀另存，本段下方历史原字节保持。

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
